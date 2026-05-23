# Copyright (c) 2026. All rights reserved.
# This file is part of the cube-recognition project.

import numpy as np
import cv2
import logging

import config


class TagDetector:

    def __init__(self, camera_matrix):
        # 初始化日志
        self.logger = logging.getLogger("TagDetector")
        self.logger.setLevel(
            logging.DEBUG if config.ENABLE_TAG_DETECTOR_DEBUG_LOG else logging.INFO
        )

        # 加载相机参数
        self.camera_matrix = camera_matrix

        # 初始化 AprilTag 检测器
        self.detector: cv2.aruco.ArucoDetector = self.__create_detector()

        self._use_umat = cv2.ocl.haveOpenCL()  # 或手动控制
        self._buf = {}  # 缓存预分配的 ndarray/UMat

    def __create_detector(self) -> cv2.aruco.ArucoDetector:
        aruco_dict = cv2.aruco.getPredefinedDictionary(config.MARKER_DETECTOR_DICT)
        params: cv2.aruco.DetectorParameters = cv2.aruco.DetectorParameters()
        params.markerBorderBits = 2  # Kalibr风格标签必须设为2
        params.useAruco3Detection = True
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 32
        params.adaptiveThreshWinSizeStep = 8
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 6.0
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        detector: cv2.aruco.ArucoDetector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return detector

    def __get_hassian_mask(self, gray: cv2.typing.MatLike):
        KERNEL_SIZE: int = 15

        # 将输入转为对应后端类型
        if isinstance(gray, cv2.UMat):
            src = gray
        else:
            src = cv2.UMat(gray) if self._use_umat else gray

        # 归一化到 [0,1] (float32)
        img_float = cv2.multiply(src, 1.0 / 255.0, dtype=cv2.CV_32F)

        # 按需创建/重用缓冲区
        h, w = gray.shape[:2]
        for name in ("Ix", "Iy", "Ixx", "Iyy", "Ixy", "det_H"):
            if name not in self._buf or self._buf[name].shape[:2] != (h, w):
                self._buf[name] = np.empty((h, w), dtype=np.float32)

        # 计算一阶导
        Ix = cv2.Sobel(
            img_float, cv2.CV_32F, 1, 0, ksize=KERNEL_SIZE, dst=self._buf["Ix"]
        )
        Iy = cv2.Sobel(
            img_float, cv2.CV_32F, 0, 1, ksize=KERNEL_SIZE, dst=self._buf["Iy"]
        )
        # 计算二阶导和混合导
        Ixx = cv2.Sobel(Ix, cv2.CV_32F, 1, 0, ksize=KERNEL_SIZE, dst=self._buf["Ixx"])
        Iyy = cv2.Sobel(Iy, cv2.CV_32F, 0, 1, ksize=KERNEL_SIZE, dst=self._buf["Iyy"])
        Ixy = cv2.Sobel(Ix, cv2.CV_32F, 0, 1, ksize=KERNEL_SIZE, dst=self._buf["Ixy"])

        # 计算Hessian行列式 det(H) = Ixx*Iyy - Ixy^2
        det_H = cv2.multiply(Ixx, Iyy, dst=self._buf['det_H'])
        cv2.subtract(det_H, cv2.multiply(Ixy, Ixy), dst=det_H)

        # 鞍点掩膜：det(H) < 0
        saddle_mask_umat = cv2.compare(det_H, 0, cv2.CMP_LT)  # 返回 0 或 255 的 uint8 掩膜

        # 响应强度 |det(H)|
        response = cv2.absdiff(det_H, 0.0)  # 逐元素绝对值

        # 找到最大值
        if self._use_umat:
            _, max_resp, _, _ = cv2.minMaxLoc(response)
            max_resp = float(max_resp[0]) if isinstance(max_resp, tuple) else max_resp
        else:
            max_resp = np.max(response)

        # 归一化响应
        if max_resp <= 0:
            self.logger.warning("Hessian响应全为零，无法检测鞍点")
            response_norm_umat = cv2.multiply(response, 0.0)  # 全零
        else:
            response_norm_umat = cv2.divide(response, max_resp)

        # 调试显示（需要时转为 numpy）
        if config.ENABLE_TAG_DETECTOR_HASSIAN_IMAGE_DEBUG:
            cv2.imshow(
                "hessian_response",
                response_norm_umat.get() if self._use_umat else response_norm_umat,
            )

        # 最终下载为 numpy 数组，保持后续代码兼容
        if self._use_umat:
            response_norm = response_norm_umat.get()
            saddle_mask = saddle_mask_umat.get().astype(bool)
        else:
            response_norm = response_norm_umat
            saddle_mask = saddle_mask_umat.astype(bool)

        return response_norm, saddle_mask

    def find_roi_cross_points(self, response_norm, strong_mask):
        """贪心最大响应 + 圆形抑制，最多返回4个点，与原始排序NMS完全等价"""
        if not np.any(strong_mask):
            return None

        roi_vals = response_norm[strong_mask]
        max_roi = np.max(roi_vals)
        if max_roi <= 0:
            return None

        # 动态阈值
        factor = max(max_roi, 0.01)
        abs_thresh = config.CORNER_CROSS_DETECT_THRESHOLD * factor

        # 构建满足阈值的掩膜
        mask = strong_mask & (response_norm > abs_thresh)
        if not np.any(mask):
            return None

        mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
        resp = response_norm
        pts = []
        min_dist = int(config.CORNER_CROSS_DETECT_MIN_DISTANCE)

        for _ in range(4):
            # 检查掩膜是否为空
            if cv2.countNonZero(mask_u8) == 0:
                break
            _, max_val, _, max_loc = cv2.minMaxLoc(resp, mask=mask_u8)
            if max_val < abs_thresh:
                break
            pts.append(max_loc)          # (x, y)
            # 抑制以该点为中心的圆形邻域
            cv2.circle(mask_u8, max_loc, min_dist, 0, -1)

        return pts if pts else None

    def detect_tags(
        self, frame: cv2.typing.MatLike
    ) -> list[dict[str, int | np.ndarray]] | None:
        gray: cv2.typing.MatLike = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(frame.shape) == 3 and frame.shape[2] == 3
            else frame
        )

        # 图像预处理 高斯模糊 + 锐化
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        preprocessed = cv2.filter2D(cv2.GaussianBlur(gray, (3, 3), 0), -1, kernel)

        # 计算Hessian响应和鞍点掩膜
        response_norm, strong_mask = self.__get_hassian_mask(preprocessed)

        # 查找AprilTag标签
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            self.logger.debug("未检测到任何标签")
            return None

        ret_corners: list[dict[str, int | np.ndarray]] = []
        for corner, id in zip(corners, ids):
            # 计算标签中心点坐标
            corner = corner.reshape(4, 2)
            p1, p2, p3, p4 = corner

            center: np.ndarray = np.mean(corner, axis=0)  # 使用四个角点的平均值作为中心
            try:
                # [AB, -CD] @ [t, u]^T = C - A
                t, u = np.linalg.solve(np.array([p3 - p1, p2 - p4]).T, (p2 - p1))

                # 判断中心点是否在标签内部
                if 0 <= t <= 1 and 0 <= u <= 1:
                    center = p1 + t * (p3 - p1)
                else:
                    self.logger.warning(f"标签 {id[0]} 的中心点不在标签内部")

                    raise ValueError("中心点不在标签内部")
            except:
                self.logger.warning(
                    f"标签 {id[0]} 的中心点计算失败，使用平均值作为中心"
                )

            # 扩展ROI区域
            expand_corner_roi = [
                center + (c - center) * config.CORNER_EXTERNAL_EXPAND_SCALE
                for c in corner
            ]
            
            reduce_corner_roi = [
                center + (c - center) * config.CORNER_EXTERNAL_REDUCE_SCALE
                for c in corner
            ]

            # 生成 ROI 掩码
            roi_mask = np.zeros(preprocessed.shape, dtype=np.uint8)

            roi_mask = cv2.fillPoly(
                roi_mask, [np.array(expand_corner_roi, dtype=np.int32)], 255
            )
            roi_mask = cv2.fillPoly(
                roi_mask, [np.array(reduce_corner_roi, dtype=np.int32)], 0
            )

            # 在 expand_corner_roi 和 reduce_corner_roi 之间的区域内寻找十字角点
            cross_points = self.find_roi_cross_points(
                response_norm, strong_mask & (roi_mask == 255)
            )

            if cross_points is not None:
                # 进行亚像素优化
                winSize = (5, 5)  # 搜索窗口为 11x11
                zeroZone = (-1, -1)  # 无死区
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.0001,
                )  # 迭代50次或精度达0.0001

                cross_points = cv2.cornerSubPix(
                    preprocessed,
                    np.array(cross_points, dtype=np.float32),
                    winSize,
                    zeroZone,
                    criteria,
                )

                # 计算替换距离限制
                # 取标签边长的对角线长度平均值的八分之一作为最大替换距离
                max_distance = 0.25 * np.mean(
                    [np.linalg.norm(p2 - p1), np.linalg.norm(p3 - p4)]
                )
                self.logger.debug(
                    f"标签 {id[0]} 的最大替换距离: {max_distance:.2f} 像素"
                )

                # 就近替换标签角点
                for subfix_point in cross_points:
                    distances = np.linalg.norm(corner - subfix_point, axis=1)
                    min_idx = np.argmin(distances)
                    if distances[min_idx] < max_distance:
                        corner[min_idx] = subfix_point
                    else:
                        self.logger.warning(
                            f"标签 {id[0]} 的亚像素角点与原角点距离过大，未替换"
                        )

            # 将优化后的角点添加到结果中
            ret_corners.append({"id": id[0], "corners": corner})

        return ret_corners if len(ret_corners) > 0 else None

    @staticmethod
    def solve_pnp(
        image_points, camera_matrix, dist_coeffs=None
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """位姿解算，主方案为 IPPE，若重投影误差过大则回退到 RANSAC+P3P"""
        image_points = np.asarray(image_points, dtype=np.float32).reshape(4, 2)

        half_length = config.MARKER_LENGTH / 2.0
        object_points = np.array(
            [
                [-half_length, half_length, 0],  # 左上
                [half_length, half_length, 0],  # 右上
                [half_length, -half_length, 0],  # 右下
                [-half_length, -half_length, 0],  # 左下
            ],
            dtype=np.float32,
        )

        # ---------- 1. 优先使用 IPPE（正方形优化）----------
        solution_num, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        best_rvec = None
        best_tvec = None
        best_err = float("inf")

        if solution_num > 0:
            # 筛选法向量朝向相机的候选
            def has_negative_projection_on_camera_z(rvec):
                R, _ = cv2.Rodrigues(rvec)
                tag_z = R[:, 2]
                return float(np.dot(tag_z, np.array([0.0, 0.0, 1.0]))) < 0.0

            candidates = []
            for rv, tv, err in zip(rvecs, tvecs, reproj_errors):
                err_val = float(np.asarray(err).reshape(-1)[0])
                if has_negative_projection_on_camera_z(rv):
                    candidates.append((rv, tv, err_val))

            if candidates:
                best_rvec, best_tvec, best_err = min(candidates, key=lambda x: x[2])
                # 非线性精化
                best_rvec, best_tvec = cv2.solvePnPRefineLM(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    best_rvec,
                    best_tvec,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        100,
                        1e-6,
                    ),
                )
                # 重新计算精化后的重投影误差
                proj, _ = cv2.projectPoints(
                    object_points, best_rvec, best_tvec, camera_matrix, dist_coeffs
                )
                best_err = (
                    cv2.norm(image_points, proj.reshape(-1, 2), cv2.NORM_L2) / 4.0
                )

        # ---------- 2. 若 IPPE 结果误差过大或失败，启用 RANSAC 鲁棒估计 ----------
        RANSAC_THRESH = 2.5  # 像素阈值，可调
        if best_rvec is None or best_err > RANSAC_THRESH:
            # 尝试用 RANSAC + P3P 估计，对离群角点鲁棒
            try:
                _, rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    iterationsCount=100,
                    reprojectionError=RANSAC_THRESH,
                    flags=cv2.SOLVEPNP_P3P,
                )
                if inliers is not None and len(inliers) >= 3:
                    # 用内点进一步精化
                    rvec_ransac, tvec_ransac = cv2.solvePnPRefineLM(
                        object_points[inliers],
                        image_points[inliers],
                        camera_matrix,
                        dist_coeffs,
                        rvec_ransac,
                        tvec_ransac,
                        criteria=(
                            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                            100,
                            1e-6,
                        ),
                    )
                    proj, _ = cv2.projectPoints(
                        object_points,
                        rvec_ransac,
                        tvec_ransac,
                        camera_matrix,
                        dist_coeffs,
                    )
                    err_ransac = (
                        cv2.norm(image_points, proj.reshape(-1, 2), cv2.NORM_L2) / 4.0
                    )
                    if err_ransac < best_err:
                        best_rvec, best_tvec, best_err = (
                            rvec_ransac,
                            tvec_ransac,
                            err_ransac,
                        )
            except cv2.error:
                pass  # RANSAC 失败则保留原结果

        if best_rvec is None:
            return None

        return best_rvec, best_tvec, best_err


if __name__ == "__main__":
    import camera

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    with camera.Camera() as cam:
        tag_detector = TagDetector(cam.camera_matrix)

        for timestamp, frame in cam:
            result = tag_detector.detect_tags(frame)
            if result is not None:
                print(f"检测到标签: {result}")

                # 可视化检测结果
                for item in result:
                    id = item["id"]
                    corners = item["corners"].astype(int)
                    cv2.polylines(
                        frame,
                        [corners],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                    cv2.putText(
                        frame,
                        f"ID:{id}",
                        tuple(corners[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    # pnp位姿估计
                    half_length = config.MARKER_LENGTH / 2.0
                    object_points = np.array(
                        [
                            [-half_length, half_length, 0],  # 左上
                            [half_length, half_length, 0],  # 右上
                            [half_length, -half_length, 0],  # 右下
                            [-half_length, -half_length, 0],  # 左下
                        ],
                        dtype=np.float32,
                    )
                    image_points = np.array(item["corners"], dtype=np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        cam.new_camera_matrix,
                        None,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,
                    )
                    if not success:
                        print(f"无法估计标签 {id} 的位姿")
                    else:
                        cv2.drawFrameAxes(
                            frame,
                            cam.new_camera_matrix,
                            None,
                            rvec,
                            tvec,
                            config.MARKER_LENGTH * 0.5,
                        )
            else:
                print("未检测到标签")

            cv2.imshow("Tag Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
