# Copyright (c) 2026. All rights reserved.
# This file is part of the cube-recognition project.

from typing import Self

import cv2
import numpy as np
import logging
import time

import config


class Camera:
    __slots__: tuple[str, ...] = (
        "_cap",
        "camera_matrix",
        "dist_coeffs",
        "new_camera_matrix",
        "roi",
        "_map1",
        "_map2",
        "_logger",
    )

    def __init__(self) -> None:
        # 初始化日志
        self._logger: logging.Logger = logging.getLogger("Camera")
        self._logger.setLevel(
            logging.DEBUG if config.ENABLE_CAMERA_DEBUG_LOG else logging.INFO
        )

        t0 = time.perf_counter()

        # 初始化摄像头（MSMF）
        # 优先使用构造参数一次性打开，减少后续多次 set 引发的协商阻塞；
        # 若后端不支持 params，再回退到传统 set。
        constructor_params: list[int | float] = [
            int(cv2.CAP_PROP_FRAME_WIDTH),
            int(config.CAMERA_WIDTH),
            int(cv2.CAP_PROP_FRAME_HEIGHT),
            int(config.CAMERA_HEIGHT),
            int(cv2.CAP_PROP_FPS),
            int(config.CAMERA_FPS),
        ]

        self._cap: cv2.VideoCapture = cv2.VideoCapture(
            config.CAMERA_INDEX,
            cv2.CAP_MSMF,
            constructor_params,
        )

        if not self._cap.isOpened():
            # 回退路径：兼容不支持 constructor params 的 OpenCV/驱动组合
            self._cap = cv2.VideoCapture(
                config.CAMERA_INDEX,
                cv2.CAP_MSMF,
            )
            _ = self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(config.CAMERA_WIDTH))
            _ = self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(config.CAMERA_HEIGHT))
            _ = self._cap.set(cv2.CAP_PROP_FPS, int(config.CAMERA_FPS))

        # 可选手动曝光
        if config.CAMERA_ENABLE_MANUAL_EXPOSURE:
            _ = self.set_manual_exposure(config.CAMERA_EXPOSURE_VALUE)

        self.new_camera_matrix: cv2.typing.MatLike | None = None
        self.roi: tuple[int, int, int, int] | None = None  # 去畸变后的相机矩阵和ROI区域
        self._map1: cv2.typing.MatLike | None = None
        self._map2: cv2.typing.MatLike | None = None

        # 加载相机参数
        calib_data = np.load(config.CAMERA_CALIB_FILE)
        self.camera_matrix: cv2.typing.MatLike | None = calib_data.get("mtx", None)
        self.dist_coeffs: cv2.typing.MatLike | None = calib_data.get("dist", None)

        self._logger.info(
            f"摄像头初始化完成: index={config.CAMERA_INDEX}, resolution=({config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT})"
        )
        self._logger.debug(
            f"摄像头打开耗时: {(time.perf_counter() - t0) * 1000.0:.1f} ms"
        )

        if (
            config.CAMERA_FRAME_DEDISTORTION
            and self.camera_matrix is not None
            and self.dist_coeffs is not None
        ):
            self._logger.info(
                f"相机参数加载成功: camera_matrix:{self.camera_matrix.shape}, dist_coeffs:{self.dist_coeffs.shape}"
            )

            # 计算新的相机矩阵以适应当前帧尺寸
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                min(
                    1.0, max(0.0, config.CAMERA_FRAME_DEDISTORTION_SCALE)
                ),  # 0.0-1.0，0表示完全裁剪，1表示保留全部像素
                (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                centerPrincipalPoint=config.CAMERA_FRAME_DEDISTORTION_CENTER_PRINCIPAL_POINT,  # 保持主点在图像中心
            )

            self._logger.debug(
                f"去畸变参数计算完成: new_camera_matrix:{self.new_camera_matrix.shape}, roi:{self.roi}"
            )
            self._logger.debug(
                f"去畸变参数: new_camera_matrix=\n{self.new_camera_matrix}, roi={self.roi}"
            )

            # 预计算去畸变映射，避免每帧 undistort 的高开销
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.dist_coeffs,
                None,
                self.new_camera_matrix,
                (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                cv2.CV_16SC2,
            )
            self._logger.debug("去畸变 remap 映射已预计算")
        else:
            if not config.CAMERA_FRAME_DEDISTORTION:
                self._logger.info("去畸变已关闭 (CAMERA_FRAME_DEDISTORTION=False)")
            else:
                self._logger.warning(
                    "相机参数缺失，去畸变功能将不可用。请检查校准文件是否正确。"
                )

        # 输出实际生效参数，便于判断驱动协商是否引入初始化延时
        try:
            real_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            real_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            self._logger.info(
                f"摄像头实际参数: resolution=({real_w}x{real_h}), fps={real_fps:.2f}"
            )
        except Exception:
            pass

    def __enter__(self) -> Self:
        if not self._cap.isOpened():
            self._logger.error(f"无法打开摄像头 {config.CAMERA_INDEX}，请检查索引")
            raise IOError(f"无法打开摄像头 {config.CAMERA_INDEX}，请检查索引")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
        self._logger.info("摄像头资源已释放")

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[float, cv2.typing.MatLike]:
        ret, frame = self._cap.read()
        current_time: int = time.monotonic_ns()
        if not ret:
            self._logger.warning("无法读取摄像头帧")
            raise StopIteration

        # 检查是否需要去畸变
        if self._map1 is not None and self._map2 is not None:
            # 使用预计算映射进行去畸变
            frame = cv2.remap(
                frame,
                self._map1,
                self._map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        return (current_time * 1e-9, frame)

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    def use_undistort(self) -> bool:
        return self.new_camera_matrix is not None and self.roi is not None

    def set_manual_exposure(self, exposure_value: float) -> bool:
        """设置手动曝光值，返回是否设置成功。"""
        if not self._cap.isOpened():
            self._logger.warning("相机未打开，无法设置曝光")
            return False

        # 不同后端对 CAP_PROP_AUTO_EXPOSURE 的取值定义不完全一致，逐个尝试。
        # 常见约定：0/1 或 0.25/0.75。
        auto_prop_candidates = [1.0, 0.25, 0.75, 0.0]
        auto_set_ok = False
        for candidate in auto_prop_candidates:
            ok = self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, candidate)
            auto_set_ok = auto_set_ok or bool(ok)

        exp_set_ok = self._cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_value))
        actual = self._cap.get(cv2.CAP_PROP_EXPOSURE)

        if exp_set_ok:
            self._logger.info(
                f"手动曝光已设置: target={float(exposure_value):.3f}, actual={float(actual):.3f}, auto_prop_set={auto_set_ok}"
            )
        else:
            self._logger.warning(
                f"手动曝光设置失败: target={float(exposure_value):.3f}, actual={float(actual):.3f}, auto_prop_set={auto_set_ok}"
            )

        return bool(exp_set_ok)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    with Camera() as cam:
        for timestamp, frame in cam:
            cv2.imshow("Camera Test", frame)
            print(f"Captured frame at {timestamp} ns with shape {frame.shape}")

            print(cam.new_camera_matrix)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
