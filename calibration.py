# Copyright (c) 2026. All rights reserved.
# This file is part of the cube-recognition project.

import cv2
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt
from math import atan2, asin, degrees


# ======================== 辅助函数 ========================
def list_available_cameras(max_index=9):
    available = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def select_camera():
    print("正在扫描可用摄像头...")
    cameras = list_available_cameras()
    if not cameras:
        print("[ERROR] 未检测到任何摄像头，将尝试使用默认摄像头 0")
        return 0
    print("检测到以下可用摄像头索引：")
    for idx in cameras:
        print(f"  [{idx}]")
    print("-" * 40)
    while True:
        user_input = input("请输入要使用的摄像头索引 (直接回车使用 0): ").strip()
        if user_input == "":
            return 0
        try:
            cam_idx = int(user_input)
            if cam_idx in cameras:
                return cam_idx
            else:
                print(f"[WARN] 索引 {cam_idx} 不可用，请从 {cameras} 中选择")
        except ValueError:
            print("[WARN] 请输入整数索引")


def calculate_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def pose_similarity(rvec1, rvec2, angle_thresh_deg=10.0):
    if rvec1 is None or rvec2 is None:
        return False
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_rel = np.dot(R1.T, R2)
    trace_val = (np.trace(R_rel) - 1) / 2
    trace_val = np.clip(trace_val, -1.0, 1.0)
    angle_rad = np.arccos(trace_val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg < angle_thresh_deg


def is_pose_novel(rvec_new, saved_rvecs, angle_thresh_deg=10.0):
    if not saved_rvecs:
        return True
    for rvec_saved in saved_rvecs:
        if pose_similarity(rvec_new, rvec_saved, angle_thresh_deg):
            return False
    return True


def calculate_min_pose_angle(rvec_new, saved_rvecs):
    if not saved_rvecs:
        return 180.0
    min_angle = 180.0
    R_new, _ = cv2.Rodrigues(rvec_new)
    for rvec_saved in saved_rvecs:
        R_saved, _ = cv2.Rodrigues(rvec_saved)
        R_rel = np.dot(R_new.T, R_saved)
        trace_val = (np.trace(R_rel) - 1) / 2
        trace_val = np.clip(trace_val, -1.0, 1.0)
        angle_rad = np.arccos(trace_val)
        angle_deg = np.degrees(angle_rad)
        if angle_deg < min_angle:
            min_angle = angle_deg
    return min_angle


def rotation_vector_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = degrees(atan2(R[2, 1], R[2, 2]))
        yaw = degrees(atan2(-R[2, 0], sy))
        roll = degrees(atan2(R[1, 0], R[0, 0]))
    else:
        pitch = degrees(atan2(-R[1, 2], R[1, 1]))
        yaw = degrees(atan2(-R[2, 0], sy))
        roll = 0.0
    return pitch, yaw, roll


def get_board_center(corners):
    return np.mean(corners, axis=0).flatten()


def get_quadrant(center_x, center_y, width, height):
    cx, cy = width / 2, height / 2
    if center_x < cx and center_y < cy:
        return "UL"
    elif center_x >= cx and center_y < cy:
        return "UR"
    elif center_x < cx and center_y >= cy:
        return "LL"
    else:
        return "LR"


def update_coverage(coverage_set, quadrant):
    coverage_set.add(quadrant)


def coverage_status(coverage_set):
    quads = ["UL", "UR", "LL", "LR"]
    status_str = " ".join([f"{q}:{'Y' if q in coverage_set else 'N'}" for q in quads])
    return status_str

# ======================== 新增：标定全面性分析函数 ========================
def calibrate_with_model(obj_points, img_points, image_size, model_type="rational"):
    """
    使用指定畸变模型进行标定
    model_type: 'simple' (k1,k2,p1,p2), 'standard' (k1,k2,p1,p2,k3), 'rational' (k1..k6)
    返回 (ret, mtx, dist, rvecs, tvecs, per_view_errors)
    """
    h, w = image_size
    init_focal = 1.2 * w
    mtx_init = np.array(
        [[init_focal, 0, w / 2], [0, init_focal, h / 2], [0, 0, 1]], dtype=np.float64
    )

    if model_type == "simple":
        dist_init = np.zeros((1, 4))
        flags = cv2.CALIB_ZERO_TANGENT_DIST  # 不包含切向畸变，实际模型为k1,k2,p1,p2
    elif model_type == "standard":
        dist_init = np.zeros((1, 5))
        flags = cv2.CALIB_FIX_K3  # 开始时不固定k3，但需要启用
        flags = 0
    else:  # rational
        dist_init = np.zeros((1, 8))
        flags = cv2.CALIB_RATIONAL_MODEL

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        (w, h),
        mtx_init,
        dist_init,
        flags=flags,
        criteria=criteria,
    )

    # 计算每张图误差
    per_view_errors = []
    for i in range(len(obj_points)):
        img_pts_proj, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], mtx, dist
        )
        err = cv2.norm(img_points[i], img_pts_proj, cv2.NORM_L2) / len(img_pts_proj)
        per_view_errors.append(err)

    return ret, mtx, dist, rvecs, tvecs, np.array(per_view_errors)


def iterative_outlier_removal(
    obj_points, img_points, image_size, max_iters=3, model="rational"
):
    """迭代剔除重投影误差过大的图像"""
    obj_curr = obj_points.copy()
    img_curr = img_points.copy()
    mtx, dist = None, None

    for i in range(max_iters):
        ret, mtx, dist, rvecs, tvecs, errors = calibrate_with_model(
            obj_curr, img_curr, image_size, model
        )
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        threshold = mean_err + 1.5 * std_err
        keep_idx = [j for j, e in enumerate(errors) if e <= threshold]
        if len(keep_idx) == len(obj_curr):
            break
        obj_curr = [obj_curr[j] for j in keep_idx]
        img_curr = [img_curr[j] for j in keep_idx]
    return mtx, dist, rvecs, tvecs, errors, obj_curr, img_curr


def visualize_pose_distribution(rvecs, tvecs, save_path=None):
    """绘制标定板在相机坐标系中的位置与姿态分布"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # 相机坐标系：原点为相机，Z轴向前
    ax.quiver(0, 0, 0, 50, 0, 0, color="r", label="X")
    ax.quiver(0, 0, 0, 0, 50, 0, color="g", label="Y")
    ax.quiver(0, 0, 0, 0, 0, 100, color="b", label="Z")

    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()
        # 标定板中心位置
        ax.scatter(t[0], t[1], t[2], c="k", marker="o", s=20)
        # 绘制标定板法向量 (Z轴方向)
        normal = R[:, 2] * 30  # 缩放显示
        ax.quiver(
            t[0], t[1], t[2], normal[0], normal[1], normal[2], color="m", length=30
        )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Calibration Board Poses in Camera Frame")
    ax.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def draw_reprojection_errors(image, img_points, proj_points, save_path):
    """在图像上绘制角点检测位置与重投影位置的连线（误差矢量）"""
    vis = image.copy()
    for pt_det, pt_proj in zip(img_points.reshape(-1, 2), proj_points.reshape(-1, 2)):
        pt_det = tuple(np.int32(pt_det))
        pt_proj = tuple(np.int32(pt_proj))
        cv2.circle(vis, pt_det, 3, (0, 255, 0), -1)  # 检测点绿色
        cv2.circle(vis, pt_proj, 3, (0, 0, 255), -1)  # 投影点红色
        cv2.line(vis, pt_det, pt_proj, (255, 0, 0), 1)  # 误差线蓝色
    cv2.imwrite(save_path, vis)


def save_calibration_report(
    report_dir, mtx, dist, errors, model_name, obj_points, img_points, image_size
):
    """保存标定详细报告（文本+图像）"""
    os.makedirs(report_dir, exist_ok=True)
    # 文本报告
    with open(os.path.join(report_dir, f"calib_report_{model_name}.txt"), "w") as f:
        f.write(f"Calibration Report - {model_name} Model\n")
        f.write("=" * 50 + "\n")
        f.write(f"Image size: {image_size}\n")
        f.write(f"Number of images: {len(obj_points)}\n")
        f.write(f"Camera Matrix:\n{mtx}\n")
        f.write(f"Distortion Coefficients:\n{dist.ravel()}\n")
        f.write(f"Mean Reprojection Error: {np.mean(errors):.4f} px\n")
        f.write(f"Std Reprojection Error: {np.std(errors):.4f} px\n")
        f.write(f"Max Error: {np.max(errors):.4f} px (image {np.argmax(errors)})\n")
        f.write(f"Min Error: {np.min(errors):.4f} px (image {np.argmin(errors)})\n")
        f.write("Per-image errors:\n")
        for i, e in enumerate(errors):
            f.write(f"  Img {i:02d}: {e:.4f} px\n")

    # 误差分布直方图
    plt.figure()
    plt.hist(errors, bins=20, edgecolor="black")
    plt.xlabel("Reprojection Error (px)")
    plt.ylabel("Frequency")
    plt.title(f"Error Distribution - {model_name}")
    plt.savefig(os.path.join(report_dir, f"error_hist_{model_name}.png"))
    plt.close()


def full_calibration_analysis(
    obj_points, img_points, image_size, save_dir_base, sample_image_path=None
):
    """
    执行全面标定分析：
      - 尝试多种畸变模型并对比
      - 迭代剔除异常值
      - 生成姿态分布图、误差报告、可视化重投影
    """
    h, w = image_size
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_root = os.path.join(save_dir_base, f"calib_analysis_{timestamp}")
    os.makedirs(report_root, exist_ok=True)

    models = {
        "simple": "4-param (k1,k2,p1,p2)",
        "standard": "5-param (k1,k2,p1,p2,k3)",
        "rational": "8-param Rational",
    }
    best_model = None
    best_mtx = None
    best_dist = None
    best_error = float("inf")
    best_errors = None
    best_obj = None
    best_img = None
    best_rvecs = None
    best_tvecs = None

    print("\n" + "=" * 60)
    print("开始全面标定分析 (多模型对比 + 异常值剔除)")
    print("=" * 60)

    for model_key, model_desc in models.items():
        print(f"\n>>> 尝试模型: {model_desc}")
        mtx, dist, rvecs, tvecs, errors, obj_f, img_f = iterative_outlier_removal(
            obj_points, img_points, image_size, max_iters=3, model=model_key
        )
        mean_err = np.mean(errors)
        print(f"    最终使用图像: {len(obj_f)} 张")
        print(f"    平均重投影误差: {mean_err:.4f} px")

        # 保存该模型的结果
        model_dir = os.path.join(report_root, model_key)
        save_calibration_report(
            model_dir, mtx, dist, errors, model_key, obj_f, img_f, (w, h)
        )

        # 更新最佳模型（按平均误差）
        if mean_err < best_error:
            best_error = mean_err
            best_model = model_key
            best_mtx = mtx
            best_dist = dist
            best_errors = errors
            best_obj = obj_f
            best_img = img_f
            best_rvecs = rvecs
            best_tvecs = tvecs

    # 输出最佳模型
    print("\n" + "=" * 60)
    print(f"最佳标定模型: {best_model} (平均误差 {best_error:.4f} px)")
    print("=" * 60)

    # 保存最佳模型参数
    np.savez(
        os.path.join(report_root, "best_calib_result.npz"),
        mtx=best_mtx,
        dist=best_dist,
        model=best_model,
    )
    # 也保存为默认文件名便于兼容
    np.savez("camera_calib_result.npz", mtx=best_mtx, dist=best_dist)

    # 绘制姿态分布图
    if best_rvecs is not None:
        pose_fig_path = os.path.join(report_root, "pose_distribution.png")
        visualize_pose_distribution(best_rvecs, best_tvecs, save_path=pose_fig_path)
        print(f"姿态分布图已保存至 {pose_fig_path}")

    # 生成重投影可视化（仅对前10张采样，避免文件过多）
    vis_dir = os.path.join(report_root, "reproj_vis")
    os.makedirs(vis_dir, exist_ok=True)
    sample_img_paths = [
        os.path.join(save_dir_base, f"calib_{i+1:02d}.jpg")
        for i in range(len(best_obj))
    ]
    for i in range(min(10, len(best_obj))):
        img_path = sample_img_paths[i]
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            proj_pts, _ = cv2.projectPoints(
                best_obj[i], best_rvecs[i], best_tvecs[i], best_mtx, best_dist
            )
            save_path = os.path.join(vis_dir, f"reproj_img{i:02d}.jpg")
            draw_reprojection_errors(img, best_img[i], proj_pts, save_path)
    print(f"重投影可视化样例已保存至 {vis_dir}")

    # 可选：去畸变对比样例
    if sample_image_path and os.path.exists(sample_image_path):
        test_img = cv2.imread(sample_image_path)
        h_test, w_test = test_img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            best_mtx, best_dist, (w_test, h_test), 1, (w_test, h_test)
        )
        dst = cv2.undistort(test_img, best_mtx, best_dist, None, new_mtx)
        x, y, w_r, h_r = roi
        dst_crop = dst[y : y + h_r, x : x + w_r]
        test_crop = test_img[y : y + h_r, x : x + w_r]
        combined = np.hstack([test_crop, dst_crop])
        comp_path = os.path.join(report_root, "undistort_comparison.jpg")
        cv2.imwrite(comp_path, combined)
        print(f"去畸变对比图保存至 {comp_path}")

    print("\n全面分析报告已生成于:", report_root)
    return best_mtx, best_dist


# ======================== 主标定函数（整合分析模块） ========================
def camera_calibration_realtime():
    # ---------- 用户可调参数 ----------
    pattern_size = (8, 12)
    square_size = 15.0
    save_dir = "./calibration_images"
    required_num = 100

    sharpness_threshold = 200.0
    motion_threshold = 0.1
    capture_cooldown = 1.5
    pose_angle_threshold = 15.0

    min_board_area_ratio = 0.05
    max_board_area_ratio = 0.4

    process_every_n_frames = 3
    resize_scale = 0.5
    # --------------------------------

    camera_index = select_camera()
    print(f"正在打开摄像头索引 {camera_index} ...")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    img_points = []
    saved_rvecs = []
    saved_centers = []
    quadrant_coverage = set()

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开摄像头索引 {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头原生分辨率: {width} x {height}")

    cv2.namedWindow("Camera Calibration - Full Feedback", cv2.WINDOW_NORMAL)

    print("=" * 50)
    print("相机内参标定 - 增强版 (采集阶段)")
    print("=" * 50)

    captured_count = 0
    save_image_idx = 0
    auto_mode = False
    last_capture_time = 0
    frame_counter = 0
    prev_gray_small = None
    motion_threshold_scaled = motion_threshold * resize_scale

    pose_feedback_text = ""
    pose_feedback_color = (255, 255, 255)
    current_pitch = current_yaw = current_roll = 0.0
    current_quadrant = ""
    coverage_text = "Coverage: UL:N UR:N LL:N LR:N"
    suggestion_text = ""

    # ---------- 实时采集循环 (与前版一致，略作整合) ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        sharpness = calculate_sharpness(gray)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            flags += cv2.CALIB_CB_EXHAUSTIVE

        ret_corners, corners = None, None
        if hasattr(cv2, "findChessboardCornersSB"):
            ret_corners, corners = cv2.findChessboardCornersSB(
                gray, pattern_size, flags + cv2.CALIB_CB_ACCURACY
            )
        else:
            ret_corners, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        corners_sub = None
        rvec_est = None
        board_center = None
        board_area_ratio = 0.0

        if ret_corners:
            criteria_subpix = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                1e-6,
            )
            corners_sub = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), criteria_subpix
            )
            cv2.drawChessboardCorners(
                display_frame, pattern_size, corners_sub, ret_corners
            )

            board_center = get_board_center(corners_sub)
            current_quadrant = get_quadrant(board_center[0], board_center[1], w, h)

            hull = cv2.convexHull(corners_sub).reshape(-1, 2)
            board_area = cv2.contourArea(hull)
            board_area_ratio = board_area / (w * h)

        motion = 0.0
        is_novel = True
        is_good_size = min_board_area_ratio <= board_area_ratio <= max_board_area_ratio

        if auto_mode and ret_corners:
            frame_counter += 1
            if frame_counter % process_every_n_frames == 0:
                gray_small = cv2.resize(gray, (0, 0), fx=resize_scale, fy=resize_scale)
                if prev_gray_small is not None:
                    diff = cv2.absdiff(gray_small, prev_gray_small)
                    motion = np.mean(diff)
                else:
                    motion = 0.0
                prev_gray_small = gray_small

                if (
                    sharpness >= sharpness_threshold
                    and motion < motion_threshold_scaled
                    and is_good_size
                ):
                    if len(obj_points) >= 2:
                        ret_temp, mtx_temp, dist_temp, _, _ = cv2.calibrateCamera(
                            obj_points, img_points, (w, h), None, None
                        )
                        if ret_temp:
                            _, rvec_est, _ = cv2.solvePnP(
                                objp, corners_sub, mtx_temp, dist_temp
                            )
                            if rvec_est is not None:
                                current_pitch, current_yaw, current_roll = (
                                    rotation_vector_to_euler(rvec_est)
                                )
                            min_angle = calculate_min_pose_angle(rvec_est, saved_rvecs)
                            if min_angle < pose_angle_threshold:
                                pose_feedback_text = (
                                    f"[REPEAT] Pose angle: {min_angle:.1f} deg"
                                )
                                pose_feedback_color = (0, 0, 255)
                                is_novel = False
                            else:
                                pose_feedback_text = (
                                    f"[NOVEL]  Pose angle: {min_angle:.1f} deg"
                                )
                                pose_feedback_color = (0, 255, 0)
                                is_novel = True
                    else:
                        pose_feedback_text = "[WAIT] Need at least 2 baseline images"
                        pose_feedback_color = (0, 255, 255)
                        is_novel = True
                else:
                    if not is_good_size:
                        pose_feedback_text = "[WARN] Board size unsuitable"
                        pose_feedback_color = (0, 165, 255)

        coverage_text = "Coverage: " + coverage_status(quadrant_coverage)
        missing = [q for q in ["UL", "UR", "LL", "LR"] if q not in quadrant_coverage]
        suggestion_text = (
            f"Move board to {missing[0]} corner" if missing else "Coverage OK"
        )

        if auto_mode and ret_corners and captured_count < required_num:
            current_time = time.time()
            if current_time - last_capture_time >= capture_cooldown:
                if (
                    sharpness >= sharpness_threshold
                    and motion < motion_threshold_scaled
                    and is_good_size
                    and is_novel
                ):
                    obj_points.append(objp)
                    img_points.append(corners_sub)
                    captured_count += 1
                    if rvec_est is not None:
                        saved_rvecs.append(rvec_est)
                    center_norm = (board_center[0] / w, board_center[1] / h)
                    saved_centers.append(center_norm)
                    update_coverage(quadrant_coverage, current_quadrant)
                    last_capture_time = current_time
                    print(f"[AUTO] 采集第 {captured_count} 张 (清晰度:{sharpness:.1f})")
                    img_path = os.path.join(save_dir, f"calib_{captured_count:02d}.jpg")
                    cv2.imwrite(img_path, frame)
                    pose_feedback_text = f"[CAPTURED] {captured_count}/{required_num}"
                    pose_feedback_color = (0, 255, 255)

        # 界面显示 ...
        mode_str = "AUTO" if auto_mode else "MANUAL"
        info_lines = [
            f"Mode: {mode_str} | Captured: {captured_count}/{required_num}",
            f"Sharpness: {sharpness:.1f} | Motion: {motion:.2f} | Area: {board_area_ratio:.3f}",
            f"Pitch: {current_pitch:6.1f}  Yaw: {current_yaw:6.1f}  Roll: {current_roll:6.1f}",
            f"Quadrant: {current_quadrant} | {coverage_text}",
        ]
        if pose_feedback_text:
            info_lines.append(pose_feedback_text)
        if auto_mode:
            info_lines.append(f"Suggestion: {suggestion_text}")
            info_lines.append("Press 'a' to switch to MANUAL")
        else:
            info_lines.append("Press SPACE to capture, 'a' for AUTO")

        for i, text in enumerate(info_lines):
            color = (255, 255, 255)
            if i == 2:
                color = (255, 255, 0)
            elif i == 3:
                color = (0, 255, 255)
            elif i >= 4 and pose_feedback_text and i == 4:
                color = pose_feedback_color
            else:
                color = (0, 255, 0) if auto_mode else (255, 255, 255)
            cv2.putText(
                display_frame,
                text,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("Camera Calibration - Full Feedback", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a"):
            auto_mode = not auto_mode
            mode_name = "AUTO" if auto_mode else "MANUAL"
            print(f"[MODE] 切换到{mode_name}模式")
            pose_feedback_text = f"Switched to {mode_name} mode"
            pose_feedback_color = (255, 255, 0)
            prev_gray_small = None
        elif key == ord("s"):
            save_image_idx += 1
            img_path = os.path.join(save_dir, f"snapshot_{save_image_idx:02d}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[SNAPSHOT] 已保存: {img_path}")
        elif key == ord(" ") and ret_corners and not auto_mode:
            obj_points.append(objp)
            img_points.append(corners_sub)
            captured_count += 1
            if len(obj_points) >= 2:
                ret_temp, mtx_temp, dist_temp, _, _ = cv2.calibrateCamera(
                    obj_points, img_points, (w, h), None, None
                )
                if ret_temp:
                    _, rvec_temp, _ = cv2.solvePnP(
                        objp, corners_sub, mtx_temp, dist_temp
                    )
                    saved_rvecs.append(rvec_temp)
            if board_center is not None:
                center_norm = (board_center[0] / w, board_center[1] / h)
                saved_centers.append(center_norm)
                update_coverage(quadrant_coverage, current_quadrant)
            print(f"[MANUAL] 采集第 {captured_count} 张")
            img_path = os.path.join(save_dir, f"calib_{captured_count:02d}.jpg")
            cv2.imwrite(img_path, frame)

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) == 0:
        print("[ERROR] 无有效图像，退出。")
        return

    # ========== 调用全面分析模块 ==========
    sample_img = os.path.join(save_dir, "calib_01.jpg")
    full_calibration_analysis(
        obj_points, img_points, (h, w), save_dir, sample_image_path=sample_img
    )

    print("\n标定与全面分析流程结束。")


if __name__ == "__main__":
    camera_calibration_realtime()
