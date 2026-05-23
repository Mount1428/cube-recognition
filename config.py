# Copyright (c) 2026. All rights reserved.
# This file is part of the cube-recognition project.

import cv2

# 相机参数
CAMERA_CALIB_FILE: str = "./camera_calib_result.npz"

# 视频流参数
CAMERA_INDEX: int = 1  # 摄像头索引
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720
CAMERA_FPS: int = 60

# 曝光参数
CAMERA_ENABLE_MANUAL_EXPOSURE: bool = False
CAMERA_EXPOSURE_VALUE: float = -6.0# 常见取值范围通常在 [-13, -1]

CAMERA_FRAME_DEDISTORTION: bool = True  # 是否进行图像去畸变
CAMERA_FRAME_DEDISTORTION_CENTER_PRINCIPAL_POINT: bool = (
    True  # 去畸变时是否将主点移至图像中心
)
CAMERA_FRAME_DEDISTORTION_SCALE: float = (
    0  # 去畸变时的缩放因子（1=保持原尺寸，<1=裁剪，>1=保留更多边缘）
)

# 标签识别参数
MARKER_LENGTH: float = 0.031  # 标签的边长（单位：米）
MARKER_DETECTOR_DICT= (
    cv2.aruco.DICT_APRILTAG_36h11
)  # 使用的标签字典

MARKER_FLOW_MAX_LOST_TIME: float = 0.2  # 标签最大光流预测时长

# 角点识别参数
CORNER_EXTERNAL_EXPAND_SCALE: float = 1.2  # 扩展 ROI 的比例（相对于标签边长）
CORNER_EXTERNAL_REDUCE_SCALE: float = 0.8  # 收缩 ROI 的比例（相对于标签边长）

CORNER_CROSS_DETECT_THRESHOLD: float = 0.01  # 十字检测的响应阈值（相对于最大响应的比例）
CORNER_CROSS_DETECT_MIN_DISTANCE: int = 12  # 十字点之间的最小距离（像素）

# 物块参数
CUBE_SIZE: float = 0.04  # 正方体边长（单位：米）
CUBE_PARAMS_DICT: list[dict[str, str | list[dict[str, str | int]]]] = [
    {
        "name": "cube_1",
        "faces": [
            # id | x轴对应正方体的哪个轴 | z轴对应正方体的哪个轴
            {"id": 7, "x_axis": "+x", "z_axis": "+z"},  # 前
            {"id": 5, "x_axis": "-y", "z_axis": "-x"},  # 左
            {"id": 21, "x_axis": "-x", "z_axis": "-z"},  # 后
            {"id": 6, "x_axis": "+z", "z_axis": "+x"},  # 右
            {"id": 22, "x_axis": "-z", "z_axis": "+y"},  # 上
            {"id": 14, "x_axis": "-z", "z_axis": "-y"},  # 下
        ],
    },
    {
        "name": "cube_2",
        "faces": [
            # id | x轴对应正方体的哪个轴 | z轴对应正方体的哪个轴
            {"id": 18, "x_axis": "+x", "z_axis": "+z"},  # 前
            {"id": 17, "x_axis": "-y", "z_axis": "-x"},  # 左
            {"id": 16, "x_axis": "-x", "z_axis": "-z"},  # 后
            {"id": 19, "x_axis": "+z", "z_axis": "+x"},  # 右
            {"id": 9, "x_axis": "+x", "z_axis": "+y"},  # 上
            {"id": 11, "x_axis": "+x", "z_axis": "-y"},  # 下
        ],
    }
]
CUBE_PREDICT_WINDOW_TIME: float = 0.2  # 预测窗口时间（秒）

# 调试参数
ENABLE_CAMERA_DEBUG_LOG: bool = False
ENABLE_TAG_DETECTOR_DEBUG_LOG: bool = False
ENABLE_TAG_MANAGER_DEBUG_LOG: bool = False
ENABLE_AR_DEBUG_LOG: bool = False
ENABLE_MODEL_DEBUG_LOG: bool = False

ENABLE_TAG_DETECTOR_LABEL_IMAGE_DEBUG: bool = False
ENABLE_TAG_DETECTOR_HASSIAN_IMAGE_DEBUG: bool = False
ENABLE_CUBE_POSE_IMAGE_DEBUG: bool = False

# ESKF 参数
ESKF_SIGMA_ACC: float = 1.5
ESKF_SIGMA_GYRO: float = 0.8
ESKF_VELOCITY_DAMPING: float = 0.5
ESKF_INNOVATION_GATE: float = 16.8

# 标签短时丢失补偿（LK 光流）
ENABLE_TAG_FLOW_FALLBACK: bool = True
TAG_FLOW_MAX_LOST_FRAMES: int = 8
TAG_FLOW_FB_ERR_PX: float = 1.5
TAG_FLOW_MIN_AREA_PX2: float = 80.0
TAG_FLOW_NOISE_SCALE_MULT: float = 2.5
