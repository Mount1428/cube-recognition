# Copyright (c) 2026. All rights reserved.
# This file is part of the cube-recognition project.

import cv2
import numpy as np
import time

import logging

import config, cube, camera, tag_detector, eskf
from ar import AROffscreenRenderer
from tag_manager import TagManager

from model import OBJModel


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s [%(name)s](%(asctime)s) %(message)s",
        level=logging.DEBUG,
    )

    # 创建 Cube 实例
    cube_instance = cube.Cube(config.CUBE_PARAMS_DICT[0])
    print(f"Cube Name: {cube_instance.name}")
    for face_id, face_info in cube_instance.faces.items():
        logging.info(
            f"Face ID: {face_id}, X Axis: {face_info['x_axis']}, Z Axis: {face_info['z_axis']}"
        )

    cube_filter = eskf.ESKF()

    with camera.Camera() as cam:
        tag_detector_instance = tag_detector.TagDetector(cam.new_camera_matrix)
        tag_manager = TagManager()
        renderer = AROffscreenRenderer(
            config.CAMERA_WIDTH, config.CAMERA_HEIGHT, cam.new_camera_matrix
        )

        obj_model = OBJModel("models/水豚嘟嘟/1.obj", scale=0.1)
        renderer.set_model_drawing_func(obj_model.draw)

        last_timestamp = None
        attach_timestamp = None
        lost_flag: bool = False
        last_fps_time = time.perf_counter()
        fps_value: float = 0.0
        tag_detect_ms: float = 0.0

        for timestamp, frame in cam:
            if last_timestamp is not None:
                dt: float = timestamp - last_timestamp
                if (
                    attach_timestamp is not None
                    and timestamp - attach_timestamp <= config.CUBE_PREDICT_WINDOW_TIME
                ):
                    cube_filter.predict(dt)
                else:
                    cube_filter.reset()  # 超过预测窗口时间，重置滤波器
                    lost_flag = True

            last_timestamp = timestamp

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            tag_t0 = time.perf_counter()
            tags = tag_detector_instance.detect_tags(frame)
            tag_t1 = time.perf_counter()
            instant_tag_ms = (tag_t1 - tag_t0) * 1000.0
            tag_detect_ms = (
                instant_tag_ms
                if tag_detect_ms <= 0.0
                else (0.9 * tag_detect_ms + 0.1 * instant_tag_ms)
            )

            tags_dict: dict[int, np.ndarray] = {}
            if tags:
                for tag in tags:
                    tags_dict[tag["id"]] = tag["corners"]

            tag_manager(timestamp, gray, tags_dict)

            if config.ENABLE_TAG_DETECTOR_LABEL_IMAGE_DEBUG:
                label_debug_frame = frame.copy()
                for id, corners in tag_manager.get_all_tags():
                    cv2.polylines(
                        label_debug_frame,
                        [corners.astype(int)],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=2,
                    )
                    cv2.putText(
                        label_debug_frame,
                        f"ID: {id}",
                        tuple(corners[0].astype(int) + np.array([0, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Tag Detection Debug", label_debug_frame)

            # 找到所有匹配cube的标签
            cube_tag = tag_manager.get_tags(set(cube_instance.faces))
            if len(cube_tag) > 0:
                # 解算所有标签的位姿
                tag_poses = [
                    (
                        id,
                        tag_detector_instance.solve_pnp(
                            corners,
                            cam.new_camera_matrix,
                        ),
                    )
                    for id, corners in cube_tag
                ]

                # 移除解算失败的标签
                tag_poses = [
                    (p[0], *p[1])  # id, rvec, tvec, err
                    for p in tag_poses
                    if p[1] is not None
                ]

                solves_result = cube_instance.solve_cube_pose(tag_poses)

                if solves_result:
                    cube_pose, solve_cov = solves_result
                    measurement_dt = (
                        timestamp - attach_timestamp if attach_timestamp is not None else None
                    )

                    cube_filter.update(
                        np.hstack(
                            [
                                cube_pose[:3, 3],
                                cv2.Rodrigues(cube_pose[:3, :3])[0].flatten(),
                            ]
                        ),
                        measurement_dt=measurement_dt,
                        measurement_covariance=solve_cov,
                    )

                    attach_timestamp = timestamp
                    lost_flag = False

            # print(cube_filter.position, "|", cube_filter.velocity)

            # 绘制立方体位置
            if config.ENABLE_CUBE_POSE_IMAGE_DEBUG:
                cube_debug_frame = frame.copy()
                if not lost_flag:
                    cube_corners_3d = np.array(
                        [
                            [-0.5, -0.5, 0.5],
                            [0.5, -0.5, 0.5],
                            [0.5, 0.5, 0.5],
                            [-0.5, 0.5, 0.5],
                            [-0.5, -0.5, -0.5],
                            [0.5, -0.5, -0.5],
                            [0.5, 0.5, -0.5],
                            [-0.5, 0.5, -0.5],
                        ],
                        dtype=np.float32,
                    ) * config.CUBE_SIZE

                    cube_corners_2d = cv2.projectPoints(
                        cube_corners_3d,
                        cv2.Rodrigues(cube_filter.orientation)[0],
                        cube_filter.position,
                        cam.new_camera_matrix,
                        None,
                    )[0].reshape(-1, 2).astype(int)

                    # 绘制立方体边框
                    for i in range(4):
                        cv2.line(
                            cube_debug_frame,
                            tuple(cube_corners_2d[i]),
                            tuple(cube_corners_2d[(i + 1) % 4]),
                            (255, 0, 255),
                            2,
                        )
                        cv2.line(
                            cube_debug_frame,
                            tuple(cube_corners_2d[i + 4]),
                            tuple(cube_corners_2d[((i + 1) % 4) + 4]),
                            (255, 0, 255),
                            2,
                        )
                        cv2.line(
                            cube_debug_frame,
                            tuple(cube_corners_2d[i]),
                            tuple(cube_corners_2d[i + 4]),
                            (255, 0, 255),
                            2,
                        )

                cv2.imshow("Cube Pose Debug", cube_debug_frame)

            # 渲染图像
            rendered_frame: cv2.typing.MatLike = (
                renderer.render(
                    frame,
                    cube_filter.position,
                    cv2.Rodrigues(cube_filter.orientation)[0].flatten(),
                )
                if not lost_flag
                else frame.copy()
            )

            # FPS 统计（指数平滑，避免抖动）
            now = time.perf_counter()
            dt_fps = now - last_fps_time
            if dt_fps > 1e-6:
                instant_fps = 1.0 / dt_fps
                fps_value = instant_fps if fps_value <= 0 else (0.9 * fps_value + 0.1 * instant_fps)
            last_fps_time = now

            cv2.putText(
                rendered_frame,
                f"FPS: {fps_value:.1f}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                rendered_frame,
                f"TagDetect: {tag_detect_ms:.2f} ms",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Camera Frame", rendered_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                renderer.close()
                break

    cv2.destroyAllWindows()
