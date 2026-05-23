# Copyright (c) 2026. All rights reserved.
# This file is part of the cube-recognition project.

import numpy as np
import cv2
from dataclasses import dataclass
import logging

import config


@dataclass(slots=True)
class Tag:
    id: int
    corners: np.ndarray

    last_attach_timestamp: float


class TagManager:

    def __init__(self) -> None:
        # 初始化日志
        self.logger: logging.Logger = logging.getLogger("TagManager")
        self.logger.setLevel(
            logging.DEBUG if config.ENABLE_TAG_MANAGER_DEBUG_LOG else logging.INFO
        )

        self.tag_map: dict[int, Tag] = {}
        self.prev_gray_frame: cv2.typing.MatLike | None = None

    def __call__(
        self,
        timestamp: float,
        frame: cv2.typing.MatLike,
        tags: dict[int, np.ndarray] | None,
    ) -> None:
        # 检查输入图像是否是灰度图像
        gray: cv2.typing.MatLike = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(frame.shape) == 3 and frame.shape[2] == 3
            else frame
        )

        # 更新已识别角点的标签
        if tags:
            for id, corners in tags.items():
                tag = self.tag_map.get(id)
                if tag:
                    tag.last_attach_timestamp = timestamp
                    tag.corners = corners
                else:
                    self.logger.info(f"新增标签 {id} ")
                    self.tag_map[id] = Tag(id, corners, timestamp)

        # 筛除超时标签
        for id in [
            id
            for id, tag in self.tag_map.items()
            if timestamp - tag.last_attach_timestamp > config.MARKER_FLOW_MAX_LOST_TIME
        ]:
            self.logger.info(f"移除标签 {id} ")
            _ = self.tag_map.pop(id)

        # 判断是否有上一帧图像
        if self.prev_gray_frame is not None:
            # 获取需要光流预测的标签
            predict_tag_map = [
                tag
                for tag in self.tag_map.values()
                if 0
                < timestamp - tag.last_attach_timestamp
                < config.MARKER_FLOW_MAX_LOST_TIME
            ]

            # 光流计算参数
            window_size = (31, 31)
            max_level = 3
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

            for tag in predict_tag_map:
                prev_pts = tag.corners.astype(np.float32).reshape(-1, 1, 2)

                # 前向预测
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray_frame,
                    gray,
                    prev_pts,
                    None,
                    winSize=window_size,
                    maxLevel=max_level,
                    criteria=criteria,
                )

                if next_pts is None or status is None or int(status.sum()) != 4:
                    self.logger.warning(f"标签 {tag.id} 光流前向预测失败")
                    continue

                # 反向传播
                back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    self.prev_gray_frame,
                    next_pts,
                    None,
                    winSize=window_size,
                    maxLevel=max_level,
                    criteria=criteria,
                )

                if (
                    back_pts is None
                    or back_status is None
                    or int(back_status.sum()) != 4
                ):
                    self.logger.warning(f"标签 {tag.id} 光流反向传播失败")
                    continue

                # 前后向一致性筛选
                fb_error = np.linalg.norm(
                    back_pts.reshape(-1, 2) - prev_pts.reshape(-1, 2), axis=1
                )
                if float(np.max(fb_error)) > config.TAG_FLOW_FB_ERR_PX:
                    self.logger.warning(f"标签 {tag.id} 光流前后向一致性筛选失败")
                    continue

                # 标签面积筛选
                if self._polygon_area(next_pts) < config.TAG_FLOW_MIN_AREA_PX2:
                    self.logger.warning(f"标签 {tag.id} 光流面积筛选失败")
                    continue

                # 更新预测角点
                tag.corners = np.asarray(next_pts, dtype=np.float32).reshape((4, 2))
                self.tag_map[tag.id] = tag

        self.prev_gray_frame = gray

    def get_tags(self, ids: set[int]) -> list[tuple[int, np.ndarray]]:
        return [(id, tag.corners) for id, tag in self.tag_map.items() if id in ids]

    def get_all_tags(self) -> list[tuple[int, np.ndarray]]:
        return [(id, tag.corners) for id, tag in self.tag_map.items()]

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        """计算四边形面积，用于剔除退化光流结果。"""
        pts: np.ndarray = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        x: np.ndarray = pts[:, 0]
        y: np.ndarray = pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
