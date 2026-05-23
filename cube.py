# Copyright (c) 2026. All rights reserved.
# This file is part of the cube-recognition project.

import cv2
import numpy as np

import config


class Cube:
    _AXIS_MAP: dict[str, tuple[int, int, int]] = {
        "+x": (1, 0, 0),
        "-x": (-1, 0, 0),
        "+y": (0, 1, 0),
        "-y": (0, -1, 0),
        "+z": (0, 0, 1),
        "-z": (0, 0, -1),
    }

    def __init__(self, params: dict[str, str | list[dict[str, int | str]]]) -> None:
        self.name: str = params.get("name", "Unknown")

        self.faces: dict[int, dict[str, str | np.ndarray]] = {}
        for face_param in params["faces"]:
            # 根据配置文件中的面的轴处于正方体的轴位置，创建 Face 到 Cube 的齐次变换矩阵
            transform = self._create_face_transform(
                face_param["x_axis"], face_param["z_axis"]
            )
            self.faces[face_param["id"]] = {
                "x_axis": face_param["x_axis"],
                "z_axis": face_param["z_axis"],
                "transform": transform,
                "inverse_transform": np.linalg.inv(transform),
            }


    def parse_axis(self, s: str) -> np.ndarray:
        return np.array(self._AXIS_MAP[s], dtype=np.float32)

    def _create_face_transform(self, x_axis: str, z_axis: str) -> np.ndarray:
        # 创建一个 4x4 的齐次变换矩阵，表示从标签坐标系到正方体坐标系的变换
        transform = np.eye(4, dtype=np.float32)

        vx = self.parse_axis(x_axis)
        vz = self.parse_axis(z_axis)
        vy = np.cross(vz, vx)  # 右手系：z × x

        transform[0:3, 0:3] = np.column_stack([vx, vy, vz])
        transform[0:3, 3] = vz * (
            config.CUBE_SIZE / 2.0
        )  # 标签位于正方体面中心，沿z轴方向偏移半个边长

        return transform

    def solve_cube_pose(
            self,
            tag_poses: list[tuple[int, np.ndarray, np.ndarray, float]],
            max_iter: int = 5,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        多标签位姿融合，带离群标签剔除与视角质量加权
        tag_poses: 列表元素为 (tag_id, rvec, tvec, reprojection_error)
        """
        if len(tag_poses) == 0:
            return None

        # ---------- 1. 将每个标签位姿转换到正方体坐标系 ----------
        cube_pose_list: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
        for tag_id, rvec, tvec, err in tag_poses:
            face_info = self.faces.get(tag_id)
            if face_info is None:
                continue

            rvec_f32 = np.asarray(rvec, dtype=np.float32).reshape(3)
            tvec_flat = np.asarray(tvec, dtype=np.float32).reshape(3)
            R, _ = cv2.Rodrigues(rvec_f32)
            R = np.asarray(R, dtype=np.float32)

            tag_to_camera = np.eye(4, dtype=np.float32)
            tag_to_camera[0:3, 0:3] = R
            tag_to_camera[0:3, 3] = tvec_flat

            tag_to_camera_inv = np.eye(4, dtype=np.float32)
            tag_to_camera_inv[0:3, 0:3] = R.T
            tag_to_camera_inv[0:3, 3] = -R.T @ tvec_flat

            cube_pose = tag_to_camera @ face_info["inverse_transform"]
            cube_pose_list.append(
                (
                    cube_pose,
                    face_info["transform"] @ tag_to_camera_inv,
                    float(err),
                    self._log_SO3(cube_pose[0:3, 0:3]),
                )
            )

        if len(cube_pose_list) == 0:
            return None

        cube_poses: np.ndarray = np.stack([item[0] for item in cube_pose_list], axis=0).astype(np.float32)
        cube_pose_inverses: np.ndarray = np.stack([item[1] for item in cube_pose_list], axis=0).astype(np.float32)
        reproj_errors: np.ndarray = np.array([item[2] for item in cube_pose_list], dtype=np.float32)
        rot_logs: np.ndarray = np.stack([item[3] for item in cube_pose_list], axis=0).astype(np.float32)

        # ---------- 2. 异常值剔除：基于平移量的中值绝对偏差 ----------
        if len(cube_poses) > 1:
            trans = cube_poses[:, :3, 3]
            median_trans = np.median(trans, axis=0)
            dists = np.linalg.norm(trans - median_trans, axis=1)
            mad = np.median(np.abs(dists - np.median(dists)))  # Median Absolute Deviation
            # 阈值：动态计算，最小 2cm，最大 8cm
            thresh = max(0.02, min(0.08, 3.0 * mad)) if mad > 0 else 0.05
            inlier_idx = [i for i, d in enumerate(dists) if d < thresh]
            if len(inlier_idx) >= 1:
                idx = np.asarray(inlier_idx, dtype=np.intp)
                cube_poses = cube_poses[idx]
                cube_pose_inverses = cube_pose_inverses[idx]
                reproj_errors = reproj_errors[idx]
                rot_logs = rot_logs[idx]

        # ---------- 计算融合权重（仅保留重投影误差） ----------
        weights: np.ndarray = 1.0 / (np.square(reproj_errors) + 1e-6)
        weights_sum = float(np.sum(weights))
        if weights_sum < 1e-12:
            weights = np.full(len(cube_poses), 1.0 / len(cube_poses), dtype=np.float32)
        else:
            weights = weights / weights_sum

        # ---------- 4. 加权平均作为初始值 ----------
        weighted_pose: np.ndarray = np.eye(4, dtype=np.float32)
        if len(cube_poses) == 1:
            weighted_pose[:] = cube_poses[0]
        else:
            weighted_pose[0:3, 0:3] = self._exp_SO3(np.sum(weights[:, None] * rot_logs, axis=0))
            weighted_pose[0:3, 3] = np.sum(weights[:, None] * cube_poses[:, 0:3, 3], axis=0)

        # ---------- 5. 图优化迭代（原有逻辑，略作整理） ----------
        def skew(v: np.ndarray) -> np.ndarray:
            return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float32)

        def compute_jacobian(v: np.ndarray, w: np.ndarray) -> np.ndarray:
            ad: np.ndarray = np.zeros((6, 6), dtype=np.float32)
            ad[:3, :3] = skew(w)
            ad[:3, 3:] = skew(v)
            ad[3:, 3:] = skew(w)
            return np.eye(6, dtype=np.float32) + 0.5 * ad

        if len(cube_poses) > 1:
            for _ in range(min(max_iter, 2)):
                hessian: np.ndarray = np.zeros((6, 6), dtype=np.float32)
                gradient: np.ndarray = np.zeros(6, dtype=np.float32)
                for Ti_inv, w in zip(cube_pose_inverses, weights):
                    e = Ti_inv @ weighted_pose
                    w_vec = self._log_SO3(e[0:3, 0:3])
                    theta = np.linalg.norm(w_vec)
                    w_skew = skew(w_vec)

                    if theta < 1e-6:
                        a_mat: np.ndarray = np.eye(3, dtype=np.float32) - 0.5 * w_skew
                    else:
                        a_mat = (
                            np.eye(3, dtype=np.float32)
                            - 0.5 * w_skew
                            + ((1 / theta ** 2) - (1 + np.cos(theta)) / (2 * theta * np.sin(theta)))
                            * w_skew
                            @ w_skew
                        )
                    v = a_mat @ e[0:3, 3]

                    J = compute_jacobian(v, w_vec)
                    hessian += w * J.T @ J
                    gradient += w * J.T @ np.hstack([v, w_vec])

                try:
                    delta = -np.linalg.solve(hessian, gradient).astype(np.float32)
                except np.linalg.LinAlgError:
                    delta = -(np.linalg.pinv(hessian).astype(np.float32) @ gradient)

                v = delta[:3]
                w_vec = delta[3:]
                theta = np.linalg.norm(w_vec)
                w_skew = skew(w_vec)

                if theta < 1e-6:
                    R_mat = np.eye(3, dtype=np.float32) + w_skew
                    v_mat = np.eye(3, dtype=np.float32) + 0.5 * w_skew
                else:
                    theta2 = theta ** 2
                    R_mat = (
                        np.eye(3, dtype=np.float32)
                        + (np.sin(theta) / theta) * w_skew
                        + ((1 - np.cos(theta)) / theta2) * w_skew @ w_skew
                    )
                    v_mat = (
                        np.eye(3, dtype=np.float32)
                        + ((1 - np.cos(theta)) / theta2) * w_skew
                        + ((theta - np.sin(theta)) / (theta ** 3)) * w_skew @ w_skew
                    )
                v = v_mat @ v

                T_delta: np.ndarray = np.eye(4, dtype=np.float32)
                T_delta[0:3, 0:3] = R_mat
                T_delta[0:3, 3] = v
                weighted_pose = T_delta @ weighted_pose

                if np.linalg.norm(delta) < 1e-6:
                    break

        # ---------- 加权样本协方差估计 ----------
        def pose_residual_se3(Ti_inv: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
            e = Ti_inv @ T_ref
            wv = self._log_SO3(e[0:3, 0:3])
            theta = np.linalg.norm(wv)
            w_skew = skew(wv)
            if theta < 1e-6:
                a_mat: np.ndarray = np.eye(3, dtype=np.float32) - 0.5 * w_skew
            else:
                a_mat = (np.eye(3, dtype=np.float32) - 0.5 * w_skew +
                         ((1 / theta ** 2) - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))) * w_skew @ w_skew)
            v = a_mat @ e[0:3, 3]
            return np.hstack([v, wv])

        residual_matrix: np.ndarray = np.vstack([pose_residual_se3(Ti_inv, weighted_pose) for Ti_inv in cube_pose_inverses])
        residual_mean: np.ndarray = np.sum(weights[:, None] * residual_matrix, axis=0)

        scatter: np.ndarray = np.zeros((6, 6), dtype=np.float32)
        for xi, w in zip(residual_matrix, weights):
            diff = xi - residual_mean
            scatter += w * np.outer(diff, diff)

        dof_factor = max(1.0 - float(np.sum(np.square(weights))), 1e-6)
        sample_cov = scatter / dof_factor

        # 物理先验（与之前相同）
        err_scale = float(np.clip(float(np.median(reproj_errors)) / 0.3, 0.5, 20.0))
        tag_count = len(cube_poses)
        trans_std_base = float(min(max(0.0015 * err_scale, 3e-4), 2e-2))
        rot_std_base = float(min(max(np.deg2rad(0.6) * err_scale, np.deg2rad(0.08)), np.deg2rad(8.0)))
        prior_diag = np.array([trans_std_base ** 2] * 3 + [rot_std_base ** 2] * 3, dtype=np.float32) / max(tag_count, 1)

        covariance = sample_cov + np.diag(prior_diag)
        covariance += np.diag(np.array([1e-8] * 3 + [np.deg2rad(0.05) ** 2] * 3, dtype=np.float32))
        covariance = 0.5 * (covariance + covariance.T)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.clip(eigvals, 1e-12, None)
        covariance = (eigvecs @ np.diag(eigvals) @ eigvecs.T).astype(np.float32)
        covariance = 0.5 * (covariance + covariance.T)

        return weighted_pose, covariance

    @staticmethod
    def _log_SO3(R: np.ndarray) -> np.ndarray:
        """将旋转矩阵映射到李代数 so(3) 向量 (axis-angle)。"""
        # 用 cv2.Rodrigues 直接得到旋转向量
        rvec, _ = cv2.Rodrigues(R)
        return np.asarray(rvec, dtype=np.float32).flatten()

    @staticmethod
    def _exp_SO3(omega: np.ndarray) -> np.ndarray:
        """将 so(3) 向量映射回旋转矩阵。"""
        R, _ = cv2.Rodrigues(np.asarray(omega, dtype=np.float32))
        return np.asarray(R, dtype=np.float32)


# 从方向矢量和位置创建齐次坐标
def make_homogeneous_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform: np.ndarray = np.eye(4, dtype=np.float32)
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32))
    transform[0:3, 0:3] = R
    transform[0:3, 3] = np.asarray(tvec, dtype=np.float32).reshape(3)

    return transform


if __name__ == "__main__":
    # 创建一个 Cube 实例，使用配置文件中的参数
    cube = Cube(config.CUBE_PARAMS_DICT[0])
    print(f"Cube Name: {cube.name}")
    for face_id, face_info in cube.faces.items():
        print(
            f"Face ID: {face_id}, X Axis: {face_info['x_axis']}, Z Axis: {face_info['z_axis']}"
        )
        print(f"Transform Matrix:\n{face_info['transform']}\n")
