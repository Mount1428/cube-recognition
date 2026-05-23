# Copyright (c) 2026. All rights reserved.
# This file is part of the cube-recognition project.

import numpy as np
import cv2

import config


class ESKF:
    def __init__(self):
        # 状态向量：位置（3维）+ 速度（3维）+ 姿态（SO(3)
        self.position: np.ndarray = np.zeros(3, dtype=np.float32)  # 位置
        self.velocity: np.ndarray = np.zeros(3, dtype=np.float32)  # 速度
        self.angular_velocity: np.ndarray = np.zeros(3, dtype=np.float32)  # 角速度
        self.orientation: np.ndarray = np.eye(3, dtype=np.float32)  # 姿态

        # 误差状态向量：位置误差（3维）+ 速度误差（3维）+ 姿态误差（3维）
        self.error_state: np.ndarray = np.zeros(9, dtype=np.float32)

        # 协方差矩阵
        self.P: np.ndarray = np.eye(9, dtype=np.float32) * np.float32(1e-2)  # 初始协方差
        self.R_base: np.ndarray = np.diag(
            np.array([2e-4, 2e-4, 4e-4, 4e-3, 4e-3, 4e-3], dtype=np.float32) * np.float32(10.0)
        )

        # 过程噪声参数（连续时间谱密度）
        self.sigma_acc: float = self._get_config_float("ESKF_SIGMA_ACC", 1.2)
        self.sigma_gyro: float = self._get_config_float("ESKF_SIGMA_GYRO", 0.8)

        # 速度阻尼可抑制无测量时速度发散
        self.velocity_damping: float = self._get_config_float("ESKF_VELOCITY_DAMPING", 0.2)
        self.angular_velocity_damping: float = self._get_config_float("ESKF_ANGULAR_VELOCITY_DAMPING", 0.2)

        # 创新门控阈值（6维观测在 99% 置信区间约为 16.81）
        self.innovation_gate: float = self._get_config_float("ESKF_INNOVATION_GATE", 16.8)

        self._prev_measurement_orientation: np.ndarray | None = None

    def reset(self):
        """重置滤波器状态和协方差"""
        self.position[:] = 0.0
        self.velocity[:] = 0.0
        self.angular_velocity[:] = 0.0
        self.orientation[:] = np.eye(3, dtype=np.float32)
        self.error_state[:] = 0.0
        self.P[:] = np.eye(9, dtype=np.float32) * np.float32(1e-2)
        self._prev_measurement_orientation = None

    def predict(self, dt: float):
        """状态预测，dt为时间步长"""
        if not np.isfinite(dt) or dt <= 0.0:
            return

        # 防止偶发大 dt 导致模型突变
        dt = float(np.clip(dt, 1e-4, 0.2))
        dt32 = np.float32(dt)

        # 常速 + 常角速度假设
        self.position += self.velocity * dt32
        self.velocity *= np.float32(max(0.0, 1.0 - self.velocity_damping * dt))

        delta_orientation = self._exp_SO3(self.angular_velocity * dt32)
        self.orientation = delta_orientation @ self.orientation
        self.orientation = self._project_to_so3(self.orientation)
        self.angular_velocity *= np.float32(max(0.0, 1.0 - self.angular_velocity_damping * dt))

        # 误差状态预测（线性化）
        F: np.ndarray = np.eye(9, dtype=np.float32)  # 状态转移矩阵
        F[0:3, 3:6] = np.eye(3, dtype=np.float32) * dt32  # 位置误差受速度误差影响

        # 离散过程噪声（CV 模型 + 姿态随机游走）
        q_acc = self.sigma_acc**2
        q_gyro = self.sigma_gyro**2

        Q = np.zeros((9, 9), dtype=np.float32)
        I3 = np.eye(3, dtype=np.float32)
        Q[0:3, 0:3] = (dt32**3 / np.float32(3.0)) * np.float32(q_acc) * I3
        Q[0:3, 3:6] = (dt32**2 / np.float32(2.0)) * np.float32(q_acc) * I3
        Q[3:6, 0:3] = (dt32**2 / np.float32(2.0)) * np.float32(q_acc) * I3
        Q[3:6, 3:6] = dt32 * np.float32(q_acc) * I3
        Q[6:9, 6:9] = dt32 * np.float32(q_gyro) * I3

        # 协方差预测
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(
        self,
        measurement: np.ndarray,
        measurement_dt: float | None = None,
        measurement_noise_scale: float = 1.0,
        measurement_covariance: np.ndarray | None = None,
    ):
        """测量更新，measurement 包含位置和姿态信息。

        当提供 measurement_covariance(6x6) 时，优先直接使用该协方差作为观测噪声R；
        否则退化为基于 measurement_noise_scale 的 R_base 缩放。
        """
        measurement = np.asarray(measurement, dtype=np.float32).reshape(-1)
        if measurement.size != 6 or not np.all(np.isfinite(measurement)):
            return

        meas_orientation = self._exp_SO3(measurement[3:6])
        if (
            measurement_dt is not None
            and measurement_dt > 0.0
            and self._prev_measurement_orientation is not None
        ):
            dt = float(np.clip(measurement_dt, 1e-4, 0.5))
            omega_meas = self._log_SO3(
                meas_orientation @ self._prev_measurement_orientation.T
            ) / np.float32(dt)
            blend = np.float32(0.5)
            self.angular_velocity = (np.float32(1.0) - blend) * self.angular_velocity + blend * omega_meas

        # 计算测量残差
        y: np.ndarray = np.zeros(6, dtype=np.float32)
        y[0:3] = measurement[0:3] - self.position  # 位置残差
        y[3:6] = self._log_SO3(meas_orientation @ self.orientation.T)  # 姿态残差

        # 计算测量矩阵（线性化）
        H: np.ndarray = np.zeros((6, 9), dtype=np.float32)
        H[0:3, 0:3] = np.eye(3, dtype=np.float32)  # 位置测量直接对应位置误差
        H[3:6, 6:9] = np.eye(3, dtype=np.float32)  # 姿态测量直接对应姿态误差

        if measurement_covariance is not None:
            cov = np.asarray(measurement_covariance, dtype=np.float32)
            if cov.shape != (6, 6) or not np.all(np.isfinite(cov)):
                return

            # 对称化并投影到半正定，避免数值误差导致R不可逆
            cov = 0.5 * (cov + cov.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 1e-10, None)
            R = (eigvecs @ np.diag(eigvals) @ eigvecs.T).astype(np.float32)
            R = 0.5 * (R + R.T)
        else:
            scale = float(np.clip(measurement_noise_scale, 0.2, 20.0))
            R = self.R_base * np.float32(scale)

        # 卡尔曼增益
        S: np.ndarray = H @ self.P @ H.T + R

        # 先做创新门控，抑制离群观测引发的跳变
        try:
            innovation_score = float(y.T @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            innovation_score = float("inf")

        if not np.isfinite(innovation_score) or innovation_score > self.innovation_gate:
            return

        try:
            K: np.ndarray = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        # 更新误差状态
        self.error_state = K @ y

        # 更新协方差矩阵
        I_KH: np.ndarray = np.eye(9, dtype=np.float32) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form 保持数值稳定
        self.P = 0.5 * (self.P + self.P.T)

        # 将误差状态应用到实际状态
        self.position += self.error_state[0:3]
        self.velocity += self.error_state[3:6]
        delta_orientation: np.ndarray = self._exp_SO3(self.error_state[6:9])
        self.orientation = delta_orientation @ self.orientation
        self.orientation = self._project_to_so3(self.orientation)

        # ESKF 中误差态每次注入后应清零
        self.error_state[:] = 0.0
        self._prev_measurement_orientation = meas_orientation

    @staticmethod
    def _project_to_so3(R: np.ndarray) -> np.ndarray:
        """用 SVD 将数值误差下的旋转矩阵投影回 SO(3)。"""
        U, _, Vt = np.linalg.svd(R)
        R_proj = U @ Vt
        if np.linalg.det(R_proj) < 0:
            U[:, -1] *= -1
            R_proj = U @ Vt
        return np.asarray(R_proj, dtype=np.float32)

    @staticmethod
    def _get_config_float(name: str, default: float) -> float:
        value = getattr(config, name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        """3维向量的反对称矩阵"""
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float32)

    @staticmethod
    def _exp_SO3(omega: np.ndarray) -> np.ndarray:
        """将 so(3) 向量映射回旋转矩阵。"""
        return np.asarray(cv2.Rodrigues(np.asarray(omega, dtype=np.float32))[0].reshape(3, 3), dtype=np.float32)

    @staticmethod
    def _log_SO3(R: np.ndarray) -> np.ndarray:
        """将旋转矩阵映射到李代数 so(3) 向量 (axis-angle)。"""
        # 用 cv2.Rodrigues 直接得到旋转向量
        return np.asarray(cv2.Rodrigues(np.asarray(R, dtype=np.float32))[0].flatten(), dtype=np.float32)
