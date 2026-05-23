# Cube Recognition

基于 **AprilTag** 和 **OpenGL** 的立方体识别与增强现实（AR）渲染项目。

通过多个 AprilTag 标签的精确定位，实时解算立方体的 6-DOF 位姿，并利用离屏 OpenGL 渲染将 3D 模型叠加到视频画面上。

---

## 特性

- **多标签位姿融合** — 利用立方体各面贴附的 AprilTag 标签，通过加权平均与图优化迭代，融合多个标签的观测得到精确的立方体位姿
- **离群标签剔除** — 基于平移量的中位数绝对偏差（MAD）自动剔除离群标签观测
- **ESKF 平滑预测** — 内嵌误差状态卡尔曼滤波器（Error State Kalman Filter），对位姿进行平滑与短时预测，应对标签短暂丢失
- **LK 光流跟踪** — 标签短暂丢失时通过 Lucas-Kanade 光流（前后向一致性校验）继续跟踪，提升鲁棒性
- **Hessian 鞍点检测** — 对标签区域进行 Hessian 行列式响应分析，辅助角点精化
- **离屏 AR 渲染** — 基于 OpenGL FBO 的离屏渲染管线，支持加载 OBJ 模型（含材质与纹理）
- **多畸变模型标定** — 配套标定工具，支持 simple / standard / rational 三种畸变模型及迭代离群剔除

---

## 项目结构

```
cube_recognition/
├── main.py              # 主程序入口
├── config.py            # 集中配置文件
├── camera.py            # 摄像头捕获与去畸变
├── cube.py              # 立方体建模与多标签位姿融合
├── tag_detector.py      # AprilTag 检测与角点精化
├── tag_manager.py       # 标签管理与光流跟踪
├── eskf.py              # 误差状态卡尔曼滤波器
├── ar.py                # 离屏 OpenGL AR 渲染器
├── model.py             # OBJ 模型加载与绘制
├── calibration.py       # 相机标定工具
├── pyproject.toml       # 项目配置（uv）
├── requirements.txt     # 依赖锁定
└── README.md
```

---

## 安装

### 环境要求

- Python >= 3.14
- 支持 OpenCL 的显卡（可选，用于加速图像处理）
- 摄像头

### 安装步骤

推荐使用 [uv](https://docs.astral.sh/uv/) 包管理器：

```bash
# 克隆项目
git clone <repository-url>
cd cube_recognition

# 使用 uv 同步依赖
uv sync

# 或使用 pip
pip install -r requirements.txt
```

---

## 使用说明

### 1. 相机标定

首次使用前需对相机进行标定：

```bash
python calibration.py
```

程序会引导选择摄像头，在棋盘格标定板前移动并采集不同角度/距离的图像，覆盖画面的四个象限。标定完成后结果保存在 `camera_calib_result.npz`，并在 `calibration_images/` 下生成标定分析报告。

### 2. 运行主程序

```bash
python main.py
```

主程序将：
1. 打开摄像头并加载标定参数
2. 逐帧检测 AprilTag 标签
3. 根据标签配置解算立方体 6-DOF 位姿
4. 通过 ESKF 进行平滑与预测
5. 将 3D 模型（OBJ）叠加到视频画面并显示

### 3. 配置说明

所有可调参数集中在 `config.py` 中，包括：

| 参数 | 说明 |
|------|------|
| `CAMERA_INDEX` | 摄像头索引 |
| `CAMERA_WIDTH/HEIGHT` | 画面分辨率 |
| `CAMERA_ENABLE_MANUAL_EXPOSURE` | 手动曝光开关 |
| `MARKER_LENGTH` | AprilTag 边长（米） |
| `MARKER_DETECTOR_DICT` | 标签字典类型 |
| `CUBE_SIZE` | 立方体边长（米） |
| `CUBE_PARAMS_DICT` | 立方体各面标签 ID 与轴向配置 |
| `ESKF_*` | 卡尔曼滤波器参数 |
| `ENABLE_*_DEBUG_LOG` | 各模块调试日志开关 |
| `ENABLE_*_IMAGE_DEBUG` | 各模块可视化调试开关 |

---

## 模块说明

### Camera (`camera.py`)
- 使用 MSMF 后端捕获视频流
- 支持构造参数一次性打开摄像头，减少驱动协商延迟
- 可选去畸变（基于预计算的 remap 映射）
- 支持手动曝光控制

### Cube (`cube.py`)
- 根据配置创建立方体模型，计算各面标签到立方体中心的齐次变换矩阵
- `solve_cube_pose()` — 多标签位姿融合，含离群剔除、加权平均和图优化迭代

### TagDetector (`tag_detector.py`)
- 基于 OpenCV 的 AprilTag 检测器（ArucoDetector）
- 检测参数针对 Kalibr 风格标签优化（markerBorderBits=2）
- 可选 OpenCL UMat 加速
- Hessian 鞍点检测用于角点评估

### TagManager (`tag_manager.py`)
- 管理所有已检测标签的生命周期
- 标签超时自动移除
- LK 光流预测（前后向一致性校验 + 面积筛选）

### ESKF (`eskf.py`)
- 19 维误差状态卡尔曼滤波器（位置/速度/角速度/姿态）
- 常速 + 常角速度运动模型
- 速度/角速度阻尼抑制无观测时的发散
- 创新门控机制抑制离群观测
- Joseph form 协方差更新保持数值稳定性

### AR Renderer (`ar.py`)
- 基于 OpenGL GLUT 的隐藏窗口 + FBO 离屏渲染
- 支持背景视频纹理叠加
- 使用相机内参设置透视投影
- 支持任意 OBJ 模型绘制回调

### OBJModel (`model.py`)
- 完整 OBJ 文件解析器
- 支持材质（MTL）与纹理贴图
- 自动构建 VBO/VAO 加速渲染
- 可选归一化到单位包围盒

---

## 标签配置

项目使用 **AprilTag 36h11** 标签族。每个立方体包含 6 个面，每个面贴附一个标签。在 `config.py` 的 `CUBE_PARAMS_DICT` 中配置：

```python
{
    "name": "cube_1",
    "faces": [
        {"id": 7,  "x_axis": "+x", "z_axis": "+z"},  # 前面
        {"id": 5,  "x_axis": "-y", "z_axis": "-x"},  # 左面
        {"id": 21, "x_axis": "-x", "z_axis": "-z"},  # 后面
        {"id": 6,  "x_axis": "+z", "z_axis": "+x"},  # 右面
        {"id": 22, "x_axis": "-z", "z_axis": "+y"},  # 上面
        {"id": 14, "x_axis": "-z", "z_axis": "-y"},  # 下面
    ],
}
```

每个面需指定标签 ID，以及标签坐标系中 x/z 轴对应立方体坐标系的哪个轴。

---

## 依赖

- Python >= 3.14
- OpenCV (opencv-python + opencv-contrib-python >= 4.13)
- PyOpenGL >= 3.10
- NumPy
- Matplotlib（标定工具用）

---

## 版权声明

Copyright (c) 2026. All rights reserved.
