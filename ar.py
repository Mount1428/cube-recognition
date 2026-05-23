# Copyright (c) 2026. All rights reserved.
# This file is part of the cube-recognition project.

import sys
import ctypes
import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GL.ARB.framebuffer_object import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import logging
import config


class AROffscreenRenderer:
    """离屏 AR 渲染器：将 3D 模型叠加到 OpenCV 图像上"""

    def __init__(self, width=640, height=480, camera_matrix=None):
        """
        参数:
            width, height : 渲染画面尺寸（需与输入图像一致）
            camera_matrix : 3x3 内参矩阵; 若为None, 使用 fx=fy=width, cx=width/2, cy=height/2
        """
        self.width = width
        self.height = height

        # 初始化日志
        self._logger: logging.Logger = logging.getLogger("AR")
        self._logger.setLevel(
            logging.DEBUG if config.ENABLE_AR_DEBUG_LOG else logging.INFO
        )

        # 相机内参
        if camera_matrix is None:
            self.fx = self.fy = width
            self.cx, self.cy = width / 2, height / 2
        else:
            self.fx = camera_matrix[0, 0]
            self.fy = camera_matrix[1, 1]
            self.cx = camera_matrix[0, 2]
            self.cy = camera_matrix[1, 2]

        # ---------- 初始化 GLUT 并创建隐藏窗口 ----------
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(width, height)
        self.window = glutCreateWindow(b"AR_Offscreen")

        glutDisplayFunc(lambda: None)

        glutHideWindow()  # 窗口不可见，仅用于 OpenGL 上下文
        glutMainLoopEvent()

        # ---------- 初始化 OpenGL 状态 ----------
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        # ---------- 创建 FBO 及相关资源 ----------
        self._init_fbo()

        # ---------- 创建背景纹理 ----------
        self.bg_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.bg_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None
        )
        glBindTexture(GL_TEXTURE_2D, 0)

        # ---------- 创建背景全屏三角形与 shader ----------
        self.bg_program = None
        self.bg_vao = None
        self.bg_vbo = None
        self.bg_uniforms = {}
        self._init_background_shader()

        # ---------- 设置投影矩阵 ----------
        self._setup_projection()

    def _init_fbo(self):
        """创建 FBO、颜色缓冲纹理、深度缓冲"""
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # 颜色纹理（用于读取渲染结果）
        self.color_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            self.width,
            self.height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_texture, 0
        )

        # 深度渲染缓冲
        self.depth_rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rb)
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.width, self.height
        )
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_rb
        )

        # 检查 FBO 完整性
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("FBO 创建失败")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _setup_projection(self):
        """使用相机内参设置透视投影"""
        near, far = 0.1, 100.0
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        left = -near * self.cx / self.fx
        right = near * (self.width - self.cx) / self.fx
        bottom = -near * self.cy / self.fy
        top = near * (self.height - self.cy) / self.fy
        glFrustum(left, right, bottom, top, near, far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def _convert_readback_rgb_to_bgr(self, raw_bytes: bytes) -> cv2.typing.MatLike:
        """将 OpenGL 回读的 RGB 字节流转换为最终输出 BGR。"""
        img_rgb = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        # OpenGL 原点在左下，且渲染链路需保持镜像；一次切片完成翻转与通道交换
        return np.ascontiguousarray(img_rgb[::-1, ::-1, ::-1])

    def _update_background_texture(self, frame_bgr):
        """将 OpenCV BGR 图像上传到背景纹理（内部转为 RGB）"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        glBindTexture(GL_TEXTURE_2D, self.bg_texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            rgb,
        )
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _init_background_shader(self):
        """初始化背景全屏三角形 shader。"""
        vs_src = """
        #version 120
        attribute vec2 a_pos;
        varying vec2 v_uv;

        void main() {
            v_uv = a_pos * 0.5 + 0.5;
            gl_Position = vec4(a_pos, 0.0, 1.0);
        }
        """

        fs_src = """
        #version 120
        varying vec2 v_uv;
        uniform sampler2D u_bg;

        void main() {
            gl_FragColor = texture2D(u_bg, vec2(v_uv.x, 1.0 - v_uv.y));
        }
        """

        try:
            vs = shaders.compileShader(vs_src, GL_VERTEX_SHADER)
            fs = shaders.compileShader(fs_src, GL_FRAGMENT_SHADER)
            self.bg_program = shaders.compileProgram(vs, fs)

            self.bg_uniforms = {
                "u_bg": glGetUniformLocation(self.bg_program, "u_bg"),
            }

            vertices = np.array(
                [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype=np.float32
            )
            self.bg_vao = glGenVertexArrays(1)
            glBindVertexArray(self.bg_vao)
            self.bg_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.bg_vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            loc = glGetAttribLocation(self.bg_program, "a_pos")
            if loc < 0:
                raise RuntimeError("背景 shader 顶点属性 a_pos 未找到")
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            glUseProgram(self.bg_program)
            glUniform1i(self.bg_uniforms["u_bg"], 0)
            glUseProgram(0)
            self._logger.info("背景全屏三角形 shader 初始化成功")
        except Exception as exc:
            self.bg_program = None
            self.bg_vao = None
            self.bg_vbo = None
            self.bg_uniforms = {}
            self._logger.warning(f"背景 shader 初始化失败，回退立即模式: {exc}")

    def _draw_background(self):
        """绘制全屏背景纹理（shader + 全屏三角形）。"""
        glDisable(GL_DEPTH_TEST)

        if self.bg_program and self.bg_vao:
            glUseProgram(self.bg_program)
            glActiveTexture(GL_TEXTURE0)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.bg_texture)
            glBindVertexArray(self.bg_vao)
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_2D, 0)
            glUseProgram(0)
            glDisable(GL_TEXTURE_2D)
        else:
            # 回退路径，避免无法创建 shader 时直接失效
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, self.width, 0, self.height, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()

            glColor3f(1, 1, 1)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.bg_texture)

            glBegin(GL_QUADS)
            glTexCoord2f(0, 1)
            glVertex3f(0, 0, 1.0)
            glTexCoord2f(1, 1)
            glVertex3f(self.width, 0, 1.0)
            glTexCoord2f(1, 0)
            glVertex3f(self.width, self.height, 1.0)
            glTexCoord2f(0, 0)
            glVertex3f(0, self.height, 1.0)
            glEnd()

            glDisable(GL_TEXTURE_2D)
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

        glEnable(GL_DEPTH_TEST)

    def _apply_pose(self, position, rvec):
        """将 OpenCV 位姿 (位置 + 旋转向量) 应用到模型视图矩阵"""
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32))
        m = np.array(
            [
                R[0, 0],
                R[1, 0],
                R[2, 0],
                0.0,
                R[0, 1],
                R[1, 1],
                R[2, 1],
                0.0,
                R[0, 2],
                R[1, 2],
                R[2, 2],
                0.0,
                position[0],
                position[1],
                position[2],
                1.0,
            ],
            dtype=np.float32,
        )
        glLoadIdentity()
        glMultMatrixf(m)

    def _draw_model(self):
        """绘制一个简单的彩色立方体（可替换）"""
        s = 0.05
        vertices = [
            [-s, -s, -s],
            [s, -s, -s],
            [s, s, -s],
            [-s, s, -s],
            [-s, -s, s],
            [s, -s, s],
            [s, s, s],
            [-s, s, s],
        ]
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 4, 7, 3],
            [1, 5, 6, 2],
            [0, 1, 5, 4],
            [3, 2, 6, 7],
        ]
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
        glBegin(GL_QUADS)
        for i, face in enumerate(faces):
            glColor3fv(colors[i])
            for v in face:
                glVertex3fv(vertices[v])
        glEnd()

    def set_model_drawing_func(self, func):
        """注入自定义无参模型绘制函数"""
        self._draw_model = func

    def render(self, frame_bgr, position, rvec) -> cv2.typing.MatLike:
        """
        离屏渲染一帧。
        参数:
            frame_bgr : OpenCV BGR 图像 (numpy array, shape=(height,width,3))
            position  : (3,) 物体在相机坐标系下的位置 (米)
            rvec      : (3,) OpenCV 旋转向量 (弧度)
        返回:
            numpy array (BGR) - 合成后的图像
        """
        # 1. 更新背景纹理
        frame_bgr = cv2.flip(frame_bgr, 1)
        self._update_background_texture(frame_bgr)

        # 2. 绑定 FBO 并渲染
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 恢复投影矩阵（因为背景绘制时会修改）
        self._setup_projection()

        # 绘制背景
        self._draw_background()

        # 应用位姿并绘制模型
        self._apply_pose(-position, rvec)
        self._draw_model()

        # 3. 同步读取当前帧像素并直接返回，避免异步管线带来的旧帧问题
        pixels = glReadPixels(
            0,
            0,
            self.width,
            self.height,
            GL_RGB,
            GL_UNSIGNED_BYTE,
        )
        output_frame = self._convert_readback_rgb_to_bgr(pixels)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        return output_frame

    def close(self):
        """销毁窗口和 OpenGL 资源（可选）"""
        if self.bg_vbo:
            glDeleteBuffers(1, [self.bg_vbo])
        if self.bg_vao:
            glDeleteVertexArrays(1, [self.bg_vao])
        if self.bg_program:
            glDeleteProgram(self.bg_program)
        if self.bg_texture:
            glDeleteTextures([self.bg_texture])
        if self.color_texture:
            glDeleteTextures([self.color_texture])
        if self.depth_rb:
            glDeleteRenderbuffers(1, [self.depth_rb])
        glutDestroyWindow(self.window)


if __name__ == "__main__":
    def simulate_pose(t):
        """模拟位姿：物体在相机前0.5m，缓慢旋转"""
        # OpenCV 相机坐标系里，物体在相机前方应为正 z
        pos = np.array([0.1 * np.cos(t), 0.1 * np.sin(t), 0.5], dtype=np.float32)
        rvec = np.array([0.0, t, 0.0], dtype=np.float32)  # 绕Y轴旋转
        return pos, rvec


    # 初始化渲染器（尺寸必须与摄像头输出一致）
    renderer = AROffscreenRenderer(width=640, height=480)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    t = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pos, rvec = simulate_pose(t)
        t += 0.03

        # 离屏渲染，返回合成好的 BGR 图像
        result = renderer.render(frame, pos, rvec)

        cv2.imshow("AR Offscreen(opencv)", result)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
            break

    renderer.close()
    cap.release()
    cv2.destroyAllWindows()
