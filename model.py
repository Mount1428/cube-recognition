# Copyright (c) 2026. All rights reserved.
# This file is part of the cube-recognition project.

import os
import ctypes
import shlex
import time
import logging
import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GL import shaders

import config


class OBJModel:
    """OBJ 模型加载与 VBO 绘制（固定管线），支持缩放、基础材质和纹理"""

    def __init__(self, obj_path, mtl_dir="", scale=1.0, normalize=True):
        """
        obj_path  : OBJ 文件路径
        mtl_dir   : 材质库与纹理的搜索目录，默认与 obj 同目录
        scale     : 最终缩放倍数
        normalize : 若为 True，先将模型缩放到单位包围盒内
        """
        self._logger: logging.Logger = logging.getLogger("OBJModel")
        self._logger.setLevel(
            logging.DEBUG if config.ENABLE_MODEL_DEBUG_LOG else logging.INFO
        )

        self.obj_path = obj_path
        self.mtl_dir = mtl_dir if mtl_dir else os.path.dirname(obj_path)
        self.scale = scale
        self.normalize = normalize

        # 原始数据
        self.vertices = []  # 位置 (x,y,z)
        self.texcoords = []  # 纹理坐标 (u,v)
        self.normals = []  # 法线 (nx,ny,nz)
        self.faces = (
            []
        )  # 未处理的原始面数据，每个元素为 (vertex_indices, texcoord_indices, normal_indices)

        # 材质
        self.materials = {}  # name -> dict
        self.current_material = None
        self.face_materials = []  # 每个面对应的材质名，平行于 self.faces

        # VBO 数据
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.material_groups = (
            []
        )  # list of dict: {'material':name, 'start':int, 'count':int}
        self.tex_id_map = {}  # material_name -> texture id (或 None)
        # 额外贴图（PBR/法线）: material_name -> {map_type: texture_id}
        self.extra_tex_id_map = {}
        self.shader_program = None
        self.shader_ready = False
        self.shader_failed = False
        self.default_tex = {}
        self.uniform_locations = {}
        self._has_pbr_maps = False
        self.material_gpu_cache = {}

        t0 = time.perf_counter()
        self._load_obj()
        t1 = time.perf_counter()
        self._build_buffers()
        self._rebuild_material_gpu_cache()
        t2 = time.perf_counter()
        self._logger.info(
            f"模型初始化完成: obj={self.obj_path}, 解析耗时={(t1 - t0):.3f}s, 缓冲构建耗时={(t2 - t1):.3f}s"
        )

    @staticmethod
    def _resolve_index(raw_str, length, fallback_to_last=True):
        """
        将 OBJ 索引字符串转换为 0‑based 整数。
        raw_str : 索引字符串（可能为空或数字）
        length  : 对应列表的长度
        fallback_to_last : 若索引越界，是否回退到最后一个有效索引（否则返回 None）
        """
        if not raw_str or length == 0:
            return None
        try:
            i = int(raw_str)
        except ValueError:
            return None
        if i > 0:
            idx = i - 1
        elif i < 0:
            idx = length + i
        else:
            return None
        if 0 <= idx < length:
            return idx
        else:
            if fallback_to_last:
                logging.getLogger("OBJModel").warning(
                    f"索引 {i} 超出范围 (0-{length})，使用 {length-1}"
                )
                return length - 1
            return None

    # ---------- 加载 ----------
    def _load_obj(self):
        current_mtl = None
        with open(self.obj_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('mtllib'):
                    mtl_file = line.split(maxsplit=1)[1]
                    self._load_mtl(os.path.join(self.mtl_dir, mtl_file))
                elif line.startswith('usemtl'):
                    current_mtl = line.split(maxsplit=1)[1] if len(line.split()) > 1 else None
                elif line.startswith('v '):
                    parts = line.split()
                    self.vertices.append(tuple(map(float, parts[1:4])))
                elif line.startswith('vt '):
                    parts = line.split()
                    tc = tuple(map(float, parts[1:]))
                    if len(tc) < 3:
                        tc = tc + (0.0,) if len(tc) == 1 else (tc[0], tc[1], 0.0)
                    self.texcoords.append(tc[:2])
                elif line.startswith('vn '):
                    parts = line.split()
                    self.normals.append(tuple(map(float, parts[1:4])))
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    v_idx, vt_idx, vn_idx = [], [], []
                    len_v = len(self.vertices)
                    len_vt = len(self.texcoords)
                    len_vn = len(self.normals)
                    for part in parts:
                        comps = part.split('/')
                        vi = self._resolve_index(comps[0], len_v, True)
                        if vi is None:
                            vi = 0   # 顶点索引绝对不能丢失
                        v_idx.append(vi)
                        vt = None
                        if len(comps) > 1 and comps[1]:
                            # UV 越界时不要回退到最后一个坐标，避免随机贴图拉裂
                            vt = self._resolve_index(comps[1], len_vt, False)
                        vt_idx.append(vt)
                        vn = None
                        if len(comps) > 2 and comps[2]:
                            vn = self._resolve_index(comps[2], len_vn, False)
                        vn_idx.append(vn)
                    self.faces.append((v_idx, vt_idx, vn_idx))
                    self.face_materials.append(current_mtl or 'default')

        # 在 _load_obj 结束前增加
        self._logger.info(
            f"OBJ 解析完成: 顶点={len(self.vertices)}, 纹理坐标={len(self.texcoords)}, 法线={len(self.normals)}, 面={len(self.faces)}"
        )
        max_vi = max((max(f[0]) for f in self.faces), default=0)
        max_vt = max((max(f[1]) for f in self.faces if any(x is not None for x in f[1])), default=-1)
        max_vn = max((max(f[2]) for f in self.faces if any(x is not None for x in f[2])), default=-1)
        self._logger.debug(f"面索引最大值: v={max_vi}, vt={max_vt}, vn={max_vn}")

    def _load_mtl(self, mtl_path):
        """解析 .mtl 文件并加载纹理"""
        if not os.path.isfile(mtl_path):
            self._logger.warning(f"mtl 文件不存在: {mtl_path}")
            return
        current_name = None
        with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("newmtl"):
                    current_name = line.split(maxsplit=1)[1]
                    self.materials[current_name] = {
                        "Ka": [0.2, 0.2, 0.2],
                        "Kd": [0.8, 0.8, 0.8],
                        "Ks": [0.0, 0.0, 0.0],
                        "Ns": 0.0,
                        "map_Kd": None,
                        "map_Pm": None,
                        "map_Pr": None,
                        "map_Bump": None,
                    }
                elif line.startswith("Ka "):
                    self._set_mtl_prop(current_name, "Ka", line)
                elif line.startswith("Kd "):
                    self._set_mtl_prop(current_name, "Kd", line)
                elif line.startswith("Ks "):
                    self._set_mtl_prop(current_name, "Ks", line)
                elif line.startswith("Ns "):
                    parts = line.split()
                    if current_name and len(parts) > 1:
                        self.materials[current_name]["Ns"] = float(parts[1])
                elif line.startswith("map_Kd "):
                    tex_file = self._parse_mtl_map_path(line, "map_Kd")
                    if current_name:
                        self.materials[current_name]["map_Kd"] = tex_file
                        # 加载纹理
                        tex_path = os.path.join(self.mtl_dir, tex_file)
                        if os.path.isfile(tex_path):
                            self.tex_id_map[current_name] = self._load_texture(tex_path)
                        else:
                            self._logger.warning(f"纹理不存在: {tex_path}")
                            self.tex_id_map[current_name] = 0
                elif line.startswith("map_Pm "):
                    self._load_extra_mtl_map(current_name, "map_Pm", line)
                elif line.startswith("map_Pr "):
                    self._load_extra_mtl_map(current_name, "map_Pr", line)
                elif line.startswith("map_Bump ") or line.startswith("bump "):
                    keyword = "map_Bump" if line.startswith("map_Bump ") else "bump"
                    self._load_extra_mtl_map(current_name, "map_Bump", line, keyword)

    def _parse_mtl_map_path(self, line, keyword):
        """从 map_XXX 行中提取纹理路径，兼容 -bm/-s/-o 等参数。"""
        try:
            tokens = shlex.split(line, comments=False, posix=True)
        except ValueError:
            # 退化到普通分割，至少保证常见场景可用
            tokens = line.split()

        if not tokens:
            return None

        # 去掉关键字（map_Kd / map_Bump / bump ...）
        if tokens[0] == keyword:
            tokens = tokens[1:]

        if not tokens:
            return None

        # 常见 MTL 选项参数个数
        opt_arity = {
            "-blendu": 1,
            "-blendv": 1,
            "-cc": 1,
            "-clamp": 1,
            "-mm": 2,
            "-o": 3,
            "-s": 3,
            "-t": 3,
            "-texres": 1,
            "-bm": 1,
            "-imfchan": 1,
            "-type": 1,
        }

        i = 0
        path_parts = []
        while i < len(tokens):
            t = tokens[i]
            if t.startswith("-") and t in opt_arity:
                i += 1 + opt_arity[t]
                continue
            if t.startswith("-") and not path_parts:
                # 未知选项，保守跳过一个参数
                i += 2
                continue
            path_parts.extend(tokens[i:])
            break

        if not path_parts:
            return None
        return " ".join(path_parts)

    def _load_extra_mtl_map(self, current_name, map_key, line, keyword=None):
        if not current_name:
            return
        keyword = keyword or map_key
        tex_file = self._parse_mtl_map_path(line, keyword)
        if not tex_file:
            return
        self.materials[current_name][map_key] = tex_file
        tex_path = os.path.join(self.mtl_dir, tex_file)
        if os.path.isfile(tex_path):
            if current_name not in self.extra_tex_id_map:
                self.extra_tex_id_map[current_name] = {}
            tex_id = self._load_texture(tex_path)
            self.extra_tex_id_map[current_name][map_key] = tex_id
            if tex_id:
                self._has_pbr_maps = True
        else:
            self._logger.warning(f"纹理不存在: {tex_path}")
            if current_name not in self.extra_tex_id_map:
                self.extra_tex_id_map[current_name] = {}
            self.extra_tex_id_map[current_name][map_key] = 0

    def _set_mtl_prop(self, name, prop, line):
        if name in self.materials:
            parts = line.split()
            self.materials[name][prop] = [float(x) for x in parts[1:4]]

    def _load_texture(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self._logger.error(f"纹理读取失败: {path}")
            return 0
        # OpenCV 图像原点在左上，OpenGL 纹理坐标原点在左下
        img = cv2.flip(img, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)    # 重要：防止边缘拉伸
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # 避免宽度不是 4 字节对齐时出现纹理错行/破碎
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
        glGenerateMipmap(GL_TEXTURE_2D)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id

    def _create_solid_texture(self, rgb):
        """创建 1x1 纯色纹理，用于缺省贴图。"""
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        data = np.array(rgb, dtype=np.uint8)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, data.tobytes())
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id

    def _ensure_shader(self):
        """懒加载 PBR shader；失败时自动回退固定管线。"""
        if self.shader_ready:
            return True
        if self.shader_failed:
            return False

        vs_src = """
        #version 120
        varying vec2 v_uv;
        varying vec3 v_pos_vs;
        varying vec3 v_t;
        varying vec3 v_b;
        varying vec3 v_n;

        void main() {
            v_uv = gl_MultiTexCoord0.xy;
            v_pos_vs = vec3(gl_ModelViewMatrix * gl_Vertex);

            vec3 n = normalize(gl_NormalMatrix * gl_Normal);
            vec3 t = normalize(gl_NormalMatrix * gl_MultiTexCoord1.xyz);
            t = normalize(t - n * dot(n, t));
            vec3 b = normalize(cross(n, t));

            v_t = t;
            v_b = b;
            v_n = n;
            gl_Position = ftransform();
        }
        """

        fs_src = """
        #version 120
        varying vec2 v_uv;
        varying vec3 v_pos_vs;
        varying vec3 v_t;
        varying vec3 v_b;
        varying vec3 v_n;

        uniform sampler2D u_albedo;
        uniform sampler2D u_pm;
        uniform sampler2D u_pr;
        uniform sampler2D u_normal;
        uniform int u_has_pm;
        uniform int u_has_pr;
        uniform int u_has_normal;
        uniform vec3 u_kd;
        uniform vec3 u_ka;
        uniform float u_metallic;
        uniform float u_roughness;
        uniform float u_normal_strength;
        uniform vec3 u_light_dir_vs;
        uniform vec3 u_light_color;

        void main() {
            vec3 albedo = texture2D(u_albedo, v_uv).rgb * u_kd;

            float metallic = u_metallic;
            if (u_has_pm == 1) {
                metallic = texture2D(u_pm, v_uv).r;
            }
            metallic = clamp(metallic, 0.0, 1.0);

            float roughness = u_roughness;
            if (u_has_pr == 1) {
                roughness = texture2D(u_pr, v_uv).r;
            }
            roughness = clamp(roughness, 0.04, 1.0);

            vec3 normal_ts = vec3(0.0, 0.0, 1.0);
            if (u_has_normal == 1) {
                normal_ts = texture2D(u_normal, v_uv).xyz * 2.0 - 1.0;
                normal_ts.xy *= u_normal_strength;
                normal_ts = normalize(normal_ts);
            }

            mat3 tbn = mat3(normalize(v_t), normalize(v_b), normalize(v_n));
            vec3 n = normalize(tbn * normal_ts);
            vec3 l = normalize(u_light_dir_vs);
            vec3 v = normalize(-v_pos_vs);
            vec3 h = normalize(l + v);

            float ndotl = max(dot(n, l), 0.0);
            float ndoth = max(dot(n, h), 0.0);
            float spec_pow = mix(96.0, 4.0, roughness);
            float spec = pow(ndoth, spec_pow);

            vec3 f0 = mix(vec3(0.04), albedo, metallic);
            vec3 diffuse = (1.0 - metallic) * albedo;
            vec3 specular = f0 * spec;
            vec3 ambient = u_ka * albedo;

            vec3 color = ambient + (diffuse + specular) * ndotl * u_light_color;
            color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
            gl_FragColor = vec4(color, 1.0);
        }
        """

        try:
            vs = shaders.compileShader(vs_src, GL_VERTEX_SHADER)
            fs = shaders.compileShader(fs_src, GL_FRAGMENT_SHADER)
            self.shader_program = shaders.compileProgram(vs, fs)

            glUseProgram(self.shader_program)
            self.uniform_locations = {
                "u_albedo": glGetUniformLocation(self.shader_program, "u_albedo"),
                "u_pm": glGetUniformLocation(self.shader_program, "u_pm"),
                "u_pr": glGetUniformLocation(self.shader_program, "u_pr"),
                "u_normal": glGetUniformLocation(self.shader_program, "u_normal"),
                "u_has_pm": glGetUniformLocation(self.shader_program, "u_has_pm"),
                "u_has_pr": glGetUniformLocation(self.shader_program, "u_has_pr"),
                "u_has_normal": glGetUniformLocation(self.shader_program, "u_has_normal"),
                "u_kd": glGetUniformLocation(self.shader_program, "u_kd"),
                "u_ka": glGetUniformLocation(self.shader_program, "u_ka"),
                "u_metallic": glGetUniformLocation(self.shader_program, "u_metallic"),
                "u_roughness": glGetUniformLocation(self.shader_program, "u_roughness"),
                "u_normal_strength": glGetUniformLocation(self.shader_program, "u_normal_strength"),
                "u_light_dir_vs": glGetUniformLocation(self.shader_program, "u_light_dir_vs"),
                "u_light_color": glGetUniformLocation(self.shader_program, "u_light_color"),
            }
            glUniform1i(self.uniform_locations["u_albedo"], 0)
            glUniform1i(self.uniform_locations["u_pm"], 1)
            glUniform1i(self.uniform_locations["u_pr"], 2)
            glUniform1i(self.uniform_locations["u_normal"], 3)
            glUseProgram(0)

            self.default_tex["white"] = self._create_solid_texture((255, 255, 255))
            self.default_tex["black"] = self._create_solid_texture((0, 0, 0))
            self.default_tex["gray"] = self._create_solid_texture((204, 204, 204))
            self.default_tex["normal"] = self._create_solid_texture((128, 128, 255))

            self.shader_ready = True
            self._logger.info("PBR shader 初始化成功")
            return True
        except Exception as e:
            self._logger.warning(f"PBR shader 初始化失败，回退固定管线: {e}")
            self.shader_failed = True
            self.shader_ready = False
            self.shader_program = None
            return False

    @staticmethod
    def _ns_to_roughness(ns):
        """将 MTL Ns(0~1000) 近似转换到粗糙度(0~1)。"""
        ns = max(0.0, float(ns))
        return float(np.sqrt(2.0 / (ns + 2.0))) if ns > 0 else 1.0

    def _rebuild_material_gpu_cache(self):
        """预计算每个材质的渲染参数，减少 draw 热路径开销。"""
        cache = {}
        material_names = set(self.materials.keys()) | set(self.face_materials)
        for name in material_names:
            material = self.materials.get(name, {})
            extra = self.extra_tex_id_map.get(name, {})
            pm_tex = extra.get("map_Pm", 0)
            pr_tex = extra.get("map_Pr", 0)
            nb_tex = extra.get("map_Bump", 0)
            kd = material.get("Kd", [0.8, 0.8, 0.8])
            ka = material.get("Ka", [0.2, 0.2, 0.2])
            cache[name] = {
                "albedo_tex": self.tex_id_map.get(name, 0),
                "pm_tex": pm_tex,
                "pr_tex": pr_tex,
                "nb_tex": nb_tex,
                "has_pm": 1 if pm_tex else 0,
                "has_pr": 1 if pr_tex else 0,
                "has_normal": 1 if nb_tex else 0,
                "kd": (float(kd[0]), float(kd[1]), float(kd[2])),
                "ka": (float(ka[0]), float(ka[1]), float(ka[2])),
                "roughness": self._ns_to_roughness(material.get("Ns", 0.0)),
            }
        self.material_gpu_cache = cache

    # ---------- 构建缓冲区 ----------
    def _build_buffers(self):
        """将原始数据重组为交错顶点数组 + 索引，并上传 VBO"""
        if not self.faces:
            self._logger.warning("无面片数据")
            return

        build_t0 = time.perf_counter()

        # 计算包围盒，用于归一化
        if self.vertices:
            v_arr = np.array(self.vertices, dtype=np.float32)
            mins = v_arr.min(axis=0)
            maxs = v_arr.max(axis=0)
            center = (mins + maxs) * 0.5
            size = (maxs - mins).max()
        else:
            mins, maxs, center, size = (0, 0, 0), (0, 0, 0), (0, 0, 0), 1.0

        scale_factor = (1.0 / size) if self.normalize and size > 0 else 1.0

        # 重组顶点
        vertices = self.vertices
        texcoords = self.texcoords
        normals = self.normals
        vertex_map = {}  # (vi, ti, ni) -> new_index
        interleaved = []  # 交错数据：px,py,pz, nx,ny,nz, u,v, tx,ty,tz
        tangent_accum = {}  # vertex_index -> [tx,ty,tz]
        indices = []
        mtl_ranges = []  # (material_name, index_start, index_count)

        for fi, face in enumerate(self.faces):
            v_idx_list, vt_idx_list, vn_idx_list = face
            num_verts = len(v_idx_list)

            # 多边形扇形三角化
            for i in range(1, num_verts - 1):
                tri_verts = [0, i, i + 1]
                tri_indices = []
                for k in tri_verts:
                    vi = v_idx_list[k]
                    ti = vt_idx_list[k] if vt_idx_list[k] is not None else -1
                    ni = vn_idx_list[k] if vn_idx_list[k] is not None else -1

                    # 处理负索引
                    if vi < 0:
                        vi += len(vertices)

                    # 纹理坐标默认值
                    if ti is None or ti < 0 or ti >= len(texcoords):
                        tex = (0.0, 0.0)
                    else:
                        tex = texcoords[ti]
                    # 法线默认值：先尝试用提供的法线，否则之后计算面法线
                    if ni is not None and 0 <= ni < len(normals):
                        normal = normals[ni]
                    else:
                        normal = (0.0, 0.0, 1.0)  # 占位，后面计算面法线

                    # 应用归一化缩放与中心位移
                    px, py, pz = vertices[vi]
                    px = (px - center[0]) * scale_factor
                    py = (py - center[1]) * scale_factor
                    pz = (pz - center[2]) * scale_factor

                    key = (vi, ti, ni)
                    if key not in vertex_map:
                        new_idx = len(interleaved) // 11
                        vertex_map[key] = new_idx
                        interleaved.extend(
                            [
                                px,
                                py,
                                pz,
                                normal[0],
                                normal[1],
                                normal[2],
                                tex[0],
                                tex[1],
                                1.0,
                                0.0,
                                0.0,
                            ]
                        )
                    tri_indices.append(vertex_map[key])

                # 如果没有提供任何法线，计算面法线并更新顶点法线
                if all(vn_idx_list[k] is None for k in tri_verts):
                    # 获取三角形三个顶点的位置（已经调整过了）
                    i0 = tri_indices[0]
                    i1 = tri_indices[1]
                    i2 = tri_indices[2]
                    b0 = 11 * i0
                    b1 = 11 * i1
                    b2 = 11 * i2
                    p0x, p0y, p0z = interleaved[b0], interleaved[b0 + 1], interleaved[b0 + 2]
                    p1x, p1y, p1z = interleaved[b1], interleaved[b1 + 1], interleaved[b1 + 2]
                    p2x, p2y, p2z = interleaved[b2], interleaved[b2 + 1], interleaved[b2 + 2]

                    e1x, e1y, e1z = p1x - p0x, p1y - p0y, p1z - p0z
                    e2x, e2y, e2z = p2x - p0x, p2y - p0y, p2z - p0z
                    nx = e1y * e2z - e1z * e2y
                    ny = e1z * e2x - e1x * e2z
                    nz = e1x * e2y - e1y * e2x
                    norm = (nx * nx + ny * ny + nz * nz) ** 0.5
                    if norm > 1e-8:
                        nx, ny, nz = nx / norm, ny / norm, nz / norm
                    else:
                        nx, ny, nz = 0.0, 0.0, 1.0
                    # 更新三个顶点的法线（平均法线更平滑，这里直接替换为面法线以保持锐边）
                    for idx in tri_indices:
                        offset = 11 * idx + 3
                        interleaved[offset : offset + 3] = [nx, ny, nz]

                # 累加切线（用于法线贴图）
                i0, i1, i2 = tri_indices
                b0 = 11 * i0
                b1 = 11 * i1
                b2 = 11 * i2
                p0x, p0y, p0z = interleaved[b0], interleaved[b0 + 1], interleaved[b0 + 2]
                p1x, p1y, p1z = interleaved[b1], interleaved[b1 + 1], interleaved[b1 + 2]
                p2x, p2y, p2z = interleaved[b2], interleaved[b2 + 1], interleaved[b2 + 2]
                uv0x, uv0y = interleaved[b0 + 6], interleaved[b0 + 7]
                uv1x, uv1y = interleaved[b1 + 6], interleaved[b1 + 7]
                uv2x, uv2y = interleaved[b2 + 6], interleaved[b2 + 7]

                e1x, e1y, e1z = p1x - p0x, p1y - p0y, p1z - p0z
                e2x, e2y, e2z = p2x - p0x, p2y - p0y, p2z - p0z
                duv1x, duv1y = uv1x - uv0x, uv1y - uv0y
                duv2x, duv2y = uv2x - uv0x, uv2y - uv0y

                denom = duv1x * duv2y - duv2x * duv1y
                if abs(denom) > 1e-8:
                    tx = (e1x * duv2y - e2x * duv1y) / denom
                    ty = (e1y * duv2y - e2y * duv1y) / denom
                    tz = (e1z * duv2y - e2z * duv1y) / denom
                else:
                    tx, ty, tz = 1.0, 0.0, 0.0

                for idx in tri_indices:
                    if idx not in tangent_accum:
                        tangent_accum[idx] = [0.0, 0.0, 0.0]
                    tangent_accum[idx][0] += tx
                    tangent_accum[idx][1] += ty
                    tangent_accum[idx][2] += tz

                # 记录当前三角形在 EBO 中的索引范围（单位：索引个数）
                tri_start = len(indices)
                indices.extend(tri_indices)
                mtl_name = self.face_materials[fi]
                mtl_ranges.append((mtl_name, tri_start, 3))

        # 合并相同材质且索引范围连续的绘制段
        merged = []
        cur_mtl = None
        cur_start = 0
        cur_count = 0
        for mtl, start, count in mtl_ranges:
            if cur_mtl is None:
                cur_mtl = mtl
                cur_start = start
                cur_count = count
            elif mtl == cur_mtl and start == cur_start + cur_count:
                cur_count += count
            else:
                merged.append(
                    {"material": cur_mtl, "start": cur_start, "count": cur_count}
                )
                cur_mtl = mtl
                cur_start = start
                cur_count = count
        if cur_mtl is not None and cur_count > 0:
            merged.append({"material": cur_mtl, "start": cur_start, "count": cur_count})

        self.material_groups = merged

        # 正交化并写回切线
        vert_count = len(interleaved) // 11
        for idx in range(vert_count):
            base = 11 * idx
            nx, ny, nz = interleaved[base + 3], interleaved[base + 4], interleaved[base + 5]
            tx, ty, tz = tangent_accum.get(idx, [1.0, 0.0, 0.0])
            n_norm = (nx * nx + ny * ny + nz * nz) ** 0.5
            if n_norm > 1e-8:
                nx, ny, nz = nx / n_norm, ny / n_norm, nz / n_norm

            dot_nt = nx * tx + ny * ty + nz * tz
            tx -= nx * dot_nt
            ty -= ny * dot_nt
            tz -= nz * dot_nt

            t_norm = (tx * tx + ty * ty + tz * tz) ** 0.5
            if t_norm <= 1e-8:
                tx, ty, tz = 1.0, 0.0, 0.0
            else:
                tx, ty, tz = tx / t_norm, ty / t_norm, tz / t_norm
            interleaved[base + 8 : base + 11] = [tx, ty, tz]

        # 上传 VBO
        if interleaved:
            self.vao = glGenVertexArrays(1)
            glBindVertexArray(self.vao)

            # 顶点 VBO
            self.vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            data = np.array(interleaved, dtype=np.float32)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

            stride = 11 * 4  # 11 个 float，每个 4 字节
            # 位置
            glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
            glEnableClientState(GL_VERTEX_ARRAY)
            # 法线
            glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(12))
            glEnableClientState(GL_NORMAL_ARRAY)
            # 纹理坐标 0: uv
            glClientActiveTexture(GL_TEXTURE0)
            glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(24))
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            # 纹理坐标 1: tangent.xyz（给 shader 使用）
            glClientActiveTexture(GL_TEXTURE1)
            glTexCoordPointer(3, GL_FLOAT, stride, ctypes.c_void_p(32))
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glClientActiveTexture(GL_TEXTURE0)

            # 索引 EBO
            self.ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            idx_data = np.array(indices, dtype=np.uint32)
            glBufferData(
                GL_ELEMENT_ARRAY_BUFFER, idx_data.nbytes, idx_data, GL_STATIC_DRAW
            )

            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        build_t1 = time.perf_counter()
        self._logger.info(
            f"缓冲构建完成: 顶点={len(interleaved)//11}, 索引={len(indices)}, 材质段={len(self.material_groups)}, 耗时={(build_t1 - build_t0):.3f}s"
        )

    # ---------- 绘制 ----------
    def draw(self):
        """绘制模型（必须在有效的 OpenGL 上下文中，且模型视图/投影矩阵已设置）"""
        if not self.vao or not self.material_groups:
            return

        glPushMatrix()
        glScalef(self.scale, self.scale, self.scale)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)

        shader_ok = self._ensure_shader()
        uloc = self.uniform_locations if shader_ok else None

        if shader_ok:
            glUseProgram(self.shader_program)

            # 光照参数对全模型统一，循环外设置
            glUniform1f(uloc["u_metallic"], 0.0)
            glUniform1f(uloc["u_normal_strength"], 1.0)
            glUniform3f(uloc["u_light_dir_vs"], 0.3, 0.6, 0.7)
            glUniform3f(uloc["u_light_color"], 1.0, 1.0, 1.0)

            for unit in (0, 1, 2, 3):
                glActiveTexture(GL_TEXTURE0 + unit)
                glEnable(GL_TEXTURE_2D)

            last_tex = [-1, -1, -1, -1]
            last_material = None

            for group in self.material_groups:
                mat_name = group["material"]
                data = self.material_gpu_cache.get(mat_name)
                if data is None:
                    data = {
                        "albedo_tex": 0,
                        "pm_tex": 0,
                        "pr_tex": 0,
                        "nb_tex": 0,
                        "has_pm": 0,
                        "has_pr": 0,
                        "has_normal": 0,
                        "kd": (0.8, 0.8, 0.8),
                        "ka": (0.2, 0.2, 0.2),
                        "roughness": 1.0,
                    }

                tex0 = data["albedo_tex"] if data["albedo_tex"] else self.default_tex["white"]
                tex1 = data["pm_tex"] if data["pm_tex"] else self.default_tex["black"]
                tex2 = data["pr_tex"] if data["pr_tex"] else self.default_tex["gray"]
                tex3 = data["nb_tex"] if data["nb_tex"] else self.default_tex["normal"]

                if tex0 != last_tex[0]:
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, tex0)
                    last_tex[0] = tex0
                if tex1 != last_tex[1]:
                    glActiveTexture(GL_TEXTURE1)
                    glBindTexture(GL_TEXTURE_2D, tex1)
                    last_tex[1] = tex1
                if tex2 != last_tex[2]:
                    glActiveTexture(GL_TEXTURE2)
                    glBindTexture(GL_TEXTURE_2D, tex2)
                    last_tex[2] = tex2
                if tex3 != last_tex[3]:
                    glActiveTexture(GL_TEXTURE3)
                    glBindTexture(GL_TEXTURE_2D, tex3)
                    last_tex[3] = tex3

                if mat_name != last_material:
                    glUniform1i(uloc["u_has_pm"], data["has_pm"])
                    glUniform1i(uloc["u_has_pr"], data["has_pr"])
                    glUniform1i(uloc["u_has_normal"], data["has_normal"])
                    glUniform3f(uloc["u_kd"], data["kd"][0], data["kd"][1], data["kd"][2])
                    glUniform3f(uloc["u_ka"], data["ka"][0], data["ka"][1], data["ka"][2])
                    glUniform1f(uloc["u_roughness"], data["roughness"])
                    last_material = mat_name

                glDrawElements(
                    GL_TRIANGLES,
                    group["count"],
                    GL_UNSIGNED_INT,
                    ctypes.c_void_p(group["start"] * 4),
                )
        else:
            # shader 不可用时回退固定管线，保证兼容
            for group in self.material_groups:
                material = self.materials.get(group["material"], {})
                tex_id = self.tex_id_map.get(group["material"], 0)
                glActiveTexture(GL_TEXTURE0)
                if tex_id:
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, tex_id)
                    glColor3f(1, 1, 1)
                    glMaterialfv(GL_FRONT, GL_AMBIENT, [1.0, 1.0, 1.0, 1.0])
                    glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
                else:
                    glDisable(GL_TEXTURE_2D)
                    ka = material.get("Ka", [0.2, 0.2, 0.2])
                    kd = material.get("Kd", [0.8, 0.8, 0.8])
                    ks = material.get("Ks", [0.0, 0.0, 0.0])
                    ns = material.get("Ns", 0.0)
                    glMaterialfv(GL_FRONT, GL_AMBIENT, [*ka, 1.0])
                    glMaterialfv(GL_FRONT, GL_DIFFUSE, [*kd, 1.0])
                    glMaterialfv(GL_FRONT, GL_SPECULAR, [*ks, 1.0])
                    glMaterialf(GL_FRONT, GL_SHININESS, min(ns, 128.0))
                    glColor3f(*kd)

                glDrawElements(
                    GL_TRIANGLES,
                    group["count"],
                    GL_UNSIGNED_INT,
                    ctypes.c_void_p(group["start"] * 4),
                )

        glUseProgram(0)
        for unit in (3, 2, 1, 0):
            glActiveTexture(GL_TEXTURE0 + unit)
            glDisable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glPopMatrix()

    def set_scale(self, scale):
        """动态设置缩放因子"""
        self.scale = scale

    def release(self):
        """释放 OpenGL 资源"""
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.ebo:
            glDeleteBuffers(1, [self.ebo])
        for tex in self.tex_id_map.values():
            if tex:
                glDeleteTextures([tex])
        for tex_map in self.extra_tex_id_map.values():
            for tex in tex_map.values():
                if tex:
                    glDeleteTextures([tex])
        for tex in self.default_tex.values():
            if tex:
                glDeleteTextures([tex])
        if self.shader_program:
            glDeleteProgram(self.shader_program)
