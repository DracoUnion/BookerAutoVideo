"""copy 子命令：提示词模板。

对应任务书第 7.3 节。提示词为全局多行常量，带 `{...}` 占位符，
LLM 输出结构化数据时用 ```json 反引号包裹，文本用 [content]...[/content] 包裹。
"""
import re
import hashlib
from typing import List
from .util import json_dump_model


def render_prompt(pmt: str, **kw) -> str:
    """将提示词模板中的 {key} 占位符替换为 kw[key]。"""
    return re.sub(r"{(\w+)}", lambda g: kw.get(g.group(1), g.group(0)), pmt)


def ext_code_block(s: str) -> str:
    """提取首个 ``` 代码块的内容。"""
    m = re.search(r'```[\w]*\n?([\s\S]+?)\n?```', s)
    return m.group(1) if m else s.strip()


def ext_cont_block(s: str) -> str:
    """提取 [content]...[/content] 内容块。"""
    m = re.search(r'\[content\]([\s\S]+)\[/content\]', s)
    return m.group(1).strip() if m else s.strip()


def gen_objs_md5(*args) -> str:
    """把参数（字符串 / pydantic 模型 / 列表）逐项 MD5 后拼接，作为缓存键。"""
    args = [
        a if isinstance(a, str) else json_dump_model(a)
        for a in args
    ]
    return '_'.join(hashlib.md5(a.encode('utf8')).hexdigest() for a in args)


def json_dump_plan(obj) -> str:
    """把 pydantic 模型 / 列表 / 普通对象转成 JSON 文本喂给提示词。"""
    return json_dump_model(obj)


# ── S3 逆向提示词：System Prompt ─────────────────────────────────

REVERSE_SYSTEM_PMT = '''
你是一名 AI 视频提示词逆向工程师。用户按时间顺序提供同一段视频的若干关键帧。
请逆向还原一段可直接用于 Seedance / 即梦文生视频模型的提示词，要求：
1. 描述主体、场景、动作、镜头运动、光线与色调、氛围；
2. 还原镜头节奏（推拉摇移、快慢切换）；
3. 给出建议时长（4/8/12 秒）与画幅（16:9 / 9:16）；
4. 只输出最终提示词，中文为主，必要时保留英文风格关键词，不要任何解释。
'''.strip()


# ── S3 逆向提示词：User Prompt 模板 ──────────────────────────────

REVERSE_USER_PMT = '''
【视频时长】{duration:.1f} 秒
【关键帧序列】共 {keyframe_count} 帧，按时间顺序如下：

{keyframe_details}

请逆向还原 Seedance 提示词。
'''.strip()


# ── 关键帧详情模板（每帧一个） ──────────────────────────────────

KEYFRAME_DETAIL_TMPL = '''
### 关键帧 {index}（t={time:.2f}s）
![关键帧 {index}]({image_data_url})
'''.strip()


# ── S3 可选润色：System Prompt ──────────────────────────────────

POLISH_SYSTEM_PMT = '''
你是一名视频生成提示词工程师。用户提供一段逆向还原的原始提示词，
以及目标生成参数（时长、分辨率、画幅、是否生成音频）。
请将其润色为一段**结构化、可直接提交给 Seedance 的最终提示词**，
并在末尾附带一个 JSON 参数块供程序解析。

要求：
1. 提示词中文为主，保留必要英文风格词（如 cinematic lighting, depth of field）；
2. 明确描述：主体、场景、动作、镜头语言、光影色调、氛围、节奏；
3. JSON 参数块包含：duration, resolution, ratio, audio, style_keywords（数组）；
4. 输出格式：先给提示词文本，再用三个反引号包裹 JSON 块。
'''.strip()


POLISH_USER_PMT = '''
【原始逆向提示词】
{reversed_prompt}

【目标生成参数】
- 时长：{duration} 秒（可选 4/8/12）
- 分辨率：{resolution}
- 画幅：{ratio}
- 音频：{'开启' if audio else '关闭'}
- 模型：{model}

请输出润色后的最终提示词 + JSON 参数块。
'''.strip()


# ── 图片编码辅助 ────────────────────────────────────────────────

def image_to_data_url(img_path: str, max_side: int = 768, quality: int = 85) -> str:
    """将图片读取、缩放、编码为 base64 data URL。

    Args:
        img_path: 图片文件路径
        max_side: 最长边限制（像素）
        quality: JPEG 质量

    Returns:
        data:image/jpeg;base64,... 格式字符串
    """
    import cv2
    import base64

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f'无法读取图片：{img_path}')

    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    b64 = base64.b64encode(buf).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def build_keyframe_details(keyframes: List, duration: float) -> str:
    """构建关键帧详情文本（含 data URL）。"""
    details = []
    for i, kf in enumerate(keyframes):
        data_url = image_to_data_url(kf.path)
        detail = render_prompt(KEYFRAME_DETAIL_TMPL, index=i, time=kf.time, image_data_url=data_url)
        details.append(detail)
    return '\n\n'.join(details)