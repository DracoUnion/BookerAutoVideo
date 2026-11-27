"""copy 子命令：提示词模板（八阶段流水线）。

对应任务书第 7.3 / 7.6 节。提示词为全局多行常量，带 `{...}` 占位符，
LLM 输出结构化数据时用 ```json 反引号包裹，文本用 [content]...[/content] 包裹。
"""
import re
import hashlib
import base64
from typing import List
from .util import json_dump_model


def render_prompt(pmt: str, **kw) -> str:
    """将提示词模板中的 {key} 占位符替换为 kw[key]（值统一转字符串）。"""
    return re.sub(r"{(\w+)}", lambda g: str(kw.get(g.group(1), g.group(0))), pmt)


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


# ── S3 逆向文生图提示词：System Prompt（固定） ───────────────────

REVERSE_IMAGE_SYSTEM_PMT = '''
你是一名 AI 绘画提示词逆向工程师。用户按时间顺序提供同一段视频的若干关键帧。
请逆向还原一段可直接用于 Seedance 文生图模型的提示词，要求：
1. 描述主体、服饰、姿态、构图、场景、背景、光线与色调、渲染风格；
2. 给出建议画幅（16:9 / 9:16）与分辨率；
3. 只输出最终提示词，中文为主，必要时保留英文风格关键词，不要任何解释。
'''.strip()


# ── S3 逆向文生图提示词：User Prompt 模板 ────────────────────────

REVERSE_IMAGE_USER_PMT = '''
【视频时长】{duration} 秒
【关键帧序列】共 {keyframe_count} 帧，按时间顺序如下：

{keyframe_details}

请逆向还原 Seedance 文生图提示词。
'''.strip()


# ── 关键帧详情模板（每帧一个，含 data URL） ──────────────────────

KEYFRAME_DETAIL_TMPL = '''
### 关键帧 {index}（t={time:.2f}s）
![关键帧 {index}]({image_data_url})
'''.strip()


# ── S3 可选润色：System Prompt ──────────────────────────────────

POLISH_SYSTEM_PMT = '''
你是一名 AI 绘画提示词工程师。用户提供一段逆向还原的原始文生图提示词，
以及目标生成参数（画幅、分辨率）。
请将其润色为一段**结构化、可直接提交给 Seedance 文生图模型的最终提示词**。

要求：
1. 提示词中文为主，保留必要英文风格词（如 cinematic lighting, depth of field）；
2. 明确描述：主体、服饰、姿态、构图、场景、背景、光线色调、渲染风格；
3. 只输出润色后的提示词文本，不要任何解释、标题或参数块。
'''.strip()


POLISH_USER_PMT = '''
【原始逆向提示词】
{reversed_prompt}

【目标生成参数】
- 画幅：{image_ratio}
- 分辨率：{image_resolution}

请输出润色后的最终文生图提示词。
'''.strip()


# ── S6 大模型改写文案：System Prompt ────────────────────────────

REWRITE_SYSTEM_PMT = '''
你是一名短视频文案改写专家。用户提供一段原始口播文案，以及改写要求。
请学习原文案的句式、节奏和包袱，改写成一段**表述不同、风格一致**的全新文案。

核心要求：
1. 学习原文案句式、节奏和包袱，不照抄内容；
2. 保留反差结构、情绪共鸣和统一结尾；
3. 每条只讲一件事、一组对比、一个笑点；
4. 像朋友吐槽，不用书面语；
5. 只输出改写后的文案，不加标题、分析或建议。
'''.strip()


REWRITE_USER_PMT = '''
【原始文案】
{original_text}

【改写要求】
{rewrite_requirement}

请输出改写后的文案。
'''.strip()


# ── 默认改写要求（--rewrite-requirement 未指定时使用） ───────────

DEFAULT_REWRITE_REQUIREMENT = '''
- 学习原文案句式、节奏和包袱，不照抄内容；
- 保留反差结构、情绪共鸣和统一结尾；
- 每条只讲一件事、一组对比、一个笑点；
- 像朋友吐槽，不用书面语；
- 只输出文案，不加标题、分析或建议。
'''.strip()


# ── 图片编码辅助（纯函数，便于单测） ─────────────────────────────

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
    """构建关键帧详情文本（含 data URL）。纯函数，便于单测。"""
    details = []
    for i, kf in enumerate(keyframes):
        data_url = image_to_data_url(kf.path)
        detail = render_prompt(KEYFRAME_DETAIL_TMPL, index=i, time=kf.time, image_data_url=data_url)
        details.append(detail)
    return '\n\n'.join(details)


def build_vision_content(keyframes: List, duration: float, detail: str = 'low') -> list:
    """构建视觉模型 user message 的 content 列表（text + 多张 image_url）。

    纯函数，便于单测。返回 OpenAI Chat Completions 多模态 content 结构。
    """
    text = render_prompt(
        REVERSE_IMAGE_USER_PMT,
        duration=f'{duration:.1f}',
        keyframe_count=len(keyframes),
        keyframe_details='（见下方按时间顺序的关键帧图片）',
    )
    content = [{'type': 'text', 'text': text}]
    for i, kf in enumerate(keyframes):
        data_url = image_to_data_url(kf.path)
        content.append({
            'type': 'image_url',
            'image_url': {'url': data_url, 'detail': detail},
        })
    return content