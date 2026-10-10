"""comic-drama 子命令：漫剧 Agent（CDA）提示词。

架构参考 remotion 子命令：提示词为全局多行常量，带 `{...}` 占位符，
LLM 输出结构化数据时用 ```json 反引号包裹，代码/文本用 [content]...[/content]。
工作流对应 CDA.md 第 4 节四个模块：ScriptBreaker → ImageGen → KeyframeGen → VideoGen。
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


# ── 模块 1：剧本拆解（ScriptBreaker）────────────────────────────

CD_BREAK_PMT = '''
你是一位漫剧（comic drama / 动态漫画短片）分镜导演。
把用户提供的文字剧本 / 故事梗概，拆解为结构化分镜方案（Storyboard）。

【视频规格】
- 风格预设：{style}
- 目标时长约 {duration} 秒
- 画布：{width} x {height}

【剧本 / 梗概】
{script}

【拆解规则】
1. 先识别登场角色，为每个角色固化一份【外观锚点】（CharacterAppearance）：
   外貌、发型、服装、主色调。后续所有图像 prompt 必须嵌入对应角色的外观描述，
   这是保证跨镜头角色一致性的核心机制，务必具体、可复用。
2. 按"同一地点 + 同一时间"划分场景（Scene），每个场景拆成若干分镜（Shot）。
3. 每个分镜给出：
   - shot_type：特写/近景/中景/全景/远景，注意景别变化有节奏。
   - camera_move：固定/推近/拉远/横移/跟拍等。
   - duration_seconds：该镜头时长（秒），所有镜头时长之和约等于目标时长。
   - dialogue：台词或旁白（可为空）。
   - image_prompt：生成该镜头图片的英文 prompt，必须写入角色外观锚点 + 场景 + 光线 + 构图。
4. 台词精炼，动作可被静帧 / 关键帧呈现。

【输出格式】
只输出一个 JSON 对象，用三个反引号包裹，结构为：
`{{"title": "...", "style_preset": "...", "characters": [{{"name":"..","appearance":"..","clothing":"..","color_palette":".."}}], "scenes": [{{"scene_id":1,"location":"..","time_of_day":"..","mood":"..","shots":[{{"shot_id":1,"scene_id":1,"shot_type":"..","camera_move":"..","duration_seconds":3.0,"dialogue":"..","image_prompt":"english prompt"}}]}}]}}`
'''

# ── 模块 1 修复：分镜校验（覆盖 / 完整性）───────────────────────

CD_BREAK_FIX_PMT = '''
你是一位漫剧分镜导演。下面这份分镜方案存在以下问题，请修正后重新输出完整的 Storyboard。

【分镜方案（当前）】
{storyboard}

【问题清单】
{problem}

请修正问题（如：某角色未固定外观锚点、镜头景别单调、时长不符、prompt 缺角色外观等），
重新输出完整 JSON，用三个反引号包裹，结构同 Storyboard。
'''

# ── 模块 2：图片素材（ImageGen）──────────────────────────────────
# 由编排器按 shot.user_image 分流；无用户素材时直接以 image_prompt 文生图，无需 LLM。
# 仅当需要 LLM 决定 prompt 增补 / 风格统一时调用：

CD_IMAGE_FIX_PMT = '''
你是一位漫剧美术。下面的分镜图片素材与整体风格不一致，请给出**统一风格后**的英文图片 prompt。

【风格预设】{style}
【角色外观锚点】
{characters}
【该分镜】shot_id={shot_id}，原 prompt：{prompt}
【问题】{problem}

输出：只输出一段统一风格后的英文图片 prompt，用 [content] 与 [/content] 包裹，
内容里必须保留该镜头涉及角色的外观锚点。
'''

# ── 模块 3：关键帧（KeyframeGen）───────────────────────────────

CD_KEYFRAME_PMT = '''
你是一位漫剧动画师。为下面这个分镜生成【起始帧】和【结束帧】两段英文图片 prompt，
两段共享角色外观、场景、光线，仅构图 / 姿态 / 机位不同，使运动自然、起止差异可控。

【风格预设】{style}
【角色外观锚点】
{characters}
【该分镜】
{shot}

【约束】
- 结束帧 prompt 必须共享同一角色与场景，限制运镜幅度，避免视频抖动。
- 输出一段 motion_description（从起始帧到结束帧发生了什么运动）。

【输出格式】
只输出一个 JSON 对象，用三个反引号包裹，结构为：
`{{"start_frame_prompt": "english", "end_frame_prompt": "english", "motion_description": "..."}}`
'''

# ── 模块 4：视频生成提示（VideoGen）────────────────────────────
# 视频后端为外部模型（Sora/Runway 等）；此提示词用于把关键帧 + 运动描述
# 组织成提交给视频后端的文本指令。

CD_VIDEO_PROMPT_PMT = '''
你是一位视频生成提示词工程师。根据关键帧与运动描述，写一段提交给视频生成后端的
英文 prompt，描述从起始帧到结束帧的运动过程（运镜、角色动作、光线变化），时长 {duration} 秒。

【该分镜】
{shot}
【运动描述】{motion}

输出：一段英文视频生成 prompt，用 [content] 与 [/content] 包裹。
'''
