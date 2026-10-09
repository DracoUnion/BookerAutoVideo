"""remotion 子命令：提示词常量与小工具。

架构参考 BookerGptTool/code2book，但本项目的 util.py 没有
render_prompt / ext_code_block / gen_objs_md5，故在此模块内置。
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


def gen_objs_md5(*args) -> str:
    """把参数（字符串 / pydantic 模型 / 列表）逐项 MD5 后拼接，作为缓存键。"""
    args = [
        a if isinstance(a, str) else json_dump_model(a)
        for a in args
    ]
    return '_'.join(hashlib.md5(a.encode('utf8')).hexdigest() for a in args)


def json_dump_plan(obj) -> str:
    """把 RemotionPlan / RemotionScene / list 等转成 JSON 文本喂给提示词。"""
    return json_dump_model(obj)


def _extract_file_tag(code: str) -> str:
    """取 `// file: <相对路径>` 标注中的路径，无则返回 ''。"""
    m = re.search(r'//\s*file:\s*(\S+)', code)
    return m.group(1) if m else ''


def _extract_files(out: str) -> List:
    """把 LLM 输出（`// file:` 标注 + ``` 代码块）解析为 [RemotionFile]。

    标注行 `// file: <路径>` 可出现在代码块前或之内；一旦标注出现，
    其后直到下一个标注（或结尾）的代码块内容都归属该文件。
    依赖 .remotion_models.RemotionFile；延迟导入避免循环。
    """
    from .remotion_models import RemotionFile
    files: List[RemotionFile] = []
    if not out:
        return files

    fname = ''
    pending: List[str] = []

    def flush():
        code = '\n'.join(pending).strip()
        if fname and code:
            files.append(RemotionFile(fname=fname, code=code))

    for line in out.splitlines():
        tag = _extract_file_tag(line)
        if tag:
            # 新文件标注：结清上一个
            flush()
            fname, pending = tag, []
            continue
        if line.strip().startswith('```'):
            continue  # 围栏本身不入内容
        if fname:
            pending.append(line)
    flush()
    return files


def _split_code_blocks(out: str) -> List:
    """兜底：按 ``` 代码块切分（无 `// file:` 标注时）。"""
    from .remotion_models import RemotionFile
    files: List[RemotionFile] = []
    blocks = re.findall(r'```[\w]*\n?([\s\S]+?)\n?```', out or '')
    for i, b in enumerate(blocks):
        fname = _extract_file_tag(b) or f'src/gen_{i}.tsx'
        files.append(RemotionFile(fname=fname, code=b.strip()))
    return files


# ── 分场景大纲 ─────────────────────────────────────────────────

REMOTION_PLAN_PMT = '''
你是一位专业的 Remotion（基于 React 的程序化视频）分镜导演。
根据用户提供的视频主题与规格，设计一个分场景（多 scene）的动画视频大纲。

【视频规格】
- 画布：{width} x {height}，{fps} fps，总时长约 {duration} 秒
- 主题 / 文案：{topic}

【设计要求】
1. 拆成 2~6 个场景（scene），每场一个独立组件，用 <Series> 或 <TransitionSeries> 串成时间轴。
2. 每个场景给出 name（用于时间轴节点的英文名）、durationInFrames（= 该场秒数 * fps）、
   points（该场要呈现的要点：文案、视觉元素、动效描述）。
3. 场景总时长（帧数之和）应约等于 duration * fps，允许少量出入。
4. 文案精炼、有节奏，适合动态呈现。

【输出格式】
只输出一个 JSON 对象，用三个反引号包裹，结构为：
`{{"title": "...", "fps": {fps}, "width": {width}, "height": {height}, "scenes": [{{"name": "Intro", "durationInFrames": 90, "points": ["..."]}}]}}`
'''

# ── 根组件（Root + index）─────────────────────────────────────

REMOTION_ROOT_PMT = '''
你是一位 Remotion 工程师。为下面的分镜大纲生成 Remotion 工程的入口文件。

【分镜大纲】
{plan}

【要生成的文件（每个都用 ```tsx 反引号包裹）】
1. src/index.ts：调用 registerRoot(RemotionRoot) 注册根组件。
2. src/Root.tsx：
   - 用 useVideoConfig 无需（Composition 的元数据直接内联）
   - 注册一个 <Composition id="{composition}" component={Main} .../>，
     其 durationInFrames / fps / width / height 取自大纲；
   - Main 组件用 <Series> 或 <TransitionSeries>（需要转场时用 @remotion/transitions）
     依次挂载各场景组件 <Series.Sequence name="..." durationInFrames={...}>。
   - 保持 <Composition> 的 id/fps/width/height/durationInFrames/defaultProps 内联（type Props，不用 interface）。

【输出格式】
依次输出文件，每个文件用 ```tsx 反引号包裹，并在代码块前单独一行写 `// file: <相对路径>` 标注文件名，例如：
```tsx
// file: src/index.ts
import {registerRoot} from "@remotion/cli";
...
```
```tsx
// file: src/Root.tsx
...
```
'''

# ── 单个场景组件 ───────────────────────────────────────────────

REMOTION_SCENE_PMT = '''
你是一位精通 Remotion 动画的 React 工程师。为下面的单个场景生成组件文件。

【分镜大纲】
{plan}

【当前场景】
{scene}

【实现规范（务必遵守 Remotion 最佳实践）】
1. 组件默认导出，文件名与场景对应（如 src/Scene{name}.tsx），用 ```tsx 包裹，块前写 `// file: src/Scene{name}.tsx`。
2. 动画全部由 useCurrentFrame() + interpolate() 驱动；不要用 CSS transition/animation。
3. 从 useVideoConfig() 取 fps；把 seconds 换算写成 fps 表达式（如 2 * fps）。
4. interpolate 必须内联在 style 属性里，输出范围/缓动用硬编码值；用 Easing.bezier / Easing.spring。
5. 变换用 scale / translate / rotate 等独立 CSS 属性，不用 transform 字符串。
6. 给可编辑的层加 name 属性；需要可编辑 props 时用 Interactive.withSchema({wrapInSequence: true})。
7. 静态文字直接内联；图片/素材放 public/ 用 staticFile() 引用。
8. 画布 {width} x {height}：1080 宽时关键文字距左右边 >=80px、距上下边 >=100px；主标题字号 >=84px，次要文字 >=44px，按宽度等比缩放。

【输出格式】
一个 ```tsx 代码块，块前单独一行写 `// file: src/Scene{name}.tsx`。
'''

# ── 修复编译错误 ───────────────────────────────────────────────

REMOTION_FIX_PMT = '''
你是一位 Remotion 工程师。下面这个文件在 TypeScript 编译 / 运行时报错，请修复。

【分镜大纲】
{plan}

【文件】{fname}
{code}

【报错信息】
{err}

请修复后输出该文件的完整内容，用 ```tsx 反引号包裹，块前单独一行写 `// file: {fname}`。
只修复报错，不改动与错误无关的逻辑；保持 Remotion 最佳实践。
'''
