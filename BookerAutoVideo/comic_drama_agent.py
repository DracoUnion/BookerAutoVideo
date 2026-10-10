"""comic-drama 子命令：封装所有 LLM 调用，每个方法对应一个独立提示词。

对应 CDA.md 模块 1（ScriptBreaker）、3（KeyframeGen）、4（VideoGen）的 LLM 部分；
模块 2（ImageGen）的文生图 / 图生图由编排器直连，仅风格统一走 LLM。
"""
import os
from os import path
from typing import List, Optional

import json_repair
from pydantic import parse_obj_as

from .util import write_yaml_model, read_yaml_model
from .openai import ask_chatgpt_retry, set_openai_props, call_tti_retry
from .comic_drama_pmt import (
    CD_BREAK_PMT, CD_BREAK_FIX_PMT, CD_IMAGE_FIX_PMT,
    CD_KEYFRAME_PMT, CD_VIDEO_PROMPT_PMT,
    render_prompt, ext_code_block, ext_cont_block, gen_objs_md5, json_dump_plan,
)
from .comic_drama_models import (
    Storyboard, CharacterAppearance, Shot, KeyframePair, VideoResult,
)


class ComicDramaAgent:
    """按 CDA 工作流调用大模型，逐步缓存。"""

    def __init__(self, args):
        self.args = args
        self.model = getattr(args, 'model', None)
        self.tt_model = getattr(args, 'tti_model', '') or getattr(args, 'im', '')
        self.pj_dir = path.abspath(getattr(args, 'out', '.'))
        self.asset_dir = path.join(self.pj_dir, 'asset')
        os.makedirs(self.asset_dir, exist_ok=True)
        set_openai_props(args)

    def _cache(self, prefix: str, key: str, ext: str = '.yaml') -> str:
        return path.join(self.asset_dir, f'{prefix}_{key}{ext}')

    # ── 模块 1：剧本拆解 ──────────────────────────────────────

    def break_script(
        self, script: str, style: str, duration: int,
        width: int, height: int, project_id: str,
    ) -> Storyboard:
        key = gen_objs_md5(script, style, duration, width, height)
        cache = self._cache('break', key)
        r = read_yaml_model(cache, Storyboard)
        if r:
            r.project_id = project_id
            return r
        ques = render_prompt(
            CD_BREAK_PMT,
            script=script, style=style,
            duration=str(duration), width=str(width), height=str(height),
        )
        parse_output = lambda s: Storyboard(
            **json_repair.loads(ext_code_block(s))
        )
        sb: Storyboard = ask_chatgpt_retry(ques, self.model, self.args, parse_output=parse_output)
        sb.project_id = project_id
        write_yaml_model(cache, sb)
        return sb

    def fix_storyboard(
        self, sb: Storyboard, problem: str,
    ) -> Storyboard:
        key = gen_objs_md5(sb, problem, 'break_fix')
        cache = self._cache('break_fix', key)
        r = read_yaml_model(cache, Storyboard)
        if r:
            return r
        ques = render_prompt(
            CD_BREAK_FIX_PMT,
            storyboard=json_dump_plan(sb), problem=problem,
        )
        parse_output = lambda s: Storyboard(
            **json_repair.loads(ext_code_block(s))
        )
        res: Storyboard = ask_chatgpt_retry(ques, self.model, self.args, parse_output=parse_output)
        res.project_id = sb.project_id
        write_yaml_model(cache, res)
        return res

    # ── 模块 2：图片素材（风格统一 / prompt 修正）──────────────

    def _characters_json(self, sb: Storyboard) -> str:
        return json_dump_plan(sb.characters)

    def fix_image_prompt(
        self, sb: Storyboard, shot: Shot, problem: str,
    ) -> str:
        key = gen_objs_md5(shot, problem, 'img_fix')
        cache = self._cache('img_fix', key, ext='.txt')
        if path.isfile(cache) and path.getsize(cache):
            return open(cache, encoding='utf8').read()
        ques = render_prompt(
            CD_IMAGE_FIX_PMT,
            style=sb.style_preset, characters=self._characters_json(sb),
            shot_id=str(shot.shot_id), prompt=shot.image_prompt, problem=problem,
        )
        res = ask_chatgpt_retry(ques, self.model, self.args, parse_output=ext_cont_block)
        open(cache, 'w', encoding='utf8').write(res)
        return res

    def gen_image_bytes(self, prompt: str, size: str = '1024x1024') -> bytes:
        """文生图（call_tti_retry，OpenAI 图像接口）。无模型时返回 None 由编排器兜底。"""
        if not self.tt_model:
            return None
        return call_tti_retry(prompt, self.tt_model, size=size, retry=self.args.retry)

    # ── 模块 3：关键帧 ────────────────────────────────────────

    def gen_keyframe_pair(self, sb: Storyboard, shot: Shot) -> KeyframePair:
        key = gen_objs_md5(shot, 'keyframe')
        cache = self._cache('keyframe', key, ext='.yaml')
        r = read_yaml_model(cache, KeyframePair)
        if r:
            return r
        ques = render_prompt(
            CD_KEYFRAME_PMT,
            style=sb.style_preset, characters=self._characters_json(sb),
            shot=json_dump_plan(shot),
        )
        parse_output = lambda s: json_repair.loads(ext_code_block(s))
        obj = ask_chatgpt_retry(ques, self.model, self.args, parse_output=parse_output)
        res = KeyframePair(
            shot_id=shot.shot_id,
            start_frame=f'keyframes/shot_{shot.shot_id:03d}_start.png',
            end_frame=f'keyframes/shot_{shot.shot_id:03d}_end.png',
            start_frame_prompt=obj.get('start_frame_prompt', ''),
            end_frame_prompt=obj.get('end_frame_prompt', ''),
            motion_description=obj.get('motion_description', ''),
        )
        write_yaml_model(cache, res)
        return res

    # ── 模块 4：视频生成 prompt ───────────────────────────────

    def gen_video_prompt(self, shot: Shot, motion: str, duration: int) -> str:
        key = gen_objs_md5(shot, motion, duration, 'video_prompt')
        cache = self._cache('video_prompt', key, ext='.txt')
        if path.isfile(cache) and path.getsize(cache):
            return open(cache, encoding='utf8').read()
        ques = render_prompt(
            CD_VIDEO_PROMPT_PMT,
            shot=json_dump_plan(shot), motion=motion, duration=str(duration),
        )
        res = ask_chatgpt_retry(ques, self.model, self.args, parse_output=ext_cont_block)
        open(cache, 'w', encoding='utf8').write(res)
        return res
