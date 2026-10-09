"""remotion 子命令：封装所有 LLM 调用，每个方法对应一个独立提示词。"""
import os
import json
from os import path
from typing import List

import json_repair
from pydantic import BaseModel, parse_obj_as

from .util import write_yaml_model, read_yaml_model
from .openai import ask_chatgpt_retry, set_openai_props
from .remotion_pmt import (
    REMOTION_PLAN_PMT, REMOTION_ROOT_PMT, REMOTION_SCENE_PMT, REMOTION_FIX_PMT,
    render_prompt, ext_code_block, gen_objs_md5,
    json_dump_plan, _extract_files, _extract_file_tag, _split_code_blocks,
)
from .remotion_models import RemotionPlan, RemotionScene, RemotionFile


class RemotionAgent:
    """按 remotion-skills 工作流调用大模型，逐文件缓存。"""

    def __init__(self, args):
        self.args = args
        self.model = getattr(args, 'model', None)
        self.pj_dir = path.abspath(args.dir)
        self.asset_dir = path.join(self.pj_dir, 'asset')
        os.makedirs(self.asset_dir, exist_ok=True)
        set_openai_props(args.key, getattr(args, 'proxy', None), args.host)

    def _cache(self, prefix: str, key: str, ext: str = '.yaml') -> str:
        return path.join(self.asset_dir, f'{prefix}_{key}{ext}')

    # ── 步骤 1：分场景大纲 ────────────────────────────────────

    def plan(self, topic: str, width: int, height: int, fps: int, duration: int) -> RemotionPlan:
        key = gen_objs_md5(topic, width, height, fps, duration)
        cache = self._cache('plan', key)
        r = read_yaml_model(cache, RemotionPlan)
        if r:
            return r
        ques = render_prompt(
            REMOTION_PLAN_PMT,
            topic=topic, width=str(width), height=str(height),
            fps=str(fps), duration=str(duration),
        )
        parse_output = lambda s: RemotionPlan(
            **json_repair.loads(ext_code_block(s))
        )
        res: RemotionPlan = ask_chatgpt_retry(ques, self.model, self.args, parse_output=parse_output)
        # 规格兜底
        res.fps = res.fps or fps
        res.width = res.width or width
        res.height = res.height or height
        write_yaml_model(cache, res)
        return res

    # ── 步骤 2：根组件（Root.tsx + index.ts）──────────────────

    def gen_root(self, plan: RemotionPlan, composition: str) -> List[RemotionFile]:
        key = gen_objs_md5(plan, composition)
        cache = self._cache('root', key, ext='.json')
        r = read_yaml_model(cache, List[RemotionFile])
        if r:
            return r
        ques = render_prompt(
            REMOTION_ROOT_PMT,
            plan=json_dump_plan(plan), composition=composition,
        )
        out = ask_chatgpt_retry(
            ques, self.model, self.args,
            parse_output=ext_code_block,
        )
        files = _extract_files(out)
        if not files:
            # 兜底：模型未按 `// file:` 标注，按代码块切分
            files = _split_code_blocks(out)
        write_yaml_model(cache, files)
        return files

    # ── 步骤 3：单个场景组件 ───────────────────────────────────

    def gen_scene(self, plan: RemotionPlan, scene: RemotionScene) -> RemotionFile:
        key = gen_objs_md5(plan, scene, 'scene')
        cache = self._cache('scene', key, ext='.json')
        r = read_yaml_model(cache, RemotionFile)
        if r:
            return r
        ques = render_prompt(
            REMOTION_SCENE_PMT,
            plan=json_dump_plan(plan), scene=json_dump_plan(scene),
            width=str(plan.width), height=str(plan.height),
        )
        out = ask_chatgpt_retry(ques, self.model, self.args, parse_output=ext_code_block)
        fname = _extract_file_tag(out) or f'src/Scene{scene.name}.tsx'
        files = _extract_files(out)
        code = files[0].code if files and files[0].code.strip() else out
        res = RemotionFile(fname=fname, code=code)
        write_yaml_model(cache, res)
        return res

    # ── 步骤 4：修复编译错误 ───────────────────────────────────

    def fix_file(self, fname: str, code: str, err: str, plan: RemotionPlan) -> RemotionFile:
        key = gen_objs_md5(fname, code, err, 'fix')
        cache = self._cache('fix', key, ext='.json')
        r = read_yaml_model(cache, RemotionFile)
        if r:
            return r
        ques = render_prompt(
            REMOTION_FIX_PMT,
            plan=json_dump_plan(plan), fname=fname, code=code, err=err,
        )
        out = ask_chatgpt_retry(ques, self.model, self.args, parse_output=ext_code_block)
        res = RemotionFile(fname=fname, code=out)
        write_yaml_model(cache, res)
        return res
