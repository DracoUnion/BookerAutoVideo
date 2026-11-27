"""copy 子命令：视频复刻数据模型（八阶段流水线）。

对应任务书第 6 节。所有模型支持 pydantic 序列化，
作为阶段产物落盘（YAML/JSON）与缓存键的标准形态。
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class VideoMeta(BaseModel):
    """视频元信息（S1 抽帧产出）。"""
    path: str = Field(..., description='视频文件路径')
    fps: float = Field(..., description='视频帧率')
    total_frames: int = Field(..., description='总帧数')
    duration: float = Field(..., description='视频时长（秒）')
    width: int = Field(..., description='视频宽度（像素）')
    height: int = Field(..., description='视频高度（像素）')


class Keyframe(BaseModel):
    """关键帧（S2 pHash 产出）。"""
    index: int = Field(..., description='关键帧序号（从 0 起）')
    time: float = Field(..., description='时间戳（秒）')
    path: str = Field(..., description='关键帧图片文件路径')
    phash: str = Field(..., description='感知哈希（十六进制字符串）')


class ImagePromptResult(BaseModel):
    """S3 逆向文生图提示词产出。"""
    reversed_prompt: str = Field(..., description='逆向还原的原始文生图提示词')
    polished_prompt: Optional[str] = Field(default=None, description='润色后的最终提示词')
    image_ratio: Optional[str] = Field(default=None, description='建议画幅（16:9 / 9:16）')
    image_resolution: Optional[str] = Field(default=None, description='建议分辨率')


class ImageResult(BaseModel):
    """S4 Seedance 文生图产出。"""
    task_id: str = Field(..., description='文生图任务 ID')
    image_url: str = Field(..., description='生成图片 URL')
    local_path: str = Field(..., description='图片本地保存路径')


class Transcript(BaseModel):
    """S5 SenseVoice 提取文案产出。"""
    text: str = Field(..., description='原始转录文案')
    language: Optional[str] = Field(default=None, description='检测到的语言')
    segments: Optional[list[dict]] = Field(default=None, description='带时间戳的分段')


class RewriteResult(BaseModel):
    """S6 大模型改写文案产出。"""
    original_text: str = Field(..., description='原始文案')
    rewritten_text: str = Field(..., description='改写后的文案')


class SeedanceImageTask(BaseModel):
    """S4 Seedance 文生图任务记录。"""
    task_id: str = Field(..., description='任务 ID')
    status: str = Field(..., description='任务状态：pending/running/succeeded/failed')
    image_url: Optional[str] = Field(default=None, description='生成图片 URL（成功时）')
    error: Optional[str] = Field(default=None, description='错误信息（失败时）')
    created_at: float = Field(..., description='创建时间戳')
    updated_at: float = Field(..., description='更新时间戳')
    prompt: str = Field(..., description='提交的文生图提示词')
    ratio: str = Field(..., description='画幅比例')
    resolution: str = Field(..., description='分辨率')
    model: str = Field(..., description='生成模型名称')


class SeedanceVideoTask(BaseModel):
    """S7 Seedance 图生视频任务记录。"""
    task_id: str = Field(..., description='任务 ID')
    status: str = Field(..., description='任务状态：pending/running/succeeded/failed')
    video_url: Optional[str] = Field(default=None, description='生成视频 URL（成功时）')
    error: Optional[str] = Field(default=None, description='错误信息（失败时）')
    created_at: float = Field(..., description='创建时间戳')
    updated_at: float = Field(..., description='更新时间戳')
    prompt: str = Field(..., description='提交的图生视频提示词（改写文案）')
    duration: int = Field(..., description='生成时长（秒）')
    resolution: str = Field(..., description='生成分辨率')
    ratio: str = Field(..., description='画幅比例')
    model: str = Field(..., description='生成模型名称')
    audio: bool = Field(..., description='是否生成音频')


class PostConfig(BaseModel):
    """S8 ffmpeg 后期配置。"""
    subtitle_path: Optional[str] = Field(default=None, description='字幕文件路径（.srt/.ass）')
    bgm_path: Optional[str] = Field(default=None, description='BGM 文件路径')
    bgm_volume: float = Field(default=0.25, description='BGM 音量')
    rhythm: Optional[Dict[str, Any]] = Field(default=None, description='节奏配置 JSON')


class CopyState(BaseModel):
    """全流程状态（断点续跑核心）。画面路线与文案路线分别记录完成状态。"""
    video_hash: str = Field(..., description='视频文件内容 sha256 前 16 位，作缓存键')
    stage: str = Field(default='init', description='已完成阶段：init/done_s1/done_s2/done_s3/done_s4/done_s5/done_s6/done_s7/done_s8')
    meta: VideoMeta = Field(..., description='视频元信息')
    keyframes: list[Keyframe] = Field(default_factory=list, description='关键帧列表')
    image_prompt: ImagePromptResult | None = Field(default=None, description='S3 逆向文生图提示词')
    image_result: ImageResult | None = Field(default=None, description='S4 文生图结果')
    transcript: Transcript | None = Field(default=None, description='S5 SenseVoice 原始文案')
    rewrite_result: RewriteResult | None = Field(default=None, description='S6 改写文案')
    image_task: SeedanceImageTask | None = Field(default=None, description='S4 文生图任务记录')
    video_task: SeedanceVideoTask | None = Field(default=None, description='S7 图生视频任务记录')
    post_config: PostConfig | None = Field(default=None, description='S8 后期配置')
    output_path: Optional[str] = Field(default=None, description='最终输出视频本地路径')
