"""copy 子命令：视频复刻数据模型。

对应任务书第 6 节。所有模型支持 pydantic 序列化，
作为阶段产物落盘（YAML）与缓存键的标准形态。
"""
from typing import List, Optional
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


class CopyState(BaseModel):
    """全流程状态（断点续跑核心）。"""
    video_hash: str = Field(..., description='视频文件内容 sha256 前 16 位，作缓存键')
    stage: str = Field(default='init', description='已完成阶段：init/done_s1/done_s2/done_s3/done_s4/done_s5')
    meta: VideoMeta = Field(..., description='视频元信息')
    keyframes: List[Keyframe] = Field(default_factory=list, description='关键帧列表')
    reversed_prompt: Optional[str] = Field(default=None, description='逆向还原的原始提示词（S3 产出）')
    polished_prompt: Optional[str] = Field(default=None, description='润色后的最终提示词（含 JSON 参数块）')
    seedance_task_id: Optional[str] = Field(default=None, description='Seedance 任务 ID')
    video_url: Optional[str] = Field(default=None, description='生成视频下载 URL')
    output_path: Optional[str] = Field(default=None, description='最终输出视频本地路径')


class SeedanceTask(BaseModel):
    """Seedance 任务记录（S4 产出）。"""
    task_id: str = Field(..., description='任务 ID')
    status: str = Field(..., description='任务状态：pending/running/succeeded/failed')
    video_url: Optional[str] = Field(default=None, description='生成视频 URL（成功时）')
    error: Optional[str] = Field(default=None, description='错误信息（失败时）')
    created_at: float = Field(..., description='创建时间戳')
    updated_at: float = Field(..., description='更新时间戳')
    prompt: str = Field(..., description='提交的提示词')
    duration: int = Field(..., description='生成时长（秒）')
    resolution: str = Field(..., description='生成分辨率')
    ratio: str = Field(..., description='画幅比例')
    model: str = Field(..., description='生成模型名称')
    audio: bool = Field(..., description='是否生成音频')