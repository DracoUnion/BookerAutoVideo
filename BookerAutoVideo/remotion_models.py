from typing import List, Optional
from pydantic import BaseModel, Field


class RemotionScene(BaseModel):
    """Remotion 视频中的一个场景（片段）。"""
    name: str = Field(..., description='场景名称，用于 Studio 时间轴的节点名')
    durationInFrames: int = Field(..., description='场景时长（帧数），按规格中的 fps 换算')
    points: List[str] = Field(default_factory=list, description='该场景需要呈现的要点列表（文案 / 视觉元素）')


class RemotionPlan(BaseModel):
    """Remotion 视频的整体分场景大纲。"""
    title: str = Field(..., description='视频标题')
    fps: int = Field(..., description='帧率')
    width: int = Field(..., description='画布宽度（像素）')
    height: int = Field(..., description='画布高度（像素）')
    scenes: List[RemotionScene] = Field(..., description='按时间轴顺序排列的场景列表')


class RemotionFile(BaseModel):
    """大模型生成的单个工程文件。"""
    fname: str = Field(..., description='工程内相对路径，如 src/SceneIntro.tsx')
    code: str = Field(..., description='文件完整文本内容')
