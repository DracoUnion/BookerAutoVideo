"""comic-drama 子命令：漫剧 Agent（CDA）数据模型。

对应 CDA.md 第 5 节数据模型设计。所有模型支持 pydantic 序列化，
作为落盘（JSON/YAML）与 LLM 结构化输出的标准形态。
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class CharacterAppearance(BaseModel):
    """角色外观锚点，跨镜头一致性核心。"""
    name: str = Field(..., description='角色名')
    appearance: str = Field(..., description='外貌描述（脸型、发型、五官、体型）')
    clothing: str = Field(..., description='服装描述')
    color_palette: str = Field(default='', description='主色调（英文色值或颜色名）')


class Shot(BaseModel):
    """单个分镜。"""
    shot_id: int = Field(..., description='分镜序号，从 1 起')
    scene_id: int = Field(..., description='所属场景序号')
    shot_type: str = Field(..., description='景别：特写/近景/中景/全景/远景')
    camera_move: str = Field(default='固定', description='运镜方式')
    duration_seconds: float = Field(default=3.0, description='分镜时长（秒）')
    dialogue: str = Field(default='', description='该镜头台词/旁白')
    image_prompt: str = Field(..., description='生成图片的英文 prompt（含角色外观锚点）')
    user_image: Optional[str] = Field(default=None, description='用户提供的图片素材路径（可选）')


class Scene(BaseModel):
    """场景：同一地点、同一时间下的一组分镜。"""
    scene_id: int = Field(..., description='场景序号，从 1 起')
    location: str = Field(..., description='地点')
    time_of_day: str = Field(default='', description='时间（白天/夜晚/黄昏等）')
    mood: str = Field(default='', description='氛围/情绪基调')
    shots: List[Shot] = Field(default_factory=list, description='场景内的分镜列表')


class Storyboard(BaseModel):
    """分镜方案（剧本拆解顶层产物）。"""
    project_id: str = Field(..., description='项目标识（用于目录）')
    title: str = Field(..., description='视频标题')
    style_preset: str = Field(default='animated comic', description='整体风格预设（英文）')
    characters: List[CharacterAppearance] = Field(default_factory=list, description='角色外观锚点列表')
    scenes: List[Scene] = Field(..., description='按时间顺序的场景列表')


class ImageAsset(BaseModel):
    """图片素材登记。"""
    shot_id: int = Field(..., description='对应分镜序号')
    image_path: str = Field(..., description='图片文件相对路径（images/shot_XXX.png）')
    source: str = Field(..., description='来源：generated / user_provided / edited')


class KeyframePair(BaseModel):
    """单个分镜的关键帧对（起始帧 + 结束帧）。"""
    shot_id: int = Field(..., description='对应分镜序号')
    start_frame: str = Field(..., description='起始帧文件相对路径')
    end_frame: str = Field(..., description='结束帧文件相对路径')
    start_frame_prompt: str = Field(default='', description='起始帧生成 prompt')
    end_frame_prompt: str = Field(default='', description='结束帧生成 prompt')
    motion_description: str = Field(default='', description='起始帧到结束帧的运动描述')


class VideoResult(BaseModel):
    """单个分镜的视频生成结果。"""
    shot_id: int = Field(..., description='对应分镜序号')
    video_path: str = Field(..., description='分镜视频文件相对路径（shots/shot_XXX.mp4）')
    duration_seconds: float = Field(default=0.0, description='分镜时长（秒）')
    model_used: str = Field(default='', description='使用的视频生成模型 / 后端')
