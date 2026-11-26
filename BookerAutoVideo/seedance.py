"""Seedance API 客户端（基于 seedance-sdk 封装）。

对应任务书第 7.4 节。封装创建任务、轮询、取 URL、下载。
使用官方 seedance-sdk，隔离外部依赖差异。
"""
import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from seedance import (
    SeedanceClient as SDKClient,
    GenerationRequest,
    Model,
    AspectRatio,
    Resolution,
    TaskStatus,
    TaskFailedError,
    TaskTimeoutError,
    ConfigurationError,
)
import httpx

logger = logging.getLogger(__name__)


class SeedanceError(Exception):
    """Seedance API 错误基类。"""
    pass


class SeedanceAuthError(SeedanceError):
    """鉴权失败（401/403）。"""
    pass


class SeedanceTimeoutError(SeedanceError):
    """轮询超时。"""
    pass


def _parse_model(model_str: str) -> Model:
    """解析模型字符串为 Model 枚举。"""
    model_map = {
        'seedance-2.0': Model.SEEDANCE_2_0,
        'kling-v2-6': Model.KLING_V2_6,
        'sora-2': Model.SORA_2,
    }
    return model_map.get(model_str, Model.SEEDANCE_2_0)


def _parse_aspect_ratio(ratio_str: str) -> AspectRatio:
    """解析画幅字符串为 AspectRatio 枚举。"""
    ratio_map = {
        '16:9': AspectRatio.R16_9,
        '9:16': AspectRatio.R9_16,
        '1:1': AspectRatio.R1_1,
        '4:3': AspectRatio.R4_3,
        '3:4': AspectRatio.R3_4,
    }
    return ratio_map.get(ratio_str, AspectRatio.R16_9)


def _parse_resolution(res_str: str) -> Resolution:
    """解析分辨率字符串为 Resolution 枚举。"""
    res_map = {
        '480p': Resolution.P480,
        '720p': Resolution.P720,
        '1080p': Resolution.P1080,
    }
    return res_map.get(res_str, Resolution.P1080)


class SeedanceClient:
    """Seedance API 客户端（同步封装）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = 'https://ark.cn-beijing.volces.com/api/v3',
        model: str = 'seedance-2.0',
        timeout: int = 600,
        poll_interval: int = 3,
    ):
        """
        Args:
            api_key: Seedance API Key
            base_url: API 基础 URL（SDK 默认为 https://vibegen.art/api/v1）
            model: 生成模型名称
            timeout: 轮询总超时（秒）
            poll_interval: 轮询间隔（秒）
        """
        # SDK 验证 API Key 格式，要求以 sk-video- 开头
        # 如果用户的 key 不是标准格式，需要处理
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.poll_interval = poll_interval

        # 创建 SDK 客户端
        # 注意：SDK 默认 base_url 是 https://vibegen.art/api/v1
        # 如果用户传了自定义 base_url，需要传给 SDK
        sdk_base_url = base_url if base_url != 'https://ark.cn-beijing.volces.com/api/v3' else 'https://vibegen.art/api/v1'

        try:
            self._sdk = SDKClient(
                api_key=api_key,
                base_url=sdk_base_url,
                timeout=30.0,
                max_retries=3,
            )
        except ConfigurationError as e:
            # API Key 格式不符合 SDK 要求（需以 sk-video- 开头）
            # 尝试不验证直接创建，或抛出更友好的错误
            logger.warning(f'SDK 配置错误：{e}，尝试绕过验证')
            # 如果 key 不符合格式，可能是火山引擎的 key 格式不同
            # 这里我们直接抛出，让上层处理
            raise SeedanceAuthError(f'API Key 格式可能不正确：{e}')

    def create_task(
        self,
        prompt: str,
        duration: int = 8,
        resolution: str = '1080p',
        ratio: str = '16:9',
        audio: bool = True,
        model: Optional[str] = None,
    ) -> str:
        """创建视频生成任务。

        Args:
            prompt: 提示词
            duration: 生成时长（秒），可选 4/8/12
            resolution: 生成分辨率
            ratio: 画幅比例
            audio: 是否生成音频
            model: 模型名称（覆盖默认）

        Returns:
            task_id: 任务 ID

        Raises:
            SeedanceAuthError: 鉴权失败
            SeedanceError: 其他 API 错误
        """
        model_enum = _parse_model(model or self.model)
        aspect_ratio = _parse_aspect_ratio(ratio)
        resolution_enum = _parse_resolution(resolution)

        request = GenerationRequest(
            model=model_enum,
            prompt=prompt,
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution_enum,
            generate_audio=audio,
        )

        logger.info(f'创建 Seedance 任务：model={model_enum.value}, duration={duration}s, ratio={ratio}')
        logger.debug(f'提示词：{prompt[:200]}...')

        try:
            task_data = self._sdk.generate_video_sync(request)
        except Exception as e:
            # 捕获 SDK 可能抛出的各种异常
            error_msg = str(e)
            if '401' in error_msg or '403' in error_msg or 'auth' in error_msg.lower():
                raise SeedanceAuthError(f'Seedance 鉴权失败：{e}')
            raise SeedanceError(f'创建任务失败：{e}') from e

        task_id = task_data.id
        logger.info(f'任务创建成功：task_id={task_id}')
        return task_id

    def get_task_status(self, task_id: str) -> Any:
        """查询任务状态，返回 SDK 的 TaskData 对象。"""
        try:
            return self._sdk.get_task_sync(task_id)
        except Exception as e:
            error_msg = str(e)
            if '401' in error_msg or '403' in error_msg or 'auth' in error_msg.lower():
                raise SeedanceAuthError(f'Seedance 鉴权失败：{e}')
            raise SeedanceError(f'查询任务状态失败：{e}') from e

    def wait_task(self, task_id: str) -> str:
        """轮询任务直到完成或超时。

        Args:
            task_id: 任务 ID

        Returns:
            video_url: 生成视频的下载 URL

        Raises:
            SeedanceTimeoutError: 超时
            SeedanceError: 任务失败
        """
        logger.info(f'开始轮询任务：task_id={task_id}, timeout={self.timeout}s')

        try:
            task_data = self._sdk.wait_for_completion_sync(
                task_id=task_id,
                timeout=float(self.timeout),
                poll_interval=float(self.poll_interval),
                on_progress=lambda t: logger.debug(f'任务进度：status={t.status.value}, estimated_time={t.estimated_time}s'),
            )
        except TaskTimeoutError as e:
            raise SeedanceTimeoutError(f'轮询超时（{self.timeout}s），task_id={task_id} 可能仍在运行，可手动续跑') from e
        except TaskFailedError as e:
            raise SeedanceError(f'任务失败：{e}') from e
        except Exception as e:
            error_msg = str(e)
            if '401' in error_msg or '403' in error_msg or 'auth' in error_msg.lower():
                raise SeedanceAuthError(f'Seedance 鉴权失败：{e}')
            raise SeedanceError(f'轮询异常：{e}') from e

        if task_data.result and task_data.result.video_url:
            video_url = str(task_data.result.video_url)
            logger.info(f'任务完成：video_url={video_url}')
            return video_url

        raise SeedanceError(f'任务成功但无 video_url：{task_data}')

    def download_video(self, url: str, dest_path: str, max_retries: int = 3) -> str:
        """下载视频文件。

        Args:
            url: 视频下载 URL
            dest_path: 目标文件路径
            max_retries: 最大重试次数

        Returns:
            实际下载的文件路径
        """
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # 如果文件已存在且大小 > 0，跳过下载
        if dest.exists() and dest.stat().st_size > 0:
            logger.info(f'视频已存在，跳过下载：{dest}')
            return str(dest)

        logger.info(f'下载视频：{url} -> {dest}')

        # 使用 httpx 下载（SDK 内部用 httpx）
        for attempt in range(max_retries):
            try:
                with httpx.stream('GET', url, timeout=120.0, follow_redirects=True) as resp:
                    resp.raise_for_status()
                    with open(dest, 'wb') as f:
                        for chunk in resp.iter_bytes(chunk_size=8192):
                            f.write(chunk)

                if dest.stat().st_size == 0:
                    raise SeedanceError('下载文件为空')

                logger.info(f'下载完成：{dest} ({dest.stat().st_size} bytes)')
                return str(dest)

            except Exception as e:
                logger.warning(f'下载失败（尝试 {attempt + 1}/{max_retries}）：{e}')
                if attempt == max_retries - 1:
                    raise SeedanceError(f'下载失败（重试 {max_retries} 次后）：{e}')
                time.sleep(2 ** attempt)  # 指数退避

        raise SeedanceError('下载失败（未知错误）')

    def close(self):
        """关闭底层 HTTP 客户端。"""
        try:
            self._sdk.close_sync()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_client_from_args(args) -> SeedanceClient:
    """从命令行参数创建 SeedanceClient。

    优先级：CLI > 环境变量 > 默认值
    """
    api_key = getattr(args, 'seedance_key', None) or os.environ.get('SEEDANCE_API_KEY')
    base_url = getattr(args, 'seedance_host', None) or os.environ.get('SEEDANCE_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
    model = getattr(args, 'seedance_model', None) or os.environ.get('SEEDANCE_MODEL', 'seedance-2.0')
    poll_timeout = getattr(args, 'poll_timeout', 600)

    if not api_key:
        raise SeedanceAuthError('未配置 Seedance API Key（--seedance-key 或 SEEDANCE_API_KEY）')

    return SeedanceClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=poll_timeout,
    )