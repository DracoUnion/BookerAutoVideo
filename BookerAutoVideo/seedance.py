"""Seedance API 客户端（基于 seedance-sdk 封装）。

对应任务书第 7.4 / 7.7 节。封装文生图任务、图生视频任务、轮询、取 URL、下载。
使用官方 seedance-sdk；SDK 不直接包裹文生图与下载时，复用其 client_sync（httpx.Client）
隔离差异，主流程只依赖抽象方法，不散落在其他模块。
"""
import os
import time
import logging
from typing import Optional
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


def _status_val(task_data) -> str:
    """提取任务状态字符串。"""
    status = getattr(task_data, 'status', None)
    if status is None:
        return str(getattr(task_data, 'status', 'unknown'))
    return status.value if hasattr(status, 'value') else str(status)


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
            api_key: Seedance API Key（SDK 要求以 sk-video- 开头）
            base_url: API 基础 URL
            model: 生成模型名称
            timeout: 轮询总超时（秒）
            poll_interval: 轮询间隔（秒）
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.poll_interval = poll_interval

        # SDK 默认 base_url 指向 https://vibegen.art/api/v1；火山引擎方使用 ark 地址时
        # 保留用户指定的 host，由 SDK 的 client_sync 承担认证头。
        sdk_base_url = base_url if base_url != 'https://ark.cn-beijing.volces.com/api/v3' \
            else 'https://vibegen.art/api/v1'

        try:
            self._sdk = SDKClient(
                api_key=api_key,
                base_url=sdk_base_url,
                timeout=30.0,
                max_retries=3,
            )
        except ConfigurationError as e:
            raise SeedanceAuthError(f'API Key 格式可能不正确：{e}')

        # client_sync 是 SDK 配置好的 httpx.Client（含 Authorization 头），复用它处理
        # SDK 未直接包裹的文生图 / 下载接口，避免在主模块散落 raw HTTP。
        self._http = self._sdk.client_sync

    # ── 文生图（S4）：SDK 未直接包裹，复用 client_sync 调用图生成端点 ────

    def create_image_task(
        self,
        prompt: str,
        ratio: str = '16:9',
        resolution: str = '1080p',
        model: Optional[str] = None,
    ) -> str:
        """创建文生图任务。

        通过 SDK 的 client_sync 调用图生成端点（isolated 在本模块）。
        """
        model_enum = _parse_model(model or self.model)
        aspect_ratio = _parse_aspect_ratio(ratio)
        resolution_enum = _parse_resolution(resolution)

        payload = {
            'model': model_enum.value,
            'prompt': prompt,
            'aspect_ratio': aspect_ratio.value,
            'resolution': resolution_enum.value,
        }

        logger.info(f'创建 Seedance 文生图任务：model={model_enum.value}, ratio={ratio}, resolution={resolution}')
        logger.debug(f'提示词：{prompt[:200]}...')

        url = f'{self.base_url}/images/generations'
        try:
            resp = self._http.post(url, json=payload, timeout=60.0)
        except Exception as e:
            raise SeedanceError(f'创建文生图任务失败：{e}') from e

        if resp.status_code in (401, 403):
            raise SeedanceAuthError(f'Seedance 鉴权失败（{resp.status_code}）：请检查 API Key 与 Base URL')
        resp.raise_for_status()

        data = resp.json()
        task_id = data.get('id') or data.get('task_id')
        if not task_id:
            raise SeedanceError(f'文生图响应中无 task_id：{data}')

        logger.info(f'文生图任务创建成功：task_id={task_id}')
        return task_id

    def wait_image_task(self, task_id: str, timeout: Optional[int] = None,
                        poll_interval: Optional[int] = None) -> str:
        """轮询文生图任务直到完成。

        Returns:
            image_url: 生成图片的下载 URL
        """
        timeout = timeout or self.timeout
        poll_interval = poll_interval or self.poll_interval
        start = time.time()
        logger.info(f'开始轮询文生图任务：task_id={task_id}, timeout={timeout}s')

        while time.time() - start < timeout:
            try:
                task_data = self._sdk.get_task_sync(task_id)
            except Exception as e:
                logger.warning(f'查询文生图任务状态失败：{e}，继续轮询')
                time.sleep(poll_interval)
                continue

            status_val = _status_val(task_data)
            logger.debug(f'文生图任务状态：{status_val}')

            if status_val == TaskStatus.COMPLETED.value:
                result = getattr(task_data, 'result', None)
                if result and getattr(result, 'image_url', None):
                    image_url = str(result.image_url)
                    logger.info(f'文生图任务完成：image_url={image_url}')
                    return image_url
                raise SeedanceError(f'文生图任务成功但无 image_url：{task_data}')

            if status_val == TaskStatus.FAILED.value:
                error = getattr(task_data, 'error', None) or 'Unknown error'
                raise SeedanceError(f'文生图任务失败：{error}')

            time.sleep(poll_interval)

        raise SeedanceTimeoutError(f'文生图轮询超时（{timeout}s），task_id={task_id} 可能仍在运行，可手动续跑')

    def download_image(self, url: str, dest_path: str) -> str:
        """下载生成的图片文件。"""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and dest.stat().st_size > 0:
            logger.info(f'图片已存在，跳过下载：{dest}')
            return str(dest)

        logger.info(f'下载图片：{url} -> {dest}')
        try:
            with self._http.stream('GET', url, timeout=120.0, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(dest, 'wb') as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            raise SeedanceError(f'下载图片失败：{e}') from e

        if dest.stat().st_size == 0:
            raise SeedanceError('下载的图片为空')

        logger.info(f'图片下载完成：{dest} ({dest.stat().st_size} bytes)')
        return str(dest)

    # ── 图生视频（S7）：使用 SDK 的 generate_video_sync / wait_for_completion_sync ────

    def create_video_task(
        self,
        image_path_or_url: str,
        prompt: str,
        duration: int = 8,
        resolution: str = '1080p',
        ratio: str = '16:9',
        audio: bool = True,
        model: Optional[str] = None,
    ) -> str:
        """创建图生视频任务（使用 seedance-sdk）。"""
        model_enum = _parse_model(model or self.model)
        aspect_ratio = _parse_aspect_ratio(ratio)
        resolution_enum = _parse_resolution(resolution)

        request = GenerationRequest(
            model=model_enum,
            prompt=prompt,
            image_urls=[image_path_or_url],
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution_enum,
            generate_audio=audio,
        )

        logger.info(f'创建 Seedance 图生视频任务：model={model_enum.value}, duration={duration}s, ratio={ratio}')
        logger.debug(f'提示词：{prompt[:200]}... 图片：{image_path_or_url}')

        try:
            task_data = self._sdk.generate_video_sync(request)
        except Exception as e:
            error_msg = str(e)
            if '401' in error_msg or '403' in error_msg or 'auth' in error_msg.lower():
                raise SeedanceAuthError(f'Seedance 鉴权失败：{e}')
            raise SeedanceError(f'创建图生视频任务失败：{e}') from e

        task_id = task_data.id
        logger.info(f'图生视频任务创建成功：task_id={task_id}')
        return task_id

    def wait_video_task(self, task_id: str, timeout: Optional[int] = None,
                        poll_interval: Optional[int] = None) -> str:
        """轮询图生视频任务直到完成（使用 seedance-sdk）。"""
        timeout = timeout or self.timeout
        poll_interval = poll_interval or self.poll_interval
        logger.info(f'开始轮询图生视频任务：task_id={task_id}, timeout={timeout}s')

        try:
            task_data = self._sdk.wait_for_completion_sync(
                task_id=task_id,
                timeout=float(timeout),
                poll_interval=float(poll_interval),
                on_progress=lambda t: logger.debug(
                    f'任务进度：status={_status_val(t)}, estimated_time={t.estimated_time}s'
                ),
            )
        except TaskTimeoutError as e:
            raise SeedanceTimeoutError(f'图生视频轮询超时（{timeout}s），task_id={task_id} 可能仍在运行，可手动续跑') from e
        except TaskFailedError as e:
            raise SeedanceError(f'图生视频任务失败：{e}') from e
        except Exception as e:
            error_msg = str(e)
            if '401' in error_msg or '403' in error_msg or 'auth' in error_msg.lower():
                raise SeedanceAuthError(f'Seedance 鉴权失败：{e}')
            raise SeedanceError(f'图生视频轮询异常：{e}') from e

        result = getattr(task_data, 'result', None)
        if result and getattr(result, 'video_url', None):
            video_url = str(result.video_url)
            logger.info(f'图生视频任务完成：video_url={video_url}')
            return video_url

        raise SeedanceError(f'图生视频任务成功但无 video_url：{task_data}')

    def download_video(self, url: str, dest_path: str, max_retries: int = 3) -> str:
        """下载视频文件。"""
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and dest.stat().st_size > 0:
            logger.info(f'视频已存在，跳过下载：{dest}')
            return str(dest)

        logger.info(f'下载视频：{url} -> {dest}')
        for attempt in range(max_retries):
            try:
                with self._http.stream('GET', url, timeout=120.0, follow_redirects=True) as resp:
                    resp.raise_for_status()
                    with open(dest, 'wb') as f:
                        for chunk in resp.iter_bytes(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                if dest.stat().st_size == 0:
                    raise SeedanceError('下载文件为空')
                logger.info(f'下载完成：{dest} ({dest.stat().st_size} bytes)')
                return str(dest)
            except Exception as e:
                logger.warning(f'下载失败（尝试 {attempt + 1}/{max_retries}）：{e}')
                if attempt == max_retries - 1:
                    raise SeedanceError(f'下载失败（重试 {max_retries} 次后）：{e}')
                time.sleep(2 ** attempt)

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
    base_url = getattr(args, 'seedance_host', None) \
        or os.environ.get('SEEDANCE_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
    model = getattr(args, 'seedance_model', None) \
        or os.environ.get('SEEDANCE_MODEL', 'seedance-2.0')
    poll_timeout = getattr(args, 'poll_timeout', 600)

    if not api_key:
        raise SeedanceAuthError('未配置 Seedance API Key（--seedance-key 或 SEEDANCE_API_KEY）')

    return SeedanceClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=poll_timeout,
    )
