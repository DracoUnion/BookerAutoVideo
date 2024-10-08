
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import soundfile as sf  # 用于读取和裁剪音频文件
import os
from os import path
import pandas as pd
import subprocess as subp
import tempfile 
import uuid
import torch
import argparse
from .util import *


def sencevoice(args):

    # 模型路径
    model_dir = args.whisper #r"D:\src\SenseVoiceSmall"
    vad_model_dir = "fsmn-vad"  # VAD模型路径
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 音频文件路径
    vid_fname = args.fname
     
    if not vid_fname.endswith('.mp3'):
        aud_fname = tempfile.gettempdir() + '/' + uuid.uuid4().hex + '.mp3'
        subp.Popen([
            'ffmpeg', '-i', vid_fname, 
            '-vn', '-acodec', 'libmp3lame', 
            aud_fname
        ], shell=True, stdin=subp.PIPE).communicate()
        if not path.isfile(aud_fname):
            raise FileNotFoundError(f'{fname} 转换失败')
    else:
        aud_fname = vid_fname
    # 加载VAD模型
    vad_model = AutoModel(
        model=vad_model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device=device,
        disable_update=True
    )
     
    # 使用VAD模型处理音频文件
    vad_res = vad_model.generate(
        input=aud_fname,
        cache={},
        max_single_segment_time=30000,  # 最大单个片段时长
    )
     
    # 从VAD模型的输出中提取每个语音片段的开始和结束时间
    segments = vad_res[0]['value']  # 假设只有一段音频，且其片段信息存储在第一个元素中
     
    # 加载原始音频数据
    audio_data, sample_rate = sf.read(aud_fname)
     
     
    # 定义一个函数来裁剪音频
    def crop_audio(audio_data, start_time, end_time, sample_rate):
        start_sample = int(start_time * sample_rate / 1000)  # 转换为样本数
        end_sample = int(end_time * sample_rate / 1000)  # 转换为样本数
        return audio_data[start_sample:end_sample]
     
     
    # 加载SenseVoice模型
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cuda:0",
        disable_update=True
    )
     
    # 对每个语音片段进行处理
    results = []
    for segment in segments:
        start_time, end_time = segment  # 获取开始和结束时间
        cropped_audio = crop_audio(audio_data, start_time, end_time, sample_rate)
     
        # 将裁剪后的音频保存为临时文件
        aud_seg_fname = tempfile.gettempdir() + '/' + uuid.uuid4().hex + '.mp3'
        sf.write(aud_seg_fname, cropped_audio, sample_rate)
     
        # 语音转文字处理
        res = model.generate(
            input=aud_seg_fname,
            cache={},
            language="auto",  # 自动检测语言
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  # 启用 VAD 断句
            merge_length_s=10000,  # 合并长度，单位为毫秒
        )
        # 处理输出结果
        text = rich_transcription_postprocess(res[0]["text"])
        # 添加时间戳
        results.append({"start": start_time / 1000, "end": end_time / 1000, "text": text})  # 转换为秒
        results[-1]['time'] = results[-1]['start']
        os.unlink(aud_seg_fname)
     
    if vid_fname != aud_fname:
        os.unlink(aud_fname)
    return results
        # 输出结果
        #print(results[0])
        # for result in results:
        #    print(f"Start: {result['start']} s, End: {result['end']} s, Text: {result['text']}")
if __name__ == '__main__':
    fname = 'C:/Users/Administrator/Videos/leemldl/【国语+资料下载】李宏毅 HYLEE ｜ 机器学习(深度学习)(2021最新·完整版) - P3：L3- 机器学习任务攻略 - ShowMeAI - BV1fM4y137M4.mp4'
    res = sencevoice(argparse.Namespace(
        fname=fname,
        model_path="D:\src\SenseVoiceSmall"
    ))
    print(res[0])