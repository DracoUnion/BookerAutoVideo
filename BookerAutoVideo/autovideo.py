import yaml
from os import path
import sys
import librosa
from io import BytesIO
from paddlespeech.cli.tts.infer import TTSExecutor
from moviepy.editor import *
from .autovideo_config import config
from EpubCrawler.util import request_retry

DIR = path.abspath(path.dirname(__file__))

def md2playbook(args):
    fname = args.fname
    if not fname.endswith('.md'):
        print('请提供 Markdown 文件')
        return

def audio_len(data):
    y, sr = librosa.load(BytesIO(data))
    return librosa.get_duration(y=y, sr=sr)

def exec_tts(text):
    tts = TTSExecutor()
    fname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.wav')
    tts(text=text, output=fname)
    data = open(fname, 'rb').read()
    os.unlink(fname)
    return data

# 素材预处理
def preproc_asset(config):
    # 加载或生成内容
    for cont in config['contents']:
        if cont['type'].endswith(':file'):
            fname = path.join(cfg_fname, cont['value'])
            cont['asset'] = open(fname, 'rb').read()
        elif cont['type'].endswith(':url'):
            url = cont['value']
            print(f'下载：{url}')
            cont['asset'] = request_retry('GET', url)
        elif cont['type'] == 'audio:tts':
            text = cont['value']
            print(f'TTS：{text}')
            cont['asset'] = exec_tts(text)
        elif cont['type'] == 'image:novelai':
            pass # TODO 待实现
            
    config['contents'] = [
        c for c in config['contents']
        if 'asset' in c
    ]
    
    # TODO 剪裁图片
    
    # 如果第一张不是图片，则提升第一个图片
    idx = -1
    for i, c in enumerate(config['contents']):
        if c['type'].startswith('image:'):
            idx = i
            break
    if idx == -1:
        print('内容中无图片，无法生成视频')
        sys.exit()
    if idx != 0:
        c = config['contents'][idx]
        del config['contents'][idx]
        config['contents'].insert(0, c)

# 内容成帧
def contents2frame(contents):
    frames = []
    for c in contents:
        if c['type'].startswith['image:']:
            frames.append({
                'image': c['asset'],
                'audios': [],
            })
        elif c['type'].startswith['audio:']:
            if len(frames) == 0: continue
            frames[-1]['audios'].append({
                'audio': c['asset'],
                'len': audio_len(c['asset']),
                'subtitle': c['value'] if c['type'] == 'audio:tts' else '',
            })
    for f in frames:
        f['len'] = sum([a['len'] for a in f['audios']])
    return frames

# 组装视频
def make_video(frames):
    clips = []
    # 图像部分
    st = 0
    for f in frames:
        clip = (
            ImageClip(f['asset'])
                .set_duration(f['len'])
                .resize(height=int(config['size'][1] * 0.9))
                .set_pos(("center", 0))
                .set_start(st)
        )
        clips.append(clip)
        st += f['len']
    # 音频部分
    st = 0
    for f in frames:
        for a in f['audios']:
            clip = AudioFileClip(a['asset']).set_start(st)
            clips.append(clip)
            if f['subtitle']:
                clip = (
                    TextClip(
                        f['subtitle'], 
                        font=path.join(DIR, 'asset', config['font']), 
                        fontsize=35, 
                        color='white', 
                        method='label'
                    )
                        .set_position(("center", "bottom"))
                        .set_duration(a['len'])
                        .set_start(st)
                )
                clips.append(clip)
            st += a['len']
    video = CompositeVideoClip(clips, size=config['size'])
    # 合并片头片尾
    if config['header']:
        header_fname = path.join(cfg_fname, config['header'])
        header = VideoFileClip(header_fname).resize(config['size'])
        video = concatenate_videoclips([header, video])
    if config['footer']:
        footer_fname = path.join(cfg_fname, config['footer'])
        footer = VideoFileClip(footer_fname).resize(config['size'])
        video = concatenate_videoclips([video, footer])
    return video

def autovideo(args):
    cfg_fname = args.config
    if not cfg_fname.endswith('.yml'):
        print('请提供 YAML 文件')
        return
    user_cfg = yaml.safe_load(cfg_fname)
    config.update(user_cfg)
    
    if not config['contents']:
        print('内容为空，无法生成')
        return
        
    # 素材预处理
    preproc_asset(config)
    # 转换成帧的形式
    frames = contents2frame(config['contents'])
    # 组装视频
    video = make_video(frames)
    # 写文件
    video_fname = fname_escape(config['name']) + '.' + config['format']
    video.write_videofile(video_fname, fps=60, remove_temp=False, verbose=True)
    print(video_fname)
    