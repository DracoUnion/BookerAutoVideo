import auditok
import os
import paddle
import re
import os
import shutil
import tempfile
import uuid
from os import path
import subprocess as subp
from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.text.infer import TextExecutor 

def check_ffmpeg():
    r = subp.Popen(
        ['ffmpeg', '-version'], 
        shell=True,
        stdout=subp.PIPE,
        stderr=subp.PIPE,
    ).communicate()
    return not r[1]

def split_audio(
    fname, target, 
    mmin_dur=1, 
    mmax_dur=100000, 
    mmax_silence=1, 
    menergy_threshold=55
):
    bname = path.basename(fname)
    bname_pre = re.sub(r'\.\w+', '', bname)

    audio_regions = list(auditok.split(
        fname,
        min_dur=mmin_dur,  # minimum duration of a valid audio event in seconds
        max_dur=mmax_dur,  # maximum duration of an event
        # maximum duration of tolerated continuous silence within an event
        max_silence=mmax_silence,
        energy_threshold=menergy_threshold  # threshold of detection
    ))

    l = len(str(len(audio_regions)))
    for i, r in enumerate(audio_regions):
        ofname = f'{bname_pre}_{i:0{l}d}_{r.meta.start:.3f}-{r.meta.end:.3f}.wav'
        ofname = path.join(target, ofname)
        print(ofname)
        r.save(ofname)

def convert_to_wav(fname):
    ofname = fname + '.wav'
    subp.Popen(
        ['ffmpeg', '-y', '-i', fname, '-vn', ofname],
        shell=True,
    ).communicate()
    return ofname

def punc_fix(words):
    text_executor = TextExecutor()
    r = ''.join([
        text_executor(
            text=w,
            task='punc',
            model='ernie_linear_p3_wudao',
            device=paddle.get_device(),
        ) for w in words
    ])
    print(r)
    return r

def audio2txt(dir):
    asr_executor = ASRExecutor()
    fnames = os.listdir(dir)
    fnames.sort()
    words = []
    for fname in fnames:
        print(f'正在识别：{fname}')
        text = asr_executor(
            audio_file=path.join(dir, fname),
            device=paddle.get_device(), force_yes=True
        )
        print(text)
        words.append(text)
    return words

def audio2txt_handle(args):
    if not check_ffmpeg():
        print('未找到 ffmpeg，请先下载并放到系统变量 PATH 路径下')
        return
    fname = args.fname
    dir = path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    os.mkdir(dir)
    if not fname.endswith('.wav'):
        fname = convert_to_wav(fname)
    split_audio(fname, dir, 1, 50)
    words = audio2txt(dir)
    text = punc_fix(words)
    open(fname + '.txt', 'w', encoding='utf8').write(text)
    print(fname + '.txt')
    shutil.rmtree(dir)
