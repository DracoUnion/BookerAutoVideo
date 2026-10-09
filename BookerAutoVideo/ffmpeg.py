from os import path
import re
import subprocess  as subp
import tempfile
import uuid
import os
import shutil

def _safe_mkdir(dir):
    try: os.mkdir(dir)
    except: pass

def _safe_remove(name):
    try: os.remove(name)
    except: pass

def _safe_rmdir(name):
    try: shutil.rmtree(name)
    except: pass


def ffmpeg_conv_fmt(video, from_, to):
    prefix = uuid.uuid4().hex
    from_fname = path.join(tempfile.gettempdir(), f'{prefix}.{from_}')
    to_fname = path.join(tempfile.gettempdir(), f'{prefix}.{to}')
    open(from_fname, 'wb').write(video)
    cmd = ['ffmpeg', '-i', from_fname, '-c', 'copy', to_fname, '-y']
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True, stdin=subp.PIPE).communicate()
    res = open(to_fname, 'rb').read()
    _safe_remove(from_fname)
    _safe_remove(to_fname)
    return res


def ffmpeg_cat(videos, fmt='mp4'):
    tmpdir = path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    _safe_mkdir(tmpdir)
    for i, video in enumerate(videos):
        fname = path.join(tmpdir, f'{i}.{fmt}')
        open(fname, 'wb').write(video)
    video_fnames = [
        'file ' + path.join(tmpdir, f'{i}.{fmt}').replace('\\', '\\\\')
        for i in range(len(videos))
    ]
    video_li_fname = path.join(tmpdir, f'list.txt') 
    open(video_li_fname, 'w', encoding='utf8').write('\n'.join(video_fnames))
    ofname = path.join(tmpdir, f'res.{fmt}')
    cmd = [
        'ffmpeg', '-f', 'concat', '-safe', '0',
        '-i', video_li_fname, '-c', 'copy', ofname, '-y',
    ]
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True, stdin=subp.PIPE).communicate()
    res = open(ofname, 'rb').read()
    _safe_rmdir(tmpdir)
    return res


def ffmpeg_add_srt(video, srt, fontname='黑体', video_fmt='mp4'):
    tmpdir = path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    _safe_mkdir(tmpdir)
    vfname = path.join(tmpdir, f'video.{video_fmt}')
    open(vfname, 'wb').write(video)
    sfname = path.join(tmpdir, f'subtitle.srt')
    open(sfname, 'w', encoding='utf8').write(srt)
    res_fname = path.join(tmpdir, f'merged.{video_fmt}')
    cmd = [
        'ffmpeg', '-i', f'video.{video_fmt}', 
        '-vf', f"subtitles=subtitle.srt:force_style='FontName={fontname}'", 
        res_fname, '-y',
    ]
    '''
    cmd = [
        'ffmpeg', '-i', vfname, '-i', sfname, 
        '-c', 'copy', res_fname, '-y',
    ]
    if video_fmt == 'mp4': cmd += ['-c:s', 'mov_text']
    '''
    print(f'cmd: {cmd}')
    subp.Popen(cmd, shell=True, cwd=tmpdir, stdin=subp.PIPE).communicate()
    res = open(res_fname, 'rb').read()
    _safe_rmdir(tmpdir)
    return res


def ffmpeg_merge_video_audio(video, audio, video_fmt='mp4', audio_fmt='mp4'):
    tmpdir = path.join(tempfile.gettempdir(), uuid.uuid4().hex)
    _safe_mkdir(tmpdir)
    vfname = path.join(tmpdir, f'video.{video_fmt}')
    v0fname = path.join(tmpdir, f'video0.{video_fmt}')
    open(vfname, 'wb').write(video)
    afname = path.join(tmpdir, f'audio.{audio_fmt}')
    a0fname = path.join(tmpdir, f'audio0.{audio_fmt}')
    open(afname, 'wb').write(audio)
    res_fname = path.join(tmpdir, f'merged.{video_fmt}')
    cmds = [
        ['ffmpeg', '-i', vfname, '-vcodec', 'copy', '-an', v0fname, '-y'],
        ['ffmpeg', '-i', afname, '-acodec', 'copy', '-vn', a0fname, '-y'],
        ['ffmpeg', '-i', a0fname, '-i', v0fname, '-c', 'copy', res_fname, '-y'],
    ]
    for cmd in cmds:
        print(f'cmd: {cmd}')
        subp.Popen(cmd, shell=True, stdin=subp.PIPE).communicate()
    res = open(res_fname, 'rb').read()
    _safe_rmdir(tmpdir)
    return res


def ffmpeg_get_info_fname(fname):
    cmd = ['ffmpeg', '-i', fname]
    print(f'cmd: {cmd}')
    r = subp.Popen(
            cmd, stdin=subp.PIPE,
            stdout=subp.PIPE, 
            stderr=subp.PIPE, shell=True
    ).communicate()
    text = r[1].decode('utf8')
    res = {}
    m = re.search(r'Duration:\x20(\d+):(\d+):(\d+)(.\d+)', text)
    if m:
        hr = int(m.group(1))
        min_ = int(m.group(2))
        sec = int(m.group(3))
        ms = float(m.group(4))
        res['duration'] = hr * 3600 + min_ * 60 + sec + ms
    m = re.search(r'(\d+)\x20fps', text)
    if m:
        res['fps'] = int(m.group(1))
    m = re.search(r'(\d+)\x20Hz', text)
    if m:
        res['sr'] = int(m.group(1))
    return res


def ffmpeg_get_info(video, fmt='mp4'):
    if isinstance(video, bytes):
        fname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.' + fmt)
        open(fname, 'wb').write(video)
    else:
        fname = video
    info = ffmpeg_get_info_fname(fname)
    
    if isinstance(video, bytes):
        _safe_remove(fname)
    return info
    
