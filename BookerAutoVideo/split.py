from .util import *


def split(args):
    fname = args.fname
    ext = extname(fname)
    info = ffmpeg_get_info_fname(fname)
    RE_NSEG = r'^(\d+)$'
    RE_HMS = r'^(\d+h)?(\d+m)?(\d+s)?$'
    if m := re.search(RE_NSEG, args.seg):
        nseg = int(m.group(1))
        dura = math.ceil(info['duration'] / nseg)
    elif m := re.search(RE_HMS, args.seg):
        dura = 0
        for tmstr in m.groups()[1:]:
            if tmstr is None: continue
            elif 'h' in tmstr:
                tm = int(tmstr[:-1]) * 3600
                dura += tm
            elif 'm' in tmstr:
                tm = int(tmstr[:-1]) * 60
                dura += tm
            elif 's' in tmstr:
                tm = int(tmstr[:-1])
                dura += tm
        nseg = info['duration'] // dura
    else:
        raise ValueError('时间格式错误，只接受表示段落数的整数，或者hms')

    print(f'nseg: {nseg}, duration: {dura}')
    for i in range(0, int(info['duration']), dura):
        st, ed = i, i + dura - 1
        ofname = fname[:-len(ext)-1] + f'_{i}.{ext}'
        print(ofname)
        cmd = [
            'ffmpeg', '-i', fname, 
            '-ss', str(st), 
            '-to', str(ed), 
            '-c', 'copy', ofname
        ]
        print(cmd)
        subp.Popen(cmd, shell=True, stdin=subp.PIPE).communicate()

    