from .util import *

def parse_seg_dura(seg: str, total: float):
    RE_NSEG = r'^(\d+)$'
    RE_HMS = r'^(\d+h)?(\d+m)?(\d+s)?$'

    if m := re.search(RE_NSEG, seg):
        nseg = int(m.group(1))
        dura = math.ceil(total / nseg)
    elif m := re.search(RE_HMS, seg):
        dura = 0
        gs = m.groups()
        assert len(gs) == 3
        if gs[0] is not None:
            tm = int(gs[0][:-1]) * 3600
            dura += tm
        if gs[1] is not None:
            tm = int(gs[1][:-1]) * 60
            dura += tm
        if gs[2] is not None:
            tm = int(gs[2][:-1])
            dura += tm
        # 至少应该有一段才对
        nseg = max(total // dura, 1)
    else:
        raise ValueError('时间格式错误，只接受表示段落数的整数，或者hms')
    return nseg, dura

def split(args):
    fname = args.fname
    ext = extname(fname)
    info = ffmpeg_get_info_fname(fname)
    nseg, dura = parse_seg_dura(args.seg, info['duration'])
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

    