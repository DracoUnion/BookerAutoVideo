from .util import *


def split(args):
    fname = args.fname
    ext = extname(fname)
    info = ffmpeg_get_info_fname(fname)
    RE_NSEG = r'^(\d+)$'
    RE_HMS = r'^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$'
    if m := re.search(RE_NSEG, args.seg):
        nseg = int(m.group(1))
        dura = math.ceil(info['duration'] / nseg)
    elif m := re.search(RE_HMS, args.seg):
        tmstr = m.group()
        dura = 0
        
        times = m.groups()[::-1]
        dura = int(times[0])
        if len(times) > 2:
            dura += int(times[1] * 60)
        if len(times) > 3:
            dura += int(times[2] * 3600)

    for i in range(0, dura, info['duration']):
        st, ed = i, i + dura - 1
        ofname = fname[:-len(ext)-1] + f'_{i}.{ext}'
        print(ofname)
        cmd = [
            'ffmpeg', '-i', fname, 
            '-ss', str(st), 
            '-to', str(ed), 
            '-c', 'copy', ofname
        ]
        subp.Popen(cmd, shell=True, stdin=subp.PIPE).communicate()

    