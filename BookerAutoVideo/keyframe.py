import av
import av.datasets
import os
import shutil
import math
import sys

def get_keyframes(fname):
    container = av.open(fname)

    stream = container.streams.video[0]
    stream.codec_context.skip_frame = 'NONKEY'

    res = []
    for frame in container.decode(stream):
        bio = BytesIO()
        frame.to_image().save(bio, format="PNG",compress=9)
        res.append({
            'idx': frame.index,
            'pts': frame.pts,
            'dts': frame.dts,
            'data': bio.getvalue(),
        })
    return res

def ext_keyframe(args):
    fname = args.file
    print(fname)
    opath = fname + "_img"
    print(opath)
    if os.path.isdir(opath):
        shutil.rmtree(opath)
    os.mkdir(opath)
    kfs = get_keyframes(fname)
    for kf in kfs:
        fname = f'idx{kf["idx"]}_pts{kf["pts"]}_dts{kf["dts"]}.png'
        open(path.join(opath, fname), 'wb').write(kf['data'])

