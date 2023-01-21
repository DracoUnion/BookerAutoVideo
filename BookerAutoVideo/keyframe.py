import av
import av.datasets
import os
import shutil
import math
import sys

def ext_keyframe(args):
    fname = args.file
    print(fname)
    img_path = fname + "_img"
    print(img_path)
    if os.path.isdir(img_path):
        shutil.rmtree(img_path)
    os.makedirs(img_path)
    container = av.open(fname)

    stream = container.streams.video[0]
    stream.codec_context.skip_frame = 'NONKEY'

    for frame in container.decode(stream):
        fname = f'idx{frame.index}_pts{frame.pts}_dts{frame.dts}.png'
        frame.to_image().save(
            os.path.join(img_path, fname), 
            compress=True,
        )

