from .util import *


def split(args):
    fname = args.fname
    info = ffmpeg_get_info_fname(fname)
    