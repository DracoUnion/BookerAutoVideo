import argparse
import sys
import math
import warnings
from . import __version__
from .autovideo import *
from .video2txt import *
from .keyframe import *
from .imgsim import *
from .clip import *
from .split import *

warnings.filterwarnings("ignore")

def main():
    openai_key = os.environ.get('OPENAI_API_KEY')
    openai_url = os.environ.get('OPENAI_BASE_URL')

    parser = argparse.ArgumentParser(prog="BookerAutoVideo", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"PYBP version: {__version__}")
    parser.add_argument("-k", "--key", default=openai_key, help="OpenAI API key")
    parser.add_argument("-H", "--host", default=openai_url, help="api host")
    parser.add_argument("-P", "--proxy", help="proxy")
    parser.set_defaults(func=lambda x: parser.print_help())
    subparsers = parser.add_subparsers()

    autovid_parser = subparsers.add_parser("gen", help="generate video")
    autovid_parser.add_argument("fname", help="src file name")
    autovid_parser.add_argument("-m", "--model", default='dall-e-3', help="model name")
    autovid_parser.add_argument("-r", "--retry", type=int, default=1_000_000, help="times of retry")
    autovid_parser.add_argument("-t", "--threads", type=int, default=8, help="num of threads")
    autovid_parser.add_argument("-p", "--one-pic", help="whether to use one pic for all frames, and it's path")
    autovid_parser.set_defaults(func=autovideo)

    video2txt_parser = subparsers.add_parser("totxt", help="convert audio to text")
    video2txt_parser.add_argument("fname", help="file name")
    video2txt_parser.add_argument("-T", "--threads", type=int, default=8, help="num of threads")
    video2txt_parser.add_argument("-I", "--no-image", action='store_true', help="whether to not catch screenshots")
    video2txt_parser.add_argument(
        "-w", "--whisper", 
        default=os.environ.get('WHISPER_CPP_MODEL_PATH', ''), 
        help="whisper.cpp model path"
    )
    video2txt_parser.add_argument("-l", "--lang", default='zh',  help="language")
    video2txt_parser.add_argument("-m", "--model-path", default=os.environ.get('PPT_MODEL_PATH', ''), help="PPT model path")
    video2txt_parser.add_argument("-s", "--batch-size", type=int, default=32, help="batch_size")
    video2txt_parser.add_argument("-t", "--diff-thres", type=float, default=0.1, help="img diff thres")
    video2txt_parser.add_argument("-p", "--ppt-thres", type=float, default=0.4, help="img ppt thres")
    video2txt_parser.add_argument("-c", "--color", type=float, default=0.4, help="color entro")
    video2txt_parser.add_argument("-H", "--hog", type=float, default=0.5, help="hog entro")
    video2txt_parser.add_argument("--left", type=float, default=0, help="left cut 0~1")
    video2txt_parser.add_argument("--right", type=float, default=0, help="right cut 0~1")
    video2txt_parser.add_argument("--bottom", type=float, default=0, help="bottom cut 0~1")
    video2txt_parser.add_argument("--top", type=float, default=0, help="top cut 0~1")
    video2txt_parser.set_defaults(
        opti_mode='quant',
        rate=0.2,
        direction=DIR_B,
        func=video2txt_handle,
    )

    kf_parser = subparsers.add_parser("ext-kf", help="extract keyframes")
    kf_parser.add_argument("fname", help="file name")
    kf_parser.add_argument("-o", "--opti-mode", default="quant", help="img opti mode, default 'none'")
    kf_parser.add_argument("-r", "--rate", type=float, default=0.2, help="how many frames to extract in 1s")
    kf_parser.add_argument("-d", "--direction", choices=[DIR_F, DIR_B, DIR_T], default=DIR_B, help="the direction used to calc frame diff")
    kf_parser.add_argument("-t", "--diff-thres", type=float, default=0.1, help="img diff thres")
    kf_parser.add_argument("-T", "--threads", type=int, default=8, help="#threads")
    kf_parser.add_argument("-p", "--ppt-thres", type=float, default=0.4, help="img ppt thres")
    kf_parser.add_argument("-m", "--model-path", default=os.environ.get('PPT_MODEL_PATH', ''), help="PPT model path")
    kf_parser.add_argument("-s", "--batch-size", type=int, default=32, help="batch_size")
    kf_parser.add_argument("-c", "--color", type=float, default=0.4, help="color entro")
    kf_parser.add_argument("-H", "--hog", type=float, default=0.5, help="hog entro")
    kf_parser.add_argument("--left", type=float, default=0, help="left cut 0~1")
    kf_parser.add_argument("--right", type=float, default=0, help="right cut 0~1")
    kf_parser.add_argument("--bottom", type=float, default=0, help="bottom cut 0~1")
    kf_parser.add_argument("--top", type=float, default=0, help="top cut 0~1")
    kf_parser.set_defaults(func=extract_keyframe_file)

    clip_parser = subparsers.add_parser("clip-test", help="test clip")
    clip_parser.add_argument("img", help="img file name for dir")
    clip_parser.add_argument("-c", "--cates", default="图文,幻灯片,人像,景物", help="cates")
    clip_parser.add_argument("-m", "--model-path", default=os.environ.get('CLIP_PATH', ''), help="clip path")
    clip_parser.add_argument("-s", "--batch-size", type=int, default=32, help="batch_size")
    clip_parser.set_defaults(func=clip_test)

    metric_parser = subparsers.add_parser("metric", help="test clip")
    metric_parser.add_argument("fname", help="img file name for dir")
    metric_parser.set_defaults(func=img_metric_handle)

    split_parser = subparsers.add_parser("split", help="split video")
    split_parser.add_argument("fname", help="video file name for dir")
    split_parser.add_argument("seg", help="nseg, or duration")
    split_parser.set_defaults(func=split)


    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__': main()