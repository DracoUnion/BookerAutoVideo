import argparse
import os
import sys
import math
import warnings
from . import __version__
from . import autovideo, video2txt, keyframe, imgsim, clip, split

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

    autovideo.reg_subparser(subparsers)
    video2txt.reg_subparser(subparsers)
    keyframe.reg_subparser(subparsers)
    imgsim.reg_subparser(subparsers)
    clip.reg_subparser(subparsers)
    split.reg_subparser(subparsers)


    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__': main()