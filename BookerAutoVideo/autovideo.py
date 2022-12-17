import yaml
from os import path
from paddlespeech.cli.tts.infer import TTSExecutor
from .autovideo_config import config
from EpubCrawler.util import request_retry

def md2playbook(args):
    fname = args.fname
    if not fname.endswith('.md'):
        print('请提供 Markdown 文件')
        return
        
def exec_tts(text):
    tts = TTSExecutor()
    fname = path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.wav')
    tts(text=text, output=fname)
    data = open(fname, 'rb').read()
    os.unlink(fname)
    return data
        
def autovideo(args):
    cfg_fname = args.fname
    if not cfg_fname.endswith('.yml'):
        print('请提供 YAML 文件')
        return
    user_cfg = yaml.safe_load(cfg_fname)
    config.update(user_cfg)
    
    if not config['contents']:
        print('内容为空，无法生成')
        return
        
    # 素材预处理
    for cont in config['contents']:
        if cont['type'].endswith(':file'):
            fname = path.join(cfg_fname, cont['value'])
            cont['asset'] = open(fname, 'rb').read()
        elif cont['type'].endswith(':url'):
            cont['asset'] = request_retry('GET', cont['value'])
        elif cont['type'] == 'audio:tts':
            cont['asset'] = exec_tts(cont['value'])
            
    config['contents'] = [
        c for c in config['contents']
        if 'asset' in c
    ]
    