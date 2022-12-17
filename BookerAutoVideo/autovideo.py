import yaml
from .autovideo_config import config
from EpubCrawler.util import request_retry

def md2playbook(args):
    fname = args.fname
    if not fname.endswith('.md'):
        print('请提供 Markdown 文件')
        return
        
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
        
    # 读取文件
    for cont in config['contents']:
        if cont['type'].endswith(':file'):
            fname = path.join(cfg_fname, cont['content'])
            cont['