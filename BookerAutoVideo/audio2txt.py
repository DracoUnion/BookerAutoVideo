import paddle
import whisper
import re
from os import path
from paddlespeech.cli.text.infer import TextExecutor 

def merge_words(words, maxl=500):
    res = []
    st = 0
    l = 0
    for i, w in enumerate(words):
        if l >= maxl:
            res.append(''.join(words[st:i]))
            st = i
            l = 0
        l += len(w)
    res.append(''.join(words[st:]))
    return res

def audio2txt_handle(args):
    fname = args.fname
    # 语音识别
    model = whisper.load_model(args.model)
    r = model.transcribe(fname, fp16=False, language='Chinese')
    words = [s['text'] for s in r['segments']]
    # 标点修正
    words = merge_words(words)
    text_executor = TextExecutor()
    text = ''.join([
        text_executor(
            text=w,
            task='punc',
            model='ernie_linear_p3_wudao',
            device=paddle.get_device(),
        ) for w in words
    ])
    # 排版
    text = re.sub(r'(.{50,100}(?:。|！|？))', r'\1\n\n', text)
    title = path.basename(fname)
    title = re.sub(r'\.\w+$', '', title)
    text = f'# {title}\n\n{text}'
    open(fname + '.md', 'w', encoding='utf8').write(text)
    print(fname + '.md')
