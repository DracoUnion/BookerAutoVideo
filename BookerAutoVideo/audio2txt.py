import paddle
import whisper
from paddlespeech.cli.text.infer import TextExecutor 

def audio2txt_handle(args):
    fname = args.fname
    model = whisper.load_model(args.model)
    r = model.transcribe(fname, fp16=False, language='Chinese')
    words = [s['text'] for s in r['segments']]
    text_executor = TextExecutor()
    text = ''.join([
        text_executor(
            text=w,
            task='punc',
            model='ernie_linear_p3_wudao',
            device=paddle.get_device(),
        ) for w in words
    ])
    open(fname + '.txt', 'w', encoding='utf8').write(text)
    print(fname + '.txt')
