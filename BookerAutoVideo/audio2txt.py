import paddle
import whisper
from paddlespeech.cli.text.infer import TextExecutor 

def audio2txt_handle(args):
    fname = args.fname
    model = whisper.load_model(args.model)
    word = model.transcribe(fname, fp16=False, language='Chinese')['text']
    text_executor = TextExecutor()
    text = text_executor(
        text=word,
        task='punc',
        model='ernie_linear_p3_wudao',
        device=paddle.get_device(),
    )
    open(fname + '.txt', 'w', encoding='utf8').write(text)
    print(fname + '.txt')
