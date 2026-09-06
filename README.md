# BookerAutoVideo

音视频自动处理工具，提供文本转视频、语音转文字、关键帧提取、图片分类、视频切分等子命令。

```bash
BookerAutoVideo [-h] [-v] [-k KEY] [-H HOST] [-P PROXY] {gen,totxt,ext-kf,clip-test,metric,split} ...
```

安装后在命令行可使用三个等价命令：`BookerAutoVideo`、`bav`、`auto-video`。

## 安装

```bash
pip install .
```

**外部依赖**（需自行安装并加入 PATH）：

+ `ffmpeg` —— 所有音视频处理都基于它
+ 语音识别模型（SenseVoice）—— `totxt` 子命令的 ASR 引擎，需要 SenseVoice 模型目录（目录内含 `fsmn-vad` 子目录），路径通过 `-w` 或环境变量 `WHISPER_CPP_MODEL_PATH` 指定

## 全局参数

| 参数 | 说明 |
| --- | --- |
| `-h, --help` | 显示帮助 |
| `-v, --version` | 显示版本号 |
| `-k KEY, --key KEY` | OpenAI API Key，默认取环境变量 `OPENAI_API_KEY` |
| `-H HOST, --host HOST` | API 地址，默认取环境变量 `OPENAI_BASE_URL` |
| `-P PROXY, --proxy PROXY` | 代理 |

## gen —— 文本转视频

把 Markdown/TXT 文本自动生成带配音、配图和字幕的视频。

```bash
BookerAutoVideo gen <fname> [options]
```

+ **输入**：`.md` 或 `.txt` 文件，按段落逐句 TTS 配音、TTI 配图，再用 ffmpeg 合成视频并烧入字幕
+ **输出**：与输入同名的 `.mp4` 视频（若同名视频已存在会报错）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `fname` | — | 源文件名（必填） |
| `-m, --model` | `OPENAI_TTI_MODEL` | 图像生成模型名 |
| `-r, --retry` | `1000000` | 图像生成失败重试次数 |
| `-t, --threads` | `8` | 并发线程数 |
| `-p, --one-pic` | — | 所有帧共用一张图片，参数为其路径 |
| `-pf, --prefix` | `''` | 发往图像生成模型的 prompt 前缀 |
| `-sf, --suffix` | `''` | 发往图像生成模型的 prompt 后缀 |

示例：

```bash
BookerAutoVideo -k $OPENAI_API_KEY gen story.md
BookerAutoVideo gen story.md -m dall-e-3 -t 4
```

## totxt —— 音视频转文字

把音频或视频转成带时间轴排版的 Markdown，视频还会提取关键帧图片插入文档。

```bash
BookerAutoVideo totxt <fname> [options]
```

+ **输入**：音频/视频文件，或一个目录（会逐个处理目录内的媒体文件）
+ **处理流程**：语音识别（SenseVoice）→ 若为视频且未加 `-I`，提取关键帧写入 `![](img/xxx.png)` → 排版保存为同名 `.md`
+ **输出**：同名 `.md` 以及同目录下的 `img/` 图片文件夹
+ 同目录已存在同名 `.srt` 时会直接解析字幕，跳过 ASR

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `fname` | — | 文件名或目录（必填） |
| `-t, --threads` | `8` | 线程数 |
| `-I, --no-image` | `False` | 不抓取视频截图，仅做语音识别 |
| `-w, --whisper` | `WHISPER_CPP_MODEL_PATH` | 语音识别模型目录（SenseVoice，含 `fsmn-vad`） |
| `-l, --lang` | `zh` | 语言 |
| `-m, --model-path` | `PPT_MODEL_PATH` | 人像/景物过滤模型路径 |
| `-s, --batch-size` | `32` | 模型推理批大小 |
| `-dt, --diff-thres` | `0.1` | 帧间差阈值 |
| `-pt, --ppt-thres` | `0.4` | PPT 过滤阈值 |
| `-c, --color` | `0.4` | 颜色熵阈值 |
| `-H, --hog` | `0.5` | HOG 熵阈值 |
| `--left / --right / --bottom / --top` | `0` | 画面裁剪比例（0~1） |

示例：

```bash
BookerAutoVideo totxt video.mp4 -w D:\src\SenseVoiceSmall
BookerAutoVideo totxt lectures/              # 批量处理整个目录
```

## ext-kf —— 关键帧提取

从视频中提取关键帧，保存为图片序列。

```bash
BookerAutoVideo ext-kf <fname> [options]
```

+ **处理流程**：按 `-r/--rate` 抽帧 → 计算 `colorfulness`/`hog` 过滤低信息量帧 → 按帧间差去重 → 可选 PPT 模型过滤人像/景物帧 → 图像优化
+ **输出**：`<name>_keyframe/` 目录，文件名为 `keyframe_<hms>.png`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `fname` | — | 视频文件名（必填） |
| `-o, --opti-mode` | `quant` | 图像优化模式：`none` / `quant` / `grid` / `trunc` / `thres` |
| `-r, --rate` | `0.2` | 每秒抽取帧数 |
| `-d, --direction` | `backward` | 帧差计算方向：`forward` / `backward` / `twoway` |
| `-dt, --diff-thres` | `0.1` | 帧间差阈值 |
| `-t, --threads` | `8` | 线程数 |
| `-pt, --ppt-thres` | `0.4` | PPT 过滤阈值 |
| `-m, --model-path` | `PPT_MODEL_PATH` | 人像/景物过滤模型路径（为空则跳过该步） |
| `-s, --batch-size` | `32` | 模型推理批大小 |
| `-c, --color` | `0.4` | 颜色熵阈值 |
| `-H, --hog` | `0.5` | HOG 熵阈值 |
| `--left / --right / --bottom / --top` | `0` | 画面裁剪比例（0~1） |

示例：

```bash
BookerAutoVideo ext-kf lecture.mp4 -r 0.5 -o quant
BookerAutoVideo ext-kf lecture.mp4 -m PPTModel -pt 0.6
```

## clip-test —— 图片分类

用中文 CLIP 模型对一张图片或一个目录里的所有图片分类。

```bash
BookerAutoVideo clip-test <img> [options]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `img` | — | 图片文件或目录（必填） |
| `-c, --cates` | `图文,幻灯片,人像,景物` | 候选类别，逗号分隔 |
| `-m, --model-path` | `CLIP_PATH` | CLIP 模型路径 |
| `-s, --batch-size` | `32` | 推理批大小 |

示例：

```bash
BookerAutoVideo clip-test photo.jpg -c 风景,人像,美食
```

## metric —— 图片指标

计算单张图片的 `colorfulness`、`hog`、`sharpness` 三个指标，用于判断画面信息量，配合 `ext-kf` 的阈值调参。

```bash
BookerAutoVideo metric <fname>
```

## split —— 视频切分

按段数或时长把视频切成多个片段。

```bash
BookerAutoVideo split <fname> <seg>
```

+ `<seg>` 为纯数字时表示**段数**；为 `hms` 格式（如 `10m`、`1h30m`）时表示**每段时长**，输出 `fname_<start>.ext`

示例：

```bash
BookerAutoVideo split lecture.mp4 8        # 切成 8 段
BookerAutoVideo split lecture.mp4 1h30m    # 每段 1 小时 30 分钟
```

## 环境变量

命令行参数缺省时，以下环境变量提供默认值：

| 变量 | 作用 |
| --- | --- |
| `OPENAI_API_KEY` | `-k` 的默认值 |
| `OPENAI_BASE_URL` | `-H` 的默认值 |
| `OPENAI_TTI_MODEL` | `gen -m` 的默认值 |
| `WHISPER_CPP_MODEL_PATH` | `totxt -w` 的默认值 |
| `PPT_MODEL_PATH` | `totxt` / `ext-kf -m` 的默认值 |
| `CLIP_PATH` | `clip-test -m` 的默认值 |