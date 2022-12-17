import os
import re
import sys
import shutil
import pyttsx3
import librosa
import subprocess
from moviepy.editor import *
from tqdm import tqdm


def path_create_is_not_exist(path, tag=0, remove=False):
    """ 创建文件或文件夹 的文件目录
    path: 表示文件/文件夹路径
    tag: 0 表示文件 catalog  1 表示文件夹 file  2 表示文件夹+文件 file
    remove:  False 表示如果存在就不操作，True 表示存在先删除后新建
    """
    if tag == 0:
        catalog = path
    else:
        catalog = "/".join(path.split("/")[:-1])

    if os.path.exists(path):
        if remove:
            if tag == 0:
                shutil.rmtree(path)
                os.makedirs(path)
                print("--- %s 目录创建成功" % path)
            else:
                # 设置文件权限，避免文件夹无法写入
                os.chmod(path, stat.S_IRWXO+stat.S_IRWXG|stat.S_IRWXU)
                os.remove(path)
                fd = open(path, mode="w", encoding="utf-8")
                fd.close()
                print("--- %s 文件创建成功" % path)
        else:
            print("--- %s 目录已经存在" % path)
    else:
        if not os.path.exists(catalog):
            os.makedirs(catalog)
            print("--- %s 目录创建成功" % catalog)
    
    if tag == 2:
        if len(path) > len(catalog):
            fd = open(path, mode="w", encoding="utf-8")
            fd.close()
            print("--- %s 文件创建成功" % path)
    # 设置文件权限，避免文件夹无法写入
    os.chmod(catalog, stat.S_IRWXO+stat.S_IRWXG+stat.S_IRWXU)



# 读取目录下的所有文件，包括嵌套的文件夹
def get_catalog_files(dir, status=0, str1=""):
    """
    dir: 目录地址
    filelist: 变量地址
    status: 表示是包含 1，取消 0，还是不包含 -1
    str1: 过滤词
    """
    def fun(dir, filelist, status, str1):
        newDir = dir
        if os.path.isfile(dir):
            filelist.append(dir)
        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                # 如果需要忽略某些文件夹，使用以下代码
                if status == 1:
                    # 如果不包含改字符串就跳过
                    if str1 not in s:
                        continue
                elif status == -1:
                    # 如果包含改字符串就跳过
                    if str1 in s:
                        continue
                newDir = os.path.join(dir, s)
                fun(newDir, filelist, status, str1)
        # 默认顺序排列
        return sorted(filelist, reverse=False)

    filelist = []
    fun(dir, filelist, status, str1)
    return filelist



def load_pkl(filename):
    """加载 pkl 文件"""
    with open(filename, 'rb') as fr:
        model = pickle.load(fr)
    return model


def save_pkl(obj, filename):
    """保存 pkl 文件"""
    with open(filename, 'wb') as fw:
        pickle.dump(obj, fw)


def step_1_copy(path_project, path_ori):
    """ 1.将原始数据copy一份到计算空间，为录制视频做准备 """
    try:
        for i in ["docs", "img"]:
            # path_create_is_not_exist("%s/%s" % (path_project, i), tag=0, remove=False)
            dir_ori = "%s/%s" % (path_ori, i) # 原始目录
            dir_tar = "%s/%s" % (path_project, i) # 目标目录，这个目的文件是不存在的，copytree会自动创建
            print("--- %s --> %s" % (dir_ori, dir_tar) )
            shutil.copytree(dir_ori, dir_tar)
    except Exception as e:
        print("Failed", e)
    path_create_is_not_exist("%s/voice" % path_project, tag=0, remove=False)


def step_2_md2mp3(engine, files, path_project, dict_md0mps_file):
    print(files)
    dict_md0mps = {}
    pbar = tqdm(files)
    for infile in pbar:
        with open(infile, encoding='utf-8') as f:
            content = f.read().split("\n")
        title = content[0].replace("# ", "").replace("丨", ".").replace("｜", ".")
        content = "\n".join(content)
        # print("-- ", content)

        pbar.set_description('转化文件中: %s' % title)
        # 找到 ![](../img/037238adc58aea08da213c048db01c31.jpg)

        regular = re.compile(r'!\[\]\(\.\.\/(img\/.+\.jpg)')
        imgs_path = ["%s/%s" % (path_project, i) for i in re.findall(regular, content)]
        print(imgs_path)

        regular = re.compile(r'!\[\]\(\.\.\/img\/.+\.jpg')
        need_replace = re.findall(regular, content)
        for i in need_replace:
            content = content.replace(i, "")
        outfile = infile.replace("/docs/", "/voice/").replace(".md", ".mp3")
        mp3_path = outfile.replace(outfile.split("/")[-1], "%s.mp3" % title)
        
        content = re.sub(r"\#|\ |\>|\*|\(|\)", "", content)
        # content = [i for i in content.split("\n") if i != ""]
        content = [i for i in re.split(r"\t|\n|\.|\。|\?|\？|;|；|:|：", content) if i != ""]
        print(content)
        dict_md0mps = {
            title: {
                "imgs_path": imgs_path,
                "mp3_path": mp3_path,
                "mp3_content": content
            }
        }
        engine.save_to_file("\n".join(content), mp3_path)
        print("文件保存成功: %s" % mp3_path)
    engine.runAndWait()
    engine.stop()
    print('finish')
    save_pkl(dict_md0mps, dict_md0mps_file)
    return dict_md0mps


def step_3_image2movie(jpg_list, mp3_time, path_tmp_1_movie):
    """
    # 将图片按照顺序逐一放入movei模板中
    """
    start_time = 0
    image_list = []
    for jpg_path in jpg_list:
        # 每段字幕时长
        every_part_duration_time = mp3_time / len(jpg_list)
        # 图片数据
        image_clip = (
            ImageClip(jpg_path)
                .set_duration(every_part_duration_time)  # 水印持续时间
                .resize(height=650)  # 水印的高度，会等比缩放
                .set_pos(("center", 0))  # 水印的位置
                .set_start(start_time)
        )
        start_time = start_time + every_part_duration_time
        image_list.append(image_clip)

    # 载入背景视频
    # path = "data/voice/start.mp4"
    # video = VideoFileClip(path).resize((1280, 720)).set_duration(mp4_duration)
    # cvc = CompositeVideoClip([video] + image_list, size=(1280, 720))
    cvc = CompositeVideoClip(image_list, size=(1280, 720))
    cvc.write_videofile(path_tmp_1_movie, fps=60, remove_temp=False, verbose=True)


def step_4_str2movie(mp3_content, mp3_time, path_tmp_1_movie, path_tmp_2_movie, font_path):
    # 合成字幕到模板视频中
    # 判断如果没有该数据的文件夹就创建
    start_time = 0
    result_list = []
    for content in mp3_content:
        # 每段字幕时长
        every_part_duration_time = mp3_time / len(mp3_content)
        # 字幕数据
        title_clip = (
            TextClip(content, font=font_path, fontsize=35, color='white', method='label')
                .set_position(("center", "bottom"))
                .set_duration(every_part_duration_time)
                .set_start(start_time)
        )
        start_time = start_time + every_part_duration_time
        result_list.append(title_clip)

    # 载入背景视频
    # path = "data/voice/mp4_1_start.mp4"
    # video_1 = VideoFileClip(path).resize((1280, 720)).set_duration(mp4_duration)
    video = VideoFileClip(path_tmp_1_movie).resize((1280, 720)).set_duration(mp3_time)
    cvc = CompositeVideoClip([video] + result_list, size=(1280, 720))
    # cvc = CompositeVideoClip([video_1, video_2] + result_list, size=(1280, 720))
    cvc.write_videofile(path_tmp_2_movie, fps=60, remove_temp=False, verbose=True)


def step_5_merge2movie(mp3_path, path_tmp_3_start, path_tmp_2_movie, path_tmp_3_end, path_tmp_4_movie, path_tmp_result):
    video_1 = VideoFileClip(path_tmp_3_start)
    video_2 = VideoFileClip(path_tmp_2_movie)
    video_3 = VideoFileClip(path_tmp_3_end)
    videoclip = concatenate_videoclips([video_1, video_2, video_3])
    # 发现没用，视频没声音，所以采用: ffmpeg
    # audio = video_1.audio
    # videoclip2 = videoclip.set_audio(my_audioclip)
    video = videoclip.without_audio()
    video.write_videofile(path_tmp_4_movie)

    audio_1 = video_1.audio
    audio_2 = AudioFileClip(mp3_path)
    audio_3 = video_3.audio
    audio = concatenate_audioclips([audio_1, audio_2, audio_3])
    path_tmp_4_movie_mp3 = path_tmp_4_movie.replace(".mp4", ".mp3")
    audio.write_audiofile(path_tmp_4_movie_mp3)
    # 将MP3合成到视频中
    cmd = 'ffmpeg -i %s -i %s -strict -2 -f mp4 %s' % (path_tmp_4_movie, path_tmp_4_movie_mp3, path_tmp_result)
    print(">>> %s " % cmd)
    subprocess.call(cmd, shell=True)
    print("将声音和音频合成到视频中去: %s" % path_tmp_result)


def main():
    engine = pyttsx3.init()
    # rate = teacher.getProperty('rate')          # 获取当前语速属性的值
    # teacher.setProperty('rate', rate-50)        # 设置语速属性为当前语速减20
    voices = {
        "zh_HK": "com.apple.speech.synthesis.voice.sin-ji",
        "zh_TW": "com.apple.speech.synthesis.voice.mei-jia",
        "zh_CN": "com.apple.speech.synthesis.voice.ting-ting.premium"
    }
    engine.setProperty('voice', voices["zh_TW"])

    path_root = "/Users/jiangzl/work/gitlab/middleware"
    path_data = "%s/data" % path_root
    path_project = "%s/biz5min" % path_data
    path_create_is_not_exist(path_project, tag=0, remove=False)

    print("\nstep 1.复制数据到项目的计算空间下")
    path_ori = "/Users/jiangzl/data/b_movie/business-5min-notes-master"
    step_1_copy(path_project, path_ori)

    print("\nstep 2.生成mp3, 记录图片路径和文本内容")
    number = 4
    files = ["/Users/jiangzl/work/gitlab/middleware/data/biz5min/docs/%s.md" % number]
    dict_md0mps_file = "%s/tmp/%s_result.pkl" % (path_project, number)
    path_create_is_not_exist(dict_md0mps_file, tag=1, remove=False)
    if not os.path.exists(dict_md0mps_file):
        step_2_md2mp3(engine, files, path_project, dict_md0mps_file)
    else:
        print("%s 文件已存在" % dict_md0mps_file)
    dict_md0mps = load_pkl(dict_md0mps_file)
    print(dict_md0mps)

    print("\nstep 3.将图片按照顺序逐一放入movie模板中")
    font_path = '%s/config/kaiti.ttf' % path_root  # 加载字体配置文件
    print("--- ", font_path)
    path_create_is_not_exist("%s/movie" % path_project, tag=0, remove=False)
    for k, v in dict_md0mps.items():
        imgs_path = v["imgs_path"]
        mp3_path  = v["mp3_path"]
        mp3_content = v["mp3_content"]

        print("\nstep 3.将图片按照顺序逐一放入movie模板中")
        mp3_time = librosa.get_duration(filename=mp3_path)
        path_tmp_1_movie = "%s/movie/%s_mp4_1_middle.mp4" % (path_project, k)
        step_3_image2movie(imgs_path, mp3_time, path_tmp_1_movie)

        print("\nstep 4.将内容按照顺序逐一放入movie模板中")
        path_tmp_2_movie = "%s/movie/%s_mp4_2_middle.mp4" % (path_project, k)
        step_4_str2movie(mp3_content, mp3_time, path_tmp_1_movie, path_tmp_2_movie, font_path)

        print("\nstep 5.将视频按照顺序合并在一起")
        path_tmp_3_start = "%s/开头.mp4" % path_project
        path_tmp_3_end   = "%s/结尾.mp4" % path_project
        path_tmp_4_movie  = "%s/movie/%s_mp4_3_middle.mp4" % (path_project, k)
        path_tmp_result  = "%s/movie/%s.mp4" % (path_project, k)
        step_5_merge2movie(mp3_path, path_tmp_3_start, path_tmp_2_movie, path_tmp_3_end, path_tmp_4_movie, path_tmp_result)


if __name__ == "__main__":
    main()
