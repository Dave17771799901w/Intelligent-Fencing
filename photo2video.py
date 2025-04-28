import cv2
import os
import datetime

def photo_to_video(input_photoPath, output_videoPath, duration=2, fps=1):
    now = datetime.datetime.now()
    nownow = now.strftime("%Y-%m-%d %H%M")
    # output_videoPath = 'invideo'
    file_name_with_ext = os.path.basename(input_photoPath)  # 获取路径的最后一部分，即文件名和后缀
    namefile, file_ext = os.path.splitext(file_name_with_ext)  # 分离文件名和后缀
    
    # 读取图片
    frame = cv2.imread(input_photoPath)
    if frame is None:
        print("无法读取图片，请检查路径是否正确。")
        return
    frame = cv2.resize(frame,(1920,1080))
    height, width, layers = frame.shape

    # 计算视频的总帧数
    total_frames = duration * fps

    # 定义视频编解码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_path = f"{output_videoPath}/{namefile}{nownow}.mp4"
    video_path = f"{output_videoPath}/{namefile}.mp4"
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # 将图片重复写入视频
    for _ in range(total_frames):
        video.write(frame)

    video.release()

    print(f"{video_path}已经生成")

     # 返回生成的视频路径
    return video_path

# 使用示例
def main():
    input_photoPath = 'RLA2.png'  # 输入照片路径
    output_videoPath = 'photo_to_video'  # 输出视频的文件夹路径
    # namefile = 'photo_video'
    photo_to_video(input_photoPath, output_videoPath)

if __name__ == "__main__":
    main()