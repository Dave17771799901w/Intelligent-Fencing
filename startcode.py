import pose_recognition as pr
import photo2video as p2v
import app

def startcode(videoInput, sideInput, heightInput):
    """
    启动比赛视频检测的任务。

    :param videoInput: 视频文件路径
    :param sideInput: 人物位置（1: 左边，2: 右边，0: 单人视频）
    :param heightInput: 人物身高（厘米）
    :return: None
    """

    pr.attacks_count = 0
    pr.global_sum_distance = 0
    pr.global_right_hand_angle = 0
    pr.global_right_ankle_angle = 0
    pr.global_left_ankle_angle = 0
    pr.global_right_knee_angle = 0
    pr.global_left_knee_angle = 0
    pr.global_right_shoulder = (0, 0)
    pr.global_left_shoulder = (0, 0)
    pr.global_right_hip = (0, 0)
    pr.global_left_hip = (0, 0)
    pr.img_base64 = ""
    pr.standardaction = ""
    pr.action = ""
    video_path = videoInput   # 视频文件路径
    side = sideInput  # 人物位置
    height_real = heightInput  # 人物身高
    detector = pr.FencingVideoDetector(video_path,height_real,side)  # 创建FencingVideoDetector实例
    success = detector.start_animation()  # 启动动画
    if success is None:
        return None # 如果动画启动失败，返回None
    return "任务完成"


def startcode_two(videoInput, sideInput, heightInput):
    """
    启动动作照片分析的任务。

    :param videoInput: 视频文件路径
    :param sideInput: 人物位置（1: 左边，2: 右边，0: 单人视频）
    :param heightInput: 人物身高（厘米）
    :return: None
    """
    pr.attacks_count = 0
    pr.global_sum_distance = 0
    pr.global_right_hand_angle = 0
    pr.global_right_ankle_angle = 0
    pr.global_left_ankle_angle = 0
    pr.global_right_knee_angle = 0
    pr.global_left_knee_angle = 0
    pr.global_right_shoulder = (0, 0)
    pr.global_left_shoulder = (0, 0)
    pr.global_right_hip = (0, 0)
    pr.global_left_hip = (0, 0)
    pr.img_base64 = ""
    pr.standardaction = ""
    pr.action = ""
    video_path = videoInput  # 视频文件路径
    side = sideInput         # 人物位置
    height_real = heightInput  # 人物身高

    detector = pr.FencingPhotoAnalyzer(video_path,height_real,side)  # 创建FencingPhotoAnalyzer实例
    # 启动动画
    success = detector.start_animation()
    if success is None:
        return None  # 如果动画启动失败，返回None
    return "任务完成"


def main():
    # 使用FencingVideoDetector类，对视频进行识别
    video_path = 'test_video\Pixel_Low.mp4'  # 替换为你的视频
    side = int(input("人物位置（0=单人视频、1=左边人物、2=右边人物）"))
    height_real = int(input("人物身高"))
    # 调用startcode函数
    startcode(video_path, side, height_real)
    return None


    # 使用FencingPhotoAnalyzer类，对视频进行识别
    # video_path = 'test_video\Pixel_Low.mp4'  # 替换为你的视频
    # side = int(input("人物位置（0=单人视频、1=左边人物、2=右边人物）"))
    # height_real = int(input("人物身高"))

    # # 调用startcode_two函数
    # startcode_two(video_path, side, height_real)
    # return None


if __name__ == '__main__':
    main()
