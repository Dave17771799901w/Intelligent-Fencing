import cv2
import numpy as np
import os
import csv
from flask import Flask, Response, Blueprint
import logging

detect_app = Blueprint('detect_app', __name__)

logging.basicConfig(level=logging.DEBUG)
detect_app.logger = logging.getLogger('detect_app')
detect_app.logger.debug("detect_app is starting...")  # 添加日志输出

def select_roi(frame):
    """
    允许用户选择ROI（感兴趣区域）。
    """
    roi = cv2.selectROI("Select ROI", frame, False, False)
    cv2.destroyWindow("Select ROI")
    detect_app.logger.debug("select_roi function called")
    return roi  # (x, y, w, h)

def save_roi_info(roi, filename='roi.csv'):
    """
    将ROI信息保存到CSV文件中。
    """
    with open(filename, 'w', newline='') as csvfile:
        roi_writer = csv.writer(csvfile)
        roi_writer.writerow(['x', 'y', 'w', 'h'])
        roi_writer.writerow([int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])])
    print(f"ROI信息已保存到 {filename}")
    detect_app.logger.debug("save_roi_info function called")


def load_roi_info(filename='roi.csv'):
    """
    从CSV文件中加载ROI信息。
    """
    if not os.path.exists(filename):
        detect_app.logger.debug("load_roi_info function called")
        return None
    with open(filename, 'r', newline='') as csvfile:
        roi_reader = csv.reader(csvfile)
        headers = next(roi_reader, None)  # 跳过标题行
        roi_values = next(roi_reader, None)
        if roi_values and len(roi_values) == 4:
            x, y, w, h = map(int, roi_values)
            print(f"已加载ROI信息：x={x}, y={y}, w={w}, h={h}")
            detect_app.logger.debug("load_roi_info function called")
            return (x, y, w, h)
        else:
            print(f"{filename} 文件格式不正确。")
            detect_app.logger.debug("load_roi_info function called")
            return None
    

def detect_colored_circles(frame, roi, color='green', dp=1.2, min_dist=50, param1=50, param2=30, min_radius=10, max_radius=50):
    """
    在指定的ROI内检测指定颜色的圆形。

    参数:
    - frame: 当前帧图像。
    - roi: 选择的感兴趣区域 (x, y, w, h)。
    - color: 要检测的颜色 ('green' 或 'red')。
    - 其他参数: 霍夫圆变换的参数。

    返回:
    - detected_circles: 检测到的圆的列表，每个圆为 (x, y, r)。
    """
    x, y, w, h = roi
    roi_frame = frame[y:y+h, x:x+w]

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    if color == 'green':
        # 定义绿色的范围
        lower_color = np.array([40, 40, 40])
        upper_color = np.array([80, 255, 255])
    elif color == 'red':
        # 扩大红色的范围
        lower_red1 = np.array([0, 50, 50])    # 增大饱和度和明度下限
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])  # 扩大红色上限范围
        upper_red2 = np.array([180, 255, 255])
    else:
        raise ValueError("Unsupported color for detection.")

    if color == 'green':
        # 创建绿色掩膜
        mask = cv2.inRange(hsv, lower_color, upper_color)
    elif color == 'red':
        # 创建红色掩膜，合并两个范围
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

    # 应用形态学操作以减少噪声
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 应用高斯模糊以进一步减少噪声
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    # 使用霍夫圆变换检测圆
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    detected_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            detected_circles.append((i[0] + x, i[1] + y, i[2]))  # 调整坐标到原图
    detect_app.logger.debug("detect_colored_circles function called")
    return detected_circles


def generate_frames():
    video_path = "detect/videos/1.mp4"  # 替换为您的视频路径
    roi_filename = 'roi.csv'    # ROI信息文件名

    # 打开视频捕获
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        detect_app.logger.error("无法打开视频文件")
        return

    # 设置视频捕获分辨率为1920x1080
    desired_width = 1920
    desired_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # 获取实际设置的分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f"实际视频分辨率: {actual_width}x{actual_height}")
    detect_app.logger.debug(f"实际视频分辨率: {actual_width}x{actual_height}")

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # 默认帧率

    cooldown_seconds = 15
    cooldown_frames = int(fps * cooldown_seconds)

    target_frame_number = 150  # 目标帧数
    current_frame = 0

    # 加载ROI信息
    roi = load_roi_info(roi_filename)

    if roi is None:
        # print("未找到ROI信息文件，将允许您选择ROI。")
        detect_app.logger.debug("未找到ROI信息文件，将允许用户选择ROI。")
        # 跳过前149帧
        while current_frame < target_frame_number - 1:
            ret, frame = cap.read()
            if not ret:
                # print(f"视频帧不足{target_frame_number}帧")
                detect_app.logger.error(f"视频帧不足{target_frame_number}帧")
                cap.release()
                return
            # 调整帧大小
            frame = cv2.resize(frame, (desired_width, desired_height))
            current_frame += 1

        # 读取第150帧
        ret, frame = cap.read()
        if not ret:
            # print(f"无法读取第{target_frame_number}帧")
            detect_app.logger.error(f"无法读取第{target_frame_number}帧")
            cap.release()
            return

        # 调整帧大小
        frame = cv2.resize(frame, (desired_width, desired_height))

        # 选择ROI
        roi = select_roi(frame)
        if roi == (0,0,0,0):
            # print("未选择ROI")
            detect_app.logger.error("未选择ROI")
            cap.release()
            return

        # 保存ROI信息
        save_roi_info(roi, roi_filename)
    else:
        # print("已加载ROI信息，将直接使用。")
        detect_app.logger.debug("已加载ROI信息，将直接使用。")
        # 跳过前149帧
        while current_frame < target_frame_number - 1:
            ret, frame = cap.read()
            if not ret:
                # print(f"视频帧不足{target_frame_number}帧")
                detect_app.logger.error(f"视频帧不足{target_frame_number}帧")
                cap.release()
                return
            # 调整帧大小
            frame = cv2.resize(frame, (desired_width, desired_height))
            current_frame += 1

        # 读取第150帧
        ret, frame = cap.read()
        if not ret:
            # print(f"无法读取第{target_frame_number}帧")
            detect_app.logger.error(f"无法读取第{target_frame_number}帧")
            cap.release()
            return

        # 调整帧大小
        frame = cv2.resize(frame, (desired_width, desired_height))

    # 初始化得分和状态
    left_score = 0
    right_score = 0
    cooldown_counter = 0  # 冷却计数器

    detect_app.logger.debug("generate_frames function called")
    while True:
        ret, frame = cap.read()
        if not ret:
            detect_app.logger.debug("End of video stream")
            break  # 视频结束

        # 调整帧大小
        frame = cv2.resize(frame, (desired_width, desired_height))

        # 检测红色圆
        red_circles = detect_colored_circles(frame, roi, color='red')

        # 检测绿色圆
        green_circles = detect_colored_circles(frame, roi, color='green')

        current_left = False
        current_right = False

        # 处理红色圆（左侧得分）
        for (cx, cy, r) in red_circles:
            # 绘制检测到的红色圆
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), 2)  # 红色圆边
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 3)  # 圆心

            # 判断圆的位置是左侧还是右侧
            roi_x, roi_y, roi_w, roi_h = roi
            center_x = cx - roi_x  # 相对于ROI的x坐标
            if center_x < roi_w // 2:
                current_left = True

        # 处理绿色圆（右侧得分）
        for (cx, cy, r) in green_circles:
            # 绘制检测到的绿色圆
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)  # 绿色圆边
            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), 3)  # 圆心

            # 判断圆的位置是左侧还是右侧
            roi_x, roi_y, roi_w, roi_h = roi
            center_x = cx - roi_x  # 相对于ROI的x坐标
            if center_x >= roi_w // 2:
                current_right = True

        # 判断是否在冷却中
        if cooldown_counter > 0:
            cooldown_counter -= 1
        else:
            if current_left:
                left_score += 1
                # print(f"左侧得分：{left_score}")
                detect_app.logger.debug(f"左侧得分：{left_score}")
                cooldown_counter = cooldown_frames
            elif current_right:
                right_score += 1
                # print(f"右侧得分：{right_score}")
                detect_app.logger.debug(f"右侧得分：{right_score}")
                cooldown_counter = cooldown_frames
            # 否则不做任何操作

        # 显示得分
        cv2.putText(frame, f"Left Score: {left_score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 红色文字
        cv2.putText(frame, f"Right Score: {right_score}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 绿色文字

        # 绘制ROI矩形
        cv2.rectangle(frame, (roi[0], roi[1]),
                      (roi[0]+roi[2], roi[1]+roi[3]),
                      (255, 255, 0), 2)  # 黄色矩形

        # 如果处于冷却中，显示冷却信息
        if cooldown_counter > 0:
            cooldown_text = f"Cooling Down: {int(cooldown_counter / fps)}s remaining"
            cv2.putText(frame, cooldown_text, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # 白色文字

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # 释放视频捕获对象
    cap.release()
    # cv2.destroyAllWindows()
    print(f"最终得分 - 左: {left_score}, 右: {right_score}")

@detect_app.route('/video_feed')
def video_feed():
    detect_app.logger.debug("video_feed route called")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

