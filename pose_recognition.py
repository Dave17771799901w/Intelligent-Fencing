import cv2   
import numpy as np
import mediapipe as mp
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
import threading
import queue
import os

import matplotlib
# matplotlib.use('Agg')  # 设置Matplotlib后端为Agg
import matplotlib.pyplot as plt

# 自己的文件
import datatreating as dt
from compare import set_compare
from compare import optical_flow_keyframes as keyframes

# Initialize global variables
human_height_pixels = 0   # 人体像素身高 全局变量
height_real = 170         # 人体实际身高 全局变量
now_ankle_pixel_coord = (0,0)  # 当前脚踝像素坐标
landmark_num_distance = 0  # 关键点移动距离

# 全局变量来存储CSV文件路径
csv_attack_path = None

side = 0  # 识别的人的位置（左边=1，右边=2，单人=0）
# 攻击时的数据
attacks_count = 0  # 攻击次数
global_sum_distance = 0  # 总移动距离
global_right_hand_angle = 0  # 右手肘角度
global_right_ankle_angle = 0  # 右脚踝角度
global_left_ankle_angle = 0  # 左脚踝角度
global_right_knee_angle = 0  # 右膝角度
global_left_knee_angle = 0  # 左膝角度
global_right_shoulder = (0,0)  # 右肩坐标
global_left_shoulder = (0,0)  # 左肩坐标
global_right_hip = (0,0)  # 右臀坐标
global_left_hip = (0,0)  # 左臀坐标
img_base64 = ""  # 图像的base64编码
standardaction = ""   # 动作得分
action = ""  # 当前动作


# 创建线程安全的队列
# data_queue = queue.Queue()
csv_path_file = 'latest_csv_path.txt'
# 定义一个姿态检测的类,用于检测比赛视频
class FencingVideoDetector:

    # 初始化函数，video_path是视频路径
    def __init__(self, video_path, height_input, side_input):
        global human_height_pixels
        global now_ankle_pixel_coord
        global landmark_num_distance
        human_height_pixels = 0
        now_ankle_pixel_coord = (0,0)
        landmark_num_distance = 0

        global height_real
        global side
        height_real = height_input
        side = side_input

        print("重置全局变量")
        global attacks_count
        global global_sum_distance
        global global_right_hand_angle
        global global_right_ankle_angle
        global global_left_ankle_angle
        global global_right_knee_angle
        global global_left_knee_angle
        global global_right_shoulder
        global global_left_shoulder
        global global_right_hip
        global global_left_hip
        global img_base64
        global standardaction
        global action
        global csv_attack_path

        attacks_count = 0
        global_sum_distance = 0
        global_right_hand_angle = 0
        global_right_ankle_angle = 0
        global_left_ankle_angle = 0
        global_right_knee_angle = 0
        global_left_knee_angle = 0
        global_right_shoulder = (0, 0)
        global_left_shoulder = (0, 0)
        global_right_hip = (0, 0)
        global_left_hip = (0, 0)
        img_base64 = ""
        standardaction = ""
        action = ""

        self.mp_pose = mp.solutions.pose   # 使用pose方法（识别姿势）
        self.mp_draw = mp.solutions.drawing_utils  # 画图方法
        # 定义关键点的绘制样式
        self.landmark_drawing_spec = self.mp_draw.DrawingSpec(thickness=-1, circle_radius=8, color=(0, 0, 255))

        # 定义连接线的绘制样式
        self.connection_drawing_spec = self.mp_draw.DrawingSpec(thickness=5, color=(255, 0, 0))  # 红色连接线

        # pose的初始化设置
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.height_real = height_real

        # 读取视频
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频的原始尺寸
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 获取视频总帧数
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        
        # 初始化matplotlib图形
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.fig)
        
        # 存储关键点数据
        self.landmarks_3d = []
        
        # 存储视角参数
        self.elev = 15
        self.azim = 90
        
        # 连接matplotlib的视角更新事件
        self.fig.canvas.mpl_connect('motion_notify_event', self._update_view)

        # 初始化CSV文件
        self.csv = dt.setup_csv()
        csv_attack_path = self.csv[6]
        # 更新全局变量csv_attack_path
        with open(csv_path_file, 'w') as f:
            f.write(csv_attack_path)  # 将 CSV 文件路径写入特定文件
        print(f"CSV文件路径: {csv_attack_path}")

        # 存放当前帧的归一化坐标
        self.current_landmarks = []
        self.pose_closed = False
        self.forefoot = None

    # 旋转3D视图
    def _update_view(self, event):
        if event.inaxes == self.ax:
            self.elev = self.ax.elev
            self.azim = self.ax.azim

    # 处理当前帧
    def process_frame(self):
        global human_height_pixels
        global now_ankle_pixel_coord
        global landmark_num_distance
        global attacks_count
        
        # 返回前端的数据
        global global_sum_distance
        global global_right_hand_angle
        global global_right_ankle_angle
        global global_left_ankle_angle
        global global_right_knee_angle
        global global_left_knee_angle
        global global_right_shoulder
        global global_left_shoulder
        global global_right_hip
        global global_left_hip, img_base64, standardaction, action

        success, image = self.cap.read()  # 读取当前帧
        if not success:
            # print("视频读取结束")
            return False
        
        self.rezise_image = cv2.resize(image, (1920, 1080))

        # 根据变量side切割图像
        image11 = dt.slicing_photo(self, side, self.rezise_image)

        # 转换颜色空间用于MediaPipe处理 将BGR格式转换为RGB格式
        image_rgb = cv2.cvtColor(image11, cv2.COLOR_BGR2RGB)

        # 进行姿态检测 处理rgb图像，并返回检测结果
        self.results = self.pose.process(image_rgb)
        
        # 在图像上绘制检测结果 
        if self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                image11, 
                self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.landmark_drawing_spec,
                connection_drawing_spec=self.connection_drawing_spec
            )
            
            # 将帧转换为 base64 编码
            self.rezise_image = cv2.resize(self.rezise_image, (1920, 1280))
            _, buffer = cv2.imencode('.jpg', self.rezise_image)
            global img_base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            if self.results.pose_world_landmarks:
                landmarks = self.results.pose_world_landmarks.landmark

                # 获取当前帧数 将帧数int类型输出
                self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # 提取所有关键点的3D坐标并进行坐标变换
                x = [-landmark.x for landmark in landmarks]
                y = [-landmark.y for landmark in landmarks]
                z = [landmark.z for landmark in landmarks]
                
                # 交换y和z坐标，使人体垂直显示
                self.landmarks_3d = [x, z, y]
            
                # 获取第一帧，用来计算 像素身高 和 比例因子
                if human_height_pixels == 0:
                    self.pixel_ratio, human_height_pixels = dt.hight_pixel_rstio(self)  # 返回值为 比例因子
                    self.pixel_ratio = float(self.pixel_ratio)

                    self.face_state = 0
                    left_heel = dt.normalized_to_pixel(self, 29)[0]
                    right_heel = dt.normalized_to_pixel(self, 30)[0]
                    left_foot_index = dt.normalized_to_pixel(self, 31)[0]
                    right_foot_index = dt.normalized_to_pixel(self, 32)[0]
                    nose = dt.normalized_to_pixel(self, 0)[0]
                    left_ear = dt.normalized_to_pixel(self, 7)[0]
                    right_ear = dt.normalized_to_pixel(self, 8)[0]
                    
                # 判定人物面向哪一边
                    '''
                    self.face_state = 1 人物在左边-左撇子
                    self.face_state = 2 人物在左边-右撇子

                    self.face_state = 3 人物在右边-左撇子
                    self.face_state = 4 人物在右边-右撇子
                    '''
                    # 人物在左边
                    if (nose[0] > left_ear[0] or nose[0] > right_ear[0]):
                        
                        if left_heel[1] > right_heel[1]:
                            self.face_state = 1  # 左左撇子
                            self.forefoot = 31  # 确认前脚 方便计算前脚移动距离
                        else:
                            self.face_state = 2  # 左右撇子
                            self.forefoot = 32     

                    # 人物在右边
                    elif (nose[0] < left_ear[0] or nose[0] < right_ear[0]):
                        if left_heel[1] < right_heel[1]:
                            self.face_state = 3  # 右左撇子
                            self.forefoot = 31
                        else:
                            self.face_state = 4  # 右右撇子
                            self.forefoot = 32

                    print(f"Face State: {self.face_state}")

                # 计算所需角度
                right_ankle_angle = dt.cal_ang(self, 26, 28, -1)   # 计算右脚踝与地面夹角
                left_ankle_angle = dt.cal_ang(self, 25, 27, -1)    # 计算左脚踝与地面夹角
                right_knee_angle = dt.cal_ang(self, 24, 26, 28)   # 计算右膝角度
                left_knee_angle = dt.cal_ang(self, 23, 25, 27)     # 计算左膝角度
                right_hand_angle = dt.cal_ang(self, 12, 14, 16)    # 计算右肘角度

                # 获取躯干四点
                right_shoulder = dt.normalized_to_pixel(self, 12)[0]  # 获取右肩坐标
                left_shoulder = dt.normalized_to_pixel(self, 11)[0]  # 获取左肩坐标
                right_hip = dt.normalized_to_pixel(self, 24)[0]      # 获取右臀坐标
                left_hip = dt.normalized_to_pixel(self, 23)[0]       # 获取左臀坐标

                # 将客户需要的数据，写入NeedData.csv文件
                self.csv[4].writerow([
                    self.frame_number,
                    left_shoulder,
                    right_shoulder,
                    left_hip,
                    right_hip,
                    left_knee_angle,
                    right_knee_angle,
                    right_hand_angle
                ])
               
               # 处理这一帧视频的所有关键点
                for landmark_value in range(33):
                    # 获取当前关键点的像素坐标和归一化坐标
                    landmark_pixel_coord, normalized_coord = dt.normalized_to_pixel(self, landmark_value)
                    # 将归一化坐标写入列表（开启动作对比功能时使用）
                    # self.current_landmarks.append([float(normalized_coord[0]),
                    #                                float(normalized_coord[1])])

                    # 将像素坐标或归一化坐标写入AllLandmarks.csv文件，只能开其中一个
                    # 将像素坐标写入AllLandmarks.csv   
                    self.csv[3].writerow([
                        self.frame_number,
                        landmark_value,
                        landmark_pixel_coord[0],
                        landmark_pixel_coord[1]
                    ])
                    # 将归一化坐标写入AllLandmarks.csv
                    # self.csv[3].writerow([self.frame_number,
                    #                       landmark_value,
                    #                       normalized_coord[0],
                    #                       normalized_coord[1]])

                    # 在某个关键点开始判断攻击动作
                    if self.forefoot is not None and self.forefoot == landmark_value:
                        # 更新前脚坐标点
                        forefoot_pixel_coord = dt.normalized_to_pixel(self, self.forefoot)[0]
                        last_ankle_pixel_coord = now_ankle_pixel_coord
                        now_ankle_pixel_coord = forefoot_pixel_coord

                        # 计算移动距离
                        # 判断向前移动
                        if (now_ankle_pixel_coord[0] <= last_ankle_pixel_coord[0] and 
                            self.face_state in [3, 4]):  # 向左移动
                        
                            # 累加每一帧距离
                            landmark_num_distance += (dt.Two_point_distance(
                                                      now_ankle_pixel_coord, last_ankle_pixel_coord) *
                                                      self.pixel_ratio)
                        
                        elif (now_ankle_pixel_coord[0] >= last_ankle_pixel_coord[0] and 
                              self.face_state in [1, 2]):  # 向右移动
                            # 累加每一帧距离
                            landmark_num_distance += (dt.Two_point_distance(
                                                      now_ankle_pixel_coord, last_ankle_pixel_coord) *
                                                      self.pixel_ratio)
                        
                        # 移动结束，进入结算
                        else:
                            if landmark_num_distance >= 20 and right_hand_angle >= 165:
                                # 累加攻击次数
                                attacks_count += 1
                                sum_distance = format(landmark_num_distance, '.2f')  # 像素移动距离*比例因子

                                print(f"第{attacks_count}次进攻：")
                                print(f"移动距离{sum_distance}\t右手夹角{right_hand_angle}°")
                                print(f"左脚与地面夹角{left_ankle_angle}°\t右脚与地面夹角{right_ankle_angle}°")
                                print(f"左膝盖夹角{left_knee_angle}°\t右膝盖夹角{right_knee_angle}°")
                                print(f"左肩膀坐标{left_shoulder}\t右肩坐标{right_shoulder}")
                                print(f"左臀部坐标{left_hip}\t右臀部坐标{right_hip}")
                                
                                # 更新全局变量
                                global_sum_distance = sum_distance
                                global_right_hand_angle = right_hand_angle
                                global_right_ankle_angle = right_ankle_angle
                                global_left_ankle_angle = left_ankle_angle
                                global_right_knee_angle = right_knee_angle
                                global_left_knee_angle = left_knee_angle
                                global_right_shoulder = right_shoulder
                                global_left_shoulder = left_shoulder
                                global_right_hip = right_hip
                                global_left_hip = left_hip
                                img_base64 = base64.b64encode(buffer).decode('utf-8')
                                # standardaction = ready_scores
                                # action = '持剑动作'
                                # print(f"更新数据: {global_sum_distance}, {global_right_hand_angle}, {action}")

                                # 将数据写入AttackData.csv文件
                                self.csv[5].writerow([
                                    attacks_count,
                                    sum_distance,
                                    right_hand_angle,
                                    right_ankle_angle,
                                    left_ankle_angle,
                                    right_knee_angle,
                                    left_knee_angle,
                                    right_shoulder,
                                    left_shoulder,
                                    right_hip,
                                    left_hip
                                ])
                            # 移动结束，距离清零
                            landmark_num_distance = 0 
                return True
        return False

    # 绘制骨骼
    def draw_skeleton(self):
        # 定义骨架连接
        connections = [
            # 头部和躯干
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            
            # 躯干
            (11, 12), (12, 14), (14, 16),  # 右臂
            (11, 13), (13, 15),  # 左臂
            (11, 23), (23, 25), (25, 27), (27, 29), (27, 31), (29,31),# 左腿
            (12, 24), (24, 26), (26, 28), (28, 30), (28,32), (30,32), # 右腿
            
            # 身体中线
            (23, 24),  # 臀部
            (11, 12)   # 肩部
        ]
        
        # 绘制连接线
        for connection in connections:
            start = connection[0]
            end = connection[1]
            x_coords = [self.landmarks_3d[0][start], self.landmarks_3d[0][end]]
            y_coords = [self.landmarks_3d[1][start], self.landmarks_3d[1][end]]
            z_coords = [self.landmarks_3d[2][start], self.landmarks_3d[2][end]]
            # 'b-'：线段选择蓝色  linewidth=2：线宽=2
            self.ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2)  

    def update_plot(self, frame):
        # success = self.process_frame()
        # if not success:
        #     return False
        if self.process_frame():
            if self.landmarks_3d:
                self.ax.clear()
                
                # 绘制关键点
                self.ax.scatter(
                    self.landmarks_3d[0],
                    self.landmarks_3d[1],
                    self.landmarks_3d[2],
                    c='red',
                    marker='o'
                )
                # 绘制骨架
                self.draw_skeleton()
                
                # 设置坐标轴范围和标签
                self.ax.set_xlim([-1, 1])
                self.ax.set_ylim([-1, 1])
                self.ax.set_zlim([-1, 1])
                self.ax.set_xlabel('X')
                self.ax.set_ylabel('Z')
                self.ax.set_zlabel('Y')
                self.ax.view_init(elev=self.elev, azim=self.azim)
                self.canvas.draw()

                # 将matplotlib动画转成base64编码
                # img_data = io.BytesIO()
                # self.fig.savefig(img_data, format='png')
                # img_data.seek(0)
                # global img_base64
                # img_base64 = base64.b64encode(img_data.read()).decode('utf-8')
                
        return True

    def get_landmarks_3d(self):
        # 返回当前帧的关键点3D坐标
        return self.landmarks_3d

    def start_animation(self):
        """
        以下是三种方法使算法逐帧读取视频
        第一种：显示matplotlib动画或保存matplotlib动画--- 有3D骨架
        第二种：直接循环update_plot，更新视频帧--- 有3D骨架
        第三种：直接调用process_frame使循环--- 没有3D骨架，只有cv2图像。运行速度最快
        """

        # 第三种
        # 直接调用process_frame使循环
        
        try:
            while True:
                success = self.process_frame()
                plt.show()
                if not success:
                    print("视频读取结束")
                    break
                print(f"处理帧: {self.frame_number}")
        finally:
            self.release_resources()


    def release_resources(self):
        # 释放视频捕捉对象
        if self.cap.isOpened():
            self.cap.release()
            print("视频捕捉资源已释放")
        
        # 关闭所有打开的CSV文件
        for csv_writer in self.csv:
            if hasattr(csv_writer, 'close'):
                csv_writer.close()
                print("CSV文件已关闭")
        
        # 关闭Matplotlib图形
        plt.close(self.fig)
        print("Matplotlib图形已关闭")

# 使用示例
def main():
    while True:
        video_path = input("请输入视频路径（或输入 'exit' 退出）：")
        if video_path.lower() == 'exit':
            print("退出程序。")
            break
        try:
            side = int(input("选择需要识别的人（左边的人=1，右边的人=2，单人视频=0）："))
            height_real = int(input("请输入真实身高（厘米）："))
        except ValueError:
            print("输入无效，请输入正确的数值。")
            continue
        
        try:
            # 调用FencingVideoDetector
            detector = FencingVideoDetector(video_path, height_real, side)
            # 启动视频处理
            detector.start_animation()
        except Exception as e:
            print(f"处理视频时出错: {e}")
            continue
        
        # 重置全局变量（如果继续使用全局变量）
        global human_height_pixels, now_ankle_pixel_coord, landmark_num_distance
        human_height_pixels = 0
        now_ankle_pixel_coord = (0,0)
        landmark_num_distance = 0
        global_sum_distance = 0
        global_right_hand_angle = 0
        global_right_ankle_angle = 0
        global_left_ankle_angle = 0
        global_right_knee_angle = 0
        global_left_knee_angle = 0
        global_right_shoulder = (0,0)
        global_left_shoulder = (0,0)
        global_right_hip = (0,0)
        global_left_hip = (0,0)
        img_base64 = ""
        standardaction = ""
        action = ""
        attacks_count = 0

        print("准备处理下一个视频。")

if __name__ == '__main__':
    main()



class FencingPhotoAnalyzer:

    # 初始化函数，video_path是视频路径
    def __init__(self, video_path, height_input, side_input):
        # 更新逻辑分析的全局变量
        global human_height_pixels
        global now_ankle_pixel_coord
        global landmark_num_distance
        global height_real
        global side
        human_height_pixels = 0
        now_ankle_pixel_coord = (0,0)
        landmark_num_distance = 0
        height_real = height_input
        side = side_input

        print("重置全局变量")
        print(f"videoPath={video_path}")
        # 更新数据请求的全局变量
        global attacks_count
        global global_sum_distance
        global global_right_hand_angle
        global global_right_ankle_angle
        global global_left_ankle_angle
        global global_right_knee_angle
        global global_left_knee_angle
        global global_right_shoulder
        global global_left_shoulder
        global global_right_hip
        global global_left_hip
        global img_base64
        global standardaction
        global action

        attacks_count = 0
        global_sum_distance = 0
        global_right_hand_angle = 0
        global_right_ankle_angle = 0
        global_left_ankle_angle = 0
        global_right_knee_angle = 0
        global_left_knee_angle = 0
        global_right_shoulder = (0, 0)
        global_left_shoulder = (0, 0)
        global_right_hip = (0, 0)
        global_left_hip = (0, 0)
        img_base64 = ""
        standardaction = ""
        action = ""

        self.mp_pose = mp.solutions.pose   # 使用pose方法（识别姿势）
        self.mp_draw = mp.solutions.drawing_utils  # 画图方法
        # 定义关键点的绘制样式
        self.landmark_drawing_spec = self.mp_draw.DrawingSpec(thickness=-1, circle_radius=8, color=(0, 0, 255))

        # 定义连接线的绘制样式
        self.connection_drawing_spec = self.mp_draw.DrawingSpec(thickness=5, color=(255, 0, 0))  # 红色连接线

        # pose的初始化设置
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.height_real = height_real

        # 读取视频
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频的原始尺寸
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 获取视频总帧数
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        
        # 初始化matplotlib图形
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.fig)
        
        # 存储关键点数据
        self.landmarks_3d = []
        
        # 存储视角参数
        self.elev = 15
        self.azim = 90
        
        # 连接matplotlib的视角更新事件
        self.fig.canvas.mpl_connect('motion_notify_event', self._update_view)

        # 初始化CSV文件
        self.csv = dt.setup_csv()
        csv_action_path = self.csv[7]
        # 更新全局变量csv_attack_path
        with open(csv_path_file, 'w') as f:
            f.write(csv_action_path)  # 将 CSV 文件路径写入特定文件
        print(f"CSV文件路径: {csv_action_path}")

        # 存放当前帧的归一化坐标
        self.current_landmarks = []
        # pose_closed的状态
        self.pose_closed = False

        # Initialize forefoot to avoid AttributeError
        self.forefoot = None

        # 增加更新全局变量状态，避免一直更新
        self.global_state = 1

        # for i in range(3):  # 最多重试3次
        #     self.cap = cv2.VideoCapture(video_path)
        #     if self.cap.isOpened():
        #         break
        #     else:
        #         print(f"视频流初始化失败，尝试重置总线... ({i+1}/3)")
        #         cv2.destroyAllWindows()
        #         time.sleep(1)
        # else:
        #     raise RuntimeError("无法建立视频通道，请检查：\n1. 视频编解码器支持\n2. 硬件加速设置\n3. 文件完整性")

    # 旋转3D视图
    def _update_view(self, event):
        if event.inaxes == self.ax:
            self.elev = self.ax.elev
            self.azim = self.ax.azim

    # 处理当前帧
    def process_frame(self):
        global human_height_pixels
        global now_ankle_pixel_coord
        global landmark_num_distance
        global attacks_count
        
        # 返回前端的数据
        global global_sum_distance
        global global_right_hand_angle
        global global_right_ankle_angle
        global global_left_ankle_angle
        global global_right_knee_angle
        global global_left_knee_angle
        global global_right_shoulder
        global global_left_shoulder
        global global_right_hip
        global global_left_hip, img_base64, standardaction, action

        success, image = self.cap.read()  # 读取当前帧
        if not success:
            # print("视频读取结束")
            return False
        
        self.rezise_image = cv2.resize(image, (1920, 1080))

        # 根据变量side切割图像
        image11 = dt.slicing_photo(self, side, self.rezise_image)

        # 转换颜色空间用于MediaPipe处理 将BGR格式转换为RGB格式
        image_rgb = cv2.cvtColor(image11, cv2.COLOR_BGR2RGB)

        # 进行姿态检测 处理rgb图像，并返回检测结果
        self.results = self.pose.process(image_rgb)
        
        # 在图像上绘制检测结果 
        if self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                image11, 
                self.results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.landmark_drawing_spec,
                connection_drawing_spec=self.connection_drawing_spec
            )

            # 保存当前姿态画面为图片
            # path = 'snapshoot/current_pose{}.png'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            snapshot_path = os.path.join(os.getcwd(), 'snapshoot/current_pose.png')
            try:
                # 创建包含姿态标注的图片
                annotated_image = image_rgb.copy()
                self.mp_draw.draw_landmarks(
                    image11,
                    self.results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.landmark_drawing_spec,
                    connection_drawing_spec=self.connection_drawing_spec
                )
                # cv2.imwrite(snapshot_path, cv2.cvtColor(image11, cv2.COLOR_RGB2BGR))
                cv2.imwrite(snapshot_path, image11)
            except Exception as e:
                print(f"姿态快照保存失败: {str(e)}")
            
            # 将帧转换为 base64 编码
            self.rezise_image = cv2.resize(self.rezise_image, (1920, 1280))
            _, buffer = cv2.imencode('.jpg', self.rezise_image)
            global img_base64
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            if self.results.pose_world_landmarks:
                landmarks = self.results.pose_world_landmarks.landmark

                # 获取当前帧数 将帧数int类型输出
                self.frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # 提取所有关键点的3D坐标并进行坐标变换
                x = [-landmark.x for landmark in landmarks]
                y = [-landmark.y for landmark in landmarks]
                z = [landmark.z for landmark in landmarks]
                
                # 交换y和z坐标，使人体垂直显示
                self.landmarks_3d = [x, z, y]
            
                # 获取第一帧，用来计算 像素身高 和 比例因子
                if human_height_pixels == 0:
                    self.pixel_ratio, human_height_pixels = dt.hight_pixel_rstio(self)  # 返回值为 比例因子
                    self.pixel_ratio = float(self.pixel_ratio)

                    self.face_state = 0
                    left_heel = dt.normalized_to_pixel(self, 29)[0]
                    right_heel = dt.normalized_to_pixel(self, 30)[0]
                    left_foot_index = dt.normalized_to_pixel(self, 31)[0]
                    right_foot_index = dt.normalized_to_pixel(self, 32)[0]
                    nose = dt.normalized_to_pixel(self, 0)[0]
                    left_ear = dt.normalized_to_pixel(self, 7)[0]
                    right_ear = dt.normalized_to_pixel(self, 8)[0]
                    right_shoulder = dt.normalized_to_pixel(self, 12)[0]  # 获取右肩坐标
                    left_shoulder = dt.normalized_to_pixel(self, 11)[0]  # 获取左肩坐标
                    
                # 判定人物面向哪一边
                    '''
                    self.face_state = 1 人物在左边-左撇子
                    self.face_state = 2 人物在左边-右撇子

                    self.face_state = 3 人物在右边-左撇子
                    self.face_state = 4 人物在右边-右撇子
                    '''
                    # 人物在左边

                    if (right_shoulder[0] > left_shoulder[0]):
                        self.face_state = 2  # 左右撇子
                        self.forefoot = 32     

                    # 人物在右边
                    elif (right_shoulder[0] < left_shoulder[0]):
                        self.face_state = 4  # 右右撇子
                        self.forefoot = 32

                    print(f"Face State: {self.face_state}")

                # 计算所需角度
                right_ankle_angle = dt.cal_ang(self, 26, 28, -1)   # 计算右脚踝与地面夹角
                left_ankle_angle = dt.cal_ang(self, 25, 27, -1)    # 计算左脚踝与地面夹角
                right_knee_angle = dt.cal_ang(self, 24, 26, 28)   # 计算右膝角度
                left_knee_angle = dt.cal_ang(self, 23, 25, 27)     # 计算左膝角度
                right_hand_angle = dt.cal_ang(self, 12, 14, 16)    # 计算右肘角度

                # 获取躯干四点
                right_shoulder = dt.normalized_to_pixel(self, 12)[0]  # 获取右肩坐标
                left_shoulder = dt.normalized_to_pixel(self, 11)[0]  # 获取左肩坐标
                right_hip = dt.normalized_to_pixel(self, 24)[0]      # 获取右臀坐标
                left_hip = dt.normalized_to_pixel(self, 23)[0]       # 获取左臀坐标
               
               # 处理这一帧视频的所有关键点
                for landmark_value in range(33):
                    # 获取当前关键点的像素坐标和归一化坐标
                    landmark_pixel_coord,normalized_coord = (dt.normalized_to_pixel(
                                                             self,landmark_value))
                    # 将归一化坐标写入列表，用来对比标准度
                    self.current_landmarks.append([float(normalized_coord[0]),
                                                   float(normalized_coord[1])])
                    
                # 将所有关键点的像素坐标或者归一化坐标写入csv，，只能开其中一个
                    # 将像素坐标写入AllLandmarks.csv   
                    self.csv[3].writerow([self.frame_number,
                                          landmark_value,
                                          landmark_pixel_coord[0],
                                          landmark_pixel_coord[1]])
                    # 将归一化坐标写入AllLandmarks.csv
                    # self.csv[3].writerow([self.frame_number,
                    #                       landmark_value,
                    #                       normalized_coord[0],
                    #                       normalized_coord[1]]) 

                # 将归一化坐标列表转换成ndarrays数组
                NPcurrent_landmarks = np.array(self.current_landmarks)   


                # 判断动作标准度
                compare_result = set_compare(self.face_state,NPcurrent_landmarks)

                global standardaction
                global action

                # 将分数改成百分制
                ready_scores_nor = float(compare_result[1])*100
                attack_scores_nor = float(compare_result[2])*100

                # 将分数分权，避免高分
                ready_scores = dt.conversion_scores(ready_scores_nor)
                attack_scores = dt.conversion_scores(attack_scores_nor)
                ready_scores = float(f"{ready_scores:.2f}")
                attack_scores = float(f"{attack_scores:.2f}")

                if right_hand_angle > 165:
                    temporary_action = '攻击动作'
                    temporary_standardaction = attack_scores
                    print(f"攻击动作标准度:{attack_scores}")

                elif right_hand_angle < 120:
                    temporary_action = '持剑动作'
                    temporary_standardaction = ready_scores
                    print(f"持剑动作标准度:{ready_scores}")
                else:
                    max_scores = ready_scores if ready_scores > attack_scores else attack_scores
                    temporary_action = '其他动作'
                    temporary_standardaction = max_scores

                if self.global_state == 1:
                    global_right_hand_angle = right_hand_angle
                    global_right_ankle_angle = right_ankle_angle
                    global_left_ankle_angle = left_ankle_angle
                    global_right_knee_angle = right_knee_angle
                    global_left_knee_angle = left_knee_angle
                    global_right_shoulder = right_shoulder
                    global_left_shoulder = left_shoulder
                    global_right_hip = right_hip
                    global_left_hip = left_hip
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    standardaction = temporary_standardaction
                    action = temporary_action
                    self.global_state = 0
                    self.csv[4].writerow([action,
                                          standardaction,
                                          global_right_hand_angle,
                                          global_right_knee_angle,
                                          global_left_knee_angle,
                                          global_right_shoulder,
                                          global_left_shoulder,
                                          global_right_hip,
                                          global_left_hip
                                          ])
                    print(f"更新数据: {global_sum_distance}, {global_right_hand_angle}, {action}")

                # 归一化坐标列表清空
                self.current_landmarks.clear() 
                
                return True
        return success

    # 绘制骨骼
    def draw_skeleton(self):
        # 定义骨架连接
        connections = [
            # 头部和躯干
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            
            # 躯干
            (11, 12), (12, 14), (14, 16),  # 右臂
            (11, 13), (13, 15),  # 左臂
            (11, 23), (23, 25), (25, 27), (27, 29), (27, 31), (29,31),# 左腿
            (12, 24), (24, 26), (26, 28), (28, 30), (28,32), (30,32), # 右腿
            
            # 身体中线
            (23, 24),  # 臀部
            (11, 12)   # 肩部
        ]
        
        # 绘制连接线
        for connection in connections:
            start = connection[0]
            end = connection[1]
            x_coords = [self.landmarks_3d[0][start], self.landmarks_3d[0][end]]
            y_coords = [self.landmarks_3d[1][start], self.landmarks_3d[1][end]]
            z_coords = [self.landmarks_3d[2][start], self.landmarks_3d[2][end]]
            # 'b-'：线段选择蓝色  linewidth=2：线宽=2
            self.ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2)  

    def update_plot(self, frame):
        success = self.process_frame()
        if not success:
            return False
        if self.landmarks_3d:
            self.ax.clear()
            
            # 绘制关键点
            self.ax.scatter(
                self.landmarks_3d[0],
                self.landmarks_3d[1],
                self.landmarks_3d[2],
                c='red',
                marker='o'
            )
            # 绘制骨架
            self.draw_skeleton()
            
            # 设置坐标轴范围和标签
            self.ax.set_xlim([-1, 1])
            self.ax.set_ylim([-1, 1])
            self.ax.set_zlim([-1, 1])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Z')
            self.ax.set_zlabel('Y')
            self.ax.view_init(elev=self.elev, azim=self.azim)
            self.canvas.draw()

            # 将matplotlib动画转成base64编码
            img_data = io.BytesIO()
            self.fig.savefig(img_data, format='png')
            img_data.seek(0)
            # global img_base64
            # img_base64 = base64.b64encode(img_data.read()).decode('utf-8')
            skeleton_path = os.path.join(os.getcwd(), 'snapshoot', 'current_3d_skeleton.png')
            with open(skeleton_path, 'wb') as f:
                f.write(img_data.read())
        return True

    def get_landmarks_3d(self):
        # 返回当前帧的关键点3D坐标
        return self.landmarks_3d

    def start_animation(self):
        """
        以下是三种方法使算法逐帧读取视频
        第一种：显示matplotlib动画或保存matplotlib动画--- 有3D骨架
        第二种：直接循环update_plot，更新视频帧--- 有3D骨架
        第三种：直接调用process_frame使循环--- 没有3D骨架，只有cv2图像。运行速度最快
        """
        # 第三种
        # 直接调用process_frame使循环
        try:
            while True:
                # success = self.process_frame()
                
                if not self.update_plot(None):
                    print("视频读取结束")
                    break
                print(f"处理帧: {self.frame_number}")
        finally:
            self.release_resources()

    def release_resources(self):
        # 释放视频捕捉对象
        if self.cap.isOpened():
            self.cap.release()
            print("视频捕捉资源已释放")
        
        # 关闭所有打开的CSV文件
        for csv_writer in self.csv:
            if hasattr(csv_writer, 'close'):
                csv_writer.close()
                print("CSV文件已关闭")
        
        # 关闭Matplotlib图形
        plt.close(self.fig)
        print("Matplotlib图形已关闭")
