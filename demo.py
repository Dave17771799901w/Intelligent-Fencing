import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class PoseDetector:
    def __init__(self, video_path):
        # 初始化MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 读取视频
        self.cap = cv2.VideoCapture(video_path)
        
        # 获取视频的原始尺寸
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 初始化matplotlib图形
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 存储关键点数据
        self.landmarks_3d = []
        
        # 存储视角参数
        self.elev = 15
        self.azim = 90
        
        # 创建OpenCV窗口并设置原始尺寸
        cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pose Detection', self.frame_width, self.frame_height)
        
        # 连接matplotlib的视角更新事件
        self.fig.canvas.mpl_connect('motion_notify_event', self._update_view)
        
    def _update_view(self, event):
        if event.inaxes == self.ax:
            self.elev = self.ax.elev
            self.azim = self.ax.azim
        
    def process_frame(self):
        success, image = self.cap.read()
        if not success:
            return False
            
        # 转换颜色空间用于MediaPipe处理
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 进行姿态检测
        results = self.pose.process(image_rgb)
        
        # 在图像上绘制检测结果
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            
        # 显示图像
        cv2.imshow('Pose Detection', image)
        cv2.waitKey(1)
        
        if results.pose_world_landmarks:
            landmarks = results.pose_world_landmarks.landmark
            
            # 提取所有关键点的3D坐标并进行坐标变换
            x = [-landmark.x for landmark in landmarks]
            y = [-landmark.y for landmark in landmarks]
            z = [landmark.z for landmark in landmarks]
            
            # 交换y和z坐标，使人体垂直显示
            self.landmarks_3d = [x, z, y]
            return True
        return True

    def draw_skeleton(self):
        # 定义骨架连接
        connections = [
            # 头部和躯干
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            
            # 躯干
            (11, 12), (12, 14), (14, 16),  # 右臂
            (11, 13), (13, 15),  # 左臂
            (11, 23), (23, 25), (25, 27), (27, 29), (27, 31),  # 左腿
            (12, 24), (24, 26), (26, 28), (28, 30), (28, 32),  # 右腿
            
            # 身体中线
            (23, 24),  # 臀部
            (11, 12),  # 肩部
        ]
        
        # 绘制连接线
        for connection in connections:
            start = connection[0]
            end = connection[1]
            x_coords = [self.landmarks_3d[0][start], self.landmarks_3d[0][end]]
            y_coords = [self.landmarks_3d[1][start], self.landmarks_3d[1][end]]
            z_coords = [self.landmarks_3d[2][start], self.landmarks_3d[2][end]]
            self.ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=2)
        
    def update_plot(self, frame):
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
                
                # 使用存储的视角参数
                self.ax.view_init(elev=self.elev, azim=self.azim)
        
    def start_animation(self):
        anim = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=1,
            blit=False
        )
        plt.show()
        
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

# 使用示例
def main():
    video_path = "D:/ICVS/face_recognition/test/ccc.mp4"  # 替换为你的视频路径
    detector = PoseDetector(video_path)
    detector.start_animation()

if __name__ == "__main__":
    main()