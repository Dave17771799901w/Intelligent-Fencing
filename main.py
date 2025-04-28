import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from moviepy.editor import *
import os
import threading
import time

# 创建GUI界面
class VideoClipperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("视频人体检测剪辑工具")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # 主框架
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 1. 文件选择部分
        file_frame = ttk.LabelFrame(main_frame, text="文件选择", padding=10)
        file_frame.pack(fill=tk.X, pady=10)
        
        self.browse_button = ttk.Button(file_frame, text="浏览文件", command=self.browse_file, width=15)
        self.browse_button.pack(side=tk.LEFT, padx=10)
        
        self.file_path_label = ttk.Label(file_frame, text="未选择文件")
        self.file_path_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # 2. 参数设置部分
        params_frame = ttk.LabelFrame(main_frame, text="参数设置", padding=10)
        params_frame.pack(fill=tk.X, pady=10)
        
        # 检测敏感度
        sensitivity_frame = ttk.Frame(params_frame)
        sensitivity_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sensitivity_frame, text="检测敏感度:").pack(side=tk.LEFT, padx=10)
        
        self.sensitivity_var = tk.DoubleVar(value=0.4)
        self.sensitivity_scale = ttk.Scale(sensitivity_frame, from_=0.1, to=1.0, 
                                         orient=tk.HORIZONTAL, length=300, 
                                         variable=self.sensitivity_var)
        self.sensitivity_scale.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        ttk.Label(sensitivity_frame, textvariable=self.sensitivity_var, width=4).pack(side=tk.LEFT)
        
        # 处理间隔帧数
        skip_frame = ttk.Frame(params_frame)
        skip_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(skip_frame, text="处理间隔帧数:").pack(side=tk.LEFT, padx=10)
        
        self.skip_var = tk.IntVar(value=5)
        self.skip_scale = ttk.Scale(skip_frame, from_=1, to=10, 
                                  orient=tk.HORIZONTAL, length=300, 
                                  variable=self.skip_var)
        self.skip_scale.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        
        ttk.Label(skip_frame, textvariable=self.skip_var, width=4).pack(side=tk.LEFT)
        
        # 3. 操作按钮部分
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        self.process_button = ttk.Button(action_frame, text="开始处理", command=self.process_video, width=15)
        self.process_button.pack(pady=5)
        
        # 4. 进度显示部分 - 固定位置
        progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding=10)
        progress_frame.pack(fill=tk.X, pady=10)
        
        # 状态信息标签 - 固定高度
        self.status_frame = ttk.Frame(progress_frame, height=40)
        self.status_frame.pack(fill=tk.X)
        self.status_frame.pack_propagate(False)  # 防止内部组件改变Frame尺寸
        
        self.status_label = ttk.Label(self.status_frame, text="就绪", anchor=tk.CENTER)
        self.status_label.pack(expand=True)
        
        # 进度条 - 固定在状态标签下方
        progress_bar_frame = ttk.Frame(progress_frame)
        progress_bar_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_bar_frame, orient="horizontal", 
                                           length=400, mode="determinate",
                                           variable=self.progress_var)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.progress_label = ttk.Label(progress_bar_frame, text="0%", width=5)
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        # 5. 输出信息部分
        output_frame = ttk.LabelFrame(main_frame, text="输出信息", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.output_text = tk.Text(output_frame, height=5, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)
        
        # 初始化变量
        self.file_path = None
        self.processing = False
        
    def log_message(self, message):
        """添加消息到输出文本框"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.output_text.config(state=tk.DISABLED)
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv")])
        if file_path:
            self.file_path = file_path
            filename = os.path.basename(file_path)
            self.file_path_label.config(text=f"已选择: {filename}")
            self.log_message(f"已选择文件: {filename}")
    
    def process_video(self):
        if not self.file_path:
            self.status_label.config(text="请先选择视频文件")
            self.log_message("错误: 请先选择视频文件")
            return
        
        if self.processing:
            return  # 防止重复点击
            
        self.processing = True
        self.process_button.config(state=tk.DISABLED)
        self.status_label.config(text="正在处理视频...")
        self.progress_var.set(0)
        self.progress_label.config(text="0%")
        self.log_message("开始处理视频...")
        
        # 在一个单独的线程中处理视频
        thread = threading.Thread(target=self.video_processing_thread)
        thread.daemon = True  # 确保程序退出时线程也终止
        thread.start()
    
    def update_progress(self, value, text=None):
        """安全地更新进度显示"""
        try:
            self.progress_var.set(value)
            self.progress_label.config(text=f"{int(value)}%")
            if text:
                self.status_label.config(text=text)
            self.root.update_idletasks()  # 更新UI但不阻塞
        except Exception as e:
            print(f"更新进度时出错: {e}")
    
    def video_processing_thread(self):
        try:
            # 使用OpenCV的HOG检测器
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            cap = cv2.VideoCapture(self.file_path)
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 在主线程中更新视频信息显示
            video_info = f"视频总帧数: {total_frames}, FPS: {fps:.2f}"
            self.root.after(0, lambda: self.status_label.config(text=video_info))
            self.root.after(0, lambda: self.log_message(video_info))
            
            # 用于跟踪人体检测结果的变量
            person_detected = False
            frames_without_person = 0
            max_frames_without_person = int(fps * 3)  # 3秒无人的帧数
            
            # 存储剪辑点
            clip_segments = []
            start_frame = None
            
            # 处理每一帧
            frame_count = 0
            skip_frames = self.skip_var.get()  # 获取用户设置的跳帧数
            detection_threshold = self.sensitivity_var.get()  # 获取检测敏感度
            
            self.root.after(0, lambda: self.log_message(
                f"使用参数: 检测敏感度={detection_threshold}, 处理间隔={skip_frames}帧"))
            
            last_update = 0  # 上次更新UI的时间
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 每隔几帧处理一次，以提高速度
                if frame_count % skip_frames == 0:
                    # 计算进度并更新UI（限制更新频率，避免UI冻结）
                    progress = int((frame_count / total_frames) * 100)
                    current_time = time.time()
                    if current_time - last_update > 0.2:  # 每200ms更新一次
                        self.root.after(0, lambda p=progress: self.update_progress(p))
                        last_update = current_time
                    
                    # 将图像调整为较小尺寸以加快检测
                    frame_resized = cv2.resize(frame, (640, 480))
                    
                    # 使用HOG检测人体
                    boxes, weights = hog.detectMultiScale(
                        frame_resized, 
                        winStride=(8, 8),
                        padding=(4, 4),
                        scale=1.05
                    )
                    
                    # 根据敏感度过滤检测结果
                    filtered_boxes = [box for box, weight in zip(boxes, weights) if weight > detection_threshold]
                    current_person_detected = len(filtered_boxes) > 0
                    
                    # 检测逻辑
                    if current_person_detected:
                        if not person_detected:
                            # 首次检测到人体，开始新片段
                            person_detected = True
                            start_frame = frame_count
                            msg = f"第 {frame_count} 帧检测到人，开始新片段"
                            self.root.after(0, lambda m=msg: self.status_label.config(text=m))
                            self.root.after(0, lambda m=msg: self.log_message(m))
                        frames_without_person = 0
                    else:
                        if person_detected:
                            # 增加无人帧计数
                            frames_without_person += skip_frames
                            
                            # 如果连续3秒无人，结束当前片段
                            if frames_without_person >= max_frames_without_person:
                                end_frame = frame_count - frames_without_person
                                if start_frame is not None and end_frame > start_frame:
                                    clip_segments.append((start_frame, end_frame))
                                    msg = f"添加片段: {start_frame}-{end_frame} (持续 {(end_frame-start_frame)/fps:.2f} 秒)"
                                    self.root.after(0, lambda m=msg: self.status_label.config(text=m))
                                    self.root.after(0, lambda m=msg: self.log_message(m))
                                person_detected = False
                                start_frame = None
                
                frame_count += 1
            
            # 处理最后一个片段
            if person_detected and start_frame is not None:
                clip_segments.append((start_frame, frame_count - 1))
                msg = f"添加最后片段: {start_frame}-{frame_count-1} (持续 {(frame_count-1-start_frame)/fps:.2f} 秒)"
                self.root.after(0, lambda m=msg: self.log_message(m))
            
            cap.release()
            
            # 在主线程中更新UI，显示剪辑信息
            summary = f"检测到 {len(clip_segments)} 个片段，正在准备剪辑..."
            self.root.after(0, lambda: self.update_progress(100, summary))
            self.root.after(0, lambda: self.log_message(summary))
            
            # 剪辑视频
            if clip_segments:
                self.create_video_clips(clip_segments, fps)
            else:
                self.root.after(0, lambda: self.update_progress(100, "未检测到人体或无需剪辑"))
                self.root.after(0, lambda: self.log_message("未检测到人体或无需剪辑"))
        except Exception as e:
            error_msg = f"处理出错: {str(e)}"
            self.root.after(0, lambda: self.status_label.config(text=error_msg))
            self.root.after(0, lambda: self.log_message(error_msg))
        finally:
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))
            self.processing = False
    
    def create_video_clips(self, clip_segments, fps):
        try:
            # 使用MoviePy剪辑视频
            input_clip = VideoFileClip(self.file_path)
            output_dir = os.path.dirname(self.file_path)
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            
            # 创建输出目录
            output_folder = os.path.join(output_dir, f"{base_name}_clips")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                
            self.root.after(0, lambda: self.log_message(f"创建输出目录: {output_folder}"))
            
            clip_count = len(clip_segments)
            valid_clips = 0
            
            # 创建每个子剪辑作为单独的文件
            for i, (start_frame, end_frame) in enumerate(clip_segments):
                start_time = start_frame / fps
                end_time = end_frame / fps
                
                # 仅处理至少1秒的片段
                if end_time - start_time < 1:
                    self.root.after(0, lambda: self.log_message(
                        f"跳过片段 {i+1} (太短: {end_time-start_time:.2f}秒)"))
                    continue
                
                valid_clips += 1
                
                # 更新进度和状态信息
                progress = int((i + 1) / clip_count * 100)
                msg = f"正在剪辑片段 {i+1}/{clip_count}"
                clip_info = f"片段 {i+1}: {start_time:.2f}s - {end_time:.2f}s (持续 {end_time-start_time:.2f}s)"
                
                self.root.after(0, lambda p=progress, m=msg: self.update_progress(p, m))
                self.root.after(0, lambda m=clip_info: self.log_message(m))
                
                # 创建子剪辑
                subclip = input_clip.subclip(start_time, end_time)
                
                # 生成输出文件名
                output_path = os.path.join(output_folder, f"{base_name}_clip_{i+1}.mp4")
                
                # 导出剪辑后的视频
                subclip.write_videofile(
                    output_path, 
                    codec="libx264", 
                    audio_codec="aac", 
                    verbose=False, 
                    logger=None,
                    threads=4  # 使用多线程加速导出
                )
                
                self.root.after(0, lambda p=output_path: self.log_message(f"已保存: {os.path.basename(p)}"))
            
            summary = f"处理完成！已保存 {valid_clips} 个视频片段到: {output_folder}"
            self.root.after(0, lambda m=summary: self.update_progress(100, "处理完成！"))
            self.root.after(0, lambda m=summary: self.log_message(m))
            input_clip.close()
        except Exception as e:
            error_msg = f"剪辑出错: {str(e)}"
            self.root.after(0, lambda m=error_msg: self.status_label.config(text=m))
            self.root.after(0, lambda m=error_msg: self.log_message(m))

if __name__ == "__main__":
    root = tk.Tk()
    # 使用ttk主题
    style = ttk.Style()
    style.theme_use('clam')  # 可以尝试不同主题: 'alt', 'default', 'classic', 'clam', 'vista'
    
    app = VideoClipperApp(root)
    root.mainloop()