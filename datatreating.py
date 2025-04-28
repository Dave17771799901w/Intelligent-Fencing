import gc
import tempfile
import traceback
import cv2
import datetime
import os
import csv
import math
from flask import app
# from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import csv
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
# from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import patheffects
import textwrap
import uuid
# 添加中文支持
try:
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
except ImportError:
    pass

try:
    from . import pose_recognition as pr
except Exception as a:   # Exception所有异常    代码错了就执行
    import pose_recognition as pr


def now_time():
        now  = datetime.datetime.now()
        nownow = now.strftime("%Y_%m_%d_%H%M")
        return nownow


    # 切割图像
def slicing_photo(self,side,image):
    """
    slicing_photo函数可以  将图片垂直切割,根据side变量选择返回左边、右边或完整的图片
    :param self:
    :param side:选择需要返回的边,0-完整  1-左边  2-右边
    :return:
    """
    # 获取当前帧的数据
    height, width = image.shape[:2]
    # 将图片垂直居中切割
    halp_width = width // 2
    
    # 判断返回哪一边的图片
    if side == 1:
        left_photo = image[:,:halp_width]
        return left_photo
    
    elif side == 2:
        right_photo = image[:,halp_width:]
        return right_photo
    
    elif side == 0:
        # print(f"+++++{image}")
        return image
    

def normalized_to_pixel(self,value):
        """
        normalized_to_pixel函数可以接收三个参数,并获取索引值的归一化坐标，并将归一化坐标转换成像素坐标
        :param results:当前帧的关键点信息
        :param self:
        :param value:需要获取的关键点索引值
        :return:返回像素坐标(元组)
        """
        landmark = self.results.pose_landmarks.landmark[value]
        normalized_coord = (landmark.x,landmark.y)
        pixel_coord = ((int(normalized_coord[0]*self.frame_width), 
                        int(normalized_coord[1]*self.frame_height)))
        return pixel_coord, normalized_coord


# 两点距离公式
def Two_point_distance(one_point,two_point):
    """
    Two_point_distance函数可以接收两个参数(坐标点),并将两个坐标点的距离返回  ---二维
    :param one_point:第一个坐标点(x, y)
    :param two_point:第二个坐标点(x, y)
    :return: 返回两点之间距离的计算结果
    """
    result_distance = math.sqrt((two_point[0]-one_point[0])**2+(two_point[1]-one_point[1])**2)
    return result_distance


# 初始化三个csv文件
def setup_csv():
    """
    setup_csv   函数可以在此py文件目录下新建一个data_csv文件夹,
                并初始化三个csv文件( AllLandmarks{nowtime}.csv  NeedData{nowtime}.csv  AttackData{nowtime}.csv)
    :param self:
    :return:
    """
# 创建 data_csv 文件夹
    # 获取当前时间
    nowtime = now_time()

    # 获取当前文件的绝对路径
    current_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    script_dirt = os.path.dirname(current_path)
    # 定义新文件名字
    new_file_name = "data_csv"
    new_file_path = os.path.join(script_dirt,new_file_name)  

    try:
        os.makedirs(new_file_path,exist_ok=True)
        # print(f"文件夹'{new_file_path}'创建成功")
    except OSError as error:
        print(f"创建文件夹出错：{error}")

#生成三个csv路径  all_landmarks_csv_path    all_NeedData_csv_path   all_AttackData_csv_path
    all_landmarks_csv_path = f"{new_file_path}\\AllLandmarks{nowtime}.csv" 
    all_NeedData_csv_path = f"{new_file_path}/ActionAnalysis{nowtime}.csv"  
    all_AttackData_csv_path = f"{new_file_path}/AttackData{nowtime}.csv" 

    # 新建三个csv文件   encoding='utf-8-sig'
    csv_AllLandmarks = open(all_landmarks_csv_path,'w',newline='', encoding='utf-8-sig') # 打开csv文件
    csv_NeedData = open(all_NeedData_csv_path,'w',newline='', encoding='utf-8-sig') # 打开csv文件
    csv_AttackData = open(all_AttackData_csv_path,'w',newline='', encoding='utf-8-sig') # 打开csv文件

    # 设置AllLandmarks.csv页眉
    csv_W_AllLandmarks = csv.writer(csv_AllLandmarks)
    csv_W_AllLandmarks.writerow(['当前帧率','关键点','X','Y'])

    # 设置NeedData.csv页眉
    csv_W_NeedData = csv.writer(csv_NeedData)
    csv_W_NeedData.writerow(['检测动作',
                             '动作得分',
                             '右手角度(°)',
                             '右膝角度(°)',
                             '左膝角度(°)',
                             '右肩坐标',
                             '左肩坐标',
                             '右臀坐标',
                             '左臀坐标'
                             ]) 
    
    # 设置AttackData.csv页眉
    csv_W_AttackData = csv.writer(csv_AttackData)
    csv_W_AttackData.writerow(['攻击次数',
                               '移动距离(cm)',
                               '右手夹角(°)',
                               '右脚夹角(°)',
                               '左脚夹角(°)',
                               '右膝夹角(°)',
                               '左膝夹角(°)',
                               '右肩坐标',
                               '左肩坐标',
                               '右臀坐标',
                               '左臀坐标'])
    return [csv_AllLandmarks,
            csv_NeedData,
            csv_AttackData,
            csv_W_AllLandmarks,
            csv_W_NeedData,
            csv_W_AttackData,
            all_AttackData_csv_path,
            all_NeedData_csv_path]


def cal_ang(self,value1, value2, value3):

    """
    根据三点坐标计算夹角
    :param value1: 第一个点的索引值
    :param value2: 第二个点的索引值
    :param value3: 第三个点的索引值，（如果为负一，没有实体点，将自动创建辅助点）
    :return: 返回任意角的夹角值,这里只是返回点2的夹角
    """
    
    # 判断1、2索引值是否有效（0~32）
    if 0 <= value1 <33 and 0 <= value2 <33:
            point_1 = normalized_to_pixel(self,value1)[0]  # 获取第一个点的像素坐标
            point_2 = normalized_to_pixel(self,value2)[0]  # 获取第二个点的像素坐标
    else:
            print("计算角度的关键点查找不到")
            return False
    
    # 判断第三个点是否有效
    if 0 <= value3 <33:
            point_3 = normalized_to_pixel(self,value3)[0]  # 获取第三个点的像素坐标
    # 如果value3为负一，为第三个角设定辅助点        
    elif value3 == -1:
            point_3 = (point_2[0]+5,point_2[1])         # 创建一个虚拟点，作为第三个点
    else:
            print("计算角度的第三点溢出")
            return False
    
    pA = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    pB = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    pC = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    
    angleA = math.degrees(math.acos((pA * pA - pB * pB - pC * pC) / (-2 * pB * pC)))
    angleB = math.degrees(math.acos((pB * pB - pA * pA - pC * pC) / (-2 * pA * pC)))
    angleC = math.degrees(math.acos((pC * pC - pA * pA - pB * pB) / (-2 * pA * pB)))
    # 左右脚踝与地面夹角，只取小角
    if value2 in [27,28] and angleB >90:
            angleB = 180-angleB

    # 格式化输出
    angleB = float(format(angleB,'.2f'))
    return angleB


def hight_pixel_rstio(self):
    """
    hight_pixel_rstio 函数可以接收两个参数,并获取鼻子、右臀、右膝盖、右脚踝四个关键点的像素坐标，
                        计算人体像素身高，最后用人体实际身高/人体像素身高，得到像素比例因子
    :param self:
    :param results: 当前帧的姿态检测结果
    :return: 返回像素比例因子
    """
    nose_pixel = normalized_to_pixel(self,0)[0]  # 鼻子像素点
    right_hip_pixel = normalized_to_pixel(self,24)[0]  # 右臀像素点
    right_knee_pixel = normalized_to_pixel(self,26)[0]  # 右膝盖像素点
    right_ankle_pixel = normalized_to_pixel(self,28)[0] # 右脚踝像素点

    nose_to_hip = Two_point_distance(nose_pixel,right_hip_pixel)  # 鼻子到右臀距离
    hip_to_knee = Two_point_distance(right_hip_pixel,right_knee_pixel) # 右臀到右膝盖距离
    knee_to_ankle = Two_point_distance(right_knee_pixel,right_ankle_pixel) # 右膝盖到右脚踝

    human_height_pixel = nose_to_hip + hip_to_knee + knee_to_ankle  # 像素点身高
    pixel_ratio = self.height_real / human_height_pixel # 人体比例因子
    return pixel_ratio,human_height_pixel


def is_similar_frame(prev_frame, current_frame, threshold):
    # 计算两帧之间的差异
    diff = np.linalg.norm(np.array(prev_frame) - np.array(current_frame))
    # 如果差异小于阈值，则认为两帧相似
    return diff < threshold

def select_keyframes(frames, threshold):
    keyframes = [frames[0]]  # 将第一帧作为关键帧
    for i in range(1, len(frames)):
        if is_similar_frame(keyframes[-1], frames[i], threshold):
            continue  # 如果当前帧与上一个关键帧相似，则跳过
        keyframes.append(frames[i])  # 将当前帧添加为关键帧
    return keyframes


def conversion_scores(scores_str):
    """
    conversion_scores 得分权重，避免整体高分
    """
    scores = float(scores_str)
    if scores >= 97:
        scores *= 1
        return scores
    if 95 <= scores < 97:
        scores *= 0.95
        return scores
    if 90 <= scores < 95:
        scores *= 0.9
        return scores
    if 85 <= scores < 90:
        scores *= 0.85
        return scores
    if 80 <= scores < 85:
        scores *= 0.8
        return scores
    if 75 <= scores < 80:
        scores *= 0.75
        return scores
    if 70 <= scores < 75:
        scores *= 0.7
        return scores
     



# CSV 文件分析
def analyze_csv_data(csv_path):
    if not os.path.exists(csv_path):
        return None
       
    # 读取 CSV 数据
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if len(rows) == 0:
            return None
        
        data = rows[0]
        action = data.get('检测动作', '')
        score = data.get('动作得分', '')
        right_hand_angle = data.get('右手角度(°)', '')
        right_knee_angle = data.get('右膝角度(°)', '')
        left_knee_angle = data.get('左膝角度(°)', '')
        right_shoulder = data.get('右肩坐标', '')
        left_shoulder = data.get('左肩坐标', '')
        right_hip = data.get('右臀坐标', '')
        left_hip = data.get('左臀坐标', '')

        # 动作分析
        evaluation = evaluate_action(
            action,
            score,
            right_hand_angle,
            right_knee_angle,
            left_knee_angle,
            right_shoulder,
            left_shoulder,
            right_hip,
            left_hip
        )
        # 添加在analyze_csv_data的返回前
        if not all([right_shoulder, left_shoulder]):
            raise ValueError("关键关节点坐标数据缺失")
        return {
            'action': action,
            'score': score,
            'right_hand_angle': right_hand_angle,
            'right_knee_angle': right_knee_angle,
            'left_knee_angle': left_knee_angle,
            'right_shoulder': right_shoulder,
            'left_shoulder': left_shoulder,
            'right_hip': right_hip,
            'left_hip': left_hip,
            'evaluation': evaluation
        }


def evaluate_action(action, score, right_hand_angle, right_knee_angle, left_knee_angle,
                    right_shoulder, left_shoulder, right_hip, left_hip):
    evaluation = []
    try:
        # 通用身体姿态分析
        def parse_coord(coord_str):
            return tuple(map(int, coord_str.strip("()").split(", ")))

        # 身体对称性分析
        r_shoulder = parse_coord(right_shoulder)
        l_shoulder = parse_coord(left_shoulder)
        shoulder_symmetry = abs(r_shoulder[0] - l_shoulder[0])

        r_hip = parse_coord(right_hip)
        l_hip = parse_coord(left_hip)
        hip_symmetry = abs(r_hip[0] - l_hip[0])

        if shoulder_symmetry > 100:
            evaluation.append("肩部不对称（差值 {}px）影响出剑稳定性".format(shoulder_symmetry))
        if hip_symmetry > 80:
            evaluation.append("髋部不对称（差值 {}px）导致重心不稳".format(hip_symmetry))

        if action == '持剑动作':
            # 持剑姿势详细分析
            hand_angle = float(right_hand_angle)
            if hand_angle < 85:
                evaluation.append("持剑手肘部过直（{}°），建议保持 85-100° 弯曲".format(hand_angle))
            elif hand_angle > 105:
                evaluation.append("持剑手肘部过屈（{}°），影响快速出击".format(hand_angle))

            # 下肢稳定性分析
            r_knee = float(right_knee_angle)
            l_knee = float(left_knee_angle)
            if r_knee < 110:
                evaluation.append("右膝弯曲不足（{}°），建议保持在 110-130°".format(r_knee))
            if l_knee > 160:
                evaluation.append("支撑腿过于直立（{}°），影响突进爆发力".format(l_knee))

            # 重心投影分析
            hip_center = (r_hip[0] + l_hip[0])/2
            foot_projection = abs(hip_center - (r_shoulder[0] + 50))  # 理想投影偏移量
            if foot_projection > 80:
                evaluation.append("重心投影偏移（{}px）易导致失衡".format(foot_projection))

        elif action == '攻击动作':
            # 攻击动作动态分析
            r_knee = float(right_knee_angle)
            if r_knee < 120:
                evaluation.append("前腿弯曲不足（{}°），应达到 120-140° 以保持冲力".format(r_knee))
            
            # 上肢协调性分析
            shoulder_hip_ratio = (r_shoulder[1] - r_hip[1]) / (l_shoulder[1] - l_hip[1])
            if shoulder_hip_ratio < 0.8:
                evaluation.append("躯干扭转不足（比率 {:.2f}），影响攻击力度".format(shoulder_hip_ratio))

            # 武器臂线性分析
            elbow_wrist_slope = (r_shoulder[1] - parse_coord(right_shoulder)[1]) / (r_shoulder[0] - parse_coord(right_shoulder)[0])
            if abs(elbow_wrist_slope) < 0.3:
                evaluation.append("剑尖指向偏离目标线（斜率 {:.2f}）".format(elbow_wrist_slope))

        # 动态平衡指标
        front_back_ratio = abs(r_shoulder[0] - l_hip[0]) / abs(l_shoulder[0] - r_hip[0])
        if front_back_ratio < 0.9 or front_back_ratio > 1.1:
            evaluation.append("前后腿负荷不均衡（比率 {:.2f}）".format(front_back_ratio))

        # 能量传递效率分析
        kinetic_chain = float(right_hand_angle)/180 * float(right_knee_angle)/140
        if kinetic_chain < 0.6:
            evaluation.append("动力链效率低下（{:.2f}），注意腿腰手的发力顺序".format(kinetic_chain))

    except Exception as e:
        evaluation.append("姿态解析异常：" + str(e))

    return "\n".join(evaluation) if evaluation else "动作表现良好，符合技术规范"


# 生成图片
# def generate_action_image(result):
#     # 创建空白图片
#     img = Image.new('RGB', (800, 600), color='white')
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", 20)  # 需要使用支持的字体


#     # 添加标题
#     draw.text((50, 50), "动作分析报告", fill='black', font=font)

#     # 添加动作信息
#     draw.text((50, 100), f"动作: {result['action']}", fill='black', font=font)
#     draw.text((50, 150), f"动作得分: {result['score']}", fill='black', font=font)
#     draw.text((50, 200), f"右手角度: {result['right_hand_angle']}°", fill='black', font=font)
#     draw.text((50, 250), f"右膝角度: {result['right_knee_angle']}°", fill='black', font=font)
#     draw.text((50, 300), f"左膝角度: {result['left_knee_angle']}°", fill='black', font=font)
#     draw.text((50, 350), f"评价: {result['evaluation']}", fill='black', font=font)

#     # 保存图片
#     image_path = os.path.join('C:/Users/lx/Desktop/archives/fencing_SourceCode/invideo', 'action_report.png')
#     os.makedirs(os.path.dirname(image_path), exist_ok=True)
#     img.save(image_path)

#     return image_path




def generate_action_image(result):
    """
    生成包含姿态分析和雷达图的综合报告
    返回生成的图片路径
    """
    # # 创建带数据填充的可视化报告
    # img = Image.new('RGB', (1000, 1600), color='white')
    # draw = ImageDraw.Draw(img)
    # # print("[DEBUG] 传入的result数据结构:")
    # # import pprint
    # # pprint.pprint(result)

    # # 检查必要字段是否存在
    # required_fields = ['action', 'score', 'right_hand_angle']
    # for field in required_fields:
    #     if field not in result:
    #         print(f"⚠️ 警告: result缺少关键字段 {field}")
    #         return os.path.join(os.getcwd(), 'static', 'error_fallback.png')
    
    # 创建带数据填充的可视化报告
    try:
        # 创建画布
        img = Image.new('RGB', (1000, 1600), color='white')
        draw = ImageDraw.Draw(img)

        # 检查必要字段是否存在
        required_fields = ['action', 'score', 'right_hand_angle']
        for field in required_fields:
            if field not in result:
                print(f"⚠️ 警告: result缺少关键字段 {field}")
                return os.path.join(os.getcwd(), 'static', 'error_fallback.png')

        # 寻找字体
        font_size = 35
        small_font_size = 30
        try:
            OSname = os.name
            # Windows系统字体路径
            if OSname == 'nt':
                font_path = "msyh.ttc"  # 微软雅黑
            # Linux/Mac系统字体路径
            else:  
                font_path = "/System/Library/Fonts/PingFang.ttc" # 苹方字体
            
            # 检查字体文件是否存在
            if not os.path.exists(font_path):
                raise FileNotFoundError(f"字体文件缺失: {font_path}")
            
            # 设置字体大小
            font = ImageFont.truetype(font_path, font_size)
            small_font = ImageFont.truetype(font_path, small_font_size)
       
        except Exception as e:
            print(f"⚠️ 无法加载指定字体: {e}, 尝试使用默认字体")
            font = ImageFont.load_default(font_size)
            small_font = ImageFont.load_default(small_font_size)

        # 主体结构划分
        sections = {
            '动作快照': (50, 50, 900, 400),
            '关节点分析': (50, 420, 440, 850),
            '生物力学指标': (510, 420, 900, 850),
            '运动学建议': (50, 870, 900, 1550)
        }
        # 绘制基本结构框架
        for title, (x1,y1,x2,y2) in sections.items():
            # 绘制带投影效果的边框
            draw.rectangle([x1+3,y1+3,x2+3,y2+3], fill='#888888')  # 投影层
            draw.rectangle([x1,y1,x2,y2], fill='white', outline='black')  # 主框
            # 跳过动作快照标题，绘制完快照再写标题
            if title == '动作快照':
                continue
            draw.text((x1+10,y1+5), title, fill='black', font=font)
        
        
        # 检查坐标格式
        def parse_coord(coord_str):
            try:
                # 兼容多种格式: (x,y) 或 (x, y) 或 x,y
                coord_str = coord_str.strip("()")
                x, y = map(int, coord_str.replace(" ", "").split(","))
                return (x, y)
            except Exception as e:
                print(f"坐标解析错误：{coord_str} → 使用默认坐标(0,0)")
                return (0, 0)  # 保障后续计算有默认值

        # 动作快照区
        # 设置快照位置的常量
        SNAPSHOT_SIZE = (850, 350)  # 快照尺比
        SNAPSHOT_POS_Y = 51  # 快照y1坐标
        SNAPSHOT_POS_X = 150  # 快照x1坐标
        try:
            # 定义快照路径
            snapshot_path = os.path.join(os.getcwd(), 'snapshoot/current_pose.png') 
            if os.path.exists(snapshot_path):
                action_img = Image.open(snapshot_path)
                # 重定义快照尺寸
                action_img = action_img.resize((648, 350), Image.Resampling.LANCZOS)
                # 将快照贴到报告上
                img.paste(action_img, (SNAPSHOT_POS_X, SNAPSHOT_POS_Y))
                # print("✅ 动作快照加载成功")
            else:
                # 生成居中占位图
                placeholder = Image.new('RGB', (850,310), (240,240,240))
                draw_ph = ImageDraw.Draw(placeholder)
                bbox = draw_ph.textbbox((0, 0), "无有效快照", font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                # 计算居中坐标
                text_x = (SNAPSHOT_SIZE[0] - text_width) // 2
                text_y = (SNAPSHOT_SIZE[1] - text_height) // 2

                draw_ph.text((text_x, text_y), "无有效快照", fill='#666', font=font)
                img.paste(placeholder, (50, SNAPSHOT_POS_Y))
        except Exception as e:
            print(f"❌ 快照加载失败: {str(e)}")
        # 绘制完快照再写标题，使标题浮于照片上。
        draw.text((155,50+5), '动作快照', fill='black', font=font)

        # 动作快照区增加3D骨骼图###################################
        try:
            # 加载3D骨架图
            skeleton_path = os.path.join(os.getcwd(), 'snapshoot/current_3d_skeleton.png')
            if os.path.exists(skeleton_path):
                # skeleton_img = Image.open(skeleton_path).convert('RGBA')
                skeleton_img = Image.open(skeleton_path)
                # 获取图片的宽度和高度
                width, height = skeleton_img.size

                # 裁剪图片
                # 计算需要裁剪的边长
                crop_width = int(width * 0.20)
                crop_height = int(height * 0.20)
                skeleton_img = skeleton_img.crop((crop_width, crop_height, width - crop_width, height - crop_height))
                # 保持宽高比的缩略图（最长边为380px）
                target_size = (380, 380)
                skeleton_img.thumbnail(
                    target_size,
                    resample=Image.Resampling.LANCZOS  # 使用最高质量的重采样过滤器
                )

                # 保存时的优化配置
                save_kwargs = {
                    'format': 'PNG',          # 推荐无损格式
                    'quality': 95,            # 如果存jpg可设置95+
                    'optimize': True          # 启用压缩优化
                }

                skeleton_img.save('output.png', **save_kwargs)
                img.paste(skeleton_img, (80,1130))
                
                # 添加标注
                # draw.text((action_img.width//2 + 20, 10), 
                #         "3D骨骼分析", 
                #         fill='white', 
                #         font=font,
                #         stroke_width=2,
                #         stroke_fill='black')
        except Exception as e:
            print(f"3D骨骼图合成失败: {str(e)}")


        # 绘制雷达图
        def generate_radar_chart(radar_path):
            # 创建完全独立的MPL上下文
            with matplotlib.rc_context(rc={'backend': 'Agg'}):
                print("\n===== 开始生成雷达图 =====")
                try:
                    # 创建画布
                    fig = plt.figure(figsize=(8,8), 
                                    facecolor='white', 
                                    edgecolor='#eeeeee',
                                    dpi=150,
                                    linewidth=0
                                    )
                    
                    # 创建画布和轴对象
                    canvas = FigureCanvasAgg(fig)
                    ax = fig.add_subplot(111, polar=True)

                    # # 设置样式（必须在绘图前）
                    plt.style.use('ggplot')    # 使用更好看的样式
                    # 设置所有元素的透明度
                    ax.set_facecolor((1,1,1,0.9))  # RGBA透明坐标区域
                    # fig.set_facecolor((0,0,0,0)) 

                    # 准备数据
                    radar_labels = ['爆发力', '稳定性', '协调性', '平衡性', '准确性']
                    radar_data = [
                        min(float(result['right_hand_angle'])/170*100, 100),  # 爆发力
                        min(float(result['right_knee_angle'])/140*100, 100),  # 稳定性
                        max(100 - abs(parse_coord(result['right_shoulder'])[0] - parse_coord(result['left_shoulder'])[0])/3, 0), # 协调性
                        max(100 - abs(parse_coord(result['right_hip'])[1] - parse_coord(result['left_hip'])[1])/2, 0),           # 平衡性
                        float(result['score'])                                                                                   # 准确性
                    ]
                    print("验证雷达数据:")
                    valid_data = [v for v in radar_data if 0 <= v <= 100]
                    if len(valid_data) != len(radar_data):
                        print(f"数据异常！有效值: {valid_data}")
                    else:
                        print("雷达数据有效！")
                    
                    
                    # 设置网格颜色
                    ax.spines['polar'].set_color('#dddddd')
                    ax.xaxis.grid(color='gray', linestyle='--', linewidth=0.8)

                    # 绘制数据
                    angles = np.linspace(0, 2*np.pi, len(radar_labels), endpoint=False).tolist()
                    angles += angles[:1]  # 闭合多边形
                    radar_data += radar_data[:1]

                    # 在绘图代码中添加混合渲染模式
                    # 绘制线条 (调整透明度)
                    ax.plot(angles, radar_data, 'o-', 
                            linewidth=2, 
                            # color='#1f77b4', 
                            color=(0.12, 0.47, 0.71, 0.8),  # RGBA格式蓝色
                            markersize=8,
                            path_effects=[patheffects.withStroke(
                                linewidth=3, 
                                foreground=(1,1,1,0.7),  
                                # alpha=0.7  # 描边透明度
                                )])
                    
                    # 填充区域透明度
                    ax.fill(angles, radar_data, 
                            alpha=0.25, 
                            color='#1f77b4',
                            edgecolor='black',  # 添加边界线
                            linewidth=0.8)
                    # 标签半透明处理
                    ax.set_thetagrids(
                        np.degrees(angles[:-1]), 
                        radar_labels,
                        fontsize=25,
                        color='#333333',  # 深灰色保证可读性
                        alpha=0.8# 文字透明度
                    )
                    ax.tick_params(axis='y', labelcolor='#666666', grid_color='#eeeeee')
                    ax.set_rlabel_position(200)

                    # 后处理样式
                    ax.set_ylim(0,100)
                    ax.set_theta_offset(np.pi / 2)
                    ax.set_theta_direction(-1)
                    ax.spines['polar'].set_color('#dddddd')
                    ax.set_facecolor('white')  # 设置绘图区背景
                    # 保存图片
                    print(f"保存雷达图到: {os.path.abspath(radar_path)}")
                    canvas.draw()  # ← 关键点：强制渲染数据

                    plt.savefig(
                        radar_path,
                        facecolor='white',   # 强制输出白色基础层
                        transparent=False,   # 关闭全图透明
                        edgecolor='none',
                        bbox_inches='tight',
                        pad_inches=0.1,      # 增加透明边缘缓冲
                        dpi=150
                    )

                    # 强制释放资源
                    print("雷达图保存完成")
                    return True

                except Exception as e:
                    print(f"\n!!! 致命错误详情 !!!")
                    print(e)
                    traceback.print_exc()  # 打印完整堆栈
                    return False
                
                finally:
                    # 显式解除引用
                    ax.cla()
                    fig.clf()
                    plt.close(fig)
                    del ax, fig, canvas
                    gc.collect()
            
                # --- 雷达图部分结束 ---

        # 生成唯一文件名防止冲突
        import uuid
        unique_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"
        save_dir = os.path.join(os.getcwd(), 'invideo')
        radar_success = False

        radar_path = os.path.join(save_dir, f"radar_{unique_id}.png")
        report_save_path = os.path.join(os.getcwd(), 'temp_reports', f"action_report_{unique_id}.png")
        # 步骤1: 生成雷达图
        radar_success = generate_radar_chart(radar_path)
        if radar_success:
            try:
                # 加入三阶段验证
                if not os.path.exists(radar_path):
                    raise FileNotFoundError("雷达图文件不存在")
                if os.path.getsize(radar_path) < 1024:
                    raise IOError("雷达图文件残缺")

                with open(radar_path, 'rb') as f:
                    header = f.read(8)
                    if not header.startswith(b'\x89PNG\r\n\x1a\n'):
                        raise ValueError("非有效的PNG文件")
                    
                # 嵌入雷达图
                radar_img = Image.open(radar_path)
                radar_img = radar_img.resize((380, 380))
                img.paste(radar_img, (490, 1130))
                print("✓ 雷达图已嵌入报告")
            except Exception as e:
                print(f"⚠️ 嵌入雷达图失败: {str(e)}")
    
        # 关节数据区
        joint_data = [
            ("动作类型", result['action']),
            ("动作评分", f"{result['score']}/100"),
            ("右手角度", f"{result['right_hand_angle']}°"),
            ("右膝角度", f"{result['right_knee_angle']}°"),
            ("左膝角度", f"{result['left_knee_angle']}°"),
            ("右肩坐标", result['right_shoulder']),
            ("左肩坐标", result['left_shoulder'])
        ]
        y_pos = 470
        # 循环写入报告
        for label, value in joint_data:
            draw.text((70, y_pos), f"{label}: {value}", fill='#333', font=small_font)
            y_pos += 40

        # 生物力学指标区
        biomechanics_metrics = [
            ("动力链效率", f"{float(result['right_hand_angle'])/180 * 100:.1f}%"),
            ("肩部对称性", f"{abs(parse_coord(result['right_shoulder'])[0] - parse_coord(result['left_shoulder'])[0])}px"),
            ("髋部稳定性", f"{abs(parse_coord(result['right_hip'])[0] - parse_coord(result['left_hip'])[0])}px"),
            ("重心偏移量", f"{abs((parse_coord(result['right_hip'])[0]+parse_coord(result['left_hip'])[0])//2 - parse_coord(result['right_shoulder'])[0])}px")
        ]
        y_pos = 470
        # 循环写入报告
        for metric, value in biomechanics_metrics:
            draw.text((530, y_pos), f"{metric}: {value}", fill='#333', font=small_font)
            y_pos += 40

        # 运动建议区
        y_pos = 930
        text_block = result['evaluation'].split('\n')
        for line in text_block:
            draw.text((70, y_pos), f"• {line}", fill='#d40000', font=small_font)
            y_pos += 45

        # 水印与边界线
        draw.text((150, 1508), f"纯态探索® 运动分析报告 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                fill='#999999', font=small_font)
        
        os.makedirs(os.path.dirname(report_save_path), exist_ok=True)
        for _ in range(3):  # 保存重试机制
            try:
                img.save(report_save_path, quality=95, optimize=True)
                print(f"✓ 报告已保存到：{os.path.abspath(report_save_path)}")
                return report_save_path
            except PermissionError:
                datetime.time.sleep(0.1)
        raise RuntimeError("无法保存最终报告")

    except Exception as e:
        print(f"!!! 报告生成致命错误 !!!")
        print(e)
        traceback.print_exc()
        # 紧急清理
        if 'radar_path' in locals() and os.path.exists(radar_path):
            try:
                os.remove(radar_path)
            except:
                pass
        return os.path.join(os.getcwd(), 'static', 'error_fallback.png')
    



    # 视频分析报告
try:
    font_path = "msyh.ttc"
    title_font = ImageFont.truetype(font_path, 48)
    header_font = ImageFont.truetype(font_path, 36)
    body_font = ImageFont.truetype(font_path, 28)
except Exception as e:
    # 备用字体设置
    title_font = ImageFont.load_default(40)
    header_font = ImageFont.load_default(30)
    body_font = ImageFont.load_default(20)


# 生成趋势图
def create_trend_chart(distances):
    
    plt.figure(figsize=(10,4))
    plt.plot(distances, marker='o', color='#2c7be5')
    plt.title('攻击移动距离趋势', pad=20, fontsize=14)
    plt.xlabel('攻击次数', labelpad=10)
    plt.ylabel('距离(cm)', labelpad=10)
    plt.grid(alpha=0.3)
    return plt.gcf()

# def compose_report_image(**kwargs):
#     # 使用PIL创建报告模板
#     img = Image.new('RGB', (1080, 1920), '#ffffff')
#     draw = ImageDraw.Draw(img)
    
#     # 添加标题部分
#     draw.text((50,50), "击剑训练分析报告", fill='#333', font=title_font)
    
#     # 添加统计数据区块
#     stats_box = create_stats_box(**kwargs)
#     img.paste(stats_box, (50, 150))
    
#     # 添加趋势图表
#     chart_img = Image.open(kwargs['chart_path'])
#     img.paste(chart_img, (50, 450))
    
#     # 添加运动建议
#     advice_box = create_advice_box(kwargs['advice']) 
#     img.paste(advice_box, (50, 900))
    
#     return img


# def generate_kinematic_advice(latest_attack):
#     advice = []
#     print("开始计算运动建议")
#     # 右手肘分析
#     try:
#         elbow_angle = float(latest_attack['右手夹角(°)'])
#         if elbow_angle > 175:
#             advice.append("右手存在过伸风险(%.1f°)，建议攻击时保持15-20°微曲以缓冲冲击" % elbow_angle)
            
#         # 膝盖角度分析
#         knee_angle = float(latest_attack['右膝夹角(°)'])
#         if knee_angle < 120:
#             advice.append("右膝弯曲不足(%.1f°)，攻击时应保持120-140°以获得更好爆发力" % knee_angle)
#         return "\\n".join(advice) if advice else "姿态指标均在安全范围内"
#     except Exception as e:
#         print("计算运动建议时报错:{}".format(e))


def generate_kinematic_advice(latest_attack):
    """生成基于生物力学的击剑动作改进建议"""
    advice = []
    try:
        # --------- 上肢分析 ---------
        # 右手肘关节角度
        elbow_angle = float(latest_attack.get('右手夹角(°)', 0))
        if elbow_angle > 170:
            advice.append("右肘过伸(%.1f°)，剑尖控制力不足，建议前臂保持15-25°微屈" % elbow_angle)
        elif elbow_angle < 90:
            advice.append("右肘过屈(%.1f°)，影响攻击速度，适当增大肘关节角度至110-130°" % elbow_angle)

        # --------- 下肢分析 ---------
        # 右膝关节
        r_knee = float(latest_attack.get('右膝夹角(°)', 0)) 
        if r_knee < 110:
            advice.append("前腿弯曲不足(%.1f°)，建议攻击时保持120-140°以获得更大推进力" % r_knee)
        elif r_knee > 150:
            advice.append("前腿过度伸展(%.1f°)，可能损伤十字韧带，进攻后应立即恢复弯曲" % r_knee)
            
        # 左膝关节
        l_knee = float(latest_attack.get('左膝夹角(°)', 0))
        if l_knee > 175:
            advice.append("后腿过直(%.1f°)，限制身体移动，建议保持160-170°作为动力储备" % l_knee)

        # --------- 踝关节分析 ---------
        # 右脚踝角度
        r_ankle = float(latest_attack.get('右脚夹角(°)', 0)) 
        if r_ankle < 80:
            advice.append("前脚掌着地角度偏小(%.1f°)，足部稳定性不足，尝试增大到85-95°" % r_ankle)
            
        # 左脚踝角度
        l_ankle = float(latest_attack.get('左脚夹角(°)', 0))
        if l_ankle > 110:
            advice.append("后脚踝过度背屈(%.1f°)，可能影响蹬地发力，建议保持95-105°" % l_ankle)

        # --------- 躯干力学分析 ---------
        # 肩髋同步性（通过坐标计算）
        try:
            # 解析肩部和臀部坐标数据
            r_shoulder = tuple(map(int, latest_attack['右肩坐标'].strip("()").split(",")))
            l_shoulder = tuple(map(int, latest_attack['左肩坐标'].strip("()").split(",")))
            r_hip = tuple(map(int, latest_attack['右臀坐标'].strip("()").split(",")))
            
            # 计算躯干扭转角度
            shoulder_center = ((r_shoulder[0]+l_shoulder[0])//2, (r_shoulder[1]+l_shoulder[1])//2)
            hip_center = ((r_hip[0]+r_hip[0])//2, (r_hip[1]+r_hip[1])//2)  # 假设为单侧髋部数据
            torso_twist = abs(shoulder_center[0] - hip_center[0])
            
            if torso_twist > 50:
                advice.append("躯干扭转不足(差值%spx)，建议增加肩髋同步旋转幅度" % torso_twist)
            elif torso_twist < 20:
                advice.append("躯干过度扭转(差值%spx)，可能导致重心失衡" % torso_twist)
                
        except (KeyError, ValueError) as e:
            print("坐标解析异常:", str(e))

        # --------- 运动表现优化 ---------
        move_distance = float(latest_attack.get('移动距离(cm)', 0))
        if move_distance > 200:
            advice.append("单次攻击距离过长(%.1fcm)，注意体力分配合理性" % move_distance)
        elif move_distance < 80:
            advice.append("攻击距离不足(%.1fcm)，建议加强腿部爆发力训练" % move_distance)

        # 当所有指标正常时显示积极反馈
        return "\n".join(advice) if advice else "动作指标在理想范围内！保持当前技术动作"
        
    except Exception as e:
        print("运动建议生成异常:", traceback.format_exc())
        return "⚠️ 建议生成失败：数据分析异常"



def create_stats_box(total_attacks, total_distance, avg_distance, best_attack, chart_path, advice): 
    """创建统计数据板块"""
    stats_img = Image.new('RGB', (980, 200), '#f8f9fa')
    draw = ImageDraw.Draw(stats_img)
    y = 20
    
    # 标题
    draw.text((20, y), "动作统计", fill='#333', font=header_font)
    y += 60
    
    # 统计项布局
    cols = [
        ("攻击次数", f"{total_attacks}次"),
        ("总移动距离", f"{total_distance:.2f}cm"),
        ("平均攻击距离", f"{avg_distance:.2f}cm", "#28a745"),
        ("最佳攻击", f"{best_attack:.2f}cm", "#dc3545")
    ]
    
    # 分列绘制
    col_width = 230
    for i, (label, value, *color) in enumerate(cols):
        x = 20 + i*col_width
        draw.text((x, y), label, fill='#666', font=body_font)
        draw.text((x, y+40), value, fill=color[0] if color else '#333', font=body_font)
        
    return stats_img

def create_advice_box(advice_text):
    """创建运动建议模块"""
    box = Image.new('RGB', (980, 600), '#fff3cd')  # 使用警示黄背景
    draw = ImageDraw.Draw(box)
    y = 20
    
    # 标题
    draw.text((20, y), "运动建议", fill='#856404', font=header_font)
    y += 70
    
    # 文本自动换行
    paragraphs = advice_text.split('\n')
    for para in paragraphs:
        lines = textwrap.wrap(para, width=45)  # 每行最多45个汉字
        for line in lines:
            draw.text((40, y), f"• {line}", fill='#856404', font=body_font)
            y += 40
        y += 20  # 段落间距
    
    return box


def compose_report_image(**kwargs):
    # 参数处理
    params = {
        'total_attacks': int(kwargs.get('total_attacks', 0)),
        'total_distance': float(kwargs.get('total_distance', 0)),
        'avg_distance': float(kwargs.get('avg_distance', 0)),
        'best_attack': float(kwargs.get('best_attack', 0)),
        'chart_path': kwargs.get('chart_path', ''),
        'advice': kwargs.get('advice', '')
    }

    try:
        print("开始初始化画布")
        # 创建画布
        img = Image.new('RGB', (1080, 1920), '#ffffff')
        draw = ImageDraw.Draw(img)
        
        # 标题部分
        draw.text((50,50), "击剑训练分析报告", fill='#2c3e50', font=title_font)
        draw.line((50, 120, 1030, 120), fill='#3498db', width=3)
    except Exception as e:
        print("初始化画布失败:{}".format(e))
    
    try:
        # 插入统计模块
        print("开始插入统计模块")
        stats_box = create_stats_box(**params)
        img.paste(stats_box, (50, 150))
    except Exception as e:
        print("插入统计模块报错:{}".format(e))

    # 插入趋势图表
    try:
        chart_img = Image.open(params['chart_path']).resize((980,400))
        img.paste(chart_img, (50, 400))
    except Exception as e:
        draw.text((50,400), "图表生成失败", fill='red', font=header_font)
    
    # 运动建议模块
    advice_box = create_advice_box(params['advice'])
    img.paste(advice_box, (50, 850))
    
    # 页脚
    # draw.text((50,1860), "生成时间：2023-07-20 14:30", fill='#666', font=body_font)
    # 水印与边界线
    draw.text((200, 1860), f"纯态探索® 运动分析报告 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fill='#999999', font=body_font)
    chart_id = str(uuid.uuid4())
    report_save_path = os.path.join(os.getcwd(), 'temp_reports', f"aaaaa_{chart_id}.png")
    img.save(report_save_path, quality=95, optimize=True)
    print(f"✓ 报告已保存到：{os.path.abspath(report_save_path)}")
    return report_save_path


# 在datatreating.py中添加以下内容
# import tempfile
# import os

# import matplotlib

# matplotlib.use('Agg')  # 确保在非GUI环境下运行?
# from matplotlib import pyplot as plt

def save_chart(figure):
    try:
        # 生成唯一文件名
        chart_id = str(uuid.uuid4())
        # temp_dir = os.path.join(tempfile.gettempdir(), "fencing_charts")
        
        # os.makedirs(temp_dir, exist_ok=True)
        # print("创建目录:{}".format(temp_dir))
        # chart_path = os.path.join(temp_dir, f"{chart_id}.png")
        chart_path = os.path.join(os.getcwd(), 'temp_reports', f"fencing_charts_{chart_id}.png")
        print("save函数的生成路径：{}".format(chart_path))
        
        # 保存参数调整
        figure.savefig(
            chart_path,
            dpi=150,
            bbox_inches='tight',
            facecolor='white',  # 强制白底
            edgecolor='none',
            transparent=False
        )
        
        # 显式释放内存
        plt.close(figure) 
        return chart_path
    except Exception as e:
        print(f"保存图表异常：{str(e)}")
        raise

