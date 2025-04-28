# 导入Flask类和相关模块
import hashlib
import tempfile
from tkinter import Image
import traceback
import cv2
from flask import Flask, request, jsonify, send_file, redirect, url_for
from flask import send_from_directory, render_template, Response, g
import os
import csv
from matplotlib import pyplot as plt
import startcode
import pose_recognition as pr
import datatreating as dt
from flask_cors import CORS
from threading import Thread
import photo2video as p2v
from detect_withroi import detect_app
import logging
import threading
from filelock import FileLock

# 创建Flask应用实例
app = Flask(__name__)
CORS(app)# 允许跨域请求

# 请求锁
report_lock = threading.Lock()
# 文件锁
REPORT_LOCK_FILE = os.path.join(tempfile.gettempdir(), 'report_gen.lock')
# 日志输出
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)
app.logger.debug("App is starting...")  # 添加日志输出

# 注册 detect_withroi 中的路由
app.register_blueprint(detect_app, url_prefix='/detect')
app.logger.debug("detect_app Blueprint registered.")  # 添加日志输出

# 特定文件路径，用于存储最新的 CSV 文件路径
csv_path_file = 'latest_csv_path.txt'

# 创建一个文件夹invideo，用来存放用户上传的文件
app.config['INVIDEO_FOLDER'] = 'invideo'
if not os.path.exists(app.config['INVIDEO_FOLDER']):
    os.makedirs(app.config['INVIDEO_FOLDER'])

# 访问根目录时，调用index()函数
@app.route('/')
def index():
    """
    处理根目录（'/'）的请求，返回首页的HTML模板。
    
    :return: 渲染后的index.html页面。
    """
    app.logger.debug("Rendering index.html")
    return render_template('index.html')


@app.route('/data', methods=['GET'])
def data():
    """
    处理 /data 路由的 GET 请求，用于从前端获取数据。
    
    这个函数的主要步骤如下：
    1. 从后端的全局变量中获取姿态识别的数据。
    2. 将这些数据封装成一个字典。
    3. 返回包含这些数据的 JSON 响应。
    
    :return: JSON 响应，包含姿态识别的数据。
    """

    # 从后端的全局变量中获取姿态识别的数据
    data = {
        'attacks_count':     pr.attacks_count,  # 进攻次数
        'sum_distance':      pr.global_sum_distance,  # 进攻移动距离
        'right_hand_angle':  pr.global_right_hand_angle,  # 右手肘角度
        'right_ankle_angle': pr.global_right_ankle_angle,  # 右脚踝角度
        'left_ankle_angle':  pr.global_left_ankle_angle,  # 左脚踝角度
        'right_knee_angle':  pr.global_right_knee_angle,  # 右膝盖角度
        'left_knee_angle':   pr.global_left_knee_angle,  # 左膝盖角度
        'right_shoulder':    pr.global_right_shoulder,  # 右肩坐标
        'left_shoulder':     pr.global_left_shoulder,  # 左肩坐标
        'right_hip':         pr.global_right_hip,  #右臀坐标
        'left_hip':          pr.global_left_hip,  # 左臀坐标
        'img_base64':        pr.img_base64,  # 姿态识别图像的 base64 编码
        'standardaction':    pr.standardaction,  # 动作标准度
        'action':            pr.action  # 检测到的动作类型
    }
    return jsonify(data)
    

@app.route('/upload', methods=['POST'])
def upload_video():
    """
    处理 /upload 路由的 POST 请求，用于上传视频或照片。
    
    这个函数的主要步骤如下：
    1. 从请求中获取视频文件、人物位置、身高、按钮类型和文件类型。
    2. 检查是否上传了文件。
    3. 根据按钮类型检查文件类型是否正确。
    4. 保存上传的文件到指定目录。
    5. 返回包含文件路径、人物位置、身高和按钮类型的JSON响应。
    
    :return: JSON响应或错误信息。
    """

    # 从请求中获取视频文件
    video = request.files.get('video')

    # 从请求表单中获取人物位置、身高、按钮类型和文件类型
    side = request.form.get('side', type=int)
    height = request.form.get('height', type=int)
    button_type = request.form.get('button_type')
    file_type = request.form.get('file_type')

     # 检查是否上传了文件
    if not video:
        return jsonify({'error': 'No video file uploaded'}), 400
    
     # 检查按钮类型和文件类型是否匹配
    if button_type == 'competition' and not file_type.startswith('video/'):
        return jsonify({'error': '比赛分析只能上传视频文件'}), 400

    if button_type == 'analysis' and not file_type.startswith('image/'):
        return jsonify({'error': '动作分析只能上传照片文件'}), 400

    # 构建保存文件的路径
    video_path = os.path.join(app.config['INVIDEO_FOLDER'], video.filename.strip())
    print(f"++++++++++{video_path}")
    video_path = video_path.replace("\\","/")
    print(f"++++++++++{video_path}")
    # 保存文件
    video.save(video_path)
     # 获取文件的绝对路径
    # absolute_video_path = os.path.abspath(video_path)
    absolute_video_path = video_path

    # 返回包含文件路径、人物位置、身高和按钮类型的JSON响应
    return jsonify({
        'video_path': absolute_video_path,
        'side': side,
        'height': height,
        'button_type': button_type
    })


@app.route('/competition')
def competition():
    """
    处理 /competition 路由的请求，用于比赛分析页面。
    
    这个函数的主要步骤如下：
    1. 从请求参数中获取视频路径、人物位置和身高。
    2. 检查视频路径是否有效。
    3. 启动一个新线程来处理姿态识别任务。
    4. 返回 `competition.html` 页面。
    
    :return: 渲染后的 `competition.html` 页面或错误信息。
    """
    try:
        # 从请求参数中获取视频路径、人物位置和身高
        video_path = request.args.get('video_path')
        side = int(request.args.get('side'))
        height = int(request.args.get('height'))
        print(f"-----------------{video_path}")

        # 检查视频路径是否有效
        if not video_path or not os.path.exists(video_path):
            return "Invalid video path", 400
        
        # 启动一个新线程来处理姿态识别任务
        thread = Thread(target=startcode.startcode, args=(video_path, side, height))
        thread.start()

        # 返回渲染后的 competition.html 页面
        return render_template('competition.html')
    except Exception as e:
        # 记录错误信息
        app.logger.error(f"Error in competition route: {e}")
        # 返回内部服务器错误响应
        return "Internal Server Error", 500

# 跳转analysis路由
@app.route('/analysis')
def analysis():
    """
    处理 /analysis 路由的请求，用于动作分析页面。
    
    这个函数的主要步骤如下：
    1. 从请求参数中获取视频路径、人物位置和身高。
    2. 调用 `photo_to_video` 函数将上传的照片转换为视频。
    3. 启动一个新线程来处理姿态识别任务。
    4. 返回 `analysis.html` 页面。
    
    :return: 渲染后的 `analysis.html` 页面或错误信息。
    """
    try:
        # 从请求参数中获取视频路径、人物位置和身高
        video_path = request.args.get('video_path')
        side = int(request.args.get('side'))
        height = int(request.args.get('height'))

        # 打印视频路径，用于调试
        print(video_path)

        # 设置输出视频的文件夹路径
        output_videoPath = 'invideo'
        # 调用 photo_to_video 函数将上传的照片转换为视频
        video_path = p2v.photo_to_video(video_path, output_videoPath)

        # 启动一个新线程来处理姿态识别任务
        thread = Thread(target=startcode.startcode_two, args=(video_path, side, height))
        thread.start()

        # 返回渲染后的 analysis.html 页面
        return render_template('analysis.html')
    except Exception as e:
        # 记录错误信息
        app.logger.error(f"Error in competition route: {e}")
        # 返回内部服务器错误响应
        return "Internal Server Error", 500


@app.route('/download_report')
def download_report():
    """
    处理/download_report路由的请求，用于下载攻击数据的CSV文件。
    
    :return: CSV文件的下载响应。
    """
    if os.path.exists(csv_path_file):
        with open(csv_path_file, 'r') as f:
            csv_attack_path = f.read().strip()  # 从特定文件中读取最新的 CSV 文件路径
        print(f"请求下载的CSV文件路径: {csv_attack_path}")
        if csv_attack_path and os.path.exists(csv_attack_path):
            return send_from_directory(os.path.dirname(csv_attack_path), os.path.basename(csv_attack_path), as_attachment=True)
    return "CSV文件路径未找到", 404

@app.route('/score')
def score():
    """
    处理 /score 路由的请求，用于显示得分页面。
    
    :return: 渲染后的 score.html 页面。
    """
    app.logger.debug("Rendering score.html")
    return render_template('score.html')

@app.route('/latest_csv_path')
def latest_csv_path():
    with open('latest_csv_path.txt', 'r') as f:
        csv_path = f.read().strip()
    return jsonify({'csv_path': csv_path})



@app.route('/generate_action_report')
def generate_action_report():
    
    # with report_lock:  # 添加互斥锁
    with FileLock(REPORT_LOCK_FILE, timeout=60):  # 最长等待1分钟
        # 添加请求指纹验证
        client_hash = hashlib.md5(f"{request.remote_addr}{request.headers.get('User-Agent')}".encode()).hexdigest()
        if hasattr(g, 'report_fingerprint') and g.report_fingerprint == client_hash:
            return jsonify({"status": "already_processing"}), 429
        g.report_fingerprint = client_hash
        try:
            app.logger.info("开始生成报告流程")
            # print("开始生成报告")

            # 1. 路径校验
            if not os.path.exists('latest_csv_path.txt'):
                app.logger.error("关键文件latest_csv_path.txt缺失")
                return jsonify({"error": "分析数据未就绪"}), 404

            with open('latest_csv_path.txt') as f:
                csv_path = f.read().strip()
                app.logger.debug(f"读取到CSV路径: {csv_path}")

            # 2. CSV文件存在性检查
            if not os.path.exists(csv_path):
                app.logger.error(f"CSV文件不存在: {csv_path}")
                return jsonify({"error": "分析数据文件丢失"}), 404

            # 3. 数据解析
            result = dt.analyze_csv_data(csv_path)
            if not result:
                app.logger.error("CSV解析失败")
                return jsonify({"error": "数据解析异常"}), 500

            # 4. 图片生成过程监控
            app.logger.debug("开始生成可视化报告图片")
            
            # 添加生成路径验证
            output_dir = os.path.join(os.getcwd(), 'invideo')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                app.logger.info(f"创建输出目录: {output_dir}")

            # 强制释放可能存在的PIL残留资源
            Image.MAX_IMAGE_PIXELS = None
            if hasattr(Image, '_decompression_bomb_check'):
                Image._decompression_bomb_check = lambda x: None
                
            image_path = dt.generate_action_image(result)
            full_image_path = os.path.join(output_dir, image_path)

            # 验证文件生成状态
            app.logger.debug(f"图像生成路径: {image_path}")
            app.logger.debug(f"文件存在性检查: {os.path.exists(full_image_path)}")
            if os.path.exists(full_image_path):
                app.logger.debug(f"文件大小: {os.path.getsize(full_image_path)} bytes")
            else:
                app.logger.error("生成的图片文件未找到！")

            return send_file(full_image_path, mimetype='image/png')

        except Exception as e:
            app.logger.error(f"报告生成严重错误: {traceback.format_exc()}")
            
            # 强制清理资源
            cv2.destroyAllWindows()
            plt.close('all')
            if 'image_path' in locals():
                try: os.remove(full_image_path)
                except: pass
                
            return jsonify({
                "error": "系统处理异常",
                "advice": [
                    "1. 确认分析文件格式正确",
                    "2. 检查磁盘剩余空间(需>100MB)",
                    "3. 重启分析服务"
                ]
            }), 500


@app.route('/report')
def report():
    """
    处理 /report 路由的请求，用于显示动作分析报告。
    
    :return: 渲染后的 report.html 页面。
    """
    return render_template('report.html')


@app.route('/system_status')
def system_status():
    """系统资源状态报告"""
    import psutil # type: ignore
    status = {
        'cpu': psutil.cpu_percent(),
        'memory': psutil.virtual_memory().percent,
        'gpu': 0
    }
    
    try:
        from GPUtil import getGPUs # type: ignore
        gpu = getGPUs()[0]
        status['gpu'] = gpu.load * 100
    except:
        pass
    
    return jsonify(status)


@app.route('/generate_competition_report')
def generate_competition_report():
    print("开始生成报告")
    with FileLock(REPORT_LOCK_FILE, timeout=60):
        try:
            # 从CSV读取比赛数据
            # 1. 路径校验
            if not os.path.exists('latest_csv_path.txt'):
                app.logger.error("关键文件latest_csv_path.txt缺失")
                return jsonify({"error": "分析数据未就绪"}), 404

            with open('latest_csv_path.txt') as f:
                csv_path = f.read().strip()
                app.logger.debug(f"读取到CSV路径: {csv_path}")

            # 2. CSV文件存在性检查
            if not os.path.exists(csv_path):
                print(f"CSV文件不存在: {csv_path}")
                app.logger.error(f"CSV文件不存在: {csv_path}")
                return jsonify({"error": "分析数据文件丢失"}), 404

            # 3. 数据解析
            result = dt.analyze_csv_data(csv_path)
            if not result:
                app.logger.error("CSV解析失败")
                return jsonify({"error": "数据解析异常"}), 500
            
            print("开始获取比赛数据")

            try:
                # 读取csv文件
                attacks = list(csv.DictReader(open(csv_path, encoding="utf-8")))
            except Exception as e:
                print("读取csv文件出错:{}".format(e))

            # 生成统计数据
            total_attacks = len(attacks)
            distances = [float(a['移动距离(cm)']) for a in attacks]
            total_distance = sum(distances)
            avg_distance = total_distance/total_attacks 
            best_attack = max(distances)

            # 生成图表
            print("开始生成趋势图")
            fig = dt.create_trend_chart(distances)
            print("保存趋势图")
            chart_path = dt.save_chart(fig)

            
            # 生成姿态建议
            try:
                # print(attacks)
                attack_count = -1
                for i in range(0, len(attacks)):
                    print(attacks[i])
                    if float(attacks[i]['右脚夹角(°)']) > 30 and float(attacks[i]['右膝夹角(°)']) > 50:
                        attack_count += 1
                        break
                if attack_count == 0:
                    attack_count = -1

                advice = dt.generate_kinematic_advice(attacks[attack_count])
            except Exception as e:
                print("生成运动建议失败：{}".format(e))

            
            # 合成报告图片
            print("开始合成报告图片")
            report_img = dt.compose_report_image(
                total_attacks=total_attacks,
                total_distance=total_distance,
                avg_distance=avg_distance,
                best_attack=best_attack,
                chart_path=chart_path,
                advice=advice
            )
            print("报告图片绘制结束")
            # 在返回前验证图表文件存在性
            if not os.path.exists(chart_path):
                print(f"图表文件不存在: {chart_path}")
                raise FileNotFoundError(f"图表文件不存在: {chart_path}")
            try:
                # 打印文件信息验证
                print(f"图表文件状态: {os.stat(chart_path)}")
                app.logger.debug(f"图表文件状态: {os.stat(chart_path)}")
                return send_file(report_img, mimetype='image/png')
            except Exception as e:
                print("渲染报告时出错！！")
                print("错误信息:{}".format(e))

        except Exception as e:
            app.logger.error(f"视频报告生成失败: {traceback.format_exc()}")
            return jsonify(error="报告生成失败"), 500


@app.route('/competition_report')
def competition_report():
    # return render_template('competition_report.html')
    # 确保模板文件路径正确
    template_path = os.path.join(app.template_folder, 'competition_report.html') 
    if not os.path.exists(template_path):
        app.logger.error(f"模板文件不存在: {template_path}")
    return send_file(template_path)  # 开发阶段直接返回文件验证



if __name__ == '__main__':
    host_address = '127.0.0.1'  # 监听本地请求
    # host_address = '0.0.0.0'   # 监听全部ip
    host_port = 5000
    print(f"访问本地服务器地址: http://{host_address}:{host_port}")
    app.logger.debug(f"Starting app on http://{host_address}:{host_port}")
    # 运行app应用
    app.run(debug=True, host=host_address, port=host_port)

