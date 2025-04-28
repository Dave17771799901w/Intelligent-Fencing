from flask import send_from_directory
import numpy as np
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
import pandas as pd

# 特定文件路径，用于存储最新的 CSV 文件路径
csv_path_file = 'latest_csv_path.txt'
def generate_report():
    try:
        # 读取最新的CSV文件路径
        with open(csv_path_file, 'r') as f:
            csv_path = f.read().strip()
        
        # 读取CSV数据
        df = pd.read_csv(csv_path)
        print(df)

        fi = open("invideo/data.txt",'w')
        fi.write(str(df))
        fi.close()
        # 生成PDF
        pdf_path = os.path.join('invideo', 'action_report.pdf')
        print(pdf_path)
        generate_analysis_pdf(df, pdf_path)
        
        return send_from_directory('invideo','action_report.pdf', as_attachment=True)
    except Exception as e:
        return str(e), 500

def generate_analysis_pdf(dataframe, filename):
    # 创建PDF文档
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # 自定义样式
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=1,  # 居中
        spaceAfter=20
    )
    
    # 内容元素列表
    elements = []
    
    # 添加标题
    elements.append(Paragraph("击剑动作分析报告", title_style))
    
    # 添加基本数据表格
    summary_data = [
        ["进攻次数", dataframe['攻击次数'].max()],
        ["总移动距离(cm)", f"{dataframe['移动距离(cm)'].sum():.1f}"],
        ["平均右手角度", f"{dataframe['右手夹角(°)'].mean():.1f}°"],
        ["最大右膝角度", f"{dataframe['右膝夹角(°)'].max():.1f}°"]
    ]
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    # 添加详细分析
    analysis = []
    
    # 右手角度分析
    right_hand_mean = dataframe['右手夹角(°)'].mean()
    if right_hand_mean < 160:
        analysis.append((
            "右手肘角度不足",
            f"平均角度 {right_hand_mean:.1f}° (建议保持160°以上)",
            "建议：保持大臂与小臂的夹角，确保攻击时的力量传递"
        ))
    
    # 膝盖角度分析
    right_knee_min = dataframe['右膝夹角(°)'].min()
    if right_knee_min < 90:
        analysis.append((
            "右膝弯曲不足",
            f"最小角度 {right_knee_min:.1f}° (建议保持90°以上)",
            "建议：降低重心，保持合理的弓步姿势"
        ))
    
    # 构建分析表格
    analysis_data = [["问题点", "数据表现", "改进建议"]]
    for item in analysis:
        analysis_data.append(item)
    
    analysis_table = Table(analysis_data, colWidths=[120, 140, 200])
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'TOP')
    ]))
    elements.append(Paragraph("动作问题分析", styles['Heading2']))
    elements.append(analysis_table)
    elements.append(Spacer(1, 20))
    
    # # 生成趋势图示例（需要matplotlib）
    # plot_path = os.path.join(app.config['INVIDEO_FOLDER'], 'trend.png')
    # dataframe['右手夹角(°)'].plot(title='右手角度变化趋势').get_figure().savefig(plot_path)
    # elements.append(Image(plot_path, width=400, height=300))
    
    # 生成PDF
    doc.build(elements)


def analyze_en_garde(df):
    """ 实战姿势专项分析 """
    analysis = []
    
    # 1. 身体前倾角度分析
    shoulder_hip_angle = calculate_torso_angle(df['右肩坐标'], df['右臀坐标'])
    if shoulder_hip_angle < 15:
        analysis.append((
            "身体直立过度",
            f"躯干前倾角 {shoulder_hip_angle:.1f}° (建议15-25°)",
            "建议：从髋部开始前倾，保持鼻尖在前膝正上方"
        ))
    elif shoulder_hip_angle > 25:
        analysis.append((
            "身体前倾过大",
            f"躯干前倾角 {shoulder_hip_angle:.1f}° (建议15-25°)",
            "建议：收紧核心肌群，保持脊柱中立位"
        ))

    # 2. 膝关节角度分析
    front_knee = df['右膝夹角(°)'].mean()  # 假设右腿为前腿
    rear_knee = df['左膝夹角(°)'].mean()
    
    if not (110 <= front_knee <= 130):
        analysis.append((
            "前膝角度异常",
            f"前膝 {front_knee:.1f}° (标准110-130°)",
            "调整站距：过小则加大前后脚距离，过大则收窄"
        ))
    
    if not (90 <= rear_knee <= 110):
        analysis.append((
            "后膝角度异常", 
            f"后膝 {rear_knee:.1f}° (标准90-110°)",
            "建议：通过深蹲训练加强腿部力量"
        ))

    # 3. 重心分布分析
    weight_distribution = calculate_weight_distribution(
        df['右踝坐标'], df['左踝坐标']
    )
    if weight_distribution < 40:
        analysis.append((
            "重心偏后",
            f"前脚承重 {weight_distribution}% (建议40-50%)",
            "练习：前后脚踩磅秤，感受重心分配"
        ))
    elif weight_distribution > 60:
        analysis.append((
            "重心前移过度",
            f"前脚承重 {weight_distribution}% (建议40-50%)",
            "注意：易导致启动速度下降，加强后腿蹬地训练"
        ))

    return analysis

def analyze_lunge(df):
    """ 攻击姿势专项分析 """
    analysis = []
    
    # 1. 弓步深度分析
    lunge_depth = calculate_lunge_depth(
        df['右髋坐标'], df['右膝坐标']  # 假设右腿为前腿
    )
    if lunge_depth < 0.8:
        analysis.append((
            "弓步不足",
            f"深度指数 {lunge_depth:.2f} (标准0.8-1.2)",
            "加强股四头肌力量训练，提高蹬伸幅度"
        ))
    
    # 2. 手臂同步性分析
    arm_extension = df['右手夹角(°)'].max()
    if arm_extension < 175:
        analysis.append((
            "手臂未完全伸展",
            f"最大角度 {arm_extension:.1f}° (应达175°+)",
            "重点练习：原地刺靶训练，强调手臂先导"
        ))
    return analysis

# 关键计算函数
def calculate_torso_angle(shoulder, hip):
    """ 通过肩髋坐标计算躯干前倾角 """
    dx = np.mean([s[0]-h[0] for s,h in zip(shoulder, hip)])
    dy = np.mean([s[1]-h[1] for s,h in zip(shoulder, hip)])
    return np.degrees(np.arctan2(dy, dx))

def calculate_weight_distribution(front_ankle, rear_ankle):
    """ 通过踝关节压力中心计算重心分布 """
    front_pressure = np.mean([p[0] for p in front_ankle])
    rear_pressure = np.mean([p[0] for p in rear_ankle])
    return front_pressure / (front_pressure + rear_pressure) * 100

def calculate_lunge_depth(hip, knee):
    """ 计算弓步深度指数（髋-膝水平距离/垂直距离） """
    horizontal = np.mean([h[0]-k[0] for h,k in zip(hip, knee)])
    vertical = np.mean([h[1]-k[1] for h,k in zip(hip, knee)])
    return abs(horizontal) / vertical