1. 概述
击剑是一项对动作准确性要求极高的运动。传统的动作识别依赖于教练和裁判的肉眼观察，存在主观性强、易出错的问题。我们将利用计算机视觉系统开发一款帮助击剑训练的小程序，专注于提升击剑训练效果。
![image](https://github.com/user-attachments/assets/94d7ae27-18d4-4a4d-8249-9cade65758ae)

2. 功能展示
Ⅰ实时动作评估
当运动员做到标准训练动作，进行计数；当运动员训练动作不标准，就进行语音提醒； 
对运动员的动作，进行实时的3D骨架绘制，方便实施观测关键点的运动轨迹。

Ⅱ训练报告生成
人体关键点数据统计
动作标准度评估
训练完成度分析

Ⅲ赛前分析
系统提取击剑选手在比赛中的人体姿态数据。
系统从前脚移动距离、重心移动比例、加速度分布以及前后腿与地面夹角四个维度提取数据
基于核心动作要点进行指导评价，提供针对性的动作纠正方法与训练方案
系统为用户制定专门应对的数据驱动战术。

Ⅳ赛后复盘
得到比赛中指定击剑动作得分率
识别和追踪比赛回放中运动员指定的进攻动作
计算该动作在比赛中的有效得分率
提供社区分享交流平台，相互分享经验、交流心得
![image](https://github.com/user-attachments/assets/c6195d77-80d8-47fa-b199-edeb5126d8d8)

MediaPipe算法框架
※ 实时识别和追踪运动员的33个人体关键点
※ 通过深度学习算法，确保在高速动作中的检测稳定性
※ 支持将2D图像转换为3D坐标系，实现立体化动作分析

数据处理
※ 自动提取和记录人体关键点坐标信息
※ 对不同场景下的坐标数据进行标准化处理
※ 将复杂的数据转化为直观的可视化图表

技术速度分析
※ 测量前脚移动距离和频率
※ 分析重心转移规律
※ 统计加速度分布特征
※研究前后腿配合模式

战术分析
※ 提取标志性动作特征
※ 识别技术弱点
※ 总结战术风格，制定应对策略


