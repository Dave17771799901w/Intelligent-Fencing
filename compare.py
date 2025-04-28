import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cosine
import cv2
# 定义身体部位的权重
KEYPOINT_WEIGHTS = {
    # 面部关键点 (0-10)
    **{i: 0.0 for i in range(11)},  # 完全忽略面部
    
    # 躯干关键点 (11-16)
    11: 1,  # 左肩
    12: 2,  # 右肩
    13: 1,  # 左肘
    14: 2,  # 右肘
    15: 1,  # 左腕
    16: 2,  # 右腕
    
    # 下半身关键点 (23-32)
    23: 0.5,  # 左髋
    24: 2,  # 右髋
    25: 0.5,  # 左膝
    26: 2,  # 右膝
    27: 0.5,  # 左踝
    28: 2,  # 右踝
    29: 0.5,  # 左脚
    30: 0.5,  # 右脚
    31: 0.5,  # 左脚跟
    32: 0.5   # 右脚跟
}

def load_pose_data(file_path):
    """加载姿势数据"""
    try:
        df = pd.read_csv(file_path, header=0, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, header=0, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, header=0, encoding='gb2312')
    
    points = list(zip(df['X'].values, df['Y'].values))
    return np.array(points)

def normalize_pose(points):
    """将姿势旋转归一化"""
    left_shoulder = points[11]
    right_shoulder = points[12]
    
    shoulder_vector = right_shoulder - left_shoulder
    angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])
    
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    
    center = (left_shoulder + right_shoulder) / 2
    normalized_points = np.array([(rotation_matrix @ (p - center)) + center for p in points])
    
    return normalized_points

def calculate_pose_similarity(standard_points, test_points):
    """计算姿势相似度，考虑关键点权重"""
    # print(f"标准{len(standard_points)}")
    # print(f"当前{len(test_points)}")
    if len(standard_points) != len(test_points):
        raise ValueError("关键点数量不匹配")
    
    standard_normalized = normalize_pose(standard_points)
    test_normalized = normalize_pose(test_points)
    
    # 计算加权欧氏距离
    weighted_euclidean_distances = []
    total_weight = 0
    
    for i, (p1, p2) in enumerate(zip(standard_normalized, test_normalized)):
        weight = KEYPOINT_WEIGHTS.get(i, 0.0)
        if weight > 0:
            dist = euclidean(p1, p2) * weight
            weighted_euclidean_distances.append(dist)
            total_weight += weight
    
    # 归一化加权欧氏距离分数
    if total_weight > 0:
        euclidean_score = 1 - np.sum(weighted_euclidean_distances) / (total_weight * np.sqrt(2))
    else:
        euclidean_score = 0
    
    # 计算加权角度相似度
    angle_similarities = []
    angle_weights = []
    
    # 只考虑有权重的关键点之间的角度
    weighted_indices = [i for i in range(len(standard_normalized)) if KEYPOINT_WEIGHTS.get(i, 0.0) > 0]
    
    for i in weighted_indices:
        for j in weighted_indices[weighted_indices.index(i)+1:]:
            weight = (KEYPOINT_WEIGHTS[i] + KEYPOINT_WEIGHTS[j]) / 2
            
            vec1_std = standard_normalized[j] - standard_normalized[i]
            vec1_test = test_normalized[j] - test_normalized[i]
            
            # 避免零向量
            if np.all(vec1_std == 0) or np.all(vec1_test == 0):
                continue
                
            cos_angle = np.dot(vec1_std, vec1_test) / (np.linalg.norm(vec1_std) * np.linalg.norm(vec1_test))
            # 处理数值误差
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            angle_similarities.append((cos_angle + 1) / 2 * weight)
            angle_weights.append(weight)
    
    if angle_weights:
        angle_score = np.sum(angle_similarities) / np.sum(angle_weights)
    else:
        angle_score = 0
    
    # 组合分数，给予角度相似度更高的权重
    final_score = 0.6 * euclidean_score + 0.4 * angle_score
    # print(euclidean_score,angle_score)
    
    return final_score

def compare_poses(standard_csv, test_points):
    
     
    try:
        standard_points = load_pose_data(standard_csv)
        # print(standard_points)
        # print(test_points)
        # print(type(standard_points))
        # test_points = load_pose_data(test_csv)
        
        similarity = calculate_pose_similarity(standard_points, test_points)
        
        return {
            'similarity_score': similarity,
            'status': 'success',
            'message': f'相似度得分: {similarity:.4f}'
        }
    except Exception as e:
        return {
            'similarity_score': 0,
            'status': 'error',
            'message': str(e)
        }

def visualize_poses(standard_points, test_points, normalized=False):
    """可视化标准姿势和测试姿势的对比"""
    import matplotlib.pyplot as plt
    
    if normalized:
        standard_points = normalize_pose(standard_points)
        test_points = normalize_pose(test_points)
    
    # 假设图像的高度是标准点的最大y值和最小y值的差
    max_y = max(max(standard_points[:, 1]), max(test_points[:, 1]))
    min_y = min(min(standard_points[:, 1]), min(test_points[:, 1]))
    image_height = max_y - min_y

    plt.figure(figsize=(12, 6))
    
    # 绘制标准姿势
    plt.subplot(121)
    weighted_points = [i for i in range(len(standard_points)) if KEYPOINT_WEIGHTS.get(i, 0.0) > 0]
    for i in range(len(standard_points)):
        if i in weighted_points:
            plt.scatter(standard_points[i, 0], image_height - standard_points[i, 1], c='blue', s=100*KEYPOINT_WEIGHTS[i])
        else:
            plt.scatter(standard_points[i, 0], image_height - standard_points[i, 1], c='lightgray', s=20)
    plt.title('Standard Pose')
    plt.axis('equal')
    
    # 绘制测试姿势
    plt.subplot(122)
    for i in range(len(test_points)):
        if i in weighted_points:
            plt.scatter(test_points[i, 0], image_height - test_points[i, 1], c='red', s=100*KEYPOINT_WEIGHTS[i])
        else:
            plt.scatter(test_points[i, 0], image_height - test_points[i, 1], c='lightgray', s=20)
    plt.title('Test Pose')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()


# 判断当前帧与标准数据的相似度
def set_compare(face_state, test_points):
    """
    判断测试数据与标准数据的相似度
    :param face: 人物在左边还是右边  left right
    :param test_points: 待测数据
    :return: 
    """
    # 判断状态
    # a = 0
    # if face == 'left':
    #     # 标准动作路径
    #     standard_left_ready_csv = 'standard_action_csv\\leftReady.csv'   # 标准左弓步
    #     standard_left_attack_csv = 'standard_action_csv\\leftAttack.csv'       # 标准左攻击
    #     # 对比相似度
    #     left_ready_result = compare_poses(standard_left_ready_csv, test_points)
    #     left_attack_result = compare_poses(standard_left_attack_csv,test_points)

    #     # print(f"准备{left_ready_result['message']}")
    #     # print(f"攻击{left_attack_result['message']}")

    #     # 只输出标准数据
    #     if left_ready_result['similarity_score'] > 0.94:
    #         print("标准的弓步！")
    #         print(f"弓步{left_ready_result['message']}")
    #         a += 1

    #     if left_attack_result['similarity_score'] > 0.94:
    #         print("标准的攻击动作！")
    #         print(f"攻击{left_attack_result['message']}")
    #         a += 1

    # else:
    #     # 标准数据路径
    #     standard_right_ready_csv = 'standard_action_csv\\rightReady.csv'  # 标准右弓步
    #     standard_right_attack_csv = 'standard_action_csv\\rightAttack.csv'  # 标准右攻击
    #     # 比较准备动作相似度
    #     right_ready_result = compare_poses(standard_right_ready_csv, test_points)
    #     # 比较攻击动作相似度
    #     right_attack_result = compare_poses(standard_right_attack_csv, test_points)

    #     # print(f"准备{right_ready_result['message']}")
    #     # print(f"攻击{right_attack_result['message']}")

    #     # 只输出标准的数据
    #     if right_ready_result['similarity_score'] > 0.94:
    #         print("标准的弓步！")
    #         print(f"弓步{right_ready_result['message']}")
    #         a += 1

    #     if right_attack_result['similarity_score'] > 0.94:
    #         print("标准的攻击动作！")
    #         print(f"攻击{right_attack_result['message']}")
    #         a += 1
    # return True if a != 0 else False

    ready_state = 0
    attack_state = 0
    if face_state == 1:
        standard_ready_csv = 'standard_action_csv/LLR.csv'  # 左.左准备
        standard_attack_csv = 'standard_action_csv/LLA.csv' # 左.左攻击
    elif face_state == 2:
        standard_ready_csv = 'standard_action_csv/LRR.csv'  # 左.右准备
        standard_attack_csv = 'standard_action_csv/LRA.csv' # 左.右攻击
    elif face_state == 3:
        standard_ready_csv = 'standard_action_csv/RLR.csv'  # 右.左准备
        standard_attack_csv = 'standard_action_csv/RLA.csv' # 右.左攻击
    elif face_state == 4:
        standard_ready_csv = 'standard_action_csv/RRR.csv'  # 右.右准备
        standard_attack_csv = 'standard_action_csv/RRA.csv' # 右.有攻击
    else: 
        return "error: error to face_state"
    
    # 计算相似度
    ready_result = compare_poses(standard_ready_csv, test_points)  # 对比准备动作
    attack_result = compare_poses(standard_attack_csv, test_points) # 对比攻击动作
    # readyNum = ready_result['similarity_score']
    # attackNum = attack_result['similarity_score']
    readyNum = f"{ready_result['similarity_score']:.5f}"
    attackNum = f"{attack_result['similarity_score']:.5f}"

    if ready_result['similarity_score'] > 0.94:
        ready_state = 1
    if attack_result['similarity_score'] > 0.94:
        attack_state = 1

    if (ready_state + attack_state) > 0:
        if (ready_state + attack_state) == 2:
            return [3,readyNum,attackNum]
        else: 
            return ([1,readyNum,attackNum] if ready_state > attack_state else [2,readyNum,attackNum])
    return [0,readyNum,attackNum]
    # print(readyNum)
    # print(attackNum)
    # return 0
        



# 判断关键帧
# 光流
def optical_flow_keyframes(prev_frame, current_frame, threshold):
    flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    if np.median(mag) > threshold:  # 如果运动的中值大于阈值，则认为当前帧是关键帧
        return True
    return False


    

def main():
    # 标准数据
    standard_csv = 'standard_action_csv/LeftAttack.csv'
    # 待测数据
    test_csv = 'standard_action_csv\LeftAttack (2).csv'

    # 数据对比
    # standard_points = load_pose_data(standard_csv)
    # test_points = load_pose_data(test_csv)
    # visualize_poses(standard_points, test_points, normalized=False)
    # visualize_poses(standard_points, test_points, normalized=True)
    

    result = compare_poses(standard_csv, test_csv)
    print(result['message'])

if __name__ == '__main__':
    main()