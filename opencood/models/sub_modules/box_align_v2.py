# -*- coding: utf-8 -*-
"""
==============================================================================
              检测框对齐模块 (Box Alignment V2) - FreeAlign 核心算法
==============================================================================

【模块功能】
    本模块实现了 FreeAlign 的核心算法: 基于检测框匹配的位姿校正。
    通过在不同智能体的检测结果之间寻找匹配关系，构建位姿图并优化，
    从而在没有外部定位设备的情况下恢复相对位姿。

【核心算法】
    1. 检测框聚类: 将不同智能体观测到的同一目标的检测框聚类
    2. 位姿图构建: 将智能体和目标都作为节点，观测关系作为边
    3. 图优化: 使用 g2o 库优化位姿图，最小化重投影误差
    4. 位姿恢复: 从优化结果中提取校正后的位姿

【算法流程图】
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     Box Alignment 算法流程                               │
    │                                                                         │
    │  输入: 各 CAV 的检测结果 (检测框 + 不确定性)                              │
    │    │                                                                    │
    │    ▼                                                                    │
    │  ┌─────────────────┐                                                    │
    │  │ 1. 坐标变换      │  将检测框变换到世界坐标系                           │
    │  │    (可选)        │                                                    │
    │  └─────────────────┘                                                    │
    │    │                                                                    │
    │    ▼                                                                    │
    │  ┌─────────────────┐                                                    │
    │  │ 2. 计算距离矩阵  │  计算所有检测框中心之间的 L2 距离                   │
    │  └─────────────────┘                                                    │
    │    │                                                                    │
    │    ▼                                                                    │
    │  ┌─────────────────┐                                                    │
    │  │ 3. 检测框聚类    │  距离 < 阈值的框聚为一类 (同一目标)                  │
    │  │    (Cluster)     │                                                    │
    │  └─────────────────┘                                                    │
    │    │                                                                    │
    │    ▼                                                                    │
    │  ┌─────────────────┐                                                    │
    │  │ 4. 构建位姿图    │  节点: CAV + Landmark (聚类中心)                    │
    │  │    (Pose Graph)  │  边: 观测关系                                      │
    │  └─────────────────┘                                                    │
    │    │                                                                    │
    │    ▼                                                                    │
    │  ┌─────────────────┐                                                    │
    │  │ 5. 图优化        │  使用 g2o 优化，最小化重投影误差                    │
    │  │    (g2o)         │                                                    │
    │  └─────────────────┘                                                    │
    │    │                                                                    │
    │    ▼                                                                    │
    │  输出: 校正后的位姿 [x, y, yaw]                                          │
    └─────────────────────────────────────────────────────────────────────────┘

【位姿图结构】
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  位姿图 (Pose Graph) 结构                                               │
    │                                                                         │
    │  节点类型:                                                              │
    │  ├── Agent 节点 (ID: 0 ~ N-1): CAV 位姿，SE(2) 表示                     │
    │  │   - Agent 0 (Ego): 固定，作为参考坐标系                              │
    │  │   - Agent 1~N-1: 待优化位姿                                          │
    │  │                                                                      │
    │  └── Landmark 节点 (ID: N ~ N+M-1): 目标位置，SE(2) 或 R² 表示          │
    │      - 每个 Landmark 对应一个聚类 (同一目标的多观测)                     │
    │                                                                         │
    │  边类型:                                                                │
    │  └── Agent-Landmark 边: 观测关系                                        │
    │      - 信息矩阵: 由检测不确定性决定                                      │
    │      - 观测值: 检测框在 Agent 坐标系中的位置                             │
    │                                                                         │
    │  图示:                                                                  │
    │                                                                         │
    │      Agent0 (Ego, 固定)                                                 │
    │         /    \                                                          │
    │        /      \                                                         │
    │    Landmark1  Landmark2                                                 │
    │      /   \       |                                                      │
    │     /     \      |                                                      │
    │  Agent1  Agent2  Agent1                                                 │
    │                                                                         │
    │  说明: Agent1 和 Agent2 都观测到 Landmark1，形成约束                    │
    └─────────────────────────────────────────────────────────────────────────┘

【关键概念】
    1. SE(2): 二维特殊欧几里得群，表示平面上的刚体变换 (x, y, yaw)
    2. Landmark: 地标，这里指被多个 CAV 观测到的目标
    3. Information Matrix: 信息矩阵，测量不确定性的逆
    4. g2o: General Graph Optimization，通用图优化库

【使用方法】
    # 单样本位姿校正
    from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
    
    refined_pose = box_alignment_relative_sample_np(
        pred_corners_list,      # 检测框列表
        noisy_lidar_pose,       # 带噪声的位姿
        uncertainty_list,       # 不确定性列表
        ...
    )
    
    # 批量处理
    refined_pose = box_alignment_relative_np(
        pred_corner3d_list,
        uncertainty_list,
        lidar_poses,
        record_len,
        ...
    )

==============================================================================
"""

# ============================================================================
#                              标准库导入
# ============================================================================

from collections import OrderedDict    # 有序字典，保持聚类顺序
import numpy as np                     # NumPy 数值计算
import torch                           # PyTorch 张量操作
import torch.nn.functional as F        # PyTorch 函数式 API
import g2o                             # 图优化库 (pip install python-g2o)
from icecream import ic                # 调试打印工具
import copy                            # 深拷贝
import os                              # 文件系统操作
import matplotlib.pyplot as plt        # 绑图库

# ============================================================================
#                           项目内部模块导入
# ============================================================================

# PoseGraphOptimization2D: 2D 位姿图优化类
# 封装了 g2o 的位姿图优化功能，支持 SE(2) 和 R² 节点
from opencood.models.sub_modules.pose_graph_optim import PoseGraphOptimization2D

# pose_to_tfm: 将 6DOF 位姿转换为 4x4 齐次变换矩阵
# 输入: [x, y, z, roll, pitch, yaw]
# 输出: 4x4 变换矩阵
from opencood.utils.transformation_utils import pose_to_tfm

# check_torch_to_numpy: 将 PyTorch Tensor 转换为 NumPy 数组
from opencood.utils.common_utils import check_torch_to_numpy

# box_utils: 检测框相关工具函数
# - project_box3d: 将检测框投影到另一个坐标系
# - corner_to_center: 角点格式转中心点格式
from opencood.utils import box_utils

# ============================================================================
#                              全局变量
# ============================================================================

# DEBUG: 调试开关，设为 True 时输出详细信息
DEBUG = False


# ============================================================================
#                          可视化函数
# ============================================================================

def vis_pose_graph(poses, pred_corner3d, save_dir_path, vis_agent=False):
    """
    ============================================================================
    位姿图可视化函数
    ============================================================================
    
    【功能说明】
        将位姿图优化过程中的检测结果可视化，保存为图片。
        用于调试和分析位姿校正效果。
    
    【参数说明】
        poses: list of np.ndarray
            位姿列表，每个元素是一次迭代/优化后的位姿
            格式: [pose_before, ..., pose_refined]
            每个 pose: [N_cav, 6] 或 [N_cav, 3]
            用于可视化优化过程
        
        pred_corner3d: list
            每个智能体的检测框角点列表
            格式: [[N1_box, 8, 3], [N2_box, 8, 3], ...]
            - N_i_box: 第 i 个 CAV 的检测框数量
            - 8: 8 个角点
            - 3: 三维坐标 (x, y, z)
        
        save_dir_path: str
            图片保存目录路径
        
        vis_agent: bool
            是否绘制智能体位置点
            - True: 用彩色点标注每个 CAV 的位置
            - False: 只绘制检测框
    
    【输出】
        在 save_dir_path 下生成多张图片:
        - 0.png: 初始位姿下的检测结果
        - 1.png, 2.png, ...: 优化过程中的状态
        - n.png: 最终优化结果
    """
    # --------------------------------------------------------------------
    # 颜色定义
    # --------------------------------------------------------------------
    # 定义每个 CAV 的颜色，用于区分不同智能体的检测框
    # 颜色顺序: 红色、绿色、蓝色、紫色、橙色
    COLOR = ['red', 'springgreen', 'dodgerblue', 'darkviolet', 'orange']
    
    # 导入相对变换计算函数
    from opencood.utils.transformation_utils import get_relative_transformation

    # --------------------------------------------------------------------
    # 创建保存目录
    # --------------------------------------------------------------------
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # --------------------------------------------------------------------
    # 遍历每次迭代的位姿，生成可视化图片
    # --------------------------------------------------------------------
    for iter, pose in enumerate(poses):
        box_idx = 0  # 检测框编号，用于图片标注
        
        # ----------------------------------------------------------------
        # 计算相对变换矩阵
        # ----------------------------------------------------------------
        # 将各 CAV 的检测框变换到 Ego (Agent 0) 坐标系
        # relative_t_matrix: [N_cav, 4, 4]
        relative_t_matrix = get_relative_transformation(pose)
        
        # N: CAV 数量
        N = pose.shape[0]
        
        # ----------------------------------------------------------------
        # 筛选有检测结果的 CAV
        # ----------------------------------------------------------------
        # nonempty_indices: 检测框数量 > 0 的 CAV 索引列表
        nonempty_indices = [idx for (idx, corners) in enumerate(pred_corner3d) 
                           if len(corners) != 0]
        
        # ----------------------------------------------------------------
        # 将检测框投影到 Ego 坐标系
        # ----------------------------------------------------------------
        # pred_corners3d_in_ego: 投影后的检测框列表
        # 格式: [[N1_box, 8, 3], [N2_box, 8, 3], ...]
        pred_corners3d_in_ego = [
            box_utils.project_box3d(pred_corner3d[i], relative_t_matrix[i]) 
            for i in nonempty_indices
        ]

        # ----------------------------------------------------------------
        # 绘制每个 CAV 的检测框
        # ----------------------------------------------------------------
        for agent_id in range(len(pred_corners3d_in_ego)):
            # 跳过没有检测框的 CAV
            if agent_id not in nonempty_indices:
                continue
            
            # 获取当前 CAV 的检测框角点
            corner3d = pred_corners3d_in_ego[agent_id]  # [N_box, 8, 3]
            
            # 获取 CAV 在 Ego 坐标系中的位置
            # relative_t_matrix[agent_id][:2,3] 取变换矩阵的平移部分 (x, y)
            agent_pos = relative_t_matrix[agent_id][:2, 3]

            # 绘制 CAV 位置点
            if vis_agent:
                plt.scatter(agent_pos[0], agent_pos[1], s=4, c=COLOR[agent_id])

            # ----------------------------------------------------------------
            # 提取 2D 检测框 (取底面四个角点)
            # ----------------------------------------------------------------
            # corner3d 的 8 个角点顺序:
            # 0-3: 底面四角, 4-7: 顶面四角
            # corner2d: [N_box, 4, 2] 取底面四角的 (x, y)
            corner2d = corner3d[:, :4, :2]
            
            # 计算检测框中心 (取四个角点的平均值)
            center2d = np.mean(corner2d, axis=1)  # [N_box, 2]
            
            # ----------------------------------------------------------------
            # 绘制每个检测框
            # ----------------------------------------------------------------
            for i in range(corner2d.shape[0]):
                # 绘制检测框的前两个角点 (用于标注编号)
                plt.scatter(corner2d[i, [0,1], 0], corner2d[i, [0,1], 1], 
                           s=2, c=COLOR[agent_id])
                
                # 绘制检测框的边框 (闭合四边形)
                # [0,1,2,3,0] 表示连接顺序: 0->1->2->3->0 (闭合)
                plt.plot(corner2d[i, [0,1,2,3,0], 0], corner2d[i, [0,1,2,3,0], 1], 
                        linewidth=1, c=COLOR[agent_id])
                
                # 标注检测框编号
                plt.text(corner2d[i, 0, 0], corner2d[i, 0, 1], 
                        s=str(box_idx), fontsize="xx-small")
                
                # 绘制 CAV 到检测框中心的连线 (虚线)
                box_center = center2d[i]  # [2,]
                connection_x = [agent_pos[0], box_center[0]]
                connection_y = [agent_pos[1], box_center[1]]
                plt.plot(connection_x, connection_y, '--', 
                        linewidth=0.5, c=COLOR[agent_id], alpha=0.3)
                
                box_idx += 1
        
        # ----------------------------------------------------------------
        # 保存图片
        # ----------------------------------------------------------------
        filename = os.path.join(save_dir_path, f"{iter}.png")
        plt.gca().invert_yaxis()  # Y 轴反转 (符合图像坐标系)
        plt.axis('equal')         # 等比例坐标轴
        plt.savefig(filename, dpi=400)
        plt.clf()  # 清空画布，准备下一张图


# ============================================================================
#                          距离计算函数
# ============================================================================

def all_pair_l2(A, B):
    """
    ============================================================================
    计算两组点之间的所有配对 L2 距离
    ============================================================================
    
    【功能说明】
        高效计算两组点之间的欧几里得距离矩阵。
        使用广播机制避免显式循环，提高计算效率。
    
    【数学原理】
        L2 距离公式:
            d(a, b) = ||a - b||_2 = sqrt(Σ(a_i - b_i)²)
        
        展开后:
            d(a, b)² = Σa_i² + Σb_i² - 2Σa_i*b_i
                     = ||a||² + ||b||² - 2<a, b>
        
        因此距离矩阵可以表示为:
            D[i,j] = sqrt(||A[i]||² + ||B[j]||² - 2*A[i]·B[j])
    
    【参数说明】
        A: np.ndarray
            第一组点，形状 [N_A, D]
            - N_A: A 组点的数量
            - D: 点的维度 (如 2 或 3)
        
        B: np.ndarray
            第二组点，形状 [N_B, D]
            - N_B: B 组点的数量
            - D: 点的维度
    
    【返回值】
        C: np.ndarray
            距离矩阵，形状 [N_A, N_B]
            - C[i, j] = L2_distance(A[i], B[j])
    
    【使用示例】
        >>> A = np.array([[0, 0], [1, 1]])
        >>> B = np.array([[1, 0], [0, 1]])
        >>> all_pair_l2(A, B)
        array([[1.        , 1.        ],
               [1.        , 1.41421356]])
    
    【复杂度】
        时间复杂度: O(N_A * N_B * D)
        空间复杂度: O(N_A * N_B)
    """
    # 计算内积项: 2 * A @ B.T
    # A @ B.T 的形状: [N_A, N_B]
    # (A @ B.T)[i, j] = Σ_k A[i,k] * B[j,k] = <A[i], B[j]>
    TwoAB = 2 * A @ B.T  # [N_A, N_B]
    
    # 计算距离矩阵
    # D[i,j] = sqrt(||A[i]||² + ||B[j]||² - 2*<A[i], B[j]>)
    C = np.sqrt(
        # ||A[i]||²: 对 A 的每个点求平方和
        # np.sum(A * A, 1, keepdims=True): [N_A, 1]
        # .repeat(N_B, axis=1): [N_A, N_B] (每行复制 N_B 次)
        np.sum(A * A, 1, keepdims=True).repeat(TwoAB.shape[1], axis=1)
        
        # ||B[j]||²: 对 B 的每个点求平方和
        # np.sum(B * B, 1, keepdims=True).T: [1, N_B]
        # .repeat(N_A, axis=0): [N_A, N_B] (每列复制 N_A 次)
        + np.sum(B * B, 1, keepdims=True).T.repeat(TwoAB.shape[0], axis=0)
        
        # -2*<A[i], B[j]>: 减去两倍的内积
        - TwoAB
    )
    
    return C


# ============================================================================
#                       核心算法: 单样本检测框对齐
# ============================================================================

def box_alignment_relative_sample_np(
            pred_corners_list,
            noisy_lidar_pose, 
            uncertainty_list=None, 
            landmark_SE2=True,
            adaptive_landmark=False,
            normalize_uncertainty=False,
            abandon_hard_cases=False,
            drop_hard_boxes=False,
            drop_unsure_edge=False,
            use_uncertainty=True,
            thres=1.5,
            yaw_var_thres=0.2,
            max_iterations=1000):
    """
    ============================================================================
    单样本检测框对齐算法 ★★★ (FreeAlign 核心)
    ============================================================================
    
    【功能说明】
        对单个样本执行检测框对齐，校正各 CAV 的相对位姿。
        这是 FreeAlign 算法的核心实现。
    
    【算法概述】
        FreeAlign 的核心思想是利用同一目标被多个 CAV 观测到的几何约束:
        1. 同一目标在世界坐标系中有唯一位置
        2. 不同 CAV 对同一目标的观测应该一致
        3. 通过最小化观测误差来校正位姿
    
    【算法步骤】
        步骤 1:  数据预处理
            - 将检测框变换到世界坐标系 (如果使用初始位姿)
            - 提取检测框中心、朝向角
        
        步骤 2:  检测框聚类
            - 计算所有检测框中心之间的距离
            - 将距离小于阈值的框聚为一类 (同一目标)
        
        步骤 3:  位姿图构建
            - 节点: CAV (待优化) + Landmark (聚类中心)
            - 边: 观测关系 (Agent 观测到 Landmark)
            - 信息矩阵: 由检测不确定性决定
        
        步骤 4:  图优化
            - 使用 g2o 库优化位姿图
            - 最小化重投影误差
        
        步骤 5:  提取结果
            - 从优化结果中提取校正后的位姿
    
    【参数说明】
        pred_corners_list: list of np.ndarray
            每个 CAV 的检测框角点列表
            格式: [[N_1, 8, 3], ..., [N_cav, 8, 3]]
            - N_i: 第 i 个 CAV 的检测框数量
            - 8: 8 个角点
            - 3: 三维坐标 (x, y, z)
            - 坐标系: 每个 CAV 的局部坐标系
        
        noisy_lidar_pose: np.ndarray
            带噪声 (或初始估计) 的位姿
            格式: [N_cav, 6]
            - 6: [x, y, z, roll, pitch, yaw]
            - 注意: 角度单位为度 (degree)
            - yaw: 航向角，车辆朝向
        
        uncertainty_list: list of np.ndarray, optional
            检测不确定性列表
            格式: [[N_1, 3], [N_2, 3], ..., [N_cav, 3]]
            - 3: [log(σ_x²), log(σ_y²), log(σ_yaw²)]
            - 不确定性越小，检测越可信
            - 用于计算信息矩阵 (信息 = 1/方差)
        
        landmark_SE2: bool
            Landmark 是否使用 SE(2) 表示
            - True: Landmark = [x, y, yaw]，同时优化位置和朝向
            - False: Landmark = [x, y]，只优化位置
            - SE(2) 更精确，但可能受检测框朝向误差影响
        
        adaptive_landmark: bool
            自适应 Landmark 类型 (仅当 landmark_SE2=True 时有效)
            - True: 当同一目标的检测框朝向差异大时，退化为 R²
            - False: 始终使用 SE(2)
            - 用于处理朝向估计不准的情况
        
        normalize_uncertainty: bool
            是否对不确定性进行归一化
            - True: certainty = sqrt(certainty)
            - False: 直接使用计算得到的 certainty
        
        abandon_hard_cases: bool
            是否放弃困难样本
            - True: 对于困难样本 (如匹配少、朝向差异大)，直接返回原位姿
            - False: 始终尝试优化
        
        drop_hard_boxes: bool
            是否丢弃困难检测框
            - True: 丢弃朝向差异大的聚类
            - False: 保留所有检测框
        
        drop_unsure_edge: bool
            是否丢弃不确定的边
            - True: 丢弃确定性低的观测边
            - False: 保留所有边
        
        use_uncertainty: bool
            是否使用不确定性信息
            - True: 使用 uncertainty_list 构建信息矩阵
            - False: 使用单位信息矩阵
        
        thres: float
            检测框聚类距离阈值 (米)
            - 两个检测框中心距离 < thres 才会被聚为一类
            - 典型值: 1.5 米
            - 太小: 匹配少；太大: 误匹配
        
        yaw_var_thres: float
            朝向角方差阈值 (弧度)
            - 同一聚类的检测框朝向方差 > yaw_var_thres 时，认为朝向不一致
            - 典型值: 0.2 弧度 (~11.5°)
            - 用于 adaptive_landmark 和 drop_hard_boxes
        
        max_iterations: int
            图优化最大迭代次数
            - 典型值: 1000
            - 迭代越多，优化越充分，但耗时越长
    
    【返回值】
        refined_lidar_poses: np.ndarray
            校正后的相对位姿
            格式: [N_cav, 3]
            - 3: [x, y, yaw]
            - yaw 单位为度 (degree)
            - 相对于 Ego (Agent 0) 坐标系
    
    【注意事项】
        1. Agent 0 (Ego) 的位姿固定为 [0, 0, 0]，作为参考坐标系
        2. 至少需要 2 个 CAV 才能进行位姿校正
        3. 匹配的检测框越多，校正效果越好
        4. 不确定性信息可以提高校正精度
    """
    
    # ========================================================================
    # 步骤 1: 参数预处理
    # ========================================================================
    
    # 如果不使用不确定性，将 uncertainty_list 设为 None
    if not use_uncertainty:
        uncertainty_list = None
    
    # --------------------------------------------------------------------
    # 数据格式设置
    # --------------------------------------------------------------------
    # order: 检测框尺寸的排列顺序
    # 'lwh': length-width-height (长-宽-高)
    # corner_to_center 函数会根据此顺序解析检测框
    order = 'lwh'
    
    # N: CAV 数量
    N = noisy_lidar_pose.shape[0]
    
    # --------------------------------------------------------------------
    # 将位姿转换为变换矩阵
    # --------------------------------------------------------------------
    # pose_to_tfm 将 6DOF 位姿 [x,y,z,roll,pitch,yaw] 转换为 4x4 齐次变换矩阵
    # lidar_pose_noisy_tfm: [N, 4, 4]
    # 用于将检测框从局部坐标系变换到世界坐标系
    lidar_pose_noisy_tfm = pose_to_tfm(noisy_lidar_pose)

    # ========================================================================
    # 步骤 2: 数据提取和预处理
    # ========================================================================
    
    # --------------------------------------------------------------------
    # 筛选有检测结果的 CAV
    # --------------------------------------------------------------------
    # nonempty_indices: 检测框数量 > 0 的 CAV 索引列表
    # 例如: 如果 CAV 1 没有检测到任何目标，则 nonempty_indices = [0, 2, ...]
    nonempty_indices = [idx for (idx, corners) in enumerate(pred_corners_list) 
                       if len(corners) != 0]
    
    # --------------------------------------------------------------------
    # 将检测框投影到世界坐标系
    # --------------------------------------------------------------------
    # pred_corners_world_list: 世界坐标系中的检测框列表
    # 格式: [[N1, 8, 3], [N2, 8, 3], ...]
    # 这一步用于计算检测框在世界坐标系中的位置和朝向
    pred_corners_world_list = [
        box_utils.project_box3d(pred_corners_list[i], lidar_pose_noisy_tfm[i]) 
        for i in nonempty_indices
    ]
    
    # --------------------------------------------------------------------
    # 将角点格式转换为中心点格式
    # --------------------------------------------------------------------
    # pred_box3d_list: 每个 CAV 的检测框 (中心点格式，局部坐标系)
    # 格式: [[N1, 7], [N2, 7], ...]
    # 7: [x, y, z, l, w, h, yaw]
    # - x, y, z: 检测框中心
    # - l, w, h: 检测框长宽高
    # - yaw: 检测框朝向角 (弧度)
    pred_box3d_list = [
        box_utils.corner_to_center(corner, order) 
        for corner in pred_corners_list if len(corner) != 0
    ]
    
    # pred_box3d_world_list: 世界坐标系中的检测框 (中心点格式)
    pred_box3d_world_list = [
        box_utils.corner_to_center(corner, order) 
        for corner in pred_corners_world_list
    ]
    
    # --------------------------------------------------------------------
    # 提取检测框中心坐标
    # --------------------------------------------------------------------
    # pred_center_list: 局部坐标系中的检测框中心
    # 格式: [[N1, 3], [N2, 3], ...]
    pred_center_list = [
        np.mean(corners, axis=1)  # 对 8 个角点求平均
        for corners in pred_corners_list if len(corners) != 0
    ]

    # --------------------------------------------------------------------
    # 提取世界坐标系中的中心坐标和朝向
    # --------------------------------------------------------------------
    # pred_center_world_list: 世界坐标系中的检测框中心
    # 格式: [[N1, 3], [N2, 3], ...]
    pred_center_world_list = [
        pred_box3d_world[:, :3]  # 取前 3 列: [x, y, z]
        for pred_box3d_world in pred_box3d_world_list
    ]
    
    # pred_yaw_world_list: 世界坐标系中的检测框朝向
    # 格式: [[N1], [N2], ...] 或 [N1,], [N2,], ...
    pred_yaw_world_list = [
        pred_box3d[:, 6]  # 取第 7 列: yaw
        for pred_box3d in pred_box3d_world_list
    ]
    
    # pred_len: 每个 CAV 的检测框数量
    # 格式: [N1, N2, ..., N_cav]
    pred_len = [len(corners) for corners in pred_corners_list]

    # ========================================================================
    # 步骤 3: 构建检测框索引映射
    # ========================================================================
    
    # box_idx_to_agent: 检测框索引到 CAV ID 的映射
    # 例如: 如果 CAV 0 有 2 个框，CAV 1 有 3 个框
    # 则 box_idx_to_agent = [0, 0, 1, 1, 1]
    # 这用于后续知道每个检测框属于哪个 CAV
    box_idx_to_agent = []
    for i in range(N):
        box_idx_to_agent += [i] * pred_len[i]  # 每个 CAV 贡献 pred_len[i] 个框
    
    # --------------------------------------------------------------------
    # 合并所有检测框数据
    # --------------------------------------------------------------------
    # pred_center_cat: 合并后的检测框中心 (局部坐标系)
    # 形状: [sum(pred_box), 3]
    pred_center_cat = np.concatenate(pred_center_list, axis=0)
    
    # pred_center_world_cat: 合并后的检测框中心 (世界坐标系)
    # 形状: [sum(pred_box), 3]
    pred_center_world_cat = np.concatenate(pred_center_world_list, axis=0)
    
    # pred_box3d_cat: 合并后的检测框 (中心点格式)
    # 形状: [sum(pred_box), 7]
    pred_box3d_cat = np.concatenate(pred_box3d_list, axis=0)
    
    # pred_yaw_world_cat: 合并后的检测框朝向 (世界坐标系)
    # 形状: [sum(pred_box)]
    pred_yaw_world_cat = np.concatenate(pred_yaw_world_list, axis=0)

    # ========================================================================
    # 步骤 4: 计算不确定性权重 (可选)
    # ========================================================================
    
    # 锚框尺寸 (用于不确定性归一化)
    # w_a: 锚框宽度 (典型车辆宽度: 1.6 米)
    # l_a: 锚框长度 (典型车辆长度: 3.9 米)
    # d_a_square: 锚框对角线长度的平方，用于归一化位置不确定性
    w_a = 1.6
    l_a = 3.9
    d_a_square = w_a ** 2 + l_a ** 2  # 1.6² + 3.9² ≈ 17.77

    if uncertainty_list is not None:
        # ----------------------------------------------------------------
        # 提取并合并不确定性
        # ----------------------------------------------------------------
        # pred_log_sigma2_cat: 合并后的对数方差
        # 形状: [sum(pred_box), 3]
        # 每行: [log(σ_x²), log(σ_y²), log(σ_yaw²)]
        pred_log_sigma2_cat = np.concatenate(
            [i for i in uncertainty_list if len(i) != 0], 
            axis=0
        )
        
        # ----------------------------------------------------------------
        # 计算确定性 (certainty = 1/variance = exp(-log_variance))
        # ----------------------------------------------------------------
        # pred_certainty_cat: 确定性权重
        # 形状: [sum(pred_box), 3]
        # certainty 越大，表示检测越可信
        pred_certainty_cat = np.exp(-pred_log_sigma2_cat)
        
        # ----------------------------------------------------------------
        # 位置不确定性归一化
        # ----------------------------------------------------------------
        # 原因: 检测网络的回归目标是归一化后的偏移量
        # x_t = (x_g - x_a) / d_a，其中 d_a 是锚框对角线长度
        # 因此: var(x) = d_a² * var(x_t)
        # 我们需要: 1/var(x) = 1/var(x_t) / d_a²
        pred_certainty_cat[:, :2] /= d_a_square  # 只对 x, y 进行归一化

        # ----------------------------------------------------------------
        # 可选: 对确定性开根号 (归一化)
        # ----------------------------------------------------------------
        if normalize_uncertainty:
            pred_certainty_cat = np.sqrt(pred_certainty_cat)

    # ========================================================================
    # 步骤 5: 计算所有检测框对之间的距离
    # ========================================================================
    
    # pred_center_allpair_dist: 距离矩阵
    # 形状: [sum(pred_box), sum(pred_box)]
    # dist[i, j] = ||center_i - center_j||_2 (世界坐标系)
    pred_center_allpair_dist = all_pair_l2(pred_center_world_cat, pred_center_world_cat)

    # --------------------------------------------------------------------
    # 将同一 CAV 的检测框对距离设为最大值
    # --------------------------------------------------------------------
    # 原因: 同一 CAV 的检测框不可能是同一目标 (一个 CAV 不会重复检测同一目标)
    # 设置为 MAX_DIST 后，在聚类时会被排除
    MAX_DIST = 10000  # 足够大的值
    cum = 0  # 累积索引
    for i in range(N):
        # 将 CAV i 的所有检测框之间的距离设为 MAX_DIST
        pred_center_allpair_dist[
            cum: cum + pred_len[i],      # CAV i 的检测框起始索引
            cum: cum + pred_len[i]       # CAV i 的检测框结束索引
        ] = MAX_DIST
        cum += pred_len[i]

    # ========================================================================
    # 步骤 6: 检测框聚类 ★ (核心步骤)
    # ========================================================================
    # 聚类目的: 找到不同 CAV 观测到的同一目标
    # 聚类方法: 距离小于阈值的检测框聚为一类
    
    # --------------------------------------------------------------------
    # 初始化聚类变量
    # --------------------------------------------------------------------
    # cluster_id: 当前聚类 ID，从 N 开始 (0 到 N-1 是 CAV 的 ID)
    cluster_id = N
    
    # cluster_dict: 聚类字典
    # 键: cluster_id
    # 值: OrderedDict，包含聚类信息
    cluster_dict = OrderedDict()
    
    # remain_box: 尚未分配到聚类的检测框索引集合
    remain_box = set(range(cum))  # cum = sum(pred_box)

    # --------------------------------------------------------------------
    # 遍历所有检测框，进行聚类
    # --------------------------------------------------------------------
    for box_idx in range(cum):
        # 如果当前检测框已经被分配，跳过
        if box_idx not in remain_box:
            continue
        
        # ----------------------------------------------------------------
        # 找到与当前检测框距离小于阈值的检测框
        # ----------------------------------------------------------------
        # within_thres_idx_tensor: 距离 < thres 的检测框索引
        within_thres_idx_tensor = (pred_center_allpair_dist[box_idx] < thres).nonzero()[0]
        within_thres_idx_list = within_thres_idx_tensor.tolist()

        # 如果没有邻近检测框，继续下一个
        if len(within_thres_idx_list) == 0:
            continue

        # ----------------------------------------------------------------
        # 广度优先搜索 (BFS) 扩展聚类
        # ----------------------------------------------------------------
        # 从当前检测框出发，找到所有属于同一目标的检测框
        # explored: 已探索的检测框索引
        explored = [box_idx]
        
        # unexplored: 待探索的检测框索引 (且在 remain_box 中)
        unexplored = [idx for idx in within_thres_idx_list if idx in remain_box]

        # BFS 循环
        while unexplored:
            idx = unexplored[0]  # 取第一个待探索的检测框
            
            # 找到与 idx 距离小于阈值的检测框
            within_thres_idx_tensor = (pred_center_allpair_dist[idx] < thres).nonzero()[0]
            within_thres_idx_list = within_thres_idx_tensor.tolist()
            
            # 将新的邻近检测框加入待探索列表
            for newidx in within_thres_idx_list:
                if (newidx not in explored) and (newidx not in unexplored) and (newidx in remain_box):
                    unexplored.append(newidx)
            
            # 将当前检测框从待探索移到已探索
            unexplored.remove(idx)
            explored.append(idx)
        
        # ----------------------------------------------------------------
        # 处理单个检测框的情况
        # ----------------------------------------------------------------
        # 如果 explored 只有一个元素，说明这是一个孤立检测框
        # 没有其他 CAV 观测到同一目标
        if len(explored) == 1:
            remain_box.remove(box_idx)
            continue
        
        # ----------------------------------------------------------------
        # 创建聚类记录
        # ----------------------------------------------------------------
        cluster_box_idxs = explored  # 聚类中的检测框索引列表

        # 初始化聚类字典
        cluster_dict[cluster_id] = OrderedDict()
        
        # 记录聚类信息
        # box_idx: 聚类中的检测框索引列表
        cluster_dict[cluster_id]['box_idx'] = [idx for idx in cluster_box_idxs]
        
        # box_center_world: 每个检测框在世界坐标系中的中心
        cluster_dict[cluster_id]['box_center_world'] = [
            pred_center_world_cat[idx] for idx in cluster_box_idxs
        ]
        
        # box_yaw: 每个检测框的朝向角
        cluster_dict[cluster_id]['box_yaw'] = [
            pred_yaw_world_cat[idx] for idx in cluster_box_idxs
        ]

        # ----------------------------------------------------------------
        # 计算朝向角方差，判断朝向是否一致
        # ----------------------------------------------------------------
        yaw_var = np.var(cluster_dict[cluster_id]['box_yaw'])
        
        # box_yaw_varies: 朝向是否差异过大
        # True: 同一目标的检测框朝向差异大，可能是误匹配或检测不准
        cluster_dict[cluster_id]['box_yaw_varies'] = yaw_var > yaw_var_thres
        
        # active: 该聚类是否参与优化
        cluster_dict[cluster_id]['active'] = True

        # ----------------------------------------------------------------
        # 确定 Landmark 表示方式
        # ----------------------------------------------------------------
        if landmark_SE2:
            if adaptive_landmark and yaw_var > yaw_var_thres:
                # 自适应模式: 朝向差异大时，使用 R² 表示
                # 只优化位置，不优化朝向
                landmark = pred_center_world_cat[box_idx][:2]  # [x, y]
                
                # 增加这些检测框的确定性权重
                # 因为朝向不可靠，但我们希望位置仍然有贡献
                for _box_idx in cluster_box_idxs:
                    pred_certainty_cat[_box_idx] *= 2
            else:
                # 使用 SE(2) 表示: [x, y, yaw]
                landmark = copy.deepcopy(pred_center_world_cat[box_idx])
                landmark[2] = pred_yaw_world_cat[box_idx]
        else:
            # 强制使用 R² 表示
            landmark = pred_center_world_cat[box_idx][:2]

        # 保存 Landmark
        cluster_dict[cluster_id]['landmark'] = landmark  # [x, y, yaw] 或 [x, y]
        cluster_dict[cluster_id]['landmark_SE2'] = True if landmark.shape[0] == 3 else False

        # ----------------------------------------------------------------
        # 调试输出 (可选)
        # ----------------------------------------------------------------
        DEBUG = False
        if DEBUG:
            from icecream import ic
            ic(cluster_dict[cluster_id]['box_idx'])
            ic(cluster_dict[cluster_id]['box_center_world'])
            ic(cluster_dict[cluster_id]['box_yaw'])
            ic(cluster_dict[cluster_id]['landmark'])

        # 更新聚类 ID
        cluster_id += 1
        
        # 将已聚类的检测框从 remain_box 中移除
        for idx in cluster_box_idxs:
            remain_box.remove(idx)

    # ========================================================================
    # 步骤 7: 统计节点数量
    # ========================================================================
    
    # vertex_num: 总节点数 (CAV + Landmark)
    vertex_num = cluster_id
    
    # agent_num: CAV 数量
    agent_num = N
    
    # landmark_num: Landmark 数量
    landmark_num = cluster_id - N

    # ========================================================================
    # 步骤 8: 处理困难样本 (可选)
    # ========================================================================
    # 困难样本定义:
    # 1. Landmark 数量太少 (匹配少)
    # 2. 大部分 Landmark 的朝向差异大
    
    if abandon_hard_cases:
        # 情况 1: Landmark 数量 <= 3
        if landmark_num <= 3:
            # 直接返回原位姿 (不做校正)
            return noisy_lidar_pose[:, [0, 1, 4]]  # [x, y, yaw]
        
        # 情况 2: 超过一半的 Landmark 朝向差异大
        yaw_varies_cnt = sum([
            cluster_dict[i]["box_yaw_varies"] 
            for i in range(agent_num, vertex_num)
        ])
        if yaw_varies_cnt >= 0.5 * landmark_num:
            return noisy_lidar_pose[:, [0, 1, 4]]

    # ========================================================================
    # 步骤 9: 丢弃困难检测框 (可选)
    # ========================================================================
    # 将朝向差异大的 Landmark 设为 inactive
    if drop_hard_boxes:
        for landmark_id in range(agent_num, vertex_num):
            if cluster_dict[landmark_id]['box_yaw_varies']:
                cluster_dict[landmark_id]['active'] = False

    # ========================================================================
    # 步骤 10: 构建位姿图并优化 ★★★
    # ========================================================================
    
    # --------------------------------------------------------------------
    # 初始化位姿图优化器
    # --------------------------------------------------------------------
    pgo = PoseGraphOptimization2D()

    # --------------------------------------------------------------------
    # 添加 CAV 节点
    # --------------------------------------------------------------------
    for agent_id in range(agent_num):
        v_id = agent_id  # 节点 ID = CAV ID
        
        # 提取位姿: [x, y, yaw]
        # noisy_lidar_pose 格式: [x, y, z, roll, pitch, yaw]
        pose_np = noisy_lidar_pose[agent_id, [0, 1, 4]]
        
        # 将 yaw 从度转换为弧度
        pose_np[2] = np.deg2rad(pose_np[2])
        
        # 创建 SE(2) 位姿对象
        v_pose = g2o.SE2(pose_np)
        
        # Agent 0 (Ego) 固定，作为参考坐标系
        # 其他 Agent 待优化
        if agent_id == 0:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=True)
        else:
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=False)

    # --------------------------------------------------------------------
    # 添加 Landmark 节点
    # --------------------------------------------------------------------
    for landmark_id in range(agent_num, vertex_num):
        v_id = landmark_id  # 节点 ID = Landmark ID
        
        # 获取 Landmark 位置
        landmark = cluster_dict[landmark_id]['landmark']  # (3,) 或 (2,)
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']

        if landmark_SE2:
            # SE(2) 节点: [x, y, yaw]
            v_pose = g2o.SE2(landmark)
        else:
            # R² 节点: [x, y]
            v_pose = landmark

        # 添加 Landmark 节点 (不固定，需要优化)
        pgo.add_vertex(id=v_id, pose=v_pose, fixed=False, SE2=landmark_SE2)

    # --------------------------------------------------------------------
    # 添加 Agent-Landmark 边 (观测约束)
    # --------------------------------------------------------------------
    for landmark_id in range(agent_num, vertex_num):
        landmark_SE2 = cluster_dict[landmark_id]['landmark_SE2']

        # 跳过 inactive 的 Landmark
        if not cluster_dict[landmark_id]['active']:
            continue

        # 遍历该 Landmark 的所有检测框
        for box_idx in cluster_dict[landmark_id]['box_idx']:
            # 获取检测该目标的 CAV ID
            agent_id = box_idx_to_agent[box_idx]
            
            if landmark_SE2:
                # SE(2) 边: 观测值为检测框在 Agent 坐标系中的 [x, y, yaw]
                e_pose = g2o.SE2(pred_box3d_cat[box_idx][[0, 1, 6]].astype(np.float64))
                
                # 信息矩阵 (对角矩阵，表示观测权重)
                info = np.identity(3, dtype=np.float64)
                
                if uncertainty_list is not None:
                    # 使用检测确定性作为信息矩阵的对角元素
                    info[[0, 1, 2], [0, 1, 2]] = pred_certainty_cat[box_idx]

                    # 可选: 丢弃确定性太低的边
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx]) < 100:
                        continue

            else:
                # R² 边: 观测值为检测框在 Agent 坐标系中的 [x, y]
                e_pose = pred_box3d_cat[box_idx][[0, 1]].astype(np.float64)
                
                # 信息矩阵 (2x2)
                info = np.identity(2, dtype=np.float64)
                
                if uncertainty_list is not None:
                    info[[0, 1], [0, 1]] = pred_certainty_cat[box_idx][:2]

                    # 可选: 丢弃确定性太低的边
                    if drop_unsure_edge and sum(pred_certainty_cat[box_idx][:2]) < 100:
                        continue

            # 添加边: agent_id -> landmark_id
            # measurement: 在 Agent 坐标系中观测到的 Landmark 位置
            # information: 观测的信息矩阵
            pgo.add_edge(
                vertices=[agent_id, landmark_id], 
                measurement=e_pose, 
                information=info, 
                SE2=landmark_SE2
            )
    
    # --------------------------------------------------------------------
    # 执行图优化
    # --------------------------------------------------------------------
    # 使用 g2o 进行非线性优化，最小化重投影误差
    # 目标函数: min Σ ||z_ij - h(x_i, l_j)||²_Ω
    # - z_ij: Agent i 对 Landmark j 的观测
    # - h(x_i, l_j): 根据 Agent 位姿和 Landmark 位置计算的预测观测
    # - Ω: 信息矩阵
    pgo.optimize(max_iterations)

    # --------------------------------------------------------------------
    # 提取优化后的位姿
    # --------------------------------------------------------------------
    pose_new_list = []
    for agent_id in range(agent_num):
        # 获取优化后的位姿 (SE(2) 向量)
        pose_new_list.append(pgo.get_pose(agent_id).vector())

    # 转换为 NumPy 数组
    refined_pose = np.array(pose_new_list)  # [N_cav, 3]
    
    # 将 yaw 从弧度转换回度
    refined_pose[:, 2] = np.rad2deg(refined_pose[:, 2])

    return refined_pose


# ============================================================================
#                       批量处理函数
# ============================================================================

def box_alignment_relative_np(pred_corner3d_list, 
                              uncertainty_list, 
                              lidar_poses, 
                              record_len, 
                              **kwargs):
    """
    ============================================================================
    批量检测框对齐
    ============================================================================
    
    【功能说明】
        对多个样本批量执行检测框对齐。
        是 box_alignment_relative_sample_np 的批量包装函数。
    
    【参数说明】
        pred_corner3d_list: list of tensors
            所有样本的检测框列表
            格式: [[N1_obj, 8, 3], [N2_obj, 8, 3], ..., [N_sumcav_obj, 8, 3]]
            - 检测框在各自 CAV 的局部坐标系中
        
        uncertainty_list: list of tensors
            检测不确定性列表
            格式: [[N1_obj, 3], ..., [N_sumcav_obj, 3]]
        
        lidar_poses: torch.Tensor
            所有 CAV 的位姿
            形状: [sum(cav), 6]
        
        record_len: torch.Tensor
            每个样本的 CAV 数量
            形状: [batch_size]
            - 例如: [2, 2, 3, 2] 表示 4 个样本，分别有 2,2,3,2 个 CAV
    
    【返回值】
        refined_lidar_pose: np.ndarray
            校正后的位姿
            形状: [sum(cav), 3]
            - 3: [x, y, yaw]
    
    【处理流程】
        1. 根据 record_len 分割数据
        2. 对每个样本调用 box_alignment_relative_sample_np
        3. 合并结果
    """
    refined_lidar_pose = []
    start_idx = 0
    
    # 遍历每个样本
    for b in record_len:
        # 对当前样本执行位姿校正
        refined_lidar_pose.append(
            box_alignment_relative_sample_np(
                pred_corner3d_list[start_idx: start_idx + b],  # 当前样本的检测框
                lidar_poses[start_idx: start_idx + b],         # 当前样本的位姿
                uncertainty_list=None if uncertainty_list is None 
                    else uncertainty_list[start_idx: start_idx + b],  # 当前样本的不确定性
                **kwargs  # 其他参数
            )
        )
        start_idx += b

    # 合并所有样本的结果
    return np.concatenate(refined_lidar_pose, axis=0)