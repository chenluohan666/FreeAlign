# -*- coding: utf-8 -*-
"""
==============================================================================
                    中间融合数据集 (Intermediate Fusion Dataset)
==============================================================================

【功能概述】
    本文件实现了协同感知中最重要的中间融合数据集类，是 FreeAlign/CoAlign 的核心数据处理模块。
    中间融合是指在特征层面进行多智能体数据融合的策略，相比早期融合和后期融合，
    中间融合能够在保持检测精度的同时减少通信带宽需求。

【核心功能】
    1. 加载多智能体 (CAV) 的传感器数据 (LiDAR/相机)
    2. 坐标变换: 将所有 CAV 的数据投影到 Ego 车辆坐标系
    3. FreeAlign 位姿校正: 使用图匹配算法校正位姿误差 (无外部定位设备)
    4. 数据预处理: 体素化、数据增强、标签生成
    5. 批处理整理: 将多个样本整理成一个 batch

【数据流程图】
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                          __getitem__(idx)                                │
    │                                                                          │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐ │
    │  │ 原始数据加载 │ -> │ 位姿噪声添加 │ -> │ FreeAlign   │ -> │ 单车处理   │ │
    │  │(父类方法)    │    │(模拟GPS误差) │    │ 位姿校正★   │    │(体素化等) │ │
    │  └─────────────┘    └─────────────┘    └─────────────┘    └───────────┘ │
    │                                                                    │     │
    │                                                                    ▼     │
    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
    │  │ 批处理整理   │ <- │ 生成标签     │ <- │ 合并多CAV数据 + GT去重      │ │
    │  │(collate_fn) │    │(正负样本分配) │    │                             │ │
    │  └─────────────┘    └─────────────┘    └─────────────────────────────┘ │
    └──────────────────────────────────────────────────────────────────────────┘

【继承关系详解】
    
    本文件使用工厂函数模式，动态创建继承自基础数据集类的中间融合数据集类。
    
    继承链:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  torch.utils.data.Dataset                                               │
    │       │                                                                  │
    │       ▼                                                                  │
    │  DAIRV2XBaseDataset / OPV2VBaseDataset / V2XSimBaseDataset              │
    │  (实现 retrieve_base_data 方法，从磁盘加载原始数据)                       │
    │       │                                                                  │
    │       ▼                                                                  │
    │  IntermediateFusionDataset (本文件)                                      │
    │  (实现 __getitem__ 方法，进行中间融合数据处理)                            │
    └─────────────────────────────────────────────────────────────────────────┘
    
    父类提供的方法 (从 DAIRV2XBaseDataset 继承):
    ├── __init__(): 初始化数据路径、预处理器、后处理器、数据增强器
    ├── retrieve_base_data(idx): 从磁盘加载一个样本的原始数据
    ├── generate_object_center_*(): 生成 GT 检测框中心
    ├── get_ext_int(): 获取相机内外参
    ├── augment(): 数据增强
    └── reinitialize(): 重新初始化 (空实现)
    
    本类新增/覆盖的方法:
    ├── __init__(): 新增 FreeAlign 相关参数初始化
    ├── __getitem__(): 核心数据处理流程 (新增)
    ├── get_item_single_car(): 单个 CAV 数据处理 (新增)
    ├── collate_batch_train(): 训练时批处理整理 (新增)
    ├── collate_batch_test(): 测试时批处理整理 (新增)
    └── post_process(): 模型输出后处理 (新增)

【使用方法】
    # 方法1: 通过 build_dataset 工厂函数创建
    from opencood.data_utils.datasets import build_dataset
    dataset = build_dataset(hypes, visualize=False, train=True)
    
    # 方法2: 直接使用工厂函数
    from opencood.data_utils.datasets.basedataset.dairv2x_basedataset import DAIRV2XBaseDataset
    from opencood.data_utils.datasets.intermediate_fusion_dataset import getIntermediateFusionDataset
    DatasetCls = getIntermediateFusionDataset(DAIRV2XBaseDataset)
    dataset = DatasetCls(hypes, visualize=False, train=True)
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_batch_train)

【关键数据结构】
    
    retrieve_base_data() 返回的数据结构 (来自父类):
    {
        'scenario_folder': "batch_1",      # 场景 ID
        'timestep': "000001",              # 时间步/帧 ID
        0: {                               # 车端数据 (Ego)
            'ego': True,                   # 是否为 Ego
            'params': {
                'lidar_pose': [x,y,z,roll,pitch,yaw],  # 6DOF 位姿
                'lidar_pose_clean': [...],             # 真实位姿 (无噪声)
                'vehicles_all': [...],                 # 360° 标注
                'vehicles_front': [...],               # 前方标注
            },
            'lidar_np': np.ndarray [N,4],  # 点云 (x,y,z,intensity)
            'camera_data': [...],          # 相机图像 (可选)
        },
        1: {                               # 路端数据 (RSU)
            'ego': False,
            ...
        }
    }
    
    __getitem__() 返回的数据结构:
    {
        'ego': {
            'timestep': "000001",
            'scenario_folder': "batch_1",
            'processed_lidar': {           # 体素化特征
                'voxel_features': [...],
                'voxel_coords': [...],
                'voxel_num_points': [...],
            },
            'object_bbx_center': np.ndarray [max_num, 7],  # GT 框
            'object_bbx_mask': np.ndarray [max_num],       # 有效框掩码
            'label_dict': {...},          # 训练标签
            'pairwise_t_matrix': np.ndarray [max_cav, max_cav, 4, 4],
            'record_len': int,            # CAV 数量
            ...
        }
    }

==============================================================================
"""

# ============================================================================
#                              标准库导入
# ============================================================================

import random                          # 随机数生成，用于数据增强随机性
import math                            # 数学函数，用于计算距离
from collections import OrderedDict    # 有序字典，保持 CAV 顺序一致性
import numpy as np                     # NumPy 数值计算库
import torch                           # PyTorch 深度学习框架
import copy                            # 深拷贝，用于复制数据
from icecream import ic                # 调试打印工具 (pip install icecream)
from PIL import Image                  # Python 图像库
import pickle as pkl                   # 序列化工具

# ============================================================================
#                           FreeAlign 核心模块导入
# ============================================================================
# 这些模块实现了 FreeAlign 的核心算法：
# 1. 图匹配：在不同 CAV 的检测框之间寻找对应关系
# 2. 位姿估计：根据匹配的检测框计算相对位姿

# get_pose_rotation: 从匹配的检测框对计算相对旋转角度
from freealign.match.match_v7_with_detection import get_pose_rotation

# get_right_box: 获取正确的检测框 (用于 GT 校正模式)
from freealign.match.match_v7_debug import get_right_box

# freealign_training: FreeAlign 训练相关函数
from freealign.match.match_v7_with_detection import freealign_training

# ============================================================================
#                           OpenCOOD 工具模块导入
# ============================================================================

# 检测框工具函数
from opencood.utils import box_utils as box_utils

# 预处理器构建函数：将点云转换为体素特征
from opencood.data_utils.pre_processor import build_preprocessor

# 后处理器构建函数：生成锚框、标签、NMS 等
from opencood.data_utils.post_processor import build_postprocessor

# 相机数据处理工具函数
from opencood.utils.camera_utils import (
    sample_augmentation,   # 随机采样数据增强参数 (resize, crop, flip, rotate)
    img_transform,         # 应用数据增强变换到图像
    normalize_img,         # 图像归一化 (均值/标准差)
    img_to_tensor,         # 图像转换为 PyTorch Tensor
)

# 异构智能体选择器 (用于处理不同类型的传感器配置)
from opencood.utils.heter_utils import AgentSelector

# 特征合并工具函数
from opencood.utils.common_utils import merge_features_to_dict

# 坐标变换工具函数
from opencood.utils.transformation_utils import (
    x1_to_x2,                    # 计算坐标系1到坐标系2的变换矩阵
    x_to_world,                  # 局部坐标系到世界坐标系的变换
    get_pairwise_transformation, # 获取所有 CAV 对之间的变换矩阵
    pose_to_tfm                  # 6DOF 位姿 [x,y,z,roll,pitch,yaw] 转 4x4 变换矩阵
)

# 位姿噪声添加函数 (模拟 GPS/RTK 定位误差)
from opencood.utils.pose_utils import add_noise_data_dict

# 点云处理工具函数
from opencood.utils.pcd_utils import (
    mask_points_by_range,       # 根据感知范围过滤点云
    mask_ego_points,            # 移除打到车辆自身的点 (避免自遮挡噪声)
    shuffle_points,             # 随机打乱点云顺序 (数据增强)
    downsample_lidar_minimum,   # 点云下采样 (减少可视化数据量)
)

# JSON 文件读取工具
from opencood.utils.common_utils import read_json

# 检测框格式转换工具
from opencood.utils.box_utils import (
    corner_to_center,    # 角点格式转中心点格式
    boxes_to_corners_3d, # 中心点格式转角点格式
    project_box3d        # 将 3D 框投影到另一个坐标系
)


# ============================================================================
#                          数据集类工厂函数
# ============================================================================

def getIntermediateFusionDataset(cls):
    """
    ============================================================================
    中间融合数据集类的工厂函数
    ============================================================================
    
    【设计模式】装饰器模式 / 工厂函数
    
    【功能说明】
        这个函数接收一个基础数据集类 (如 DAIRV2XBaseDataset)，
        动态创建一个继承自该基础类的中间融合数据集类。
        
    【为什么使用工厂函数】
        1. 代码复用：同样的中间融合逻辑可以应用于不同的数据集
           - DAIR-V2X: 真实世界车路协同数据
           - OPV2V: 仿真多车协同数据
           - V2X-Sim: 仿真车路协同数据
        
        2. 解耦设计：基础数据加载和融合逻辑分离
           - 父类负责：从磁盘加载原始数据
           - 子类负责：数据处理、融合、标签生成
    
    【参数说明】
        cls: BaseDataset 子类
            基础数据集类，必须实现以下方法：
            - retrieve_base_data(idx): 加载原始数据
            - generate_object_center(): 生成 GT 框
            - get_ext_int(): 获取相机参数
    
    【返回值】
        IntermediateFusionDataset: 包装后的中间融合数据集类
        
    【使用示例】
        # 在 opencood/data_utils/datasets/__init__.py 中
        def build_dataset(hypes, visualize, train):
            # 根据配置选择基础数据集类
            if dataset_name == 'dairv2x':
                base_cls = DAIRV2XBaseDataset
            elif dataset_name == 'opv2v':
                base_cls = OPV2VBaseDataset
            
            # 根据融合策略选择包装函数
            if fusion_method == 'intermediate':
                dataset_cls = getIntermediateFusionDataset(base_cls)
            
            return dataset_cls(hypes, visualize, train)
    """
    class IntermediateFusionDataset(cls):
        """
        ====================================================================
        中间融合数据集类
        ====================================================================
        
        【类说明】
            这个类继承自传入的基础数据集类 (如 DAIRV2XBaseDataset)，
            并在此基础上实现中间融合所需的所有数据处理逻辑。
        
        【核心方法】
            1. __init__(): 初始化，包括 FreeAlign 参数
            2. __getitem__(): 核心数据处理流程
            3. get_item_single_car(): 单个 CAV 数据处理
            4. collate_batch_train(): 训练批处理
            5. collate_batch_test(): 测试批处理
        """
        
        # ====================================================================
        # 初始化函数
        # ====================================================================
        def __init__(self, params, visualize, train=True):
            """
            ====================================================================
            初始化中间融合数据集
            ====================================================================
            
            【初始化流程】
                1. 调用父类初始化 (加载基础配置和数据索引)
                2. 初始化融合策略参数
                3. 生成锚框
                4. 初始化 FreeAlign 位姿校正参数
            
            【参数说明】
                params: dict
                    配置字典，来自 YAML 配置文件
                    示例配置结构:
                    ```yaml
                    model:
                      args:
                        supervise_single: true
                    fusion:
                      args:
                        proj_first: false
                    box_align:
                      no_pose: true
                      min_anchor: 4
                      ...
                    ```
                
                visualize: bool
                    是否启用可视化模式
                    - True: 保存额外的可视化数据 (原始点云等)
                    - False: 仅保存训练所需数据
                    
                train: bool
                    是否为训练模式
                    - True: 使用训练集，启用数据增强
                    - False: 使用验证/测试集，禁用数据增强
            """
            # ----------------------------------------------------------------
            # 第一步: 调用父类初始化
            # ----------------------------------------------------------------
            # 父类 __init__ 会执行以下操作:
            # 1. 保存配置参数 self.params = params
            # 2. 构建预处理器 self.pre_processor = build_preprocessor(...)
            # 3. 构建后处理器 self.post_processor = build_postprocessor(...)
            # 4. 构建数据增强器 self.data_augmentor = DataAugmentor(...)
            # 5. 设置输入源 self.load_lidar_file / self.load_camera_file
            # 6. 加载数据索引 self.split_info / self.co_data
            # 7. 初始化噪声配置 self.params['noise_setting']
            super().__init__(params, visualize, train)

            # ----------------------------------------------------------------
            # 第二步: 初始化融合策略参数
            # ----------------------------------------------------------------
            
            # supervise_single: 是否使用单车监督损失
            # 用于 DiscoNet 等需要单车检测监督的方法
            # True: 计算每个 CAV 单独的检测损失，帮助学习更好的单车特征
            # False: 仅计算协同后的检测损失
            self.supervise_single = True if ('supervise_single' in params['model']['args'] 
                                             and params['model']['args']['supervise_single']) \
                                    else False
            
            # proj_first: 是否先将点云投影到 Ego 坐标系再体素化
            # 这涉及两种通信模式:
            # - False (默认, 1-round 通信):
            #   1. 每个 CAV 本地体素化
            #   2. 发送特征图
            #   3. 接收方用 warp 操作对齐特征图
            #   优点: 通信量小，仅需传输特征图
            # 
            # - True (2-round 通信):
            #   1. 发送方先发送位姿
            #   2. 接收方计算变换矩阵并返回
            #   3. 发送方变换点云后再体素化
            #   优点: 特征对齐更精确
            #   缺点: 需要两轮通信
            self.proj_first = False if 'proj_first' not in params['fusion']['args'] \
                                    else params['fusion']['args']['proj_first']

            # ----------------------------------------------------------------
            # 第三步: 生成锚框
            # ----------------------------------------------------------------
            # 锚框是目标检测中预定义的候选框，用于预测实际目标的位置和大小
            # 
            # generate_anchor_box() 根据配置生成锚框:
            # - 锚框尺寸: l=3.9m, w=1.6m, h=1.56m (典型车辆尺寸)
            # - 锚框方向: 0° 和 90° (两个方向)
            # - 特征图步长: 2 (每个锚框对应特征图上的一个位置)
            # 
            # 输出形状: [N_anchors, 7]
            # - 7 维: [x, y, z, l, w, h, yaw]
            #   - x, y, z: 锚框中心坐标
            #   - l, w, h: 锚框长宽高
            #   - yaw: 锚框朝向角
            self.anchor_box = self.post_processor.generate_anchor_box()
            
            # 将 NumPy 数组转换为 PyTorch Tensor (用于模型推理)
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            # ----------------------------------------------------------------
            # 第四步: 知识蒸馏标志
            # ----------------------------------------------------------------
            # kd_flag: 知识蒸馏标志
            # 用于 DiscoNet 等需要教师网络监督的方法
            # True: 加载教师网络的预测结果作为额外监督
            # False: 不使用知识蒸馏
            self.kd_flag = params.get('kd_flag', False)

            # ----------------------------------------------------------------
            # 第五步: FreeAlign 位姿校正参数初始化 ★ (核心创新点)
            # ----------------------------------------------------------------
            # FreeAlign 的核心创新: 在没有外部定位设备 (GPS/RTK) 的情况下，
            # 通过识别多智能体感知数据中的几何模式来进行时空对齐
            self.box_align = False
            
            # 如果配置文件中有 box_align 部分，则启用 FreeAlign
            if "box_align" in params:
                self.box_align = True
                
                # --------------------- Stage1 检测结果路径 ---------------------
                # FreeAlign 需要两阶段训练:
                # Stage 1: 使用预训练模型生成检测框 (pose_graph_pre_calc.py)
                # Stage 2: 使用 Stage1 检测框进行图匹配和位姿校正
                # 
                # stage1_boxes.json 包含:
                # - pred_corner3d_np_list: 各 CAV 的检测框角点
                # - uncertainty_np_list: 检测不确定性
                # - cav_id_list: CAV ID 列表
                self.stage1_result_path = params['box_align']['train_result'] if train \
                                          else params['box_align']['val_result']
                self.stage1_result = read_json(self.stage1_result_path)
                
                # --------------------- 位姿图优化参数 ---------------------
                # 传递给 box_alignment_relative_sample_np 函数
                # 包含:
                # - use_uncertainty: 是否使用不确定性权重
                # - landmark_SE2: 是否使用 SE(2) 位姿估计
                # - abandon_hard_cases: 是否放弃困难样本
                self.box_align_args = params['box_align']['args']
                
                # --------------------- no_pose 核心参数 ★ ---------------------
                # no_pose 是 FreeAlign 与 CoAlign 的关键区别:
                # 
                # no_pose=True (FreeAlign 模式):
                #   - 完全不使用外部位姿信息
                #   - 位姿初始化为零，完全依赖图匹配恢复
                #   - 适用于: 无 GPS/RTK 设备场景
                # 
                # no_pose=False (CoAlign 模式):
                #   - 使用外部位姿作为初始值
                #   - 通过图匹配校正位姿误差
                #   - 适用于: 有 GPS/RTK 但精度有限的场景
                self.no_pose = params['box_align']['no_pose']
                
                # --------------------- MASS 算法参数 ---------------------
                # MASS (Multi-Anchor Subgraph Searching) 是 FreeAlign 的核心算法
                # 用于在两个 CAV 的检测框集合中找到匹配的子图
                
                # min_anchor: 最小锚点数量
                # 至少需要多少对匹配的检测框才能进行位姿估计
                # 值越大，估计越可靠，但成功率越低
                self.min_anchor = params['box_align']['min_anchor']
                
                # anchor_error: 锚点匹配误差阈值 (米)
                # 两检测框中心距离小于此阈值才认为是同一目标
                self.anchor_error = params['box_align']['anchor_error']
                
                # box_error: 检测框匹配误差阈值 (米)
                # 用于 MASS 算法中的子图搜索
                self.box_error = params['box_align']['box_error']
                
                # --------------------- GNN 边特征学习 (可选) ---------------------
                # use_gnn: 是否使用图神经网络学习检测框之间的边特征
                # True: 使用预训练的 GNN 提取更准确的匹配特征
                # False: 使用简单的几何距离作为匹配特征
                self.use_gnn = params['box_align']['gnn']
                
                # gt_correct: 是否使用 GT 框进行校正 (仅用于调试/消融实验)
                self.gt_correct = params['box_align']['gt_correct']
                
                # 如果启用 GNN，加载预训练的 GNN 模型
                if self.use_gnn:
                    self.gnn_error = params['box_align']['gnn_error']
                    self.gnn_extractor = torch.load(params['box_align']['gnn_model_path'])


        # ====================================================================
        # 单车数据处理函数
        # ====================================================================
        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            ====================================================================
            处理单个 CAV (Connected Autonomous Vehicle) 的数据
            ====================================================================
            
            【功能说明】
                这是对每个智能体数据的核心处理函数。
                在 __getitem__ 中会被循环调用，处理每个通信范围内的 CAV。
            
            【处理流程】
                1. 提取位姿信息，计算变换矩阵
                2. LiDAR 点云预处理 (打乱、过滤、投影、体素化)
                3. 相机数据处理 (可选)
                4. 生成单车 GT 标签
                5. 生成协同 GT 标签
            
            【参数说明】
                selected_cav_base: dict
                    单个 CAV 的原始数据字典，来自父类 retrieve_base_data() 的输出
                    结构示例:
                    {
                        'params': {
                            'lidar_pose': [x, y, z, roll, pitch, yaw],      # 6DOF 位姿
                            'lidar_pose_clean': [...],                      # 真实位姿
                            'vehicles_all': [...],                          # 360° 标注
                            'vehicles_front': [...],                        # 前方标注
                            'camera0': {'extrinsic': ..., 'intrinsic': ...} # 相机参数
                        },
                        'lidar_np': np.ndarray [N, 4],  # 点云数据
                        'camera_data': [img1, img2, ...]  # 相机图像 (可选)
                    }
                
                ego_cav_base: dict
                    Ego 车辆的数据字典，结构同上
                    用于获取 Ego 位姿，计算相对变换
            
            【返回值】
                selected_cav_processed: dict
                    处理后的数据字典，包含:
                    - 'processed_features': 体素化后的点云特征
                    - 'object_bbx_center': GT 检测框中心 [N, 7]
                    - 'object_bbx_mask': 有效框掩码 [N]
                    - 'transformation_matrix': 变换矩阵 [4, 4]
                    - 'image_inputs': 相机输入 (可选)
                    - 'single_*': 单车监督相关标签
            """
            # ----------------------------------------------------------------
            # 初始化输出字典
            # ----------------------------------------------------------------
            selected_cav_processed = {}
            
            # ----------------------------------------------------------------
            # 第一步: 提取位姿信息
            # ----------------------------------------------------------------
            # lidar_pose: 当前使用的位姿 (可能带有噪声或经过 FreeAlign 校正)
            #             用于特征变换，将 CAV 数据对齐到 Ego 坐标系
            # lidar_pose_clean: 真实位姿 (Ground Truth，无噪声)
            #                   仅用于生成 GT 标签，确保标签质量
            ego_pose = ego_cav_base['params']['lidar_pose']
            ego_pose_clean = ego_cav_base['params']['lidar_pose_clean']

            # ----------------------------------------------------------------
            # 第二步: 计算变换矩阵
            # ----------------------------------------------------------------
            # 变换矩阵 T_ego_cav: 将 CAV 坐标系的点变换到 Ego 坐标系
            # 数学表示: P_ego = T_ego_cav @ P_cav
            
            # transformation_matrix: 使用当前位姿计算 (带噪声或校正后)
            # 这是实际用于特征变换的矩阵
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],  # CAV 位姿
                        ego_pose)                                      # Ego 位姿
            # x1_to_x2 函数内部:
            # 1. 将两个 6DOF 位姿转换为 4x4 变换矩阵
            # 2. 计算 T_ego_cav = T_ego_world @ T_world_cav
            #    = T_ego_world @ T_cav_world^(-1)
            
            # transformation_matrix_clean: 使用真实位姿计算
            # 仅用于生成 GT 标签，确保标签坐标正确
            transformation_matrix_clean = \
                x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                        ego_pose_clean)
            
            # ----------------------------------------------------------------
            # 第三步: LiDAR 点云处理
            # ----------------------------------------------------------------
            if self.load_lidar_file or self.visualize:
                # 3.1 获取原始点云
                # lidar_np: [N, 4] 形状的 NumPy 数组
                # 每行: [x, y, z, intensity]
                # - x, y, z: 点在 LiDAR 坐标系中的三维坐标
                # - intensity: 反射强度 (与物体材质相关)
                lidar_np = selected_cav_base['lidar_np']
                
                # 3.2 打乱点云顺序
                # 数据增强，防止模型过拟合点云顺序
                # 注意: 点云是无序的，打乱不会影响语义信息
                lidar_np = shuffle_points(lidar_np)
                
                # 3.3 移除打到车辆自身的点
                # 这些点是由于激光打到车辆自身部件 (车顶、后视镜等) 产生的
                # 它们不是环境感知的一部分，会产生噪声
                # mask_ego_points 会根据车辆尺寸过滤这些点
                lidar_np = mask_ego_points(lidar_np)
                
                # 3.4 将点云投影到 Ego 坐标系
                # 这是协同感知的关键步骤: 统一到同一坐标系
                # project_points_by_matrix_torch 内部:
                # 1. 提取点的前 3 维 (x, y, z)
                # 2. 添加齐次坐标 (变为 [N, 4])
                # 3. 应用变换矩阵: P_ego = T @ P_cav
                # 4. 返回变换后的坐标 [N, 3]
                projected_lidar = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                             transformation_matrix)
                
                # 3.5 处理 proj_first 模式
                # 如果 proj_first=True，在体素化前就将点云坐标更新为投影后的坐标
                # 这意味着体素化是在 Ego 坐标系中进行的
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar

                # 3.6 保存投影后的点云用于可视化
                if self.visualize:
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                # 3.7 知识蒸馏模式下的点云处理
                # DiscoNet 需要教师网络的点云作为额外监督
                if self.kd_flag:
                    lidar_proj_np = copy.deepcopy(lidar_np)
                    lidar_proj_np[:,:3] = projected_lidar
                    selected_cav_processed.update({'projected_lidar': lidar_proj_np})

                # 3.8 预处理点云 (体素化)
                # 这是最关键的预处理步骤
                # 
                # preprocess() 内部流程:
                # 1. 定义体素网格:
                #    - 根据 voxel_size [0.4, 0.4, 4] 定义网格分辨率
                #    - 根据 cav_lidar_range [-140.8, -40, -3, 140.8, 40, 1] 定义范围
                # 
                # 2. 将点分配到体素:
                #    - 计算每个点所属的体素坐标 (voxel_x, voxel_y, voxel_z)
                #    - 统计每个体素内的点数
                # 
                # 3. 体素特征提取:
                #    - 对每个体素内的点，取最多 max_points_per_voxel 个点
                #    - 计算每个点的特征: (x, y, z, intensity, x_offset, y_offset, z_offset)
                #    - 或者使用 PointNet-like 网络提取特征
                # 
                # 输出字典包含:
                # - voxel_features: [N_voxel, max_points, C] 体素特征
                # - voxel_coords: [N_voxel, 4] 体素坐标 (batch, z, y, x)
                # - voxel_num_points: [N_voxel] 每个体素的点数
                processed_lidar = self.pre_processor.preprocess(lidar_np)
                selected_cav_processed.update({'processed_features': processed_lidar})

            # ----------------------------------------------------------------
            # 第四步: 生成单车 GT 标签
            # ----------------------------------------------------------------
            # 单车标签用于 supervise_single 模式
            # 注意: 参考坐标系是该 CAV 自身，不是 Ego
            # 这允许模型学习每个 CAV 的单车检测能力
            
            # generate_object_center 由父类实现
            # 根据标注生成 GT 检测框中心
            # 参数:
            # - [selected_cav_base]: CAV 数据列表 (单个元素)
            # - selected_cav_base['params']['lidar_pose']: 参考坐标系位姿
            # 
            # 返回:
            # - object_bbx_center: [N, 7] GT 框中心 (x,y,z,l,w,h,yaw)
            # - object_bbx_mask: [N] 有效框掩码
            # - object_ids: 物体 ID 列表 (用于去重)
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], selected_cav_base['params']['lidar_pose']
            )
            
            # generate_label 根据锚框和 GT 框生成训练标签
            # 内部流程:
            # 1. 计算 GT 框与每个锚框的 IoU
            # 2. 根据阈值分配正/负样本:
            #    - pos_threshold=0.6: IoU > 0.6 为正样本
            #    - neg_threshold=0.45: IoU < 0.45 为负样本
            # 3. 计算回归目标: dx, dy, dz, dl, dw, dh, d_yaw
            # 
            # 返回:
            # - pos_equal_one: 正样本掩码
            # - neg_equal_one: 负样本掩码
            # - targets: 回归目标
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, 
                anchors=self.anchor_box, 
                mask=object_bbx_mask
            )
            
            # 保存单车标签
            selected_cav_processed.update({
                "single_label_dict": label_dict,
                "single_object_bbx_center": object_bbx_center,
                "single_object_bbx_mask": object_bbx_mask
            })

            # ----------------------------------------------------------------
            # 第五步: 相机数据处理 (可选)
            # ----------------------------------------------------------------
            if self.load_camera_file:
                camera_data_list = selected_cav_base["camera_data"]
                params = selected_cav_base["params"]
                
                # 初始化相机数据收集列表
                imgs = []           # 图像 Tensor 列表
                rots = []           # 旋转矩阵列表
                trans = []          # 平移向量列表
                intrins = []        # 内参矩阵列表
                extrinsics = []     # 外参矩阵列表
                post_rots = []      # 数据增强旋转矩阵列表
                post_trans = []     # 数据增强平移向量列表

                # 遍历所有相机 (DAIR-V2X 车端有 1 个相机，路端有 1 个相机)
                for idx, img in enumerate(camera_data_list):
                    # 5.1 获取相机外参和内参
                    # get_ext_int 由父类实现
                    # camera_to_lidar: 相机坐标系到 LiDAR 坐标系的变换矩阵 [4, 4]
                    # camera_intrinsic: 相机内参矩阵 [3, 3]
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    # 5.2 转换为 PyTorch Tensor
                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(camera_to_lidar[:3, :3])   # 旋转部分 R_wc
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])   # 平移部分 T_wc

                    # 5.3 初始化数据增强后处理矩阵
                    post_rot = torch.eye(2)       # 2x2 单位矩阵
                    post_tran = torch.zeros(2)    # 2D 零向量

                    img_src = [img]

                    # 5.4 加载深度图 (可选，当前未启用)
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # 5.5 数据增强
                    # sample_augmentation 随机采样增强参数:
                    # - resize: 缩放比例
                    # - resize_dims: 缩放后尺寸
                    # - crop: 裁剪区域
                    # - flip: 是否翻转
                    # - rotate: 旋转角度
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        self.data_aug_conf, self.train
                    )
                    
                    # 5.6 应用数据增强变换
                    # img_transform 对图像进行:
                    # - resize: 调整图像大小
                    # - crop: 裁剪感兴趣区域
                    # - flip: 水平/垂直翻转
                    # - rotate: 旋转图像
                    # 同时更新后处理矩阵 (用于坐标映射)
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src, post_rot, post_tran,
                        resize=resize, resize_dims=resize_dims,
                        crop=crop, flip=flip, rotate=rotate,
                    )
                    
                    # 5.7 构造 3x3 增强矩阵
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # 5.8 图像归一化和转 Tensor
                    # normalize_img: 将像素值归一化到 [0, 1] 或标准化
                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    # 5.9 收集数据
                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)

                # 5.10 整理相机输入字典
                selected_cav_processed.update({
                    "image_inputs": {
                        "imgs": torch.stack(imgs),              # [Ncam, 3or4, H, W]
                        "intrins": torch.stack(intrins),        # [Ncam, 3, 3]
                        "extrinsics": torch.stack(extrinsics),  # [Ncam, 4, 4]
                        "rots": torch.stack(rots),              # [Ncam, 3, 3]
                        "trans": torch.stack(trans),            # [Ncam, 3]
                        "post_rots": torch.stack(post_rots),    # [Ncam, 3, 3]
                        "post_trans": torch.stack(post_trans),  # [Ncam, 3]
                    }
                })

            # ----------------------------------------------------------------
            # 第六步: 保存锚框
            # ----------------------------------------------------------------
            selected_cav_processed.update({"anchor_box": self.anchor_box})

            # ----------------------------------------------------------------
            # 第七步: 生成协同 GT 标签
            # ----------------------------------------------------------------
            # 注意: 参考坐标系是 Ego (协同检测的统一坐标系)
            # 这是最终用于损失计算的标签
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], ego_pose_clean  # 使用 Ego 的真实位姿
            )

            # 保存处理结果
            selected_cav_processed.update({
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],  # 只保留有效框
                "object_bbx_mask": object_bbx_mask,
                "object_ids": object_ids,
                'transformation_matrix': transformation_matrix,
                'transformation_matrix_clean': transformation_matrix_clean
            })

            return selected_cav_processed


        # ====================================================================
        # 数据获取入口 (__getitem__)
        # ====================================================================
        def __getitem__(self, idx):
            """
            ====================================================================
            PyTorch Dataset 的核心方法: 根据索引获取一个样本
            ====================================================================
            
            【执行流程】
                1. 加载原始数据 (调用父类 retrieve_base_data)
                2. 添加位姿噪声 (模拟真实场景)
                3. 找到 Ego 车辆
                4. 筛选通信范围内的 CAV
                5. FreeAlign 位姿校正 ★ (核心创新)
                6. 计算成对变换矩阵
                7. 处理每个 CAV 的数据
                8. 整理单车监督标签
                9. 去重 GT 目标
                10. Padding 到固定大小
                11. 合并特征
                12. 生成标签
                13. 组装输出字典
            
            【参数说明】
                idx: int
                    样本索引，由 DataLoader 提供
                    范围: [0, len(dataset))
            
            【返回值】
                processed_data_dict: OrderedDict
                    处理后的数据字典，包含模型训练所需的所有数据
            """
            # ----------------------------------------------------------------
            # 第一步: 加载原始数据
            # ----------------------------------------------------------------
            # retrieve_base_data 由父类实现 (如 DAIRV2XBaseDataset)
            # 从磁盘加载:
            # - 点云数据 (.pcd 文件)
            # - 相机图像 (.jpg 文件)
            # - 位姿信息 (.json 标定文件)
            # - GT 标注 (.json 标签文件)
            base_data_dict = self.retrieve_base_data(idx)
            
            # 提取元数据 (场景 ID 和时间步)
            timestep = base_data_dict.pop('timestep', None)          # 如 "000001"
            scenario_folder = base_data_dict.pop('scenario_folder', None)  # 如 "batch_1"
            
            # 删除非数字键 (清理数据字典)
            # base_data_dict 的键应该是 CAV ID (0, 1, 2, ...)
            # 但可能包含其他元数据键，需要清理
            del_key_list = []
            for key in base_data_dict.keys():
                try:
                    int(key)  # 尝试转换为整数
                except:
                    del_key_list.append(key)  # 转换失败则标记删除
            for del_key in del_key_list:
                del base_data_dict[del_key]
            
            # ----------------------------------------------------------------
            # 第二步: 添加位姿噪声
            # ----------------------------------------------------------------
            # 模拟真实场景中的位姿误差
            # GPS/RTK 定位存在误差:
            # - 位置误差: 通常 0.1-1.0 米 (取决于 GPS 精度)
            # - 旋转误差: 通常 0.1-0.5 弧度
            # 
            # add_noise_data_dict 会:
            # 1. 复制 lidar_pose 到 lidar_pose_clean (保存真实值)
            # 2. 对 lidar_pose 添加高斯噪声
            # 3. 噪声参数来自配置文件 noise_setting.args
            base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])

            # ----------------------------------------------------------------
            # 第三步: 初始化输出字典
            # ----------------------------------------------------------------
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}
            processed_data_dict['ego']['timestep'] = timestep
            processed_data_dict['ego']['scenario_folder'] = scenario_folder

            # ----------------------------------------------------------------
            # 第四步: 找到 Ego 车辆
            # ----------------------------------------------------------------
            # Ego 车辆是协同感知的参考中心
            # 其他所有 CAV 的数据都会变换到 Ego 坐标系
            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # 遍历所有 CAV，找到标记为 ego 的车辆
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:  # ego 标志在父类加载数据时设置
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break
            
            # 断言检查 (确保数据正确)
            assert cav_id == list(base_data_dict.keys())[0], \
                "The first element in the OrderedDict must be ego"
            assert ego_id != -1  # 必须找到 Ego
            assert len(ego_lidar_pose) > 0  # Ego 位姿不能为空

            # ----------------------------------------------------------------
            # 第五步: 初始化数据收集列表
            # ----------------------------------------------------------------
            # 这些列表用于收集所有 CAV 的处理结果
            agents_image_inputs = []              # 相机输入列表
            processed_features = []               # 处理后的特征列表
            object_stack = []                     # GT 框列表
            object_id_stack = []                  # 物体 ID 列表
            single_label_list = []                # 单车标签列表
            single_object_bbx_center_list = []    # 单车 GT 框中心列表
            single_object_bbx_mask_list = []      # 单车 GT 框掩码列表
            too_far = []                          # 超出通信范围的 CAV ID 列表
            lidar_pose_list = []                  # 位姿列表 (用于特征变换)
            lidar_pose_clean_list = []            # 真实位姿列表 (用于评估)
            cav_id_list = []                      # 通信范围内的 CAV ID 列表
            projected_lidar_clean_list = []       # 投影后的点云列表 (DiscoNet 用)

            # 可视化/知识蒸馏模式下的点云收集
            if self.visualize or self.kd_flag:
                projected_lidar_stack = []

            # ----------------------------------------------------------------
            # 第六步: 筛选通信范围内的 CAV
            # ----------------------------------------------------------------
            # 协同感知假设 CAV 之间可以通过 V2X 通信
            # 但通信距离有限 (通常 100-200 米)
            # 超出范围的 CAV 无法共享信息
            for cav_id, selected_cav_base in base_data_dict.items():
                # 计算该 CAV 与 Ego 的平面距离 (忽略高度)
                distance = math.sqrt(
                    (selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 +
                    (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2
                )

                # 超出通信范围则跳过
                # comm_range 来自配置文件，通常为 100 米
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    continue

                # 记录通信范围内 CAV 的位姿
                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])
                cav_id_list.append(cav_id)

            # 移除超出通信范围的 CAV
            for cav_id in too_far:
                base_data_dict.pop(cav_id)

            # =================================================================
            # 第七步: FreeAlign 位姿校正 ★★★ (核心创新点)
            # =================================================================
            # 这是 FreeAlign 的核心算法
            # 使用 Stage1 检测框进行图匹配，校正位姿误差
            # 
            # 算法流程:
            # 1. 加载 Stage1 预计算的检测结果
            # 2. 对每个 CAV，提取其检测框
            # 3. 使用 MASS 算法在不同 CAV 的检测框之间找匹配
            # 4. 使用 SVD/ICP 计算相对位姿
            # 5. 位姿图优化 (可选)
            if self.box_align and str(idx) in self.stage1_result.keys():
                # 导入位姿图优化模块
                from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
                
                stage1_content = self.stage1_result[str(idx)]
                
                if stage1_content is not None:
                    # --------------------- 获取 Stage1 检测结果 ---------------------
                    # all_agent_id_list: 所有 CAV 的 ID 列表
                    # all_agent_corners_list: 每个 CAV 检测框的角点
                    # all_agent_uncertainty_list: 每个 CAV 检测的不确定性
                    all_agent_id_list = stage1_content['cav_id_list']
                    all_agent_corners_list = stage1_content['pred_corner3d_np_list']
                    all_agent_uncertainty_list = stage1_content['uncertainty_np_list']

                    # --------------------- 获取当前通信范围内的智能体 ---------------------
                    cur_agent_id_list = cav_id_list
                    cur_agent_pose_list = [base_data_dict[cav_id]['params']['lidar_pose'] 
                                           for cav_id in cav_id_list]
                    cur_agent_pose = np.array(cur_agent_pose_list)
                    
                    # 索引当前智能体在 all_agent_list 中的位置
                    # 用于提取对应的检测框
                    cur_agent_in_all_agent = [all_agent_id_list.index(cur_agent) 
                                              for cur_agent in cur_agent_id_list]

                    # 提取当前智能体的检测框和不确定性
                    pred_corners_list = [np.array(all_agent_corners_list[cur_in_all_ind], dtype=np.float64) 
                                         for cur_in_all_ind in cur_agent_in_all_agent]
                    uncertainty_list = [np.array(all_agent_uncertainty_list[cur_in_all_ind], dtype=np.float64) 
                                        for cur_in_all_ind in cur_agent_in_all_agent]

                    # 保存 GT 位姿 (用于评估)
                    gt_pose = pose_to_tfm(cur_agent_pose)
                    
                    # --------------------- 关键: no_pose 模式 ★ ---------------------
                    # 这是 FreeAlign 的核心特性
                    # no_pose=True: 完全不使用外部位姿，从零开始恢复
                    # 
                    # 位姿初始化:
                    # - x, y, z = 0 (位置未知)
                    # - roll, pitch = 0 (假设地面平坦)
                    # - yaw = 1.0 (初始朝向，特殊编码)
                    if self.no_pose:
                        cur_agent_pose = np.zeros((len(cur_agent_id_list), 6))
                        cur_agent_pose[:,4] = 1.0  # 初始 yaw 角 (特殊值)
                    else:
                        # CoAlign 模式: 使用带噪声的外部位姿作为初始值
                        cur_agent_pose = np.array(copy.deepcopy(
                            [base_data_dict[cav_id]['params']['lidar_pose'] 
                             for cav_id in cav_id_list]))

                    # --------------------- GT 校正模式 (调试用) ---------------------
                    # 使用 GT 框替代预测框，用于消融实验
                    if self.gt_correct:
                        selected_cav_base = base_data_dict[cav_id_list[0]]
                        selected_cav_processed = self.get_item_single_car(
                            selected_cav_base, ego_cav_base)
                        
                        ego_corner_box_gt = boxes_to_corners_3d(
                            selected_cav_processed['object_bbx_center'], 'hwl')
                        
                        for i in range(1, len(cur_agent_pose)):
                            pred_corners_list[i] = np.array(get_right_box(
                                torch.tensor(pred_corners_list[i], dtype=torch.float32),
                                torch.tensor(ego_corner_box_gt, dtype=torch.float32),
                                torch.tensor(np.expand_dims(
                                    get_pairwise_transformation(base_data_dict, self.max_cav, self.proj_first), 0),
                                    dtype=torch.float32)))

                    # --------------------- 位姿图优化 ---------------------
                    # box_alignment_relative_sample_np 实现了 MASS 算法
                    # 
                    # 算法细节:
                    # 1. 多锚点子图搜索 (MASS):
                    #    - 选择一个"锚点对"作为匹配起点
                    #    - 贪婪扩展子图，添加匹配的框对
                    #    - 直到无法找到更多匹配
                    # 
                    # 2. 相对位姿计算:
                    #    - 对匹配的框对，使用 SVD 计算刚体变换
                    #    - 或使用 ICP 迭代优化
                    # 
                    # 3. 位姿图优化:
                    #    - 使用 g2o 库进行位姿图优化
                    #    - 最小化重投影误差
                    if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(
                            pred_corners_list,      # 各 CAV 的检测框角点
                            cur_agent_pose,         # 初始位姿
                            uncertainty_list=uncertainty_list,  # 不确定性权重
                            **self.box_align_args   # 其他参数
                        )
                        
                        # 更新校正后的位姿
                        # refined_pose 只包含 [x, y, yaw]
                        # z, roll, pitch 保持不变
                        cur_agent_pose[:,[0,1,4]] = refined_pose

                        # 将校正后的位姿写回数据字典
                        for i, cav_id in enumerate(cav_id_list):
                            lidar_pose_list[i] = cur_agent_pose[i].tolist()
                            base_data_dict[cav_id]['params']['lidar_pose'] = cur_agent_pose[i].tolist()

                # 移除校正失败的 CAV (位姿全为零表示失败)
                indices_to_delete = []
                for i, cav_id in enumerate(cav_id_list):
                    if not cur_agent_pose[i].any():
                        base_data_dict.pop(cav_id)
                        indices_to_delete.append(i)

                # 更新 CAV ID 列表
                new_list = [element for index, element in enumerate(cav_id_list) 
                            if index not in indices_to_delete]
                cav_id_list = new_list

            # ----------------------------------------------------------------
            # 第八步: 计算成对变换矩阵
            # ----------------------------------------------------------------
            # pairwise_t_matrix: [max_cav, max_cav, 4, 4]
            # pairwise_t_matrix[i][j] 表示 CAV_i 到 CAV_j 的变换矩阵
            # 
            # 这个矩阵用于:
            # 1. 特征融合时的坐标对齐
            # 2. 模型中的注意力机制
            pairwise_t_matrix = get_pairwise_transformation(
                base_data_dict, self.max_cav, self.proj_first
            )

            # 整理位姿数组
            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)      # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)
            
            # 记录 CAV 数量
            cav_num = len(cav_id_list)

            # ----------------------------------------------------------------
            # 第九步: 处理每个 CAV
            # ----------------------------------------------------------------
            for _, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                
                # 调用单车处理函数 (详见该函数的注释)
                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base, ego_cav_base)
                
                # 收集处理结果
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                
                # 收集 LiDAR 特征
                if self.load_lidar_file:
                    processed_features.append(selected_cav_processed['processed_features'])
                
                # 收集相机特征
                if self.load_camera_file:
                    agents_image_inputs.append(selected_cav_processed['image_inputs'])

                # 收集可视化数据
                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
                
                # 收集单车监督数据
                if self.supervise_single:
                    single_label_list.append(selected_cav_processed['single_label_dict'])
                    single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])

            # ----------------------------------------------------------------
            # 第十步: 整理单车监督标签
            # ----------------------------------------------------------------
            if self.supervise_single:
                # 批量整理单车标签
                single_label_dicts = self.post_processor.collate_batch(single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict['ego'].update({
                    "single_label_dict_torch": single_label_dicts,
                    "single_object_bbx_center_torch": single_object_bbx_center,
                    "single_object_bbx_mask_torch": single_object_bbx_mask,
                })

            # 知识蒸馏模式
            if self.kd_flag:
                # 合并所有 CAV 的点云
                stack_lidar_np = np.vstack(projected_lidar_stack)
                # 按范围过滤
                stack_lidar_np = mask_points_by_range(
                    stack_lidar_np, self.params['preprocess']['cav_lidar_range'])
                # 体素化
                stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                processed_data_dict['ego'].update({
                    'teacher_processed_lidar': stack_feature_processed
                })

            # ----------------------------------------------------------------
            # 第十一步: 去重 GT 目标
            # ----------------------------------------------------------------
            # 多个 CAV 可能观测到同一目标
            # 通过 object_id 去重，避免重复计算
            unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]

            # ----------------------------------------------------------------
            # 第十二步: Padding 到固定大小
            # ----------------------------------------------------------------
            # PyTorch 批处理要求数据维度一致
            # max_num 定义了每帧最大目标数量
            object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            # ----------------------------------------------------------------
            # 第十三步: 合并特征
            # ----------------------------------------------------------------
            if self.load_lidar_file:
                # 合并所有 CAV 的体素特征
                merged_feature_dict = merge_features_to_dict(processed_features)
                processed_data_dict['ego'].update({'processed_lidar': merged_feature_dict})
            
            if self.load_camera_file:
                # 合并所有 CAV 的相机特征
                merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
                processed_data_dict['ego'].update({'image_inputs': merged_image_inputs_dict})

            # ----------------------------------------------------------------
            # 第十四步: 生成标签
            # ----------------------------------------------------------------
            # 生成训练所需的标签:
            # - pos_equal_one: 正样本掩码
            # - neg_equal_one: 负样本掩码
            # - targets: 回归目标
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=self.anchor_box,
                mask=mask
            )

            # ----------------------------------------------------------------
            # 第十五步: 组装输出字典
            # ----------------------------------------------------------------
            processed_data_dict['ego'].update({
                'object_bbx_center': object_bbx_center,       # GT 框 [max_num, 7]
                'object_bbx_mask': mask,                       # 有效框掩码 [max_num]
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,                      # 训练标签
                'cav_num': cav_num,                            # CAV 数量
                'pairwise_t_matrix': pairwise_t_matrix,        # 成对变换矩阵
                'lidar_poses_clean': lidar_poses_clean,        # 真实位姿
                'lidar_poses': lidar_poses                     # 校正后位姿
            })

            # 可视化数据
            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar': projected_lidar_stack})

            # 元数据
            processed_data_dict['ego'].update({
                'sample_idx': idx,
                'cav_id_list': cav_id_list
            })

            return processed_data_dict


        # ====================================================================
        # 批处理整理函数 (训练)
        # ====================================================================
        def collate_batch_train(self, batch):
            """
            ====================================================================
            将多个样本整理成一个 batch (训练模式)
            ====================================================================
            
            【功能说明】
                这是 PyTorch DataLoader 的 collate_fn 参数使用的函数。
                由于协同感知数据格式复杂 (不同样本的 CAV 数量不同)，
                需要自定义批处理逻辑。
            
            【处理流程】
                1. 遍历 batch 中的每个样本
                2. 收集各类数据到列表
                3. 转换为 Tensor 并合并
                4. 组装输出字典
            
            【参数说明】
                batch: list of dict
                    多个样本的列表，每个样本是 __getitem__ 的输出
                    batch_size 由 DataLoader 决定
            
            【返回值】
                output_dict: dict
                    整理后的 batch 数据字典
            """
            output_dict = {'ego': {}}

            # 初始化收集列表
            object_bbx_center = []           # GT 框中心
            object_bbx_mask = []             # GT 框掩码
            object_ids = []                  # 物体 ID
            processed_lidar_list = []        # LiDAR 特征
            image_inputs_list = []           # 相机输入
            record_len = []                  # 每个样本的 CAV 数量
            label_dict_list = []             # 标签字典
            lidar_pose_list = []             # 位姿
            origin_lidar = []                # 原始点云 (可视化)
            lidar_pose_clean_list = []       # 真实位姿
            time_step_list = []              # 时间步
            scenario_folder_list = []        # 场景 ID
            pairwise_t_matrix_list = []      # 成对变换矩阵
            teacher_processed_lidar_list = [] # 教师网络特征

            # 单车监督相关
            if self.supervise_single:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []

            # 遍历 batch 中的每个样本
            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                
                # 收集数据
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                time_step_list.append(ego_dict['timestep'])
                scenario_folder_list.append(ego_dict['scenario_folder'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses'])
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])
                
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs'])
                
                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                if self.kd_flag:
                    teacher_processed_lidar_list.append(ego_dict['teacher_processed_lidar'])

                if self.supervise_single:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])
                    object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                    object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])

            # 转换为 Tensor
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))  # [B, max_num, 7]
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))      # [B, max_num]

            # 合并 LiDAR 特征
            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)
                processed_lidar_torch_dict = self.pre_processor.collate_batch(merged_feature_dict)
                output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

            # 合并相机特征
            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')
                output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            
            # 整理其他数据
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
            label_torch_dict = self.post_processor.collate_batch(label_dict_list)

            # 添加 GT 框到标签字典
            label_torch_dict.update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask
            })

            # 成对变换矩阵 [B, max_cav, max_cav, 4, 4]
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len

            # 组装输出字典
            output_dict['ego'].update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask,
                'record_len': record_len,
                'label_dict': label_torch_dict,
                'object_ids': object_ids[0],
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_pose_clean': lidar_pose_clean,
                'lidar_pose': lidar_pose,
                'anchor_box': self.anchor_box_torch,
                'timestep': time_step_list,
                'scenario_folder': scenario_folder_list
            })

            if self.visualize:
                origin_lidar = downsample_lidar_minimum(pcd_np_list=origin_lidar[0])
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.kd_flag:
                teacher_processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(teacher_processed_lidar_list)
                output_dict['ego'].update({
                    'teacher_processed_lidar': teacher_processed_lidar_torch_dict
                })

            if self.supervise_single:
                output_dict['ego'].update({
                    "label_dict_single": {
                        "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                        "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                        "targets": torch.cat(targets_single, dim=0),
                        "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                        "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                    },
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0)
                })

            return output_dict


        # ====================================================================
        # 批处理整理函数 (测试)
        # ====================================================================
        def collate_batch_test(self, batch):
            """
            ====================================================================
            测试时的批处理函数
            ====================================================================
            
            【与训练模式的区别】
                - batch size 必须为 1
                - 保存额外的元数据 (sample_idx, cav_id_list)
            
            【参数说明】
                batch: list of dict
                    单个样本的列表 (batch_size=1)
            
            【返回值】
                output_dict: dict
                    整理后的数据字典
            """
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            
            output_dict = self.collate_batch_train(batch)
            
            if output_dict is None:
                return None

            # 添加锚框
            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box': self.anchor_box_torch})

            # 单位变换矩阵 (测试时所有预测都在 Ego 坐标系)
            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = torch.from_numpy(np.identity(4)).float()

            output_dict['ego'].update({
                'transformation_matrix': transformation_matrix_torch,
                'transformation_matrix_clean': transformation_matrix_clean_torch,
            })

            output_dict['ego'].update({
                "sample_idx": batch[0]['ego']['sample_idx'],
                "cav_id_list": batch[0]['ego']['cav_id_list']
            })

            return output_dict


        # ====================================================================
        # 后处理函数
        # ====================================================================
        def post_process(self, data_dict, output_dict):
            """
            ====================================================================
            模型输出的后处理函数
            ====================================================================
            
            【功能说明】
                1. 解码模型输出为 3D 检测框
                2. 应用 NMS 去除重复检测
                3. 生成 GT 框用于评估
            
            【参数说明】
                data_dict: dict
                    输入数据字典
                
                output_dict: dict
                    模型输出字典，包含:
                    - 'cls_preds': 分类预测 [B, num_anchors, num_classes]
                    - 'reg_preds': 回归预测 [B, num_anchors, 7]
                    - 'dir_preds': 方向预测 [B, num_anchors, 2]
            
            【返回值】
                pred_box_tensor: 预测框 Tensor
                pred_score: 预测分数
                gt_box_tensor: GT 框 Tensor
            """
            # 后处理流程:
            # 1. 解码回归预测 (将偏移量转换为绝对坐标)
            # 2. 过滤低置信度检测 (score_threshold)
            # 3. NMS 去重 (nms_thresh)
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict, output_dict)
            
            # 生成 GT 框 (用于评估)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor


    return IntermediateFusionDataset
