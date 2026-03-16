# -*- coding: utf-8 -*-
"""
==============================================================================
中间融合数据集 (Intermediate Fusion Dataset)
==============================================================================

功能概述:
    本文件实现了协同感知中最重要的中间融合数据集类，是 FreeAlign/CoAlign 的核心数据处理模块。

核心功能:
    1. 加载多智能体 (CAV) 的传感器数据 (LiDAR/相机)
    2. 坐标变换: 将所有 CAV 的数据投影到 Ego 车辆坐标系
    3. FreeAlign 位姿校正: 使用图匹配算法校正位姿误差
    4. 数据预处理: 体素化、数据增强、标签生成
    5. 批处理整理: 将多个样本整理成一个 batch

数据流程:
    原始数据 → 噪声添加 → FreeAlign位姿校正 → 单车处理 → 批处理整理

继承关系:
    BaseDataset (基础数据集)
        └── IntermediateFusionDataset (中间融合数据集)
                ├── OPV2VBaseDataset
                ├── DAIRV2XBaseDataset
                └── V2XSimBaseDataset

使用方法:
    from opencood.data_utils.datasets import build_dataset
    dataset = build_dataset(hypes, visualize=False, train=True)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_batch_train)

==============================================================================
"""

# ============================================================================
# 导入依赖
# ============================================================================

import random
import math
from collections import OrderedDict   # 有序字典，保持 CAV 顺序
import numpy as np
import torch
import copy
from icecream import ic               # 调试打印工具
from PIL import Image
import pickle as pkl

# ============================================================================
# FreeAlign 核心模块导入
# ============================================================================

# FreeAlign 位姿校正算法
from freealign.match.match_v7_with_detection import get_pose_rotation
from freealign.match.match_v7_debug import get_right_box
from freealign.match.match_v7_with_detection import freealign_training

# ============================================================================
# OpenCOOD 工具模块导入
# ============================================================================

from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor   # 预处理器
from opencood.data_utils.post_processor import build_postprocessor # 后处理器

# 相机数据处理工具
from opencood.utils.camera_utils import (
    sample_augmentation,   # 数据增强参数采样
    img_transform,         # 图像变换
    normalize_img,         # 图像归一化
    img_to_tensor,         # 图像转 Tensor
)

from opencood.utils.heter_utils import AgentSelector   # 异构智能体选择器
from opencood.utils.common_utils import merge_features_to_dict   # 特征合并
from opencood.utils.transformation_utils import (
    x1_to_x2,                    # 计算两个坐标系之间的变换矩阵
    x_to_world,                  # 局部坐标到世界坐标
    get_pairwise_transformation, # 获取成对变换矩阵
    pose_to_tfm                  # 6DOF 位姿转 4x4 变换矩阵
)
from opencood.utils.pose_utils import add_noise_data_dict   # 添加位姿噪声

# 点云处理工具
from opencood.utils.pcd_utils import (
    mask_points_by_range,       # 按范围过滤点云
    mask_ego_points,            # 移除打到自身的点
    shuffle_points,             # 打乱点云顺序
    downsample_lidar_minimum,   # 最小下采样
)

from opencood.utils.common_utils import read_json
from opencood.utils.box_utils import corner_to_center, boxes_to_corners_3d, project_box3d


# ============================================================================
# 数据集类工厂函数
# ============================================================================

def getIntermediateFusionDataset(cls):
    """
    中间融合数据集类的工厂函数。

    使用装饰器模式，将基础数据集类包装成中间融合数据集类。
    这样可以在不同数据集 (OPV2V, DAIR-V2X, V2X-Sim) 上复用相同的中间融合逻辑。

    参数:
        cls: BaseDataset 子类，如 DAIRV2XBaseDataset, OPV2VBaseDataset

    返回:
        IntermediateFusionDataset: 包装后的中间融合数据集类

    示例:
        # 在 __init__.py 中使用
        base_cls = DAIRV2XBaseDataset
        dataset_cls = getIntermediateFusionDataset(base_cls)
        dataset = dataset_cls(hypes, visualize=False, train=True)
    """
    class IntermediateFusionDataset(cls):
        # ====================================================================
        # 初始化函数
        # ====================================================================
        def __init__(self, params, visualize, train=True):
            """
            初始化中间融合数据集。

            参数:
                params: 配置字典，来自 YAML 文件
                visualize: 是否启用可视化模式
                train: 是否为训练模式 (True=训练, False=验证/测试)
            """
            # 调用父类初始化 (BaseDataset)
            # 父类会初始化数据路径、预处理器、后处理器等
            super().__init__(params, visualize, train)

            # ------------------------- 融合参数 -------------------------
            # supervise_single: 是否使用单车监督损失 (DiscoNet 等方法需要)
            self.supervise_single = True if ('supervise_single' in params['model']['args'] 
                                             and params['model']['args']['supervise_single']) \
                                    else False
            
            # proj_first: 是否先将点云投影到 Ego 坐标系再处理
            # True: 2-round 通信模式 (先变换点云)
            # False: 1-round 通信模式 (变换特征图，默认)
            self.proj_first = False if 'proj_first' not in params['fusion']['args'] \
                                    else params['fusion']['args']['proj_first']

            # ------------------------- 锚框生成 -------------------------
            # 生成预定义的锚框，用于目标检测
            # anchor_box: numpy 数组，形状 [N_anchors, 7] (x,y,z,l,w,h,yaw)
            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            # ------------------------- 知识蒸馏标志 -------------------------
            # kd_flag: 是否使用知识蒸馏 (DiscoNet 需要)
            self.kd_flag = params.get('kd_flag', False)

            # ------------------------- FreeAlign 位姿校正参数 ★ -------------------------
            # 这是 FreeAlign 的核心配置，用于在无外部定位设备时校正位姿
            self.box_align = False
            
            if "box_align" in params:
                self.box_align = True
                
                # Stage1 检测结果路径 (预计算的检测框)
                # 训练时使用 train_result，验证时使用 val_result
                self.stage1_result_path = params['box_align']['train_result'] if train \
                                          else params['box_align']['val_result']
                self.stage1_result = read_json(self.stage1_result_path)
                
                # 位姿图优化参数
                self.box_align_args = params['box_align']['args']
                
                # no_pose: True 表示完全不使用外部位姿 (FreeAlign 核心特性)
                #          False 表示使用带噪声的外部位姿 (CoAlign 行为)
                self.no_pose = params['box_align']['no_pose']
                
                # MASS 算法参数
                self.min_anchor = params['box_align']['min_anchor']      # 最小锚点数量
                self.anchor_error = params['box_align']['anchor_error']  # 锚点匹配误差阈值 (米)
                self.box_error = params['box_align']['box_error']        # 检测框匹配误差阈值 (米)
                
                # GNN 边特征学习 (可选)
                self.use_gnn = params['box_align']['gnn']
                self.gt_correct = params['box_align']['gt_correct']
                
                if self.use_gnn:
                    self.gnn_error = params['box_align']['gnn_error']
                    self.gnn_extractor = torch.load(params['box_align']['gnn_model_path'])


        # ====================================================================
        # 单车数据处理函数
        # ====================================================================
        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            处理单个 CAV (Connected Autonomous Vehicle) 的数据。

            这是对每个智能体数据的核心处理函数，包括:
            1. 计算变换矩阵 (CAV 坐标系 → Ego 坐标系)
            2. 点云预处理 (打乱、过滤、体素化)
            3. 相机数据处理 (可选)
            4. 标签生成 (GT 框)

            参数:
                selected_cav_base: dict
                    单个 CAV 的原始数据字典，包含:
                    - 'params': 位姿、传感器参数等元数据
                    - 'lidar_np': 点云数据 [N, 4] (x,y,z,intensity)
                    - 'camera_data': 相机图像列表 (可选)
                
                ego_cav_base: dict
                    Ego 车辆的数据字典，用于获取 Ego 位姿

            返回:
                selected_cav_processed: dict
                    处理后的数据字典，包含:
                    - 'processed_features': 体素化后的点云特征
                    - 'object_bbx_center': GT 检测框中心 [N, 7]
                    - 'transformation_matrix': 变换矩阵 [4, 4]
                    - 'image_inputs': 相机输入 (可选)
            """
            selected_cav_processed = {}
            
            # ------------------------- 提取位姿信息 -------------------------
            # lidar_pose: 可能带有噪声的位姿 (用于特征变换)
            # lidar_pose_clean: 真实位姿 (用于生成 GT 标签)
            ego_pose = ego_cav_base['params']['lidar_pose']
            ego_pose_clean = ego_cav_base['params']['lidar_pose_clean']

            # ------------------------- 计算变换矩阵 -------------------------
            # transformation_matrix: T_ego_cav，将 CAV 坐标系的点变换到 Ego 坐标系
            # 使用带噪声的位姿 (模拟真实场景)
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose)
            
            # transformation_matrix_clean: 使用真实位姿计算的变换矩阵
            # 仅用于生成 GT 标签，确保标签质量
            transformation_matrix_clean = \
                x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                        ego_pose_clean)
            
            # ------------------------- LiDAR 点云处理 -------------------------
            if self.load_lidar_file or self.visualize:
                # 1. 获取原始点云
                lidar_np = selected_cav_base['lidar_np']  # [N, 4]
                
                # 2. 打乱点云顺序 (数据增强，防止过拟合)
                lidar_np = shuffle_points(lidar_np)
                
                # 3. 移除打到车辆自身的点 (避免自遮挡噪声)
                lidar_np = mask_ego_points(lidar_np)
                
                # 4. 将点云投影到 Ego 坐标系
                #    projected_lidar: [N, 3] (x, y, z in ego frame)
                projected_lidar = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                             transformation_matrix)
                
                # 5. 如果 proj_first=True，先用投影后的点云坐标
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar

                # 保存投影后的点云用于可视化
                if self.visualize:
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                # 知识蒸馏模式下的点云处理
                if self.kd_flag:
                    lidar_proj_np = copy.deepcopy(lidar_np)
                    lidar_proj_np[:,:3] = projected_lidar
                    selected_cav_processed.update({'projected_lidar': lidar_proj_np})

                # 6. 预处理点云 (体素化)
                #    输出: voxel_features, voxel_coords, voxel_num_points
                processed_lidar = self.pre_processor.preprocess(lidar_np)
                selected_cav_processed.update({'processed_features': processed_lidar})

            # ------------------------- 生成单车 GT 标签 -------------------------
            # 注意: 参考坐标系是该 CAV 自身 (用于单车监督)
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], selected_cav_base['params']['lidar_pose']
            )
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, 
                anchors=self.anchor_box, 
                mask=object_bbx_mask
            )
            
            # 保存单车标签 (用于 supervise_single 模式)
            selected_cav_processed.update({
                "single_label_dict": label_dict,
                "single_object_bbx_center": object_bbx_center,
                "single_object_bbx_mask": object_bbx_mask
            })

            # ------------------------- 相机数据处理 (可选) -------------------------
            if self.load_camera_file:
                camera_data_list = selected_cav_base["camera_data"]
                params = selected_cav_base["params"]
                
                # 初始化相机数据列表
                imgs = []
                rots = []          # 旋转矩阵
                trans = []         # 平移向量
                intrins = []       # 内参矩阵
                extrinsics = []    # 外参矩阵
                post_rots = []     # 数据增强后的旋转
                post_trans = []    # 数据增强后的平移

                # 遍历所有相机
                for idx, img in enumerate(camera_data_list):
                    # 获取相机外参和内参
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(camera_to_lidar[:3, :3])   # R_wc
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])   # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # 加载深度图 (可选)
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # 数据增强: 随机 resize, crop, flip, rotate
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        self.data_aug_conf, self.train
                    )
                    
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src, post_rot, post_tran,
                        resize=resize, resize_dims=resize_dims,
                        crop=crop, flip=flip, rotate=rotate,
                    )
                    
                    # 构造 3x3 增强矩阵
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # 图像归一化和转 Tensor
                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    # 收集数据
                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)

                # 整理相机输入字典
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

            # ------------------------- 锚框 -------------------------
            selected_cav_processed.update({"anchor_box": self.anchor_box})

            # ------------------------- 生成协同 GT 标签 -------------------------
            # 注意: 参考坐标系是 Ego (用于协同检测)
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], ego_pose_clean
            )

            selected_cav_processed.update({
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
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
            PyTorch Dataset 的核心方法，根据索引获取一个样本。

            执行流程:
                1. 加载原始数据 (点云、相机、位姿、标签)
                2. 添加位姿噪声 (模拟真实场景)
                3. FreeAlign 位姿校正 ★ (核心创新)
                4. 处理每个 CAV 的数据
                5. 合并多 CAV 数据
                6. 生成标签

            参数:
                idx: 样本索引

            返回:
                processed_data_dict: 处理后的数据字典
            """
            # ------------------------- 第一步: 加载原始数据 -------------------------
            # retrieve_base_data 由父类实现，从磁盘加载数据
            base_data_dict = self.retrieve_base_data(idx)
            
            # 提取元数据
            timestep = base_data_dict.pop('timestep', None)
            scenario_folder = base_data_dict.pop('scenario_folder', None)
            
            # 删除非数字键 (清理数据字典)
            del_key_list = []
            for key in base_data_dict.keys():
                try:
                    int(key)
                except:
                    del_key_list.append(key)
            for del_key in del_key_list:
                del base_data_dict[del_key]
            
            # ------------------------- 第二步: 添加位姿噪声 -------------------------
            # 模拟真实场景中的位姿误差 (GPS/RTK 噪声)
            base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])

            # 初始化输出字典
            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}
            processed_data_dict['ego']['timestep'] = timestep
            processed_data_dict['ego']['scenario_folder'] = scenario_folder

            # ------------------------- 第三步: 找到 Ego 车辆 -------------------------
            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # 遍历所有 CAV，找到标记为 ego 的车辆
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break
            
            # 断言检查
            assert cav_id == list(base_data_dict.keys())[0], \
                "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            # ------------------------- 初始化数据收集列表 -------------------------
            agents_image_inputs = []
            processed_features = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            too_far = []             # 超出通信范围的 CAV
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []
            projected_lidar_clean_list = []

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []

            # ------------------------- 第四步: 筛选通信范围内的 CAV -------------------------
            for cav_id, selected_cav_base in base_data_dict.items():
                # 计算该 CAV 与 Ego 的距离
                distance = math.sqrt(
                    (selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2 +
                    (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2
                )

                # 超出通信范围则跳过
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
            # 第五步: FreeAlign 位姿校正 ★ (核心创新点)
            # =================================================================
            # 使用 Stage1 检测框进行图匹配，校正位姿误差
            if self.box_align and str(idx) in self.stage1_result.keys():
                from opencood.models.sub_modules.box_align_v2 import box_alignment_relative_sample_np
                
                stage1_content = self.stage1_result[str(idx)]
                
                if stage1_content is not None:
                    # 获取所有智能体的 Stage1 检测结果
                    all_agent_id_list = stage1_content['cav_id_list']
                    all_agent_corners_list = stage1_content['pred_corner3d_np_list']
                    all_agent_uncertainty_list = stage1_content['uncertainty_np_list']

                    # 获取当前通信范围内的智能体
                    cur_agent_id_list = cav_id_list
                    cur_agent_pose_list = [base_data_dict[cav_id]['params']['lidar_pose'] 
                                           for cav_id in cav_id_list]
                    cur_agent_pose = np.array(cur_agent_pose_list)
                    
                    # 索引当前智能体在 all_agent_list 中的位置
                    cur_agent_in_all_agent = [all_agent_id_list.index(cur_agent) 
                                              for cur_agent in cur_agent_id_list]

                    # 提取当前智能体的检测框和不确定性
                    pred_corners_list = [np.array(all_agent_corners_list[cur_in_all_ind], dtype=np.float64) 
                                         for cur_in_all_ind in cur_agent_in_all_agent]
                    uncertainty_list = [np.array(all_agent_uncertainty_list[cur_in_all_ind], dtype=np.float64) 
                                        for cur_in_all_ind in cur_agent_in_all_agent]

                    gt_pose = pose_to_tfm(cur_agent_pose)
                    
                    # --------------------- 关键: no_pose 模式 ---------------------
                    # no_pose=True: 完全不使用外部位姿，初始化为零
                    # 这是 FreeAlign 的核心特性，实现无外部定位的对齐
                    if self.no_pose:
                        cur_agent_pose = np.zeros((len(cur_agent_id_list), 6))
                        cur_agent_pose[:,4] = 1.0  # 初始 yaw 角
                    else:
                        # 使用带噪声的外部位姿 (CoAlign 行为)
                        cur_agent_pose = np.array(copy.deepcopy(
                            [base_data_dict[cav_id]['params']['lidar_pose'] 
                             for cav_id in cav_id_list]))

                    # GT 校正模式 (用于调试/消融实验)
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
                    # 调用 box_alignment_relative_sample_np 进行位姿校正
                    # 算法流程:
                    #   1. MASS 算法找匹配的检测框对
                    #   2. 使用 SVD/ICP 计算相对位姿
                    #   3. 位姿图优化 (g2o)
                    if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(
                            pred_corners_list,
                            cur_agent_pose,
                            uncertainty_list=uncertainty_list,
                            **self.box_align_args
                        )
                        
                        # 更新校正后的位姿 (只更新 x, y, yaw)
                        cur_agent_pose[:,[0,1,4]] = refined_pose

                        # 将校正后的位姿写回数据字典
                        for i, cav_id in enumerate(cav_id_list):
                            lidar_pose_list[i] = cur_agent_pose[i].tolist()
                            base_data_dict[cav_id]['params']['lidar_pose'] = cur_agent_pose[i].tolist()

                # 移除校正失败的 CAV
                indices_to_delete = []
                for i, cav_id in enumerate(cav_id_list):
                    if not cur_agent_pose[i].any():
                        base_data_dict.pop(cav_id)
                        indices_to_delete.append(i)

                new_list = [element for index, element in enumerate(cav_id_list) 
                            if index not in indices_to_delete]
                cav_id_list = new_list

            # ------------------------- 第六步: 计算成对变换矩阵 -------------------------
            # pairwise_t_matrix: [max_cav, max_cav, 4, 4]
            # 表示每对 CAV 之间的坐标变换
            pairwise_t_matrix = get_pairwise_transformation(
                base_data_dict, self.max_cav, self.proj_first
            )

            # 整理位姿数组
            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)      # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)
            
            cav_num = len(cav_id_list)

            # ------------------------- 第七步: 处理每个 CAV -------------------------
            for _, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                
                # 调用单车处理函数
                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base, ego_cav_base)
                
                # 收集处理结果
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                
                if self.load_lidar_file:
                    processed_features.append(selected_cav_processed['processed_features'])
                
                if self.load_camera_file:
                    agents_image_inputs.append(selected_cav_processed['image_inputs'])

                if self.visualize or self.kd_flag:
                    projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
                
                if self.supervise_single:
                    single_label_list.append(selected_cav_processed['single_label_dict'])
                    single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])

            # ------------------------- 第八步: 整理单车监督标签 -------------------------
            if self.supervise_single:
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
                stack_lidar_np = np.vstack(projected_lidar_stack)
                stack_lidar_np = mask_points_by_range(
                    stack_lidar_np, self.params['preprocess']['cav_lidar_range'])
                stack_feature_processed = self.pre_processor.preprocess(stack_lidar_np)
                processed_data_dict['ego'].update({
                    'teacher_processed_lidar': stack_feature_processed
                })

            # ------------------------- 第九步: 去重 GT 目标 -------------------------
            # 多个 CAV 可能观测到同一目标，需要去重
            unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]

            # ------------------------- 第十步: Padding 到固定大小 -------------------------
            # 确保所有样本的 GT 框数量一致，便于批处理
            object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            # ------------------------- 第十一步: 合并特征 -------------------------
            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_features)
                processed_data_dict['ego'].update({'processed_lidar': merged_feature_dict})
            
            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
                processed_data_dict['ego'].update({'image_inputs': merged_image_inputs_dict})

            # ------------------------- 第十二步: 生成标签 -------------------------
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=self.anchor_box,
                mask=mask
            )

            # ------------------------- 第十三步: 组装输出字典 -------------------------
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

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar': projected_lidar_stack})

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
            将多个样本整理成一个 batch。

            这是 PyTorch DataLoader 的 collate_fn 参数使用的函数。
            由于协同感知数据格式复杂 (不同样本的 CAV 数量不同)，
            需要自定义批处理逻辑。

            参数:
                batch: list of dict
                    多个样本的列表，每个样本是 __getitem__ 的输出

            返回:
                output_dict: dict
                    整理后的 batch 数据字典，包含:
                    - 'processed_lidar': 体素化特征
                    - 'record_len': 每个样本的 CAV 数量
                    - 'pairwise_t_matrix': 成对变换矩阵
                    - 'label_dict': 训练标签
                    - 'object_bbx_center': GT 框
                    - 'object_bbx_mask': GT 框掩码
            """
            output_dict = {'ego': {}}

            # 初始化收集列表
            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            image_inputs_list = []
            record_len = []               # 每个样本的 CAV 数量
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []
            lidar_pose_clean_list = []
            time_step_list = []
            scenario_folder_list = []
            pairwise_t_matrix_list = []
            teacher_processed_lidar_list = []

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
            测试时的批处理函数。

            与训练时的区别:
                - batch size 必须为 1
                - 保存额外的元数据 (sample_idx, cav_id_list)

            参数:
                batch: list of dict
                    单个样本的列表 (batch_size=1)

            返回:
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
            模型输出的后处理函数。

            功能:
                1. 解码模型输出为 3D 检测框
                2. 应用 NMS 去除重复检测
                3. 生成 GT 框用于评估

            参数:
                data_dict: dict
                    输入数据字典
                output_dict: dict
                    模型输出字典，包含:
                    - 'cls_preds': 分类预测
                    - 'reg_preds': 回归预测
                    - 'dir_preds': 方向预测

            返回:
                pred_box_tensor: 预测框 Tensor
                pred_score: 预测分数
                gt_box_tensor: GT 框 Tensor
            """
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict, output_dict)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor


    return IntermediateFusionDataset