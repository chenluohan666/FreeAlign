# -*- coding: utf-8 -*-
"""
==============================================================================
DAIR-V2X 基础数据集类 (DAIR-V2X Base Dataset)
==============================================================================

功能概述:
    本文件实现了 DAIR-V2X 数据集的基础数据加载类。
    DAIR-V2X 是真实世界的车路协同感知数据集，包含:
    - 车端 (Vehicle-side): 自动驾驶车辆的传感器数据
    - 路端 (Infrastructure-side): 路侧单元 (RSU) 的传感器数据

数据集特点:
    1. 真实世界数据 (非仿真)
    2. 车路协同场景 (V2I)
    3. 包含 LiDAR 和相机两种传感器
    4. 提供精确的标定参数

数据目录结构:
    dataset/my_dair_v2x/v2x_c/
    ├── vehicle-side/              # 车端数据
    │   ├── data/                  # 原始数据
    │   ├── calib/                 # 标定文件
    │   └── label/                 # 标注文件
    ├── infrastructure-side/       # 路端数据
    │   ├── data/
    │   ├── calib/
    │   └── label/
    └── cooperative/               # 协同标注
        └── data_info.json

使用方法:
    # 在 __init__.py 中通过工厂函数使用
    from opencood.data_utils.datasets import build_dataset
    dataset = build_dataset(hypes, visualize=False, train=True)

==============================================================================
"""

# ============================================================================
# 标准库导入
# ============================================================================
import os                              # 操作系统接口，用于文件路径操作
from collections import OrderedDict     # 有序字典，保持数据顺序一致性
import cv2                              # OpenCV，用于图像读取和处理
import h5py                             # HDF5 文件格式支持，用于存储大型数据
import torch                            # PyTorch 深度学习框架
import numpy as np                      # NumPy，数值计算库
from torch.utils.data import Dataset    # PyTorch 数据集基类
from PIL import Image                   # Python 图像库，用于图像处理
import random                           # 随机数生成，用于数据增强

# ============================================================================
# OpenCOOD 内部模块导入
# ============================================================================

# 点云处理工具
import opencood.utils.pcd_utils as pcd_utils

# 数据增强器
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor

# YAML 配置加载
from opencood.hypes_yaml.yaml_utils import load_yaml

# 点云下采样工具
from opencood.utils.pcd_utils import downsample_lidar_minimum

# 相机数据处理工具
from opencood.utils.camera_utils import (
    load_camera_data,           # 加载相机图像数据
    load_intrinsic_DAIR_V2X     # 加载 DAIR-V2X 相机内参
)

# 通用工具函数
from opencood.utils.common_utils import read_json    # JSON 文件读取

# 坐标变换工具函数
from opencood.utils.transformation_utils import (
    tfm_to_pose,                                    # 4x4 变换矩阵转 6DOF 位姿
    rot_and_trans_to_trasnformation_matrix          # 旋转和平移转变换矩阵
)
from opencood.utils.transformation_utils import (
    veh_side_rot_and_trans_to_trasnformation_matrix  # 车端位姿变换矩阵计算
)
from opencood.utils.transformation_utils import (
    inf_side_rot_and_trans_to_trasnformation_matrix  # 路端位姿变换矩阵计算
)

# 预处理器和后处理器构建函数
from opencood.data_utils.pre_processor import build_preprocessor    # 构建预处理器
from opencood.data_utils.post_processor import build_postprocessor  # 构建后处理器


# ============================================================================
# DAIR-V2X 数据集类定义
# ============================================================================

class DAIRV2XBaseDataset(Dataset):
    """
    DAIR-V2X 基础数据集类。
    
    继承自 PyTorch 的 Dataset 类，实现 DAIR-V2X 数据集的加载功能。
    主要功能:
        1. 加载车端和路端的传感器数据 (LiDAR/相机)
        2. 加载标定参数 (内参/外参)
        3. 加载 3D 检测标注
        4. 计算各智能体的位姿
    
    注意:
        - 此类是基础类，通常与 IntermediateFusionDataset 组合使用
        - __getitem__ 方法在此类中为空，由子类实现
    """
    
    def __init__(self, params, visualize, train=True):
        """
        初始化 DAIR-V2X 数据集。
        
        参数:
            params: dict
                配置参数字典，来自 YAML 配置文件
            visualize: bool
                是否启用可视化模式
                True: 保存额外的可视化数据 (如原始点云)
                False: 仅保存训练所需数据
            train: bool
                是否为训练模式
                True: 使用训练集数据
                False: 使用验证/测试集数据
        """
        # ------------------------- 保存基本参数 -------------------------
        self.params = params        # 保存完整的配置参数字典
        self.visualize = visualize  # 保存可视化标志
        self.train = train          # 保存训练/验证标志

        # ------------------------- 构建预处理器 -------------------------
        # 预处理器负责:
        #   1. 点云体素化 (Voxelization)
        #   2. 特征散射到 BEV (Bird's Eye View)
        # 参数来自配置文件的 'preprocess' 部分
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        
        # ------------------------- 构建后处理器 -------------------------
        # 后处理器负责:
        #   1. 生成锚框 (Anchor Boxes)
        #   2. 生成训练标签 (正/负样本分配)
        #   3. NMS 后处理 (推理时)
        # 参数来自配置文件的 'postprocess' 部分
        self.post_processor = build_postprocessor(params["postprocess"], train)
        
        # 设置 GT 框生成方法为基于 IoU 的方法
        # generate_gt_bbx_by_iou: 通过计算预测框与 GT 框的 IoU 来匹配
        self.post_processor.generate_gt_bbx = self.post_processor.generate_gt_bbx_by_iou
        
        # ------------------------- 构建数据增强器 -------------------------
        # 数据增强器负责:
        #   1. 随机翻转 (Random Flip)
        #   2. 随机旋转 (Random Rotation)
        #   3. 随机缩放 (Random Scaling)
        # 参数来自配置文件的 'data_augment' 部分
        self.data_augmentor = DataAugmentor(params['data_augment'], train)

        # ------------------------- 点云裁剪配置 -------------------------
        # clip_pc: 是否对点云进行裁剪
        # True: 根据感知范围裁剪点云
        # False: 不裁剪
        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False

        # ------------------------- 最大 CAV 数量配置 -------------------------
        # max_cav: 每帧最大智能体数量
        # DAIR-V2X 默认为 2 (1 个车端 + 1 个路端)
        # OPV2V 可能有更多智能体
        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2    # 默认值: 2 个智能体 (车 + 路)
        else:
            self.max_cav = params['train_params']['max_cav']

        # ------------------------- 输入源配置 -------------------------
        # 根据配置决定加载哪些传感器数据
        # load_lidar_file: 是否加载 LiDAR 点云
        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        
        # load_camera_file: 是否加载相机图像
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        
        # load_depth_file: 是否加载深度图 (暂不支持)
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        # 断言: 目前不支持深度图输入
        assert self.load_depth_file is False

        # ------------------------- 标签类型配置 -------------------------
        # label_type: 标签来源
        # 'lidar': 使用 LiDAR 标注 (360° 全方位标注)
        # 'camera': 使用相机标注 (仅在相机视野内的标注)
        self.label_type = params['label_type']
        
        # 根据标签类型选择对应的标签生成函数
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                                    else self.generate_object_center_camera

        # ------------------------- 相机数据增强配置 -------------------------
        # 如果加载相机数据，需要读取数据增强配置
        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        # ------------------------- 数据分割路径配置 -------------------------
        # 根据训练/验证模式选择对应的数据分割文件
        if self.train:
            split_dir = params['root_dir']      # 训练集路径
        else:
            split_dir = params['validate_dir']  # 验证/测试集路径

        # ------------------------- 数据根目录 -------------------------
        self.root_dir = params['data_dir']  # DAIR-V2X 数据根目录
        
        # ------------------------- 加载车端帧信息 -------------------------
        # 读取车端数据信息 JSON 文件
        # 包含每个帧的图像路径、批次 ID 等元数据
        v_data_info = read_json(os.path.join(self.root_dir, 'vehicle-side/data_info.json'))

        # 构建帧 ID 到场景 ID 的映射字典
        # v_frame2scene: {帧ID: 场景ID}
        self.v_frame2scene = OrderedDict()
        for v_frame in v_data_info:
            # 从图像路径中提取帧 ID (去除 .jpg 后缀)
            v_frame_id = v_frame["image_path"].split("/")[-1].replace(".jpg", "")
            # 建立帧 ID 到批次 ID (场景 ID) 的映射
            self.v_frame2scene[v_frame_id] = v_frame["batch_id"]

        # ------------------------- 加载数据分割信息 -------------------------
        # split_info: 包含当前分割 (train/val/test) 的帧 ID 列表
        # 例如: ["000001", "000002", ...]
        self.split_info = read_json(split_dir)

        # ------------------------- 加载协同数据信息 -------------------------
        # 读取协同数据信息 JSON 文件
        # 包含车端-路端配对的帧信息、标定路径、标签路径等
        co_datainfo = read_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        
        # 构建车端帧 ID 到协同数据信息的映射字典
        # co_data: {车端帧ID: 协同数据信息}
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            # 从车端图像路径中提取帧 ID
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            # 建立映射
            self.co_data[veh_frame_id] = frame_info

        # ------------------------- 噪声配置初始化 -------------------------
        # 如果配置中没有噪声设置，初始化为不添加噪声
        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False


    def reinitialize(self):
        """
        重新初始化数据集。
        
        此方法在每个 epoch 结束后被调用，用于:
        1. 重新随机化数据增强参数
        2. 重新加载数据索引
        
        DAIR-V2X 数据集是静态的，不需要重新初始化，所以此方法为空。
        """
        pass  # DAIR-V2X 数据集不需要重新初始化


    def retrieve_base_data(self, idx):
        """
        根据索引获取基础数据。
        
        这是数据加载的核心方法，负责从磁盘加载一个样本的所有原始数据。
        
        注意:
            此方法与 Intermediate Fusion 和 Early Fusion 不同，
            标签不是协同的，而是分别为车端和路端加载各自的标签。
        
        参数:
            idx: int
                数据加载器提供的样本索引
        
        返回:
            data: OrderedDict
                包含加载的 YAML 参数和每个 CAV 的 LiDAR 数据的字典
                
        数据结构:
            data = {
                'scenario_folder': 场景ID (如 "batch_1"),
                'timestep': 帧ID (如 "000001"),
                0: {  # 车端数据 (Ego)
                    'ego': True,
                    'params': {
                        'lidar_pose': [x, y, z, roll, pitch, yaw],  # 6DOF 位姿
                        'vehicles_all': [...],    # 360° 标注
                        'vehicles_front': [...],  # 相机视野内标注
                        ...
                    },
                    'lidar_np': np.ndarray [N, 4],  # 点云数据
                    'camera_data': [...]            # 相机图像 (可选)
                },
                1: {  # 路端数据 (RSU)
                    'ego': False,
                    'params': {...},
                    'lidar_np': np.ndarray [N, 4],
                    ...
                }
            }
        """
        # ====================================================================
        # 第一步: 获取帧信息
        # ====================================================================
        
        # 根据 idx 从分割信息中获取车端帧 ID
        veh_frame_id = self.split_info[idx]  # 例如 "000001"
        
        # 从协同数据中获取该帧的完整信息
        # 包含: 车端/路端图像路径、点云路径、标签路径、标定路径等
        frame_info = self.co_data[veh_frame_id]
        
        # 系统误差偏移量 (路端与车端之间的时间/空间偏差)
        # 用于修正路端位姿
        system_error_offset = frame_info["system_error_offset"]
        
        # ====================================================================
        # 第二步: 初始化输出数据字典
        # ====================================================================
        
        data = OrderedDict()  # 使用有序字典保持数据顺序
        
        # 保存场景 ID (批次 ID)
        data['scenario_folder'] = self.v_frame2scene[veh_frame_id]
        
        # 保存时间步 (帧 ID)
        data['timestep'] = veh_frame_id
        
        # 初始化车端数据 (ID=0, 作为 Ego)
        data[0] = OrderedDict()
        data[0]['ego'] = True  # 标记为 Ego 车辆
        
        # 初始化路端数据 (ID=1, 作为 RSU)
        data[1] = OrderedDict()
        data[1]['ego'] = False  # 标记为非 Ego
        
        # 初始化参数字典
        data[0]['params'] = OrderedDict()
        data[1]['params'] = OrderedDict()
        
        # ====================================================================
        # 第三步: 加载车端位姿
        # ====================================================================
        
        # 读取 LiDAR 到 Novatel (GPS/IMU) 的标定
        # lidar_to_novatel: LiDAR 坐标系到 Novatel 坐标系的变换
        lidar_to_novatel = read_json(
            os.path.join(self.root_dir, 'vehicle-side/calib/lidar_to_novatel/' + str(veh_frame_id) + '.json')
        )
        
        # 读取 Novatel 到世界坐标系的变换
        # novatel_to_world: Novatel 坐标系到世界坐标系的变换 (由 GPS/IMU 提供)
        novatel_to_world = read_json(
            os.path.join(self.root_dir, 'vehicle-side/calib/novatel_to_world/' + str(veh_frame_id) + '.json')
        )
        
        # 计算车端的变换矩阵 (LiDAR 坐标系 → 世界坐标系)
        # transformation_matrix: 4x4 齐次变换矩阵
        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel, novatel_to_world)
        
        # 将变换矩阵转换为 6DOF 位姿 [x, y, z, roll, pitch, yaw]
        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)
        
        # ====================================================================
        # 第四步: 加载路端位姿
        # ====================================================================
        
        # 从协同数据中获取路端帧 ID
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        
        # 读取虚拟 LiDAR 到世界坐标系的变换
        # virtuallidar_to_world: 路端 LiDAR 坐标系到世界坐标系的变换
        virtuallidar_to_world = read_json(
            os.path.join(self.root_dir, 'infrastructure-side/calib/virtuallidar_to_world/' + str(inf_frame_id) + '.json')
        )
        
        # 计算路端的变换矩阵 (考虑系统误差偏移)
        transformation_matrix = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world, system_error_offset)
        
        # 将变换矩阵转换为 6DOF 位姿
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)
        
        # ====================================================================
        # 第五步: 加载协同标注 (GT 检测框)
        # ====================================================================
        
        # 加载车端的前方车辆标注 (仅相机视野内)
        # 注意: 使用 label_world_backup 作为 vehicles_front
        data[0]['params']['vehicles_front'] = read_json(
            os.path.join(self.root_dir, frame_info['cooperative_label_path'].replace("label_world", "label_world_backup"))
        )
        
        # 加载车端的全方位标注 (360°)
        # 这是 FreeAlign 补充标注后的完整标注
        data[0]['params']['vehicles_all'] = read_json(
            os.path.join(self.root_dir, frame_info['cooperative_label_path'])
        )
        
        # 路端不加载协同标注 (协同标注统一在车端管理)
        data[1]['params']['vehicles_front'] = []  # 空列表
        data[1]['params']['vehicles_all'] = []    # 空列表
        
        # ====================================================================
        # 第六步: 加载相机数据 (可选)
        # ====================================================================
        
        if self.load_camera_file:
            # ------------------------- 车端相机数据 -------------------------
            # 加载车端相机图像
            data[0]['camera_data'] = load_camera_data(
                [os.path.join(self.root_dir, frame_info["vehicle_image_path"])]
            )
            
            # 初始化车端相机参数字典
            data[0]['params']['camera0'] = OrderedDict()
            
            # 加载车端相机外参 (LiDAR 到相机的变换矩阵)
            data[0]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix(
                read_json(os.path.join(self.root_dir, 'vehicle-side/calib/lidar_to_camera/' + str(veh_frame_id) + '.json'))
            )
            
            # 加载车端相机内参 (焦距、光心等)
            data[0]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X(
                read_json(os.path.join(self.root_dir, 'vehicle-side/calib/camera_intrinsic/' + str(veh_frame_id) + '.json'))
            )
            
            # ------------------------- 路端相机数据 -------------------------
            # 加载路端相机图像
            data[1]['camera_data'] = load_camera_data(
                [os.path.join(self.root_dir, frame_info["infrastructure_image_path"])]
            )
            
            # 初始化路端相机参数字典
            data[1]['params']['camera0'] = OrderedDict()
            
            # 加载路端相机外参
            data[1]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix(
                read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/virtuallidar_to_camera/' + str(inf_frame_id) + '.json'))
            )
            
            # 加载路端相机内参
            data[1]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X(
                read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/camera_intrinsic/' + str(inf_frame_id) + '.json'))
            )
        
        # ====================================================================
        # 第七步: 加载 LiDAR 点云数据 (可选)
        # ====================================================================
        
        if self.load_lidar_file or self.visualize:
            # 加载车端 LiDAR 点云
            # lidar_np: numpy 数组，形状 [N, 4] (x, y, z, intensity)
            data[0]['lidar_np'], _ = pcd_utils.read_pcd(
                os.path.join(self.root_dir, frame_info["vehicle_pointcloud_path"])
            )
            
            # 加载路端 LiDAR 点云
            data[1]['lidar_np'], _ = pcd_utils.read_pcd(
                os.path.join(self.root_dir, frame_info["infrastructure_pointcloud_path"])
            )
        
        # ====================================================================
        # 第八步: 加载单车端标注 (用于单车监督)
        # ====================================================================
        
        # 车端单车标注 (前方)
        data[0]['params']['vehicles_single_front'] = read_json(
            os.path.join(self.root_dir, 'vehicle-side/label/lidar_backup/{}.json'.format(veh_frame_id))
        )
        
        # 车端单车标注 (全方位)
        data[0]['params']['vehicles_single_all'] = read_json(
            os.path.join(self.root_dir, 'vehicle-side/label/lidar/{}.json'.format(veh_frame_id))
        )
        
        # 路端单车标注 (前方) - 与全方位相同
        data[1]['params']['vehicles_single_front'] = read_json(
            os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id))
        )
        
        # 路端单车标注 (全方位)
        data[1]['params']['vehicles_single_all'] = read_json(
            os.path.join(self.root_dir, 'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id))
        )
        
        return data  # 返回完整的数据字典


    def __len__(self):
        """
        返回数据集的样本数量。
        
        返回:
            int: 当前分割 (train/val/test) 中的样本数量
        """
        return len(self.split_info)


    def __getitem__(self, idx):
        """
        根据索引获取一个样本。
        
        注意:
            此方法在基础类中为空，由子类 (如 IntermediateFusionDataset) 实现。
            这样设计是为了支持不同的融合策略:
            - IntermediateFusionDataset: 中间特征融合
            - EarlyFusionDataset: 早期数据融合
            - LateFusionDataset: 后期决策融合
        
        参数:
            idx: int
                样本索引
        
        返回:
            由子类实现
        """
        pass  # 由子类实现


    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        """
        生成 LiDAR 标签的物体中心坐标。
        
        使用 360° 全方位的 LiDAR 标注生成 GT 检测框。
        
        参数:
            cav_contents: list
                CAV 内容列表，每个元素是一个 CAV 的数据字典
            reference_lidar_pose: list
                参考坐标系 (通常是 Ego) 的 LiDAR 位姿 [x, y, z, roll, pitch, yaw]
        
        返回:
            object_bbx_center: np.ndarray
                GT 检测框中心坐标 [N, 7] (x, y, z, l, w, h, yaw)
            object_bbx_mask: np.ndarray
                有效框掩码 [N]
            object_ids: list
                物体 ID 列表
        """
        # 将全方位标注赋值给 'vehicles' 字段
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_all']
        
        # 调用后处理器生成物体中心
        return self.post_processor.generate_object_center_dairv2x(cav_contents, reference_lidar_pose)


    def generate_object_center_camera(self, cav_contents, reference_lidar_pose):
        """
        生成相机标签的物体中心坐标。
        
        仅使用相机视野内的标注生成 GT 检测框。
        
        参数:
            cav_contents: list
                CAV 内容列表
            reference_lidar_pose: list
                参考坐标系的 LiDAR 位姿
        
        返回:
            object_bbx_center: np.ndarray
                GT 检测框中心坐标 [N, 7]
            object_bbx_mask: np.ndarray
                有效框掩码 [N]
            object_ids: list
                物体 ID 列表
        """
        # 将前方标注赋值给 'vehicles' 字段
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_front']
        
        # 调用后处理器生成物体中心
        return self.post_processor.generate_object_center_dairv2x(cav_contents, reference_lidar_pose)


    def generate_object_center_single(self, cav_contents, reference_lidar_pose, **kwargs):
        """
        生成单车端的物体中心坐标。
        
        用于单车监督训练 (DiscoNet 等方法)。
        标注在各自 CAV 的坐标系中，而非 Ego 坐标系。
        
        参数:
            cav_contents: list
                CAV 内容列表
            reference_lidar_pose: list
                参考坐标系的 LiDAR 位姿 (此处不使用)
            **kwargs: dict
                额外参数
        
        返回:
            object_bbx_center: np.ndarray
                GT 检测框中心坐标
            object_bbx_mask: np.ndarray
                有效框掩码
            object_ids: list
                物体 ID 列表
        """
        suffix = "_single"  # 后缀，用于区分单车标注
        
        # 根据标签类型选择对应的单车标注
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if self.label_type == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        
        # 调用后处理器生成单车物体中心
        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)


    def get_ext_int(self, params, camera_id):
        """
        获取相机的外参和内参矩阵。
        
        参数:
            params: dict
                相机参数字典，包含 'camera0', 'camera1' 等
            camera_id: int
                相机 ID (0 表示第一个相机)
        
        返回:
            camera_to_lidar: np.ndarray
                相机到 LiDAR 的变换矩阵 [4, 4]
            camera_intrinsic: np.ndarray
                相机内参矩阵 [3, 3]
        """
        # 获取 LiDAR 到相机的变换矩阵 (外参)
        # R_cw: 相机坐标系到世界坐标系的旋转 (此处世界坐标系即 LiDAR 坐标系)
        lidar_to_camera = params["camera%d" % camera_id]['extrinsic'].astype(np.float32)
        
        # 计算相机到 LiDAR 的变换矩阵 (外参的逆)
        # R_wc: 世界坐标系 (LiDAR) 到相机坐标系的变换
        camera_to_lidar = np.linalg.inv(lidar_to_camera)
        
        # 获取相机内参矩阵
        # 包含: 焦距 (fx, fy), 光心 (cx, cy), 畸变参数等
        camera_intrinsic = params["camera%d" % camera_id]['intrinsic'].astype(np.float32)
        
        return camera_to_lidar, camera_intrinsic


    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        数据增强函数。
        
        对点云和检测框进行随机增强，包括:
        1. 随机翻转 (沿 x/y 轴)
        2. 随机旋转
        3. 随机缩放
        
        参数:
            lidar_np: np.ndarray
                点云数据，形状 [N, 4] (x, y, z, intensity)
            object_bbx_center: np.ndarray
                GT 检测框中心，形状 [N, 7] (x, y, z, l, w, h, yaw)
            object_bbx_mask: np.ndarray
                有效框掩码，形状 [N]
        
        返回:
            lidar_np: np.ndarray
                增强后的点云数据
            object_bbx_center: np.ndarray
                增强后的 GT 检测框中心
            object_bbx_mask: np.ndarray
                有效框掩码 (不变)
        """
        # 将数据打包成字典
        tmp_dict = {
            'lidar_np': lidar_np,              # 点云
            'object_bbx_center': object_bbx_center,  # 检测框
            'object_bbx_mask': object_bbx_mask       # 掩码
        }
        
        # 调用数据增强器进行增强
        tmp_dict = self.data_augmentor.forward(tmp_dict)
        
        # 解包增强后的数据
        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']
        
        return lidar_np, object_bbx_center, object_bbx_mask
