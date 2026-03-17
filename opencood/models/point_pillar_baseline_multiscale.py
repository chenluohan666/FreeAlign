# -*- coding: utf-8 -*-
"""
==============================================================================
        PointPillar 多尺度融合模型 (PointPillar Baseline Multiscale)
==============================================================================

【作者信息】
    Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

【功能概述】
    本文件实现了一个集成多种简单融合方法的多尺度协同感知模型。
    基于 PointPillar 检测器，支持多种特征融合策略。

【支持的融合方法】
    1. F-Cooper (Max Fusion): 最大值融合，取各 CAV 特征的最大值
    2. Self-Att (Attention Fusion): 注意力融合，学习各 CAV 特征的权重
    3. DiscoNet (无知识蒸馏): 基于蒸馏的融合网络
    4. V2VNet: 图神经网络融合
    5. V2X-ViT: Transformer 融合
    6. When2comm: 空间置信度融合

【模型架构】
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                          模型整体架构                                    │
    │                                                                         │
    │  输入: 体素化点云数据                                                    │
    │    │                                                                    │
    │    ▼                                                                    │
    │  ┌─────────────┐                                                        │
    │  │  PillarVFE  │  体素特征编码 (PointNet-like)                          │
    │  │  [N,4]→[N,C]│                                                        │
    │  └─────────────┘                                                        │
    │    │                                                                    │
    │    ▼                                                                    │
    │  ┌─────────────┐                                                        │
    │  │  Scatter    │  散射到伪图像 (BEV 特征图)                              │
    │  │  [N,C]→[B,C,H,W]│                                                     │
    │  └─────────────┘                                                        │
    │    │                                                                    │
    │    ▼                                                                    │
    │  ┌─────────────┐                                                        │
    │  │  Backbone   │  多尺度特征提取 (ResNet)                                │
    │  │  提取多尺度   │  [C1,H1,W1], [C2,H2,W2], [C3,H3,W3]                   │
    │  └─────────────┘                                                        │
    │    │                                                                    │
    │    ▼                                                                    │
    │  ┌─────────────┐                                                        │
    │  │ Fusion Net  │  多尺度特征融合 ★                                       │
    │  │ (Max/Att)   │  使用 t_matrix 对齐后融合                               │
    │  └─────────────┘                                                        │
    │    │                                                                    │
    │    ▼                                                                    │
    │  ┌─────────────┐                                                        │
    │  │  Decode     │  多尺度特征解码 (上采样 + 拼接)                          │
    │  └─────────────┘                                                        │
    │    │                                                                    │
    │    ▼                                                                    │
    │  ┌─────────────┐                                                        │
    │  │  Heads      │  检测头 (分类 + 回归 + 方向)                             │
    │  │  cls/reg/dir│                                                        │
    │  └─────────────┘                                                        │
    │    │                                                                    │
    │    ▼                                                                    │
    │  输出: 检测预测                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

【多尺度融合详解】
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Scale 1 (1/2): 低分辨率，大感受野，适合检测大目标                        │
    │  Scale 2 (1/4): 中分辨率，中感受野                                        │
    │  Scale 3 (1/8): 高分辨率，小感受野，适合检测小目标                         │
    │                                                                         │
    │  每个尺度独立融合后，再上采样拼接                                          │
    └─────────────────────────────────────────────────────────────────────────┘

【配置文件示例】
    ```yaml
    model:
      core_method: point_pillar_baseline_multiscale
      args:
        voxel_size: [0.4, 0.4, 4]
        lidar_range: [-140.8, -40, -3, 140.8, 40, 1]
        anchor_number: 2
        fusion_method: max  # or att
        
        pillar_vfe:
          use_norm: true
          with_distance: false
          use_absolute_xyz: true
          num_filters: [64]
        
        point_pillar_scatter:
          num_features: 64
        
        base_bev_backbone:
          layer_nums: [3, 5, 8]        # 每个阶段的层数
          layer_strides: [2, 2, 2]     # 每个阶段的下采样率
          num_filters: [64, 128, 256]  # 每个阶段的通道数
          upsample_strides: [1, 2, 4]  # 上采样倍数
          num_upsample_filter: [128, 128, 128]  # 上采样后的通道数
    ```

【使用方法】
    # 在 train.py 中自动加载
    python opencood/tools/train.py --hypes_yaml config.yaml

==============================================================================
"""

# ============================================================================
#                              标准库导入
# ============================================================================

import torch.nn as nn        # PyTorch 神经网络模块，提供各种网络层
from icecream import ic      # 调试打印工具 (pip install icecream)

# ============================================================================
#                           子模块导入 (模型组件)
# ============================================================================

# PillarVFE: Pillar 体素特征编码器
# 功能: 对每个体素内的点进行特征编码 (类似 PointNet)
# 输入: 体素内点的特征 [N_voxel, max_points, C]
# 输出: 体素特征向量 [N_voxel, C]
from opencood.models.sub_modules.pillar_vfe import PillarVFE

# PointPillarScatter: 特征散射模块
# 功能: 将稀疏的体素特征散射到密集的 BEV 伪图像
# 输入: 体素特征 [N_voxel, C]
# 输出: BEV 特征图 [B, C, H, W]
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

# ResNetBEVBackbone: ResNet 风格的 BEV 主干网络
# 功能: 提取多尺度 BEV 特征
# 特点: 更强的特征提取能力，支持残差连接
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 

# BaseBEVBackbone: 基础 BEV 主干网络
# 功能: 提取多尺度 BEV 特征
# 特点: 轻量级，参数少
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone 

# DownsampleConv: 下采样卷积模块
# 功能: 减小特征图尺寸，降低计算量
# 用于 shrink_header 配置
from opencood.models.sub_modules.downsample_conv import DownsampleConv

# NaiveCompressor: 朴素特征压缩器
# 功能: 压缩特征图通道数，减少通信带宽
# 用于带宽受限的协同感知场景
from opencood.models.sub_modules.naive_compress import NaiveCompressor

# ============================================================================
#                           融合模块导入
# ============================================================================

# 从 fusion_in_one 导入所有融合方法
from opencood.models.fuse_modules.fusion_in_one import (
    MaxFusion,          # 最大值融合: 取各 CAV 特征的逐元素最大值
    AttFusion,          # 注意力融合: 学习各 CAV 的注意力权重
    DiscoFusion,        # DiscoNet 融合: 基于知识蒸馏的融合
    V2VNetFusion,       # V2VNet 融合: 图神经网络融合
    V2XViTFusion,       # V2X-ViT 融合: Transformer 融合
    When2commFusion     # When2comm 融合: 空间置信度融合
)

# ============================================================================
#                           工具函数导入
# ============================================================================

# normalize_pairwise_tfm: 归一化成对变换矩阵
# 功能: 将 4x4 变换矩阵转换为特征图上的仿射变换参数
# 输入: 变换矩阵 [B, max_cav, max_cav, 4, 4], 特征图尺寸
# 输出: 归一化变换矩阵 (用于 grid_sample)
from opencood.utils.transformation_utils import normalize_pairwise_tfm


# ============================================================================
#                          主模型类定义
# ============================================================================

class PointPillarBaselineMultiscale(nn.Module):
    """
    ============================================================================
    PointPillar 多尺度融合模型
    ============================================================================
    
    【模型说明】
        这是一个基于 PointPillar 的协同感知模型，支持多尺度特征融合。
        是 FreeAlign/CoAlign 的核心检测模型。
    
    【继承关系】
        nn.Module (PyTorch 基类)
            └── PointPillarBaselineMultiscale
    
    【核心组件】
        1. pillar_vfe: 体素特征编码器
        2. scatter: 特征散射模块
        3. backbone: BEV 主干网络 (ResNet 或基础版)
        4. fusion_net: 多尺度融合模块列表
        5. cls_head: 分类预测头
        6. reg_head: 回归预测头
        7. dir_head: 方向预测头 (可选)
    
    【数据流】
        体素数据 → VFE → Scatter → Backbone → Fusion → Heads → 预测
    """
    
    def __init__(self, args):
        """
        ====================================================================
        模型初始化
        ====================================================================
        
        【参数说明】
            args: dict
                模型配置字典，来自 YAML 配置文件的 model.args 部分
                必需字段:
                - pillar_vfe: VFE 配置
                - point_pillar_scatter: Scatter 配置
                - base_bev_backbone: 主干网络配置
                - voxel_size: 体素尺寸 [vx, vy, vz]
                - lidar_range: LiDAR 感知范围
                - anchor_number: 每个位置的锚框数量
                - fusion_method: 融合方法 ("max" 或 "att")
        
        【初始化流程】
            1. 调用父类初始化
            2. 创建 VFE 模块
            3. 创建 Scatter 模块
            4. 创建 Backbone 模块
            5. 创建融合模块列表
            6. 创建压缩模块 (可选)
            7. 创建下采样模块 (可选)
            8. 创建检测头
        """
        # --------------------------------------------------------------------
        # 调用父类 nn.Module 的初始化方法
        # 这是 PyTorch 模型的标准做法
        # --------------------------------------------------------------------
        super(PointPillarBaselineMultiscale, self).__init__()

        # --------------------------------------------------------------------
        # 第一步: 创建 PillarVFE (体素特征编码器)
        # --------------------------------------------------------------------
        # PillarVFE 的作用:
        # 1. 对每个体素内的点进行特征提取
        # 2. 使用类似 PointNet 的结构: MLP -> MaxPool -> MLP
        # 3. 输出每个体素的特征向量
        # 
        # 参数说明:
        # - args['pillar_vfe']: VFE 配置字典
        #   - use_norm: 是否使用 BatchNorm
        #   - with_distance: 是否使用点距离作为特征
        #   - use_absolute_xyz: 是否使用绝对坐标
        #   - num_filters: MLP 隐藏层维度列表
        # - num_point_features=4: 输入点特征维度 (x, y, z, intensity)
        # - voxel_size: 体素尺寸 [vx, vy, vz]，用于计算体素坐标
        # - point_cloud_range: 感知范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        self.pillar_vfe = PillarVFE(
            args['pillar_vfe'],              # VFE 配置
            num_point_features=4,            # 点特征维度: (x, y, z, intensity)
            voxel_size=args['voxel_size'],   # 体素尺寸: [0.4, 0.4, 4]
            point_cloud_range=args['lidar_range']  # 感知范围
        )
        # 输入: voxel_features [N_voxel, max_points, 4] (每个体素内的点)
        # 输出: voxel_features [N_voxel, C] (每个体素的特征向量)

        # --------------------------------------------------------------------
        # 第二步: 创建 PointPillarScatter (特征散射模块)
        # --------------------------------------------------------------------
        # Scatter 的作用:
        # 1. 将稀疏的体素特征"散射"到密集的 BEV 网格中
        # 2. 创建一个"伪图像"(pseudo-image)，可以用 2D CNN 处理
        # 
        # 原理:
        # - 体素特征是稀疏的 (只有非空体素有特征)
        # - Scatter 根据体素坐标，将特征放到对应的空间位置
        # - 空白位置填充零
        # 
        # 参数说明:
        # - args['point_pillar_scatter']: Scatter 配置
        #   - num_features: 输出特征通道数 (与 VFE 输出维度一致)
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # 输入: 体素特征 [N_voxel, C]
        # 输出: BEV 特征图 [B, C, H, W]

        # --------------------------------------------------------------------
        # 第三步: 创建 Backbone (BEV 主干网络)
        # --------------------------------------------------------------------
        # Backbone 的作用:
        # 1. 提取多尺度特征 (不同感受野)
        # 2. 低层特征: 高分辨率，小感受野，适合小目标
        # 3. 高层特征: 低分辨率，大感受野，适合大目标
        # 
        # 可选两种 Backbone:
        # - ResNetBEVBackbone: ResNet 结构，性能更强
        # - BaseBEVBackbone: 基础结构，更轻量
        # 
        # 判断是否使用 ResNet
        is_resnet = args['base_bev_backbone'].get("resnet", True)  # 默认使用 ResNet
        
        if is_resnet:
            # 使用 ResNet 风格的 Backbone
            # 参数:
            # - args['base_bev_backbone']: 主干网络配置
            # - 64: 输入通道数 (来自 Scatter 的输出)
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64)
        else:
            # 使用基础 Backbone
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        
        # Backbone 配置示例:
        # layer_nums: [3, 5, 8]        # 每个 stage 的卷积层数
        # layer_strides: [2, 2, 2]     # 每个 stage 的下采样率
        # num_filters: [64, 128, 256]  # 每个 stage 的输出通道数
        # upsample_strides: [1, 2, 4]  # 解码时的上采样倍数
        # num_upsample_filter: [128, 128, 128]  # 上采样后的通道数
        
        # 保存体素尺寸，用于后续计算变换矩阵
        self.voxel_size = args['voxel_size']

        # --------------------------------------------------------------------
        # 第四步: 创建融合模块列表
        # --------------------------------------------------------------------
        # 多尺度融合: 对每个尺度的特征独立进行融合
        # 
        # 为什么多尺度融合?
        # - 不同尺度的特征有不同的语义信息
        # - 低层特征: 位置准确，但语义信息少
        # - 高层特征: 语义丰富，但位置模糊
        # - 多尺度融合可以兼顾两者
        # 
        # 创建一个 ModuleList 来存储各尺度的融合模块
        self.fusion_net = nn.ModuleList()
        
        # 根据 backbone 的 stage 数量创建融合模块
        # len(layer_nums) = 融合模块数量 = 3 (典型配置)
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            # 根据融合方法选择对应的融合模块
            if args['fusion_method'] == "max":
                # MaxFusion: 逐元素取最大值
                # 公式: fused = max(feat_1, feat_2, ..., feat_N)
                # 优点: 简单高效，无额外参数
                # 缺点: 可能丢失信息
                self.fusion_net.append(MaxFusion())
            
            if args['fusion_method'] == "att":
                # AttFusion: 注意力加权融合
                # 公式: fused = Σ(α_i * feat_i), 其中 α_i 由网络学习
                # 优点: 自适应权重，可以学习重要性
                # 缺点: 增加参数量和计算量
                # 参数: args['att']['feat_dim'][i] - 当前尺度的特征维度
                self.fusion_net.append(AttFusion(args['att']['feat_dim'][i]))
        
        # 计算输出通道数
        # 输出通道数 = 所有上采样特征通道数之和
        # 例如: [128, 128, 128] -> 128 + 128 + 128 = 384
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        # --------------------------------------------------------------------
        # 第五步: 创建下采样模块 (可选)
        # --------------------------------------------------------------------
        # Shrink (压缩) 模块的作用:
        # 1. 减小特征图尺寸，降低检测头计算量
        # 2. 进一步融合多尺度特征
        # 
        # 配置示例:
        # shrink_header:
        #   kernal_size: [3]
        #   stride: [2]
        #   dim: [256]  # 输出通道数
        self.shrink_flag = False  # 默认不启用
        
        if 'shrink_header' in args:
            self.shrink_flag = True
            # 创建下采样卷积模块
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            # 更新输出通道数
            self.out_channel = args['shrink_header']['dim'][-1]

        # --------------------------------------------------------------------
        # 第六步: 创建压缩模块 (可选)
        # --------------------------------------------------------------------
        # Compression (压缩) 模块的作用:
        # 1. 减少通信带宽
        # 2. 在带宽受限的场景下，压缩特征后再传输
        # 
        # 原理: 使用 1x1 卷积减少通道数
        # 例如: 64 -> 32 通道，带宽减少一半
        self.compression = False  # 默认不启用
        
        if "compression" in args:
            self.compression = True
            # 创建朴素压缩器
            # 参数:
            # - 64: 输入通道数
            # - args['compression']: 压缩后的通道数
            self.naive_compressor = NaiveCompressor(64, args['compression'])

        # --------------------------------------------------------------------
        # 第七步: 创建检测头
        # --------------------------------------------------------------------
        # 检测头是模型的最后几层，负责输出检测结果
        # PointPillar 使用 1x1 卷积作为检测头
        
        # 分类头 (cls_head)
        # 功能: 预测每个锚框是否包含目标 (二分类) 或目标类别 (多分类)
        # 输出通道数: anchor_number (每个位置 2 个锚框)
        # 例如: 2 个锚框 -> 输出 [B, 2, H, W]
        self.cls_head = nn.Conv2d(
            self.out_channel,           # 输入通道数 (来自 Backbone 或 Shrink)
            args['anchor_number'],      # 输出通道数 = 锚框数量
            kernel_size=1               # 1x1 卷积，逐点预测
        )
        
        # 回归头 (reg_head)
        # 功能: 预测边界框的偏移量 (dx, dy, dz, dw, dl, dh, d_yaw)
        # 输出通道数: 7 * anchor_number (每个锚框预测 7 个偏移量)
        # 例如: 2 个锚框 -> 输出 [B, 14, H, W]
        self.reg_head = nn.Conv2d(
            self.out_channel,                    # 输入通道数
            7 * args['anchor_number'],           # 输出通道数 = 7 * 锚框数量
            kernel_size=1                        # 1x1 卷积
        )
        
        # 方向头 (dir_head) - 可选
        # 功能: 预测目标朝向的方向分类 (前向/后向)
        # 解决 180° 方向歧义问题
        # 
        # 为什么需要方向头?
        # - 回归头预测的 yaw 角在 ±π 附近不连续
        # - 例如: π 和 -π 实际上是同一方向
        # - 方向头将方向离散化为多个 bin，分类预测
        self.use_dir = False  # 默认不启用方向预测
        
        if 'dir_args' in args.keys():
            self.use_dir = True
            # 方向分类头
            # num_bins: 方向 bin 数量 (通常为 2，即前向/后向)
            # 输出通道数: num_bins * anchor_number
            self.dir_head = nn.Conv2d(
                self.out_channel,                              # 输入通道数
                args['dir_args']['num_bins'] * args['anchor_number'],  # 输出通道数
                kernel_size=1                                  # 1x1 卷积
            )  # BIN_NUM = 2 (前向/后向)
 
        # --------------------------------------------------------------------
        # 第八步: 固定 Backbone 参数 (可选)
        # --------------------------------------------------------------------
        # 用于微调 (Fine-tuning) 场景
        # 固定 Backbone 参数，只训练检测头
        # 常见用途:
        # 1. 迁移学习: 在新数据集上微调
        # 2. 时间延迟补偿: 微调融合模块
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()


    def backbone_fix(self):
        """
        ====================================================================
        固定 Backbone 参数
        ====================================================================
        
        【功能说明】
            在微调 (Fine-tuning) 时，固定 Backbone 的参数不参与训练。
            只训练融合模块和检测头。
        
        【使用场景】
            1. 迁移学习: 预训练模型在新数据集上微调
            2. 时间延迟补偿: 固定特征提取器，微调融合策略
            3. 减少过拟合: 减少 trainable 参数数量
        
        【实现方式】
            将 requires_grad 设置为 False，梯度不会计算和更新
        """
        # 固定 VFE 参数
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False  # 禁用梯度计算

        # 固定 Scatter 参数
        for p in self.scatter.parameters():
            p.requires_grad = False

        # 固定 Backbone 参数
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 固定压缩模块参数 (如果启用)
        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        
        # 固定下采样模块参数 (如果启用)
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        # 固定分类头参数
        for p in self.cls_head.parameters():
            p.requires_grad = False
        
        # 固定回归头参数
        for p in self.reg_head.parameters():
            p.requires_grad = False


    def forward(self, data_dict):
        """
        ====================================================================
        前向传播 (核心推理流程)
        ====================================================================
        
        【功能说明】
            模型的前向传播函数，实现从输入数据到检测预测的完整流程。
        
        【处理流程】
            1. 提取体素数据
            2. VFE 编码
            3. Scatter 散射
            4. 计算变换矩阵
            5. 压缩特征 (可选)
            6. 多尺度特征提取
            7. 多尺度特征融合
            8. 解码多尺度特征
            9. 下采样 (可选)
            10. 检测预测
        
        【参数说明】
            data_dict: dict
                输入数据字典，来自 DataLoader
                必需字段:
                - 'processed_lidar': 体素化数据
                    - 'voxel_features': 体素内点特征 [N_voxel, max_points, 4]
                    - 'voxel_coords': 体素坐标 [N_voxel, 4] (batch, z, y, x)
                    - 'voxel_num_points': 每个体素的点数 [N_voxel]
                - 'record_len': 每个 batch 的 CAV 数量 [B]
                - 'pairwise_t_matrix': 成对变换矩阵 [B, max_cav, max_cav, 4, 4]
        
        【返回值】
            output_dict: dict
                输出预测字典，包含:
                - 'cls_preds': 分类预测 [B, anchor_num, H, W]
                - 'reg_preds': 回归预测 [B, 7*anchor_num, H, W]
                - 'dir_preds': 方向预测 [B, num_bins*anchor_num, H, W] (可选)
        """
        # ====================================================================
        # 第一步: 提取体素数据
        # ====================================================================
        # 从数据字典中提取预处理好的体素数据
        # 这些数据由 IntermediateFusionDataset 的 collate_batch_train 生成
        
        # voxel_features: 体素内点的特征
        # 形状: [N_voxel, max_points, C]
        # - N_voxel: 总体素数 (所有 batch 所有 CAV 的体素总和)
        # - max_points: 每个体素内最大点数 (如 32)
        # - C: 点特征维度 (4: x, y, z, intensity) 或更多
        voxel_features = data_dict['processed_lidar']['voxel_features']
        
        # voxel_coords: 体素的坐标
        # 形状: [N_voxel, 4]
        # - 每行: [batch_idx, z, y, x]
        # - 用于 Scatter 将特征放到正确的空间位置
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        
        # voxel_num_points: 每个体素内实际点数
        # 形状: [N_voxel]
        # - 用于 VFE 中的有效点 mask
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        
        # record_len: 每个 batch 样本的 CAV 数量
        # 形状: [B]
        # - 例如: [2, 2, 3, 2] 表示 4 个 batch，分别有 2,2,3,2 个 CAV
        # - 用于融合时区分不同 CAV 的特征
        record_len = data_dict['record_len']

        # ====================================================================
        # 第二步: 组装 batch 字典
        # ====================================================================
        # 将体素数据打包成字典，传递给 VFE 和 Scatter
        batch_dict = {
            'voxel_features': voxel_features,      # 体素特征
            'voxel_coords': voxel_coords,          # 体素坐标
            'voxel_num_points': voxel_num_points,  # 体素点数
            'record_len': record_len               # CAV 数量
        }
        
        # ====================================================================
        # 第三步: VFE 编码
        # ====================================================================
        # PillarVFE 对每个体素内的点进行特征编码
        # 
        # 内部流程:
        # 1. 对每个点计算特征:
        #    - 原始特征: x, y, z, intensity
        #    - 相对特征: x - x_mean, y - y_mean, z - z_mean (相对体素中心)
        #    - 绝对特征: x, y, z (如果 use_absolute_xyz=True)
        # 2. MLP 编码: [N_points, C_in] -> [N_points, C_out]
        # 3. MaxPool: [N_points, C_out] -> [1, C_out] (取每个特征维度的最大值)
        # 4. 输出: [N_voxel, C]
        batch_dict = self.pillar_vfe(batch_dict)
        # 输入: [N_voxel, max_points, 4] (每个体素内的点特征)
        # 输出: [N_voxel, 64] (每个体素的特征向量)
        
        # ====================================================================
        # 第四步: Scatter 散射
        # ====================================================================
        # PointPillarScatter 将稀疏体素特征散射到密集 BEV 网格
        # 
        # 原理:
        # 1. 创建空白张量: [B, C, H, W] (H, W 由感知范围和体素大小决定)
        # 2. 根据 voxel_coords 将特征放到对应位置
        # 3. 输出 "伪图像" (pseudo-image)，可以用 2D CNN 处理
        batch_dict = self.scatter(batch_dict)
        # 输入: [N_voxel, 64] (稀疏体素特征)
        # 输出: [N, C, H, W] (密集 BEV 特征图，N = 所有 CAV 的总数)
        # 例如: [8, 64, 216, 704] (8 个 CAV，64 通道，216x704 特征图)
        
        # ====================================================================
        # 第五步: 计算归一化变换矩阵
        # ====================================================================
        # 变换矩阵用于特征对齐
        # 
        # 为什么需要变换?
        # - 不同 CAV 在不同位置，观测角度不同
        # - 融合前需要对齐到同一坐标系 (Ego 坐标系)
        # - 使用仿射变换 (warp) 将其他 CAV 的特征变换到 Ego 视角
        # 
        # normalize_pairwise_tfm 的作用:
        # - 将 4x4 齐次变换矩阵转换为 grid_sample 可用的归一化坐标
        # - 归一化范围: [-1, 1] (PyTorch grid_sample 要求)
        _, _, H0, W0 = batch_dict['spatial_features'].shape  # 获取原始特征图尺寸
        # H0, W0: 特征图高度和宽度
        # 例如: H0=216, W0=704 (取决于感知范围和体素大小)
        
        # 计算归一化变换矩阵
        # t_matrix: [B, max_cav, max_cav, 2, 3]
        # - 用于 grid_sample 的仿射变换参数
        # - t_matrix[b, i, j] 表示 batch b 中 CAV_i 到 CAV_j 的变换
        t_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'],  # 原始变换矩阵 [B, max_cav, max_cav, 4, 4]
            H0, W0,                           # 特征图尺寸
            self.voxel_size[0]                # 体素大小 (用于坐标缩放)
        )

        # ====================================================================
        # 第六步: 获取 BEV 空间特征
        # ====================================================================
        # spatial_features: BEV 特征图
        # 形状: [N, C, H, W]
        # - N: 所有 CAV 的总数 (B * max_cav 或更少)
        # - C: 特征通道数 (64)
        # - H, W: 特征图尺寸
        spatial_features = batch_dict['spatial_features']

        # ====================================================================
        # 第七步: 特征压缩 (可选)
        # ====================================================================
        # 在带宽受限场景下，压缩特征减少通信量
        if self.compression:
            # 使用 1x1 卷积压缩通道数
            # 例如: [N, 64, H, W] -> [N, 32, H, W]
            spatial_features = self.naive_compressor(spatial_features)

        # ====================================================================
        # 第八步: 多尺度特征提取
        # ====================================================================
        # Backbone 提取多尺度特征
        # 
        # 多尺度特征的优点:
        # - 低层特征 (高分辨率): 空间信息丰富，适合小目标
        # - 高层特征 (低分辨率): 语义信息丰富，适合大目标
        # 
        # get_multiscale_feature 返回特征列表:
        # - feature_list[0]: Scale 1, [N, C1, H1, W1]
        # - feature_list[1]: Scale 2, [N, C2, H2, W2]
        # - feature_list[2]: Scale 3, [N, C3, H3, W3]
        # 
        # 典型配置:
        # - Scale 1: [N, 64, 108, 352] (下采样 2x)
        # - Scale 2: [N, 128, 54, 176] (下采样 4x)
        # - Scale 3: [N, 256, 27, 88] (下采样 8x)
        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        
        # 注释掉的代码: 直接解码 (不融合)
        # feature = self.backbone.decode_multiscale_feature(feature_list) 

        # ====================================================================
        # 第九步: 多尺度特征融合 ★ (核心步骤)
        # ====================================================================
        # 对每个尺度独立进行融合
        # 
        # 融合流程 (以 MaxFusion 为例):
        # 1. 对于每个 CAV 的特征，使用 t_matrix 进行仿射变换 (warp)
        #    - 将其他 CAV 的特征变换到 Ego 坐标系
        #    - 使用 PyTorch 的 grid_sample 实现
        # 2. 对变换后的特征取逐元素最大值
        #    - fused = max(warped_feat_1, warped_feat_2, ...)
        # 
        # 注意: 融合只保留 Ego 的特征
        # - 输入: [N, C, H, W] (N 个 CAV 的特征)
        # - 输出: [B, C, H, W] (B 个 batch，每个只保留 Ego)
        fused_feature_list = []  # 存储融合后的多尺度特征
        
        for i, fuse_module in enumerate(self.fusion_net):
            # 对第 i 个尺度进行融合
            # feature_list[i]: [N, Ci, Hi, Wi] - 第 i 尺度的所有 CAV 特征
            # record_len: [B] - 每个 batch 的 CAV 数量
            # t_matrix: [B, max_cav, max_cav, 2, 3] - 变换矩阵
            fused_feature_list.append(
                fuse_module(feature_list[i], record_len, t_matrix)
            )
            # 融合后: [B, Ci, Hi, Wi]
        
        # 注释掉的代码: 不融合，只取第一个 CAV (Ego) 的特征
        # fused_feature_list.append(feature_list[i][0].unsqueeze(0))
        
        # ====================================================================
        # 第十步: 解码多尺度特征
        # ====================================================================
        # 将融合后的多尺度特征上采样到相同尺寸，然后拼接
        # 
        # decode_multiscale_feature 流程:
        # 1. 对每个尺度的特征进行上采样 (Upsample)
        #    - 使用转置卷积或双线性插值
        # 2. 将上采样后的特征拼接 (Concatenate)
        #    - 例如: [B, 128, H, W] + [B, 128, H, W] + [B, 128, H, W]
        #           -> [B, 384, H, W]
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list)
        # 输出: [B, 384, H, W] (B 个 batch，384 通道，原始特征图尺寸)

        # ====================================================================
        # 第十一步: 下采样 (可选)
        # ====================================================================
        # 如果启用 shrink，进一步下采样减少计算量
        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
            # 例如: [B, 384, H, W] -> [B, 256, H/2, W/2]

        # ====================================================================
        # 第十二步: 检测预测
        # ====================================================================
        # 使用 1x1 卷积进行逐点预测
        
        # 分类预测 (psm: Probability Score Map)
        # 预测每个位置每个锚框的目标存在概率
        # 输出: [B, anchor_num, H, W]
        psm = self.cls_head(fused_feature)
        # 例如: [B, 2, H, W] -> 每个位置 2 个锚框的分类分数
        
        # 回归预测 (rm: Regression Map)
        # 预测每个位置每个锚框的边界框偏移
        # 输出: [B, 7*anchor_num, H, W]
        # 7 个值: dx, dy, dz, dw, dl, dh, d_yaw (相对锚框的偏移)
        rm = self.reg_head(fused_feature)
        # 例如: [B, 14, H, W] -> 每个位置 2 个锚框，每个 7 个回归值

        # ====================================================================
        # 第十三步: 组装输出字典
        # ====================================================================
        output_dict = {
            'cls_preds': psm,    # 分类预测 [B, anchor_num, H, W]
            'reg_preds': rm      # 回归预测 [B, 7*anchor_num, H, W]
        }

        # 方向预测 (可选)
        if self.use_dir:
            # 方向分类预测
            # 输出: [B, num_bins*anchor_num, H, W]
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})

        return output_dict


    def save_single(self, data_dict):
        """
        ====================================================================
        保存单车特征 (用于特征提取/调试)
        ====================================================================
        
        【功能说明】
            保存每个 CAV 的单独特征，用于:
            1. 特征分析
            2. 知识蒸馏的教师网络特征提取
            3. 调试和可视化
        
        【参数说明】
            data_dict: dict
                输入数据字典
        
        【返回值】
            output_dict: dict 或 None
                如果只有 1 个 CAV，返回 None
                否则返回每个 CAV 的检测预测
        """
        # --------------------------------------------------------------------
        # 提取体素数据
        # --------------------------------------------------------------------
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        
        # 如果只有 1 个 CAV，不保存
        if record_len[0] == 1:
            return None
        
        # --------------------------------------------------------------------
        # 前向传播 (与 forward 类似)
        # --------------------------------------------------------------------
        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len
        }
        
        # VFE 编码
        batch_dict = self.pillar_vfe(batch_dict)
        
        # Scatter 散射
        batch_dict = self.scatter(batch_dict)
        
        # 计算变换矩阵
        _, _, H0, W0 = batch_dict['spatial_features'].shape
        t_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0]
        )

        spatial_features = batch_dict['spatial_features']

        # 压缩 (可选)
        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)
        
        # 多尺度特征提取
        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        
        # 保存特征列表 (用于后续分析)
        save_feature_list = []
        for feature in feature_list:
            # detach(): 分离计算图
            # cpu(): 移到 CPU
            save_feature_list.append(feature.detach().cpu())
        
        # 解码多尺度特征
        feature = self.backbone.decode_multiscale_feature(feature_list) 
        
        # --------------------------------------------------------------------
        # 保存特征到文件
        # --------------------------------------------------------------------
        import os
        import torch
        
        # 构建保存路径
        save_path = os.path.join(
            'opencood/logs/dairv2x_point_pillar_lidar_max_2023_04_13_13_34_49',
            "test",
            "featurelist",
            data_dict['scenario_folder'][0]
        )

        # 创建目录
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except FileExistsError:
                pass
        
        # 保存特征文件
        save_path = os.path.join(save_path, data_dict['timestep'][0] + ".pt")
        torch.save(save_feature_list, save_path)
        
        # 下采样 (可选)
        if self.shrink_flag:
            feature = self.shrink_conv(feature)

        # 检测预测
        psm = self.cls_head(feature)
        rm = self.reg_head(feature)

        if self.use_dir:
            dm = self.dir_head(feature)
        
        # 为每个 CAV 生成单独的输出
        output_dict = {}
        for i, cav_id in enumerate(data_dict['cav_id_list']):   
            output_dict[cav_id] = {
                'cls_preds': psm[i].unsqueeze(0),
                'reg_preds': rm[i].unsqueeze(0),
                'dir_preds': dm[i].unsqueeze(0)
            }

        return output_dict