# -*- coding: utf-8 -*-
# 本模块是数据集包的初始化文件
# 负责导入各种融合策略的数据集工厂函数和不同数据集的基础类
# 并提供统一的build_dataset接口来构建数据集实例

# ==================== 融合策略数据集工厂函数导入 ====================
# 协作感知中的融合策略决定了多智能体之间如何交换和融合信息

# 导入后期融合数据集的工厂函数
# 后期融合(Late Fusion): 各智能体独立完成检测，最后融合检测结果
# 优点: 通信带宽需求小；缺点: 丢失中间特征信息
from opencood.data_utils.datasets.late_fusion_dataset import getLateFusionDataset

# 导入早期融合数据集的工厂函数
# 早期融合(Early Fusion): 在原始数据层面进行融合，如合并点云
# 优点: 信息保留最完整；缺点: 通信带宽需求最大
from opencood.data_utils.datasets.early_fusion_dataset import getEarlyFusionDataset

# 导入中间融合数据集的工厂函数
# 中间融合(Intermediate Fusion): 在特征层面进行融合
# 平衡了通信带宽和信息保留，是协作感知的主流方法
# FreeAlign论文采用的就是中间融合策略
from opencood.data_utils.datasets.intermediate_fusion_dataset import getIntermediateFusionDataset

# 导入两阶段中间融合数据集的工厂函数
# 两阶段中间融合: 
#   Stage 1: 各智能体独立检测，生成检测框
#   Stage 2: 使用FreeAlign进行时空对齐后融合特征
# 这是FreeAlign的核心数据集类型
from opencood.data_utils.datasets.intermediate_2stage_fusion_dataset import getIntermediate2stageFusionDataset

# ==================== 数据集基础类导入 ====================
# 不同的数据集有不同的数据格式和加载方式，需要对应的基础类

# 导入OPV2V数据集的基础类
# OPV2V: Open Passive Vehicle-to-Vehicle，是一个仿真协作感知数据集
# 使用CARLA和OpenCDA联合仿真生成，支持多智能体协作场景
from opencood.data_utils.datasets.basedataset.opv2v_basedataset import OPV2VBaseDataset

# 导入V2X-Sim数据集的基础类
# V2X-Sim: 另一个基于CARLA仿真的V2X协作感知数据集
# 提供多种传感器数据和多智能体协作场景
from opencood.data_utils.datasets.basedataset.v2xsim_basedataset import V2XSIMBaseDataset

# 导入DAIR-V2X数据集的基础类
# DAIR-V2X: 第一个真实世界的V2X协作感知数据集
# 由中国智能网联汽车研究院发布，包含真实道路场景数据
# 包括车端和路端的多模态传感器数据
from opencood.data_utils.datasets.basedataset.dairv2x_basedataset import DAIRV2XBaseDataset

# 导入V2XSet数据集的基础类
# V2XSet: 另一个大规模V2X协作感知数据集
# 提供丰富的协作感知场景和标注
from opencood.data_utils.datasets.basedataset.v2xset_basedataset import V2XSETBaseDataset


def build_dataset(dataset_cfg, visualize=False, train=True):
    """
    构建数据集实例的工厂函数。
    
    该函数根据配置参数动态创建合适的数据集实例，
    支持多种融合策略和数据集类型的组合。
    
    构建过程采用工厂模式:
    1. 根据fusion_name获取对应的融合策略工厂函数
    2. 根据dataset_name获取对应的数据集基础类
    3. 组合两者创建具体的数据集实例

    Parameters
    ----------
    dataset_cfg : dict
        数据集配置字典，包含以下关键字段:
        - fusion['core_method']: 融合策略名称
          可选值: 'late', 'intermediate', 'intermediate2stage', 'early'
        - fusion['dataset']: 数据集名称
          可选值: 'opv2v', 'v2xsim', 'dairv2x', 'v2xset'
        - 其他预处理、后处理等配置参数
        
    visualize : bool, 可选
        是否启用可视化模式。默认为False。
        启用后会生成数据可视化结果，用于调试和分析。
        
    train : bool, 可选
        是否为训练模式。默认为True。
        - True: 加载训练集，可能包含数据增强
        - False: 加载验证/测试集，不进行数据增强

    Returns
    -------
    dataset : object
        构建好的数据集实例，继承自对应的基础数据集类，
        并应用了指定的融合策略。可直接用于PyTorch DataLoader。

    Example
    -------
    >>> config = {
    ...     'fusion': {
    ...         'core_method': 'intermediate2stage',
    ...         'dataset': 'opv2v'
    ...     }
    ... }
    >>> dataset = build_dataset(config, visualize=False, train=True)
    >>> dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    """
    # 从配置字典中提取融合策略名称
    # fusion_name决定了智能体之间如何交换和融合信息
    fusion_name = dataset_cfg['fusion']['core_method']
    
    # 从配置字典中提取数据集名称
    # dataset_name决定了数据来源和格式
    dataset_name = dataset_cfg['fusion']['dataset']

    # 验证融合策略名称是否有效
    # 确保配置中指定的融合策略是被支持的
    # 'late': 后期融合 - 融合检测结果
    # 'intermediate': 中间融合 - 融合特征
    # 'intermediate2stage': 两阶段中间融合 - FreeAlign使用
    # 'early': 早期融合 - 融合原始数据
    assert fusion_name in ['late', 'intermediate', 'intermediate2stage', 'early']
    
    # 验证数据集名称是否有效
    # 确保配置中指定的数据集是被支持的
    # 'opv2v': OPV2V仿真数据集
    # 'v2xsim': V2X-Sim仿真数据集
    # 'dairv2x': DAIR-V2X真实世界数据集
    # 'v2xset': V2XSet数据集
    assert dataset_name in ['opv2v', 'v2xsim', 'dairv2x', 'v2xset']

    # 动态构建融合策略工厂函数的名称
    # 例如: fusion_name='intermediate2stage' -> 'getIntermediate2stageFusionDataset'
    # capitalize()将首字母大写，其余小写
    fusion_dataset_func = "get" + fusion_name.capitalize() + "FusionDataset"
    
    # 使用eval将字符串转换为实际的函数对象
    # eval会查找当前命名空间中对应名称的函数
    # 这样可以根据配置动态选择工厂函数，无需大量的if-else判断
    fusion_dataset_func = eval(fusion_dataset_func)
    
    # 动态构建数据集基础类的名称
    # 例如: dataset_name='opv2v' -> 'OPV2VBaseDataset'
    # upper()将字符串转换为大写
    base_dataset_cls = dataset_name.upper() + "BaseDataset"
    
    # 使用eval将字符串转换为实际的类对象
    # 这样可以根据配置动态选择数据集类
    base_dataset_cls = eval(base_dataset_cls)

    # 构建最终的数据集实例
    # 这是一个两步调用:
    # 1. fusion_dataset_func(base_dataset_cls) 返回一个数据集类
    #    工厂函数会将基础数据集类与融合策略组合
    # 2. 使用参数实例化该数据集类
    dataset = fusion_dataset_func(base_dataset_cls)(
        params=dataset_cfg,    # 传入完整的数据集配置
        visualize=visualize,   # 是否启用可视化
        train=train           # 是否为训练模式
    )

    # 返回构建好的数据集实例
    return dataset