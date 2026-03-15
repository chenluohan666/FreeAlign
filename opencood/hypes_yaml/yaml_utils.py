# -*- coding: utf-8 -*-
# 作者: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>
# 许可证: TDG-Attribution-NonCommercial-NoDistrib
# 本模块提供YAML配置文件的加载、解析和保存功能
# 主要用于协作感知系统中的参数配置管理

# 导入正则表达式模块，用于模式匹配和字符串处理
import re
# 导入PyYAML模块，用于YAML文件的解析和生成
import yaml
# 导入操作系统模块，用于文件路径操作
import os
# 导入数学模块，提供数学运算函数
import math

# 导入NumPy数值计算库，用于数组操作和数值计算
import numpy as np


def load_yaml(file, opt=None):
    """
    加载YAML配置文件并返回参数字典。
    
    该函数是配置系统的核心入口，负责：
    1. 从指定路径读取YAML文件
    2. 配置YAML解析器以正确处理浮点数格式
    3. 可选地调用额外的解析器进行参数后处理

    Parameters
    ----------
    file : string
        YAML配置文件的完整路径。
        
    opt : argparser, 可选
        命令行参数解析器对象，包含运行时配置选项。
        如果提供了opt.model_dir，将从该目录加载config.yaml。

    Returns
    -------
    param : dict
        包含所有配置参数的嵌套字典结构。
    """
    # 如果提供了命令行参数且指定了模型目录，则从模型目录加载配置文件
    # 这允许在测试/推理时使用与训练时相同的配置
    if opt and opt.model_dir:
        # 拼接模型目录路径和配置文件名，构建完整配置文件路径
        file = os.path.join(opt.model_dir, 'config.yaml')
    
    # 以只读模式打开YAML配置文件
    stream = open(file, 'r')
    
    # 获取YAML的Loader类，用于自定义解析行为
    loader = yaml.Loader
    
    # 添加隐式解析器，用于正确识别和处理各种浮点数格式
    # 这解决了YAML默认解析器对某些浮点数格式处理不当的问题
    loader.add_implicit_resolver(
        # 指定解析器的标签类型为浮点数
        u'tag:yaml.org,2002:float',
        # 定义匹配浮点数的正则表达式模式，支持多种格式：
        # 1. 带小数点的常规浮点数: 如 1.23, -1.23e10
        # 2. 科学计数法表示: 如 1e10, -1E-5
        # 3. 以小数点开头的浮点数: 如 .5, .5e10
        # 4. 时间格式: 如 1:30:45.5
        # 5. 无穷大: 如 .inf, .Inf, .INF
        # 6. 非数: 如 .nan, .NaN, .NAN
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),  # re.X标志启用详细模式，允许正则跨行和添加注释
        # 指定触发该解析器的首字符集合
        list(u'-+0123456789.'))
    
    # 使用配置好的Loader加载YAML文件内容，解析为Python字典
    param = yaml.load(stream, Loader=loader)
    
    # 检查配置中是否指定了额外的YAML解析器函数名
    # 这允许用户自定义参数后处理逻辑
    if "yaml_parser" in param:
        # 使用eval动态调用指定的解析函数，对参数进行后处理
        # 例如: param["yaml_parser"] = "load_point_pillar_params"
        # 将会调用 load_point_pillar_params(param)
        param = eval(param["yaml_parser"])(param)

    # 返回处理完成的参数字典
    return param


def load_voxel_params(param):
    """
    根据激光雷达范围和体素分辨率，计算锚框参数和目标分辨率。
    
    该函数为基于体素(Voxel)的3D目标检测模型计算必要的空间参数：
    - 体素网格尺寸 (W, H, D)
    - 体素在各个维度的大小 (vw, vh, vd)
    
    这些参数用于将连续的3D空间离散化为体素网格，
    是PointPillar、VoxelNet等模型的基础配置。

    Parameters
    ----------
    param : dict
        原始加载的参数字典，包含preprocess和postprocess配置。

    Returns
    -------
    param : dict
        修改后的参数字典，新增了anchor_args中的W、H、D属性。
        - W: X轴方向的体素数量（网格宽度）
        - H: Y轴方向的体素数量（网格高度）
        - D: Z轴方向的体素数量（网格深度）
    """
    # 从后处理配置中获取锚框相关参数
    anchor_args = param['postprocess']['anchor_args']
    
    # 获取激光雷达的感知范围
    # cav_lidar_range格式: [x_min, y_min, z_min, x_max, y_max, z_max]
    # 表示激光雷达能探测到的3D空间边界
    cav_lidar_range = anchor_args['cav_lidar_range']
    
    # 从预处理配置中获取体素大小
    # voxel_size格式: [voxel_width, voxel_height, voxel_depth]
    # 定义每个体素在三个维度上的物理尺寸（单位：米）
    voxel_size = param['preprocess']['args']['voxel_size']

    # 提取X方向的体素尺寸（米/体素）
    vw = voxel_size[0]
    # 提取Y方向的体素尺寸（米/体素）
    vh = voxel_size[1]
    # 提取Z方向的体素尺寸（米/体素）
    vd = voxel_size[2]

    # 将体素尺寸保存到锚框参数中，便于后续使用
    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    # 计算X方向的体素网格数量
    # (x_max - x_min) / voxel_width = X方向的体素数量
    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    
    # 计算Y方向的体素网格数量
    # (y_max - y_min) / voxel_height = Y方向的体素数量
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    
    # 计算Z方向的体素网格数量
    # (z_max - z_min) / voxel_depth = Z方向的体素数量
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    # 更新后处理配置中的锚框参数
    param['postprocess'].update({'anchor_args': anchor_args})

    # 有时我们只想可视化数据而不运行模型
    # 因此需要检查'model'键是否存在
    if 'model' in param:
        # 将网格尺寸传递给模型配置，用于构建神经网络结构
        param['model']['args']['W'] = anchor_args['W']
        param['model']['args']['H'] = anchor_args['H']
        param['model']['args']['D'] = anchor_args['D']
    
    # 如果配置中包含box_align_pre_calc（FreeAlign的预计算配置）
    # 这用于两阶段检测中的第一阶段后处理配置
    if 'box_align_pre_calc' in param:
        # 更新第一阶段后处理器的锚框参数
        param['box_align_pre_calc']['stage1_postprocessor_config'].update({'anchor_args': anchor_args})

    # 返回更新后的参数字典
    return param


def load_point_pillar_params(param):
    """
    根据激光雷达范围和体素分辨率，计算PointPillar模型的参数。
    
    PointPillar是一种高效的3D目标检测方法，它：
    1. 将点云转换为伪图像（pseudo-image）
    2. 使用2D卷积进行处理
    3. 在BEV（Bird's Eye View）空间进行检测
    
    本函数计算PointPillar特有的网格参数，使用math.ceil向上取整，
    确保所有点云数据都能被覆盖。

    Parameters
    ----------
    param : dict
        原始加载的参数字典。

    Returns
    -------
    param : dict
        修改后的参数字典，新增了网格尺寸和锚框参数。
    """
    # 获取激光雷达感知范围
    # 格式: [x_min, y_min, z_min, x_max, y_max, z_max]
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    
    # 获取体素尺寸配置
    voxel_size = param['preprocess']['args']['voxel_size']

    # 计算体素网格尺寸（用体素数量表示）
    # grid_size = (range_max - range_min) / voxel_size
    # 分别计算X、Y、Z三个方向的体素数量
    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    
    # 将网格尺寸四舍五入为64位整数，用于神经网络中的张量维度
    grid_size = np.round(grid_size).astype(np.int64)
    
    # 将网格尺寸保存到模型的PointPillar Scatter层配置中
    # Scatter层负责将体素特征散射到空间网格上
    param['model']['args']['point_pillar_scatter']['grid_size'] = grid_size

    # 从后处理配置中获取锚框参数
    anchor_args = param['postprocess']['anchor_args']

    # 提取各个维度的体素尺寸
    vw = voxel_size[0]  # X方向体素宽度
    vh = voxel_size[1]  # Y方向体素高度
    vd = voxel_size[2]  # Z方向体素深度

    # 将体素尺寸保存到锚框参数
    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    # 计算X方向的网格宽度（使用ceil向上取整，确保覆盖所有区域）
    # W对应图像宽度，但在激光雷达坐标系中沿X轴方向
    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    
    # 计算Y方向的网格高度（对应激光雷达坐标系的Y轴）
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    
    # 计算Z方向的网格深度（对应激光雷达坐标系的Z轴）
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    # 更新后处理配置
    param['postprocess'].update({'anchor_args': anchor_args})

    # 返回更新后的参数字典
    return param


def load_second_params(param):
    """
    根据激光雷达范围和体素分辨率，计算SECOND模型的参数。
    
    SECOND (Sparsely Embedded Convolutional Detection) 是一种
    基于3D稀疏卷积的目标检测方法。与PointPillar不同，
    SECOND使用真正的3D卷积处理体素特征。

    Parameters
    ----------
    param : dict
        原始加载的参数字典。

    Returns
    -------
    param : dict
        修改后的参数字典，新增了网格尺寸和锚框参数。
    """
    # 获取激光雷达感知范围
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    
    # 获取体素尺寸
    voxel_size = param['preprocess']['args']['voxel_size']

    # 计算体素网格尺寸
    # 通过范围差除以体素尺寸得到各方向的体素数量
    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    
    # 四舍五入并转换为整数类型
    grid_size = np.round(grid_size).astype(np.int64)
    
    # 将网格尺寸保存到模型配置中，用于3D稀疏卷积
    param['model']['args']['grid_size'] = grid_size

    # 获取锚框配置
    anchor_args = param['postprocess']['anchor_args']

    # 提取各维度的体素尺寸
    vw = voxel_size[0]  # X方向
    vh = voxel_size[1]  # Y方向
    vd = voxel_size[2]  # Z方向

    # 保存体素尺寸到锚框配置
    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    # 计算各方向的体素网格数量（使用int截断取整）
    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    # 更新后处理配置
    param['postprocess'].update({'anchor_args': anchor_args})

    # 返回更新后的参数字典
    return param


def load_bev_params(param):
    """
    加载BEV（Bird's Eye View，鸟瞰图）相关的几何参数。
    
    BEV表示是将3D场景投影到2D俯视图的方法，常用于：
    - 2D目标检测
    - 语义分割
    - 运动规划
    
    本函数计算BEV表示所需的几何参数，包括：
    - 边界范围
    - 分辨率
    - 输入形状
    - 标签形状

    Parameters
    ----------
    param : dict
        原始加载的参数字典。

    Returns
    -------
    param : dict
        修改后的参数字典，新增了geometry_param属性。
    """
    # 获取BEV表示的分辨率（米/像素）
    # 定义BEV图中每个像素对应的实际物理距离
    res = param["preprocess"]["args"]["res"]
    
    # 解包激光雷达感知范围
    # L1, W1, H1: X、Y、Z方向的最小值
    # L2, W2, H2: X、Y、Z方向的最大值
    L1, W1, H1, L2, W2, H2 = param["preprocess"]["cav_lidar_range"]
    
    # 获取下采样率
    # 用于从输入特征图生成标签时降采样
    # 例如：downsample_rate=4 表示标签尺寸是输入的1/4
    downsample_rate = param["preprocess"]["args"]["downsample_rate"]

    # 定义辅助函数：计算给定范围内的网格数量
    # low: 范围最小值，high: 范围最大值，r: 分辨率
    def f(low, high, r):
        # 返回从low到high，以r为间隔的网格数量
        return int((high - low) / r)

    # 计算输入特征图的形状（输入到网络的BEV图尺寸）
    input_shape = (
        int((f(L1, L2, res))),  # X方向的像素数
        int((f(W1, W2, res))),  # Y方向的像素数
        int((f(H1, H2, res)) + 1)  # Z方向的像素数（+1是为了包含边界）
    )
    
    # 计算标签图的形状（检测头的输出尺寸）
    # 标签图比输入图小，因为经过了下采样
    label_shape = (
        int(input_shape[0] / downsample_rate),  # X方向
        int(input_shape[1] / downsample_rate),  # Y方向
        7  # 通道数：7个参数表示检测框信息
        # 通常为: [x, y, z, w, l, h, heading] 或 [dx, dy, dz, dw, dl, dh, dheading]
    )
    
    # 将所有几何参数打包成字典
    geometry_param = {
        'L1': L1,  # X方向最小值
        'L2': L2,  # X方向最大值
        'W1': W1,  # Y方向最小值
        'W2': W2,  # Y方向最大值
        'H1': H1,  # Z方向最小值
        'H2': H2,  # Z方向最大值
        "downsample_rate": downsample_rate,  # 下采样率
        "input_shape": input_shape,  # 输入特征图形状
        "label_shape": label_shape,  # 标签形状
        "res": res  # 分辨率
    }
    
    # 将几何参数保存到预处理配置中
    param["preprocess"]["geometry_param"] = geometry_param
    # 将几何参数保存到后处理配置中
    param["postprocess"]["geometry_param"] = geometry_param
    # 将几何参数保存到模型配置中
    param["model"]["args"]["geometry_param"] = geometry_param
    
    # 返回更新后的参数字典
    return param


def save_yaml(data, save_name):
    """
    将字典数据保存为YAML文件。
    
    用于保存训练配置、模型参数等信息，
    便于后续复现和部署。

    Parameters
    ----------
    data : dict
        包含所有待保存数据的字典。
        
    save_name : string
        输出YAML文件的完整路径。
    """
    # 以写入模式打开文件
    with open(save_name, 'w') as outfile:
        # 将字典序列化为YAML格式并写入文件
        # default_flow_style=False 使用块格式，更易读
        # 块格式示例:
        # key:
        #   subkey: value
        # 而非流格式: key: {subkey: value}
        yaml.dump(data, outfile, default_flow_style=False)


def load_point_pillar_params_stage1(param):
    """
    为FreeAlign两阶段检测的第一阶段计算PointPillar参数。
    
    FreeAlign采用两阶段检测策略：
    - Stage 1: 使用PointPillar进行初步检测，生成候选框
    - Stage 2: 使用FreeAlign进行空间对齐和特征融合
    
    本函数专门配置第一阶段模型的参数。

    Parameters
    ----------
    param : dict
        原始加载的参数字典。

    Returns
    -------
    param : dict
        修改后的参数字典，配置了stage1的网格和锚框参数。
    """
    # 获取激光雷达感知范围
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    
    # 获取体素尺寸
    voxel_size = param['preprocess']['args']['voxel_size']

    # 计算体素网格尺寸
    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    
    # 四舍五入并转换为整数
    grid_size = np.round(grid_size).astype(np.int64)
    
    # 将网格尺寸保存到第一阶段模型的Scatter层配置
    param['box_align_pre_calc']['stage1_model_config']['point_pillar_scatter']['grid_size'] = grid_size

    # 获取第一阶段后处理器的锚框配置
    anchor_args = param['box_align_pre_calc']['stage1_postprocessor_config']['anchor_args']

    # 提取各维度的体素尺寸
    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    # 保存体素尺寸
    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    # 计算各方向的体素网格数量
    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    # 更新第一阶段后处理器配置
    param['box_align_pre_calc']['stage1_postprocessor_config'].update({'anchor_args': anchor_args})

    # 返回更新后的参数字典
    return param


def load_lift_splat_shoot_params(param):
    """
    根据检测范围和体素分辨率，计算Lift-Splat-Shoot (LSS)模型的参数。
    
    LSS是一种将多相机图像转换为BEV特征表示的方法：
    1. Lift: 为每个图像像素预测深度分布，"提升"到3D
    2. Splat: 将3D特征"投影"到BEV网格
    3. Shoot: 在BEV特征上进行目标检测
    
    本函数配置LSS所需的网格参数。

    Parameters
    ----------
    param : dict
        原始加载的参数字典。

    Returns
    -------
    param : dict
        修改后的参数字典，新增了网格尺寸和锚框参数。
    """
    # 获取激光雷达感知范围（用作BEV网格的范围）
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    
    # 获取体素/网格尺寸
    voxel_size = param['preprocess']['args']['voxel_size']

    # 计算体素网格尺寸
    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    
    # 四舍五入并转换为整数
    grid_size = np.round(grid_size).astype(np.int64)
    
    # 获取锚框配置
    anchor_args = param['postprocess']['anchor_args']

    # 提取各维度的体素尺寸
    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    # 保存体素尺寸
    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    # 计算各方向的体素网格数量（使用ceil向上取整）
    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    # 更新后处理配置
    param['postprocess'].update({'anchor_args': anchor_args})

    # 返回更新后的参数字典
    return param


def load_point_pillar_lss_params(param):
    """
    为结合PointPillar和LSS的多模态模型计算参数。
    
    该模型融合激光雷达点云（通过PointPillar）和
    相机图像（通过LSS）的特征，实现多模态协作感知。
    
    本函数配置lidar分支的PointPillar参数。

    Parameters
    ----------
    param : dict
        原始加载的参数字典。

    Returns
    -------
    param : dict
        修改后的参数字典，新增了网格尺寸和锚框参数。
    """
    # 获取激光雷达感知范围
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    
    # 获取体素尺寸
    voxel_size = param['preprocess']['args']['voxel_size']

    # 计算体素网格尺寸
    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    
    # 四舍五入并转换为整数
    grid_size = np.round(grid_size).astype(np.int64)
    
    # 将网格尺寸保存到激光雷达分支的PointPillar配置中
    # 注意路径: model -> args -> lidar_args -> point_pillar_scatter
    param['model']['args']['lidar_args']['point_pillar_scatter']['grid_size'] = grid_size

    # 获取锚框配置
    anchor_args = param['postprocess']['anchor_args']

    # 提取各维度的体素尺寸
    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    # 保存体素尺寸
    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    # 计算各方向的体素网格数量（使用ceil向上取整）
    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    # 更新后处理配置
    param['postprocess'].update({'anchor_args': anchor_args})

    # 返回更新后的参数字典
    return param
