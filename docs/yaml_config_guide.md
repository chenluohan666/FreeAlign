# FreeAlign YAML 配置文件完整指南

本文档详细说明了 FreeAlign 项目中所有 YAML 配置文件的作用、用途和配置参数。

---

## 目录结构总览

```
D:\FreeAlign\
├── point_pillar_intermediate_fusion_zh.yaml    # 根目录配置模板（中文注释）
├── environment.yml                              # Conda 环境配置
└── opencood/
    └── hypes_yaml/                              # 配置文件主目录
        ├── dairv2x/                             # DAIR-V2X 数据集配置
        ├── opv2v/                               # OPV2V 数据集配置
        ├── v2xset/                              # V2X-Set 数据集配置
        └── v2xsim/                              # V2X-Sim 数据集配置
```

---

## 一、根目录配置文件

### 1.1 `point_pillar_intermediate_fusion_zh.yaml`

**用途**: 带有详细中文注释的中间融合配置模板，适合初学者理解配置参数。

**主要功能**:
- 定义模型训练的基础参数（batch_size、epoches、学习率等）
- 配置数据预处理流程（体素化参数）
- 设置数据增强策略（翻转、旋转、缩放）
- 定义模型架构（PointPillar 骨干网络）
- 配置损失函数和优化器

**适用场景**: 作为自定义配置的参考模板，理解各参数含义。

### 1.2 `environment.yml`

**用途**: Conda 环境配置文件，定义项目运行所需的 Python 环境和依赖包。

**内容**:
```yaml
name: opencood                    # 环境名称
channels:
  - defaults
dependencies:
  - pip==21.1.2
  - python==3.7.11                # Python 版本
  - pip:
    - matplotlib==3.4.2
    - numpy
    - open3d                      # 3D 点云可视化
    - opencv-python==4.5.5.62
    - cython
    - pygame
    - tensorboardX                # 训练可视化
    - shapely
    - einops
```

**使用方法**:
```bash
conda env create -f environment.yml
conda activate opencood
```

---

## 二、DAIR-V2X 数据集配置 (`opencood/hypes_yaml/dairv2x/`)

DAIR-V2X 是真实世界的车路协同感知数据集，采集自中国道路场景。

### 2.1 目录结构

```
dairv2x/
├── pointpillar_max_freealign.yaml       # FreeAlign 核心配置
├── where2comm.yaml                       # Where2comm 方法配置
├── visualization_dair.yaml               # 可视化专用配置
└── lidar_only_with_noise/               # 带噪声实验配置
    ├── coalign/                          # CoAlign 方法配置
    │   ├── pointpillar_coalign.yaml      # CoAlign 完整版
    │   ├── pointpillar_coalign_woba.yaml # CoAlign 无位姿校正版
    │   ├── pointpillar_uncertainty.yaml  # 不确定性估计版
    │   ├── precalc.yaml                  # 预计算配置
    │   └── SECOND_uncertainty.yaml       # SECOND 骨干网络版
    ├── pointpillar_early.yaml            # 早期融合
    ├── pointpillar_single.yaml           # 单车检测
    ├── pointpillar_v2vnet.yaml           # V2VNet 融合
    ├── pointpillar_v2vnet_robust.yaml    # V2VNet 鲁棒版
    ├── pointpillar_v2xvit.yaml           # V2X-ViT 融合
    ├── pointpillar_selfatt.yaml          # 自注意力融合
    ├── pointpillar_fcooper.yaml          # F-Cooper 融合
    ├── pointpillar_mash.yaml             # MASH 融合
    ├── pointpillar_disconet.yaml         # DiscoNet 知识蒸馏
    ├── SECOND.yaml                        # SECOND 检测器
    ├── SECOND_early.yaml                 # SECOND 早期融合
    ├── fpvrcnn.yaml                      # FPV-RCNN 检测器
    └── fvoxelrcnn.yaml                   # F-VoxelRCNN 检测器
```

### 2.2 核心配置文件详解

#### 2.2.1 `pointpillar_max_freealign.yaml` - FreeAlign 核心配置

**用途**: FreeAlign 论文的核心实验配置，实现无外部定位设备的时空对齐。

**关键特性**:
- `no_pose: true` - 完全不使用外部位姿，从零恢复
- `box_align` 模块 - 检测框对齐算法配置
- 最大融合策略 (`fusion_method: max`)

**核心参数**:
```yaml
box_align:
  train_result: "opencood/logs/coalign_precalc/dairv2x/train/stage1_boxes.json"
  val_result: "opencood/logs/coalign_precalc/dairv2x/val/stage1_boxes.json"
  free_align: "train"          # 启用 FreeAlign
  gnn: false                   # 不使用 GNN 边特征学习
  no_pose: true                # 不使用外部位姿（核心参数）
  min_anchor: 4                # 最小锚点数量
  anchor_error: 0.3            # 锚点匹配误差阈值(米)
  box_error: 0.5               # 检测框匹配误差阈值(米)
  args:
    use_uncertainty: true      # 使用检测不确定性
    landmark_SE2: true         # 使用 SE(2) 位姿表示
    abandon_hard_cases: true   # 放弃困难样本
```

**适用场景**: 
- 论文复现实验
- 无 GPS/RTK 设备的协同感知
- 位姿噪声鲁棒性测试

#### 2.2.2 `where2comm.yaml` - Where2comm 配置

**用途**: Where2comm 方法的配置，通过空间置信度图指导通信内容选择。

**核心特点**:
- 空间置信度图生成
- 自适应通信选择
- 带宽效率优化

**关键参数**:
```yaml
model:
  core_method: point_pillar_where2comm
  args:
    fusion_args:
      in_channels: 256
      n_head: 8
      only_attention: true
      multi_scale: true
```

**适用场景**: 带宽受限的协同感知场景。

#### 2.2.3 `lidar_only_with_noise/coalign/pointpillar_coalign.yaml`

**用途**: CoAlign 方法的完整配置，使用外部位姿作为初始值进行校正。

**与 FreeAlign 的区别**:
```yaml
# FreeAlign
no_pose: true      # 完全不使用外部位姿

# CoAlign
# 无 no_pose 参数，使用带噪声的外部位姿作为初始值
noise_setting:
  add_noise: true  # 添加位姿噪声
```

**适用场景**: 有 GPS/RTK 但精度有限的场景。

#### 2.2.4 `lidar_only_with_noise/coalign/pointpillar_coalign_woba.yaml`

**用途**: CoAlign 基线配置，禁用 box_align 模块。

**特点**:
- `box_align` 模块被注释掉
- 仅使用注意力融合，无位姿校正
- 用于消融实验对比

### 2.3 融合方法配置对比

| 配置文件 | 融合方法 | 核心特点 | 通信轮次 |
|---------|---------|---------|---------|
| `pointpillar_early.yaml` | early | 原始点云融合 | 1-round |
| `pointpillar_single.yaml` | late | 单车独立检测后融合 | 0-round |
| `pointpillar_v2vnet.yaml` | intermediate | 图神经网络融合 | 1-round |
| `pointpillar_v2xvit.yaml` | intermediate | Vision Transformer 融合 | 1-round |
| `pointpillar_selfatt.yaml` | intermediate | 自注意力融合 | 1-round |
| `pointpillar_fcooper.yaml` | intermediate | 特征级最大融合 | 1-round |
| `pointpillar_disconet.yaml` | intermediate | 知识蒸馏融合 | 1-round |

### 2.4 检测器配置对比

| 配置文件 | 检测器 | 体素大小 | 特点 |
|---------|-------|---------|------|
| `pointpillar_*.yaml` | PointPillar | [0.4, 0.4, 4] | 高效柱状特征 |
| `SECOND.yaml` | SECOND | [0.1, 0.1, 0.1] | 稀疏卷积，更精细 |
| `fpvrcnn.yaml` | FPV-RCNN | [0.1, 0.1, 0.1] | 两阶段，高精度 |

---

## 三、OPV2V 数据集配置 (`opencood/hypes_yaml/opv2v/`)

OPV2V 是仿真多车协同感知数据集，基于 CARLA 模拟器生成。

### 3.1 目录结构

```
opv2v/
├── config.yaml                          # 默认基础配置
├── eval_intermediate.yaml               # 中间融合评估配置
├── visualization_opv2v.yaml             # 可视化配置
└── lidar_only_with_noise/              # 带噪声实验配置
    ├── coalign/                         # CoAlign 配置（同 dairv2x）
    ├── pointpillar_early.yaml           # 早期融合
    ├── pointpillar_single.yaml          # 单车检测
    ├── pointpillar_v2vnet.yaml          # V2VNet
    ├── pointpillar_v2vnet_robust.yaml   # V2VNet 鲁棒版
    ├── pointpillar_v2xvit.yaml          # V2X-ViT
    ├── pointpillar_selfatt.yaml         # 自注意力
    ├── pointpillar_selfatt_singlescale.yaml # 单尺度自注意力
    ├── pointpillar_fcooper.yaml         # F-Cooper
    ├── pointpillar_mash.yaml            # MASH
    ├── pointpillar_disconet.yaml        # DiscoNet
    ├── SECOND.yaml                       # SECOND
    ├── SECOND_early.yaml                # SECOND 早期融合
    ├── fpvrcnn.yaml                     # FPV-RCNN
    └── fvoxelrcnn.yaml                  # F-VoxelRCNN
```

### 3.2 核心配置文件详解

#### 3.2.1 `config.yaml` - 默认基础配置

**用途**: OPV2V 数据集的默认配置，使用最大融合策略。

**数据路径**:
```yaml
root_dir: dataset/OPV2V/train
validate_dir: dataset/OPV2V/validate
test_dir: dataset/OPV2V/test
```

**特点**:
- `add_noise: false` - 不添加噪声，使用 GT 位姿
- `fusion_method: max` - 最大融合
- `comm_range: 70` - 通信范围 70 米

#### 3.2.2 `lidar_only_with_noise/pointpillar_v2vnet.yaml`

**用途**: V2VNet 图神经网络融合配置。

**核心模块**:
```yaml
model:
  args:
    fusion_method: v2vnet
    v2vnet:
      num_iteration: 2          # 图传播迭代次数
      in_channels: 256
      gru_flag: true            # 使用 GRU 更新节点特征
      agg_operator: "max"       # 聚合操作：max 或 avg
      conv_gru:
        H: 50
        W: 176
        num_layers: 1
        kernel_size: [[3,3]]
```

**适用场景**: 多车协同场景，需要考虑车辆间关系建模。

#### 3.2.3 `lidar_only_with_noise/pointpillar_v2xvit.yaml`

**用途**: V2X-ViT Vision Transformer 融合配置。

**核心模块**:
```yaml
model:
  args:
    fusion_method: v2xvit
    v2xvit:
      transformer:
        encoder:
          num_blocks: 1         # 每层融合块数量
          depth: 3              # 编码器层数
          use_roi_mask: true    # 使用 ROI 掩码
          cav_att_config:       # 智能体注意力
            dim: 256
            heads: 8
            dim_head: 32
            dropout: 0.3
          pwindow_att_config:   # 空间窗口注意力
            dim: 256
            heads: [16, 8, 4]
            dim_head: [16, 32, 64]
            window_size: [4, 8, 16]
```

**适用场景**: 大规模车路协同场景，需要处理异构智能体。

#### 3.2.4 `lidar_only_with_noise/pointpillar_disconet.yaml`

**用途**: DiscoNet 知识蒸馏融合配置。

**核心特点**:
```yaml
kd_flag:
  teacher_model: point_pillar_disconet_teacher
  teacher_model_config: *model_args
  teacher_path: "opencood/logs/opv2v_point_pillar_lidar_early_xxx.pth"

loss:
  core_method: point_pillar_disconet_loss
  args:
    kd:
      weight: 10000             # 知识蒸馏损失权重
```

**适用场景**: 需要从教师网络学习协同策略。

#### 3.2.5 `visualization_opv2v.yaml`

**用途**: 可视化专用配置，用于生成检测结果可视化。

**特点**:
```yaml
name: visualization
input_source: ['camera']         # 使用相机输入
label_type: 'camera'
only_vis_ego: true               # 仅可视化 Ego
add_data_extension: ['bev_visibility.png']  # 添加 BEV 可见性图
```

---

## 四、V2X-Set 数据集配置 (`opencood/hypes_yaml/v2xset/`)

V2X-Set 是 V2X-Sim 数据集的子集，专注于车路协同场景。

### 4.1 目录结构

```
v2xset/
├── pointpillar_max_multiscale.yaml    # 多尺度最大融合配置
└── SECOND_uncertainty.yaml            # SECOND 不确定性估计配置
```

### 4.2 核心配置文件详解

#### 4.2.1 `pointpillar_max_multiscale.yaml`

**用途**: V2X-Set 数据集的基础配置，多尺度特征融合。

**数据路径**:
```yaml
root_dir: "dataset/V2XSET/train"
validate_dir: "dataset/V2XSET/validate"
test_dir: "dataset/V2XSET/test"
```

**特点**:
- `add_noise: false` - 不添加噪声
- `fusion_method: max` - 最大融合
- `comm_range: 70` - 通信范围 70 米

---

## 五、V2X-Sim 数据集配置 (`opencood/hypes_yaml/v2xsim/`)

V2X-Sim 是仿真车路协同感知数据集。

### 5.1 目录结构

```
v2xsim/
├── v2xsim_mutiscale_max.yaml          # 多尺度最大融合配置
├── visualization.yaml                  # 可视化配置
└── lidar_only_with_noise/             # 带噪声实验配置
    ├── coalign/                        # CoAlign 配置
    └── ... (其他配置与 dairv2v/opv2v 类似)
```

### 5.2 核心配置文件详解

#### 5.2.1 `v2xsim_mutiscale_max.yaml`

**用途**: V2X-Sim 数据集的多尺度最大融合配置。

**特点**: 与其他数据集配置类似，针对 V2X-Sim 数据路径调整。

---

## 六、配置文件通用结构

所有配置文件都遵循以下通用结构：

```yaml
# ==================== 基本信息 ====================
name: 实验名称                      # 用于保存模型和日志
root_dir: 训练数据路径
validate_dir: 验证数据路径
test_dir: 测试数据路径

# ==================== 噪声设置 ====================
noise_setting:
  add_noise: true/false            # 是否添加位姿噪声
  args:
    pos_std: 位置噪声标准差(米)
    rot_std: 旋转噪声标准差(弧度)

# ==================== 训练参数 ====================
train_params:
  batch_size: 批次大小
  epoches: 训练轮数
  eval_freq: 评估频率
  save_freq: 保存频率
  max_cav: 最大协同车辆数

# ==================== 融合配置 ====================
fusion:
  core_method: 'intermediate/early/late'
  dataset: '数据集名称'
  args:
    proj_first: false/true        # 是否先投影再体素化

# ==================== FreeAlign 配置 ====================
box_align:
  train_result: 训练集 Stage1 检测结果路径
  val_result: 验证集 Stage1 检测结果路径
  free_align: "train/val/none"
  gnn: true/false
  no_pose: true/false             # 核心参数
  min_anchor: 最小锚点数
  anchor_error: 锚点误差阈值
  box_error: 检测框误差阈值

# ==================== 预处理配置 ====================
preprocess:
  core_method: 'SpVoxelPreprocessor'
  args:
    voxel_size: [x, y, z]         # 体素尺寸
    max_points_per_voxel: 每体素最大点数
    max_voxel_train: 训练时最大体素数
    max_voxel_test: 测试时最大体素数
  cav_lidar_range: [xmin, ymin, zmin, xmax, ymax, zmax]

# ==================== 数据增强配置 ====================
data_augment:
  - NAME: random_world_flip
    ALONG_AXIS_LIST: ['x']
  - NAME: random_world_rotation
    WORLD_ROT_ANGLE: [min, max]
  - NAME: random_world_scaling
    WORLD_SCALE_RANGE: [min, max]

# ==================== 后处理配置 ====================
postprocess:
  core_method: 'VoxelPostprocessor'
  anchor_args:
    l: 锚框长度
    w: 锚框宽度
    h: 锚框高度
    r: [朝向角度]
  target_args:
    pos_threshold: 正样本阈值
    neg_threshold: 负样本阈值
  nms_thresh: NMS 阈值

# ==================== 模型配置 ====================
model:
  core_method: 模型名称
  args:
    pillar_vfe: VFE 配置
    base_bev_backbone: 骨干网络配置
    fusion_method: 融合方法

# ==================== 损失函数配置 ====================
loss:
  core_method: 损失函数名称
  args:
    cls: 分类损失配置
    reg: 回归损失配置
    dir: 方向损失配置

# ==================== 优化器配置 ====================
optimizer:
  core_method: 'Adam'
  lr: 学习率
  args:
    weight_decay: 权重衰减

# ==================== 学习率调度器配置 ====================
lr_scheduler:
  core_method: 'multistep'
  gamma: 衰减系数
  step_size: [衰减节点]
```

---

## 七、配置文件选择指南

### 7.1 按数据集选择

| 数据集 | 配置目录 | 特点 |
|-------|---------|------|
| DAIR-V2X | `dairv2x/` | 真实世界数据，车路协同 |
| OPV2V | `opv2v/` | 仿真多车协同 |
| V2X-Set | `v2xset/` | V2X-Sim 子集 |
| V2X-Sim | `v2xsim/` | 仿真车路协同 |

### 7.2 按融合方法选择

| 融合方法 | 配置文件关键词 | 特点 |
|---------|--------------|------|
| 早期融合 | `*_early.yaml` | 原始数据融合，精度高，通信大 |
| 后期融合 | `*_single.yaml`, `late` | 检测结果融合，通信小，精度低 |
| 中间融合 | `intermediate` | 特征级融合，平衡精度和通信 |

### 7.3 按应用场景选择

| 应用场景 | 推荐配置 | 理由 |
|---------|---------|------|
| 论文复现 | `pointpillar_max_freealign.yaml` | FreeAlign 核心配置 |
| 无 GPS 场景 | `no_pose: true` 配置 | 完全无外部位姿 |
| 有 GPS 但精度有限 | CoAlign 配置 | 外部位姿作为初始值 |
| 带宽受限 | Where2comm 配置 | 通信效率优化 |
| 多车协同 | V2VNet/V2X-ViT 配置 | 图结构建模 |

---

## 八、自定义配置指南

### 8.1 创建新配置文件

1. 复制最接近的现有配置文件
2. 修改 `name` 参数
3. 调整数据路径
4. 修改模型/融合方法参数
5. 调整训练超参数

### 8.2 关键参数调优建议

| 参数 | 调优建议 |
|------|---------|
| `batch_size` | 根据 GPU 显存调整，越大越稳定 |
| `lr` | Adam 推荐 0.001-0.003 |
| `voxel_size` | 越小越精细，但计算量大 |
| `min_anchor` | 越大越可靠，但成功率降低 |
| `anchor_error` | 根据检测精度调整 |

---

## 九、常见问题

### Q1: FreeAlign 和 CoAlign 的区别？

| 特性 | FreeAlign | CoAlign |
|------|-----------|---------|
| `no_pose` | `true` | 无此参数 |
| 位姿初始化 | 全零 | 使用外部值 |
| 适用场景 | 无 GPS/RTK | 有 GPS 但精度有限 |

### Q2: 如何选择融合方法？

- **精度优先**: 早期融合或 V2X-ViT
- **带宽优先**: Where2comm 或后期融合
- **平衡方案**: 中间融合（最大融合/注意力融合）

### Q3: 如何添加新的数据集？

1. 创建新的配置目录
2. 复制现有配置文件
3. 修改数据路径和参数
4. 调整 `fusion.dataset` 参数

---

## 十、配置文件索引

### 10.1 完整文件列表

| 文件路径 | 用途 |
|---------|------|
| `point_pillar_intermediate_fusion_zh.yaml` | 中文注释模板 |
| `environment.yml` | Conda 环境 |
| `dairv2x/pointpillar_max_freealign.yaml` | FreeAlign 核心 |
| `dairv2x/where2comm.yaml` | Where2comm |
| `dairv2x/lidar_only_with_noise/coalign/*.yaml` | CoAlign 系列 |
| `opv2v/config.yaml` | OPV2V 默认 |
| `opv2v/lidar_only_with_noise/*.yaml` | 各融合方法 |
| `v2xset/pointpillar_max_multiscale.yaml` | V2X-Set 基础 |
| `v2xsim/v2xsim_mutiscale_max.yaml` | V2X-Sim 基础 |

### 10.2 按功能分类

**FreeAlign/CoAlign 相关**:
- `dairv2x/pointpillar_max_freealign.yaml`
- `*/lidar_only_with_noise/coalign/*.yaml`

**融合方法对比**:
- `*/pointpillar_early.yaml` - 早期融合
- `*/pointpillar_single.yaml` - 单车检测
- `*/pointpillar_v2vnet.yaml` - V2VNet
- `*/pointpillar_v2xvit.yaml` - V2X-ViT
- `*/pointpillar_selfatt.yaml` - 自注意力

**检测器对比**:
- `*/pointpillar_*.yaml` - PointPillar
- `*/SECOND*.yaml` - SECOND
- `*/fpvrcnn.yaml` - FPV-RCNN

---

*本文档由 FreeAlign 项目生成，最后更新: 2026-03-17*
