# FreeAlign (ICRA 2024)

**Robust Collaborative Perception without External Localization and Clock Devices**

![FreeAlign Banner](images/newbanner.png)

[Paper](https://arxiv.org/pdf/2405.02965) | [Project Page](https://siheng-chen.github.io/) | [中文文档](README_CN.md)

## 目录

- [项目简介](#项目简介)
- [安装指南](#安装指南)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [训练指南](#训练指南)
- [推理与测试](#推理与测试)
- [FreeAlign 配置详解](#freealign-配置详解)
- [项目结构](#项目结构)
- [支持的模型与方法](#支持的模型与方法)
- [可视化工具](#可视化工具)
- [常见问题](#常见问题)
- [致谢](#致谢)
- [引用](#引用)

---

## 项目简介

FreeAlign 是一种新颖的协同感知系统，能够在**不依赖外部定位和时钟设备**的情况下实现鲁棒的时空对齐。其核心思想是通过识别多个代理感知数据中固有的几何模式来进行对齐。

### 核心特性

- 无需 GPS/RTK 定位设备
- 无需同步时钟
- 基于图匹配的时空对齐
- 可与现有协同感知方法无缝集成

### 核心模块

| 模块 | 论文对应 | 功能 |
|------|----------|------|
| Salient-Object Graph Learning | Section IV-A | 使用 GNN 学习检测框之间的边特征 |
| Multi-Anchor Subgraph Searching (MASS) | Section IV-B | 寻找两个图之间的最大公共子图 |
| Relative Transformation Calculation | Section IV-C | 计算相对位姿和时间延迟 |

---

## 安装指南

### 环境要求

- Python >= 3.7
- PyTorch >= 1.10
- CUDA >= 11.3

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/chenluohan666/FreeAlign.git
cd FreeAlign

# 2. 创建 Conda 环境
conda env create -f environment.yml
conda activate freealign

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 spconv (稀疏卷积库)
pip install spconv-cu113  # 根据 CUDA 版本选择

# 5. 安装 g2o (位姿图优化)
# 参考: https://github.com/uoip/g2opy

# 6. 编译自定义 CUDA 算子
cd opencood/pcdet_utils/iou3d_nms
python setup.py install

cd ../pointnet2
python setup.py install

cd ../roiaware_pool3d
python setup.py install
```

### 详细安装指南

更多安装细节请参考：
- [OpenCOOD 安装文档](https://opencood.readthedocs.io/en/latest/md_files/installation.html)
- [CoAlign 安装指南](https://udtkdfu8mk.feishu.cn/docx/LlMpdu3pNoCS94xxhjMcOWIynie)

---

## 数据准备

### 支持的数据集

| 数据集 | 类型 | 链接 |
|--------|------|------|
| OPV2V | 仿真 | [下载](https://opv2v.github.io/) |
| DAIR-V2X | 真实世界 | [下载](https://thudairv2x.netlify.app/) |
| V2XSet | 仿真 | [下载](https://xzhis.me/V2XSet/) |
| V2X-Sim | 仿真 | [下载](https://sites.google.com/view/v2x-sim/home) |

### DAIR-V2X-C 补充标注

原始 DAIR-V2X 仅标注车辆侧相机视野内的 3D 框，我们补充了缺失的标注以支持 360 度检测。

**下载链接**: [Google Drive](https://drive.google.com/file/d/13g3APNeHBVjPcF-nTuUoNOSGyTzdfnUK/view?usp=sharing)

### 数据目录结构

```
dataset/
├── my_dair_v2x/
│   └── v2x_c/
│       └── cooperative-vehicle-infrastructure/
│           ├── train.json
│           ├── val.json
│           └── data/
│               ├── vehicle-side/
│               └── infrastructure-side/
├── OPV2V/
│   └── train/
│   └── validate/
│   └── test/
└── V2XSET/
    └── train/
    └── validate/
    └── test/
```

---

## 快速开始

### 下载预训练模型

从百度网盘下载预训练模型：
- [百度网盘](https://pan.baidu.com/s/16JvC7aVTuobUUL99l6NmQA?pwd=5j6c)

### 快速推理

```bash
python opencood/tools/inference.py --model_dir opencood/logs/dairv2x_point_pillar_lidar_max_freealign
```

---

## 训练指南

### 训练流程概览

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Stage 1    │ --> │  Stage 2    │ --> │  Inference  │
│ 预计算检测框 │     │  训练主模型  │     │   测试评估   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Stage 1: 预计算检测框

FreeAlign 需要第一阶段检测结果进行图匹配。

```bash
# 预计算 Stage 1 检测框
python opencood/tools/pose_graph_pre_calc.py \
    --hypes_yaml opencood/hypes_yaml/dairv2x/pointpillar_max_freealign.yaml
```

### Stage 2: 训练主模型

#### 单 GPU 训练

```bash
python opencood/tools/train.py \
    --hypes_yaml opencood/hypes_yaml/dairv2x/pointpillar_max_freealign.yaml
```

#### 多 GPU 分布式训练

```bash
# 使用 torch.distributed.launch
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    opencood/tools/train_ddp.py \
    --hypes_yaml opencood/hypes_yaml/dairv2x/pointpillar_max_freealign.yaml

# 或使用 torchrun (推荐 PyTorch >= 1.10)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    opencood/tools/train_ddp.py \
    --hypes_yaml opencood/hypes_yaml/dairv2x/pointpillar_max_freealign.yaml
```

#### 从检查点恢复训练

```bash
python opencood/tools/train.py \
    --hypes_yaml opencood/hypes_yaml/dairv2x/pointpillar_max_freealign.yaml \
    --model_dir opencood/logs/dairv2x_point_pillar_lidar_max_freealign
```

### 训练参数说明

| 参数 | 说明 |
|------|------|
| `--hypes_yaml` | 配置文件路径 |
| `--model_dir` | 恢复训练的检查点目录 |
| `--fusion_method` | 融合方法: `intermediate`, `late`, `early`, `no` |

---

## 推理与测试

### 标准推理

```bash
python opencood/tools/inference.py \
    --model_dir opencood/logs/dairv2x_point_pillar_lidar_max_freealign \
    --fusion_method intermediate
```

### 带噪声推理 (测试鲁棒性)

```bash
python opencood/tools/inference_w_noise.py \
    --model_dir opencood/logs/dairv2x_point_pillar_lidar_max_freealign \
    --fusion_method intermediate
```

### 推理参数说明

| 参数 | 说明 |
|------|------|
| `--model_dir` | 模型检查点目录 |
| `--fusion_method` | 融合方法 |
| `--save_vis_interval` | 可视化保存间隔 |
| `--save_npy` | 是否保存预测结果为 npy |
| `--no_score` | 是否打印预测分数 |

### 融合方法选项

| 方法 | 说明 |
|------|------|
| `intermediate` | 中间特征融合 (默认) |
| `late` | 后期融合 |
| `early` | 早期融合 |
| `no` | 无融合 (单代理) |
| `single` | 单代理检测 |

---

## FreeAlign 配置详解

### 配置文件示例

配置文件位于 `opencood/hypes_yaml/dairv2x/pointpillar_max_freealign.yaml`

### FreeAlign 核心配置

```yaml
box_align:
  # Stage 1 检测结果路径
  train_result: "opencood/logs/coalign_precalc/dairv2x/train/stage1_boxes.json"
  val_result: "opencood/logs/coalign_precalc/dairv2x/val/stage1_boxes.json"
  test_result: "opencood/logs/coalign_precalc/dairv2x/test/stage1_boxes.json"
  
  # FreeAlign 模式
  free_align: "train"  # "train", "val", "none"
  
  # 是否使用 GNN 学习边特征
  gnn: false
  
  # 是否不使用外部位姿 (True = 完全依赖 FreeAlign)
  no_pose: true
  
  # 最小锚点数量 (MASS 算法参数)
  min_anchor: 4
  
  # 锚点匹配误差阈值 (米)
  anchor_error: 0.3
  
  # 检测框匹配误差阈值 (米)
  box_error: 0.5
  
  # 位姿图优化参数
  args:
    use_uncertainty: true      # 使用不确定性权重
    landmark_SE2: true         # 使用 SE(2) 位姿估计
    adaptive_landmark: false   # 自适应锚点选择
    normalize_uncertainty: false
    abandon_hard_cases: true   # 放弃困难样本
    drop_hard_boxes: true      # 丢弃难以匹配的框
```

### 噪声配置 (测试鲁棒性)

```yaml
noise_setting:
  add_noise: false  # 是否添加噪声
  args:
    pos_std: 0.2    # 位置噪声标准差 (米)
    rot_std: 0.2    # 旋转噪声标准差 (弧度)
    pos_mean: 0     # 位置噪声均值
    rot_mean: 0     # 旋转噪声均值
```

---

## 项目结构

```
FreeAlign/
├── freealign/                    # FreeAlign 核心实现
│   ├── models/                   # 匹配网络模型
│   │   ├── matching.py           # 匹配网络主入口
│   │   ├── superbox.py           # 检测框描述子提取
│   │   ├── superbevglue.py       # BEV 图匹配网络
│   │   ├── superglue.py          # SuperGlue 匹配网络
│   │   └── graph/                # 图神经网络模块
│   ├── graph/                    # 图匹配算法
│   │   ├── greedy_match.py       # 贪婪子图匹配 (MASS)
│   │   └── greedy_match_triangular.py
│   ├── match/                    # 匹配策略
│   │   ├── match_v7_with_detection.py  # FreeAlign 训练实现
│   │   └── match_superglue.py    # SuperGlue 匹配
│   └── optimize.py               # 相对位姿优化 (SVD/ICP)
│
├── opencood/                     # OpenCOOD 框架
│   ├── tools/                    # 工具脚本
│   │   ├── train.py              # 单 GPU 训练
│   │   ├── train_ddp.py          # 多 GPU 分布式训练
│   │   ├── inference.py          # 标准推理
│   │   ├── inference_w_noise.py  # 带噪声推理
│   │   ├── pose_graph_pre_calc.py  # Stage 1 预计算
│   │   └── pose_graph_evaluate.py  # 位姿图评估
│   ├── models/                   # 模型实现
│   │   ├── point_pillar_baseline_multiscale.py  # 主模型
│   │   ├── point_pillar_coalign.py
│   │   ├── sub_modules/          # 子模块
│   │   │   └── box_align_v2.py   # 位姿图优化核心
│   │   └── fuse_modules/         # 融合模块
│   ├── data_utils/               # 数据处理
│   │   └── datasets/
│   │       └── intermediate_fusion_dataset.py  # FreeAlign 调用入口
│   ├── hypes_yaml/               # 配置文件
│   │   ├── dairv2x/              # DAIR-V2X 配置
│   │   ├── opv2v/                # OPV2V 配置
│   │   ├── v2xset/               # V2XSet 配置
│   │   └── v2xsim/               # V2X-Sim 配置
│   ├── loss/                     # 损失函数
│   └── utils/                    # 工具函数
│
├── docs/                         # 文档
├── images/                       # 图片资源
└── requirements.txt              # 依赖列表
```

---

## 支持的模型与方法

### 检测器

| 模型 | 说明 |
|------|------|
| PointPillar | 基线检测器 |
| SECOND | 稀疏卷积检测器 |
| F-PVRCNN | 高精度检测器 |

### 协同方法

| 方法 | 配置文件 | 说明 |
|------|----------|------|
| Late Fusion | `*_late.yaml` | 后期融合 |
| F-Cooper | `*_fcooper.yaml` | 特征融合 |
| V2VNet | `*_v2vnet.yaml` | 图神经网络融合 |
| V2X-ViT | `*_v2xvit.yaml` | Transformer 融合 |
| Where2comm | `where2comm.yaml` | 空间置信度融合 |
| DiscoNet | `*_disconet.yaml` | 知识蒸馏融合 |
| CoAlign | `*_coalign.yaml` | 协同对齐 |
| **FreeAlign** | `*_freealign.yaml` | 无外部设备的鲁棒对齐 |

### 配置文件路径

```bash
# DAIR-V2X 数据集
opencood/hypes_yaml/dairv2x/

# OPV2V 数据集
opencood/hypes_yaml/opv2v/

# V2X-Sim 数据集
opencood/hypes_yaml/v2xsim/

# V2XSet 数据集
opencood/hypes_yaml/v2xset/
```

---

## 可视化工具

### 生成可视化结果

```bash
# 保存推理可视化
python opencood/tools/save_vis.py \
    --model_dir opencood/logs/dairv2x_point_pillar_lidar_max_freealign
```

### 可视化配置

在配置文件中设置：

```yaml
postprocess:
  gt_range: *cav_lidar
  # ... 其他参数

# 可视化脚本会自动生成 BEV 视图和 3D 视图
```

---

## 常见问题

### Q1: 如何切换不同的协同方法？

修改配置文件中的 `fusion_method`：

```yaml
model:
  args:
    fusion_method: max  # 可选: max, att
```

### Q2: 如何测试不同噪声水平下的性能？

修改配置文件中的噪声设置：

```yaml
noise_setting:
  add_noise: true
  args:
    pos_std: 0.5  # 调整噪声标准差
    rot_std: 0.5
```

### Q3: 如何禁用 FreeAlign？

设置 `free_align: "none"` 或删除 `box_align` 配置块。

### Q4: Stage 1 检测框如何生成？

运行 `pose_graph_pre_calc.py`，它会使用预训练模型生成检测框并保存为 JSON 文件。

### Q5: 训练时显存不足怎么办？

- 减小 `batch_size`
- 减小 `max_voxel_train`
- 使用 `train_ddp.py` 进行多卡训练并减小单卡 batch_size

---

## 致谢

本项目基于以下开源工作：

- [CoAlign](https://github.com/yifanlu0227/CoAlign) - 协同对齐框架
- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) - 协同感知框架
- [g2opy](https://github.com/uoip/g2opy) - 图优化库
- [d3d](https://github.com/cmpute/d3d) - 深度 3D 检测

---

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@inproceedings{lei2024freealign,
  title={Robust Collaborative Perception without External Localization and Clock Devices},
  author={Lei, Zixing and Ni, Zhenyang and Han, Ruize and Tang, Shuo and Wang, Dingju and Feng, Chen and Chen, Siheng and Wang, Yanfeng},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024}
}
```

---

## License

本项目采用 MIT License，详见 [LICENSE](LICENSE) 文件。