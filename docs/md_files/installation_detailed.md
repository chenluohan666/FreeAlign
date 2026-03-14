# OpenCOOD框架安装详细教程

## 概述

OpenCOOD（Open Collaborative Object Detection）是一个开源的协作式3D目标检测框架，专为车联网（V2V）场景设计，支持多个智能车辆（CAV）协同进行目标检测。

## 系统/硬件要求

### 系统要求
- **操作系统**：OpenCOOD在Ubuntu 18.04下经过测试
  - 虽然官方仅在Ubuntu 18.04下测试，但通常可以在其他Linux发行版上运行
  - Windows用户建议使用WSL2或虚拟机来运行

### 硬件要求
- **GPU**：建议至少6GB显存
  - 3D目标检测模型通常需要大量GPU内存
  - 更大的显存可以支持更大的批处理大小，提高训练效率
  - 推荐使用NVIDIA GPU，并安装相应的CUDA驱动

- **硬盘空间**：建议预留100GB用于数据下载
  - OPV2V数据集（开放车辆到车辆协作感知数据集）体积庞大
  - 训练过程中还会产生模型检查点、日志等文件
  - 预留空间有助于避免训练过程中因空间不足而中断

- **Python版本**：必须使用Python 3.7
  - 框架特定依赖项可能与Python 3.7兼容性最佳
  - 使用其他版本的Python可能导致依赖安装失败

## 安装步骤

### 1. 依赖安装

首先，如果尚未下载OpenCOOD，需要从GitHub克隆项目：

```bash
git clone https://github.com/DerrickXuNu/OpenCOOD.git
cd OpenCOOD
```

接下来创建conda环境并安装依赖：

```bash
conda env create -f environment.yml
conda activate opencood
python setup.py develop
```

**详细步骤说明：**

- `conda env create -f environment.yml`：根据environment.yml文件创建一个包含所有依赖项的conda环境
  - environment.yml文件定义了项目所需的所有Python包
  - conda会自动解析依赖关系并安装兼容版本

- `conda activate opencood`：激活名为opencood的conda环境
  - 所有后续操作都应在该环境中执行
  - 确保使用正确的Python解释器和包

- `python setup.py develop`：以开发模式安装OpenCOOD
  - 开发模式允许在修改源代码后无需重新安装即可生效
  - 方便开发和调试

如果conda安装失败，可以通过pip安装：

```bash
pip install -r requirements.txt
```

- `requirements.txt`：包含项目所需的所有Python包及其版本
- 当conda环境创建失败时的备用方案

### 2. PyTorch安装（>=1.8）

前往 https://pytorch.org/ 安装PyTorch CUDA版本。

**PyTorch安装详细说明：**

PyTorch是深度学习框架，OpenCOOD基于PyTorch实现。需要安装支持CUDA的版本以利用GPU加速。

在PyTorch官网选择：
- 稳定版本（Stable）而非预览版本
- PyTorch版本（1.8或更高）
- 操作系统（Linux/Windows）
- 包管理器（Conda/Pip）
- 语言（Python）
- CUDA版本（根据系统中的CUDA版本选择）

例如，对于CUDA 10.2，安装命令可能是：
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

### 3. Spconv安装（1.2.1版本）

OpenCOOD目前使用旧版spconv来生成体素特征，未来将升级到spconv 2.0。

要安装spconv 1.2.1，请按照官方指南：https://github.com/traveller59/spconv/tree/v1.2.1

#### 安装spconv 1.2.1的提示：

1. **CMake版本**：确保cmake版本>=3.13.2
   - 检查版本：`cmake --version`
   - 如需升级：`conda install cmake` 或从官网下载

2. **CUDA环境**：CUDNN和CUDA运行时库需要安装在机器上
   - 检查CUDA版本：`nvcc --version`
   - 检查CUDNN版本：`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`

**Spconv安装详细步骤：**

1. 克隆spconv仓库：
   ```bash
   git clone https://github.com/traveller59/spconv.git
   cd spconv
   git checkout v1.2.1
   ```

2. 安装依赖：
   ```bash
   pip install spconv-cu102  # 对于CUDA 10.2
   # 或者手动编译
   python setup.py bdist_wheel
   pip install dist/spconv-*.whl
   ```

### 4. Bbx IOU CUDA版本编译

安装边界框IOU（交并比）计算的CUDA版本：

```bash
python opencood/utils/setup.py build_ext --inplace
```

**编译说明：**

- 此步骤编译CUDA加速的NMS（非最大抑制）计算
- NMS是目标检测中的关键步骤，用于去除冗余的边界框
- CUDA加速可以显著提高计算速度
- `--inplace`参数表示在当前目录编译，而不是安装到系统路径

## 常见问题及解决方案

### 1. 环境兼容性问题
- 确保所有依赖项版本兼容
- 如果遇到问题，尝试创建新的conda环境

### 2. CUDA相关问题
- 确保CUDA、CUDNN和PyTorch版本兼容
- 检查环境变量设置（如LD_LIBRARY_PATH）

### 3. 编译错误
- 确保已安装编译工具（如gcc、g++）
- 检查CUDA开发工具包是否正确安装

## 验证安装

安装完成后，可以通过以下方式验证：

1. 检查Python环境：
   ```python
   import torch
   print(torch.__version__)  # 应显示PyTorch版本号
   print(torch.cuda.is_available())  # 应返回True
   ```

2. 测试OpenCOOD导入：
   ```python
   import opencood
   print(opencood.__file__)  # 应显示OpenCOOD的安装路径
   ```

## 后续步骤

安装完成后，您可以：
1. 下载OPV2V数据集
2. 配置模型参数
3. 运行示例训练脚本
4. 开始进行协作感知实验

安装是使用OpenCOOD框架的第一步，确保所有依赖正确安装对后续的实验至关重要。