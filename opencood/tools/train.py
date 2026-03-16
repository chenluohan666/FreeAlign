# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
==============================================================================
OpenCOOD/FreeAlign 单机训练脚本
==============================================================================

功能概述:
    本脚本是协同感知模型的单 GPU 训练入口，支持从头训练和断点续训。

使用方法:
    # 从头训练
    python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/opv2v/pointpillar_intermediate.yaml

    # 断点续训
    python opencood/tools/train.py --model_dir opencood/logs/your_model --hypes_yaml config.yaml

训练流程:
    1. 解析命令行参数
    2. 加载 YAML 配置文件
    3. 构建训练/验证数据集
    4. 创建模型、损失函数、优化器
    5. 训练循环 (前向传播 → 计算损失 → 反向传播 → 参数更新)
    6. 验证 & 保存检查点
    7. 训练完成后自动运行推理测试

注意事项:
    - 多 GPU 训练请使用 train_ddp.py
    - FreeAlign 需要先运行 pose_graph_pre_calc.py 生成 Stage1 检测框
==============================================================================
"""

# ============================================================================
# 导入依赖
# ============================================================================

import argparse          # 命令行参数解析
import os                # 文件路径操作
import statistics        # 统计计算 (计算均值)

import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter   # TensorBoard 日志记录

# OpenCOOD 内部模块
import opencood.hypes_yaml.yaml_utils as yaml_utils   # YAML 配置解析
from opencood.tools import train_utils                 # 训练工具函数
from opencood.data_utils.datasets import build_dataset # 数据集构建
import glob
from icecream import ic    # 调试打印工具


# ============================================================================
# 命令行参数解析
# ============================================================================

def train_parser():
    """
    解析命令行参数。

    参数说明:
        --hypes_yaml, -y: 
            YAML 配置文件路径 (必需)
            示例: opencood/hypes_yaml/dairv2x/pointpillar_max_freealign.yaml

        --model_dir: 
            预训练模型目录，用于断点续训 (可选)
            示例: opencood/logs/dairv2x_point_pillar_lidar_max_freealign

        --fusion_method, -f: 
            融合方法，传递给推理脚本 (默认: intermediate)
            可选值: intermediate, late, early, no, single

    返回:
        opt: argparse.Namespace 对象，包含解析后的参数
    """
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    训练主函数。

    执行流程:
        1. 解析参数 → 2. 加载配置 → 3. 构建数据集 → 4. 创建模型
        5. 训练循环 → 6. 验证保存 → 7. 推理测试
    """

    # ========================================================================
    # 第一步: 解析命令行参数并加载配置文件
    # ========================================================================
    opt = train_parser()
    
    # 加载 YAML 配置文件，返回配置字典 hypes
    # hypes 包含: model, loss, optimizer, preprocess, postprocess 等配置
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # ========================================================================
    # 第二步: 构建数据集
    # ========================================================================
    print('Dataset Building')
    
    # 构建训练数据集
    # build_dataset 会根据 hypes['fusion']['core_method'] 自动选择数据集类:
    #   - IntermediateFusionDataset (中间融合，最常用)
    #   - LateFusionDataset (后期融合)
    #   - EarlyFusionDataset (早期融合)
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    
    # 构建验证数据集
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    # ========================================================================
    # 第三步: 创建数据加载器 (DataLoader)
    # ========================================================================
    
    # 训练数据加载器
    train_loader = DataLoader(
        opencood_train_dataset,
        batch_size=hypes['train_params']['batch_size'],  # 批次大小，从配置文件读取
        num_workers=4,                                    # 数据加载进程数
        collate_fn=opencood_train_dataset.collate_batch_train,  # 批处理函数，整理多个样本
        shuffle=True,       # 每个 epoch 打乱数据顺序
        pin_memory=True,    # 锁页内存，加速 GPU 数据传输
        drop_last=True,     # 丢弃不完整的最后一批
        prefetch_factor=2   # 预取因子，每个 worker 预取的批次数
    )
    
    # 验证数据加载器
    val_loader = DataLoader(
        opencood_validate_dataset,
        batch_size=hypes['train_params']['batch_size'],
        num_workers=4,
        collate_fn=opencood_train_dataset.collate_batch_train,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2
    )

    # ========================================================================
    # 第四步: 创建模型
    # ========================================================================
    print('Creating Model')
    
    # 根据配置创建模型
    # create_model 会根据 hypes['model']['core_method'] 动态导入模型类:
    #   - point_pillar_baseline_multiscale (FreeAlign 默认)
    #   - point_pillar_coalign (CoAlign)
    #   - point_pillar_v2vnet (V2VNet)
    #   - 等等...
    model = train_utils.create_model(hypes)
    
    # 设置设备 (优先使用 GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 记录最低验证损失，用于保存最佳模型
    lowest_val_loss = 1e5      # 初始化为很大的值
    lowest_val_epoch = -1      # 记录最佳 epoch

    # ========================================================================
    # 第五步: 创建损失函数
    # ========================================================================
    
    # 根据配置创建损失函数
    # create_loss 会根据 hypes['loss']['core_method'] 选择:
    #   - point_pillar_loss: 分类(Focal) + 回归(Smooth L1) + 方向(CE)
    #   - 其他损失函数...
    criterion = train_utils.create_loss(hypes)

    # ========================================================================
    # 第六步: 创建优化器和学习率调度器
    # ========================================================================
    
    # 优化器设置
    optimizer = train_utils.setup_optimizer(hypes, model)

    # ========================================================================
    # 第七步: 处理断点续训或从头训练
    # ========================================================================
    
    if opt.model_dir:
        # ----------------- 断点续训模式 -----------------
        saved_path = opt.model_dir
        
        # 加载已保存的模型权重，返回起始 epoch
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        
        # 创建学习率调度器，从 init_epoch 开始
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")

    else:
        # ----------------- 从头训练模式 -----------------
        init_epoch = 0
        
        # 创建保存模型的目录
        # 目录名格式: {name}_{timestamp}
        saved_path = train_utils.setup_train(hypes)
        
        # 创建学习率调度器
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # ========================================================================
    # 第八步: 将模型移至 GPU
    # ========================================================================
    if torch.cuda.is_available():
        model.to(device)
        
    # 创建 TensorBoard 日志记录器
    writer = SummaryWriter(saved_path)

    # ========================================================================
    # 第九步: 训练主循环
    # ========================================================================
    print('Training start')
    
    # 从配置文件读取总训练轮数
    epoches = hypes['train_params']['epoches']
    
    # 检查是否使用单车监督 (某些方法如 DiscoNet 需要)
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") \
                            else opencood_train_dataset.supervise_single

    # ==================== Epoch 循环 ====================
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        
        # 打印当前学习率
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        
        # ==================== Batch 循环 ====================
        for i, batch_data in enumerate(train_loader):
            
            # 跳过空数据或无目标的数据
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            
            # 设置模型为训练模式 (启用 Dropout, BatchNorm 更新)
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            
            # 将数据移至 GPU
            batch_data = train_utils.to_device(batch_data, device)
            
            # 将当前 epoch 传入数据字典 (某些模块可能需要)
            batch_data['ego']['epoch'] = epoch
            
            # ==================== 前向传播 ====================
            # model 的 forward 流程:
            #   1. pillar_vfe: 体素特征编码
            #   2. scatter: 散射到伪图像 (BEV 特征图)
            #   3. backbone: 多尺度特征提取
            #   4. fusion_net: 多代理特征融合 (使用校正后的位姿)
            #   5. detection_head: 分类 + 回归 + 方向预测
            ouput_dict = model(batch_data['ego'])
            
            # ==================== 计算损失 ====================
            # final_loss = cls_loss + reg_loss + dir_loss
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            
            # 记录损失到 TensorBoard
            criterion.logging(epoch, i, len(train_loader), writer)

            # 如果启用单车监督，添加额外的单车损失
            if supervise_single_flag:
                final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single")
                criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

            # ==================== 反向传播 ====================
            final_loss.backward()   # 计算梯度
            optimizer.step()        # 更新参数

            # 清空 GPU 缓存，防止显存溢出
            torch.cuda.empty_cache()

        # ====================================================================
        # 验证阶段 (每隔 eval_freq 个 epoch 执行一次)
        # ====================================================================
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():  # 禁用梯度计算，节省显存
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()  # 设置为评估模式 (禁用 Dropout, 固定 BatchNorm)

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())

            # 计算平均验证损失
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            
            # 记录验证损失到 TensorBoard
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            # ==================== 保存最佳模型 ====================
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                
                # 保存当前最佳模型
                torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                
                # 删除之前的最佳模型 (只保留最新的最佳)
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1

        # ==================== 定期保存检查点 ====================
        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
        
        # 更新学习率
        scheduler.step(epoch)

        # 重新初始化数据集 (用于数据增强等)
        opencood_train_dataset.reinitialize()

    # ========================================================================
    # 训练完成
    # ========================================================================
    print('Training Finished, checkpoints saved to %s' % saved_path)

    # ========================================================================
    # 第十步: 清理多余的最佳模型文件 (DDP 训练可能产生多个)
    # ========================================================================
    run_test = True    
    bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
    
    if len(bestval_model_list) > 1:
        import numpy as np
        bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
        ascending_idx = np.argsort(bestval_model_epoch_list)
        for idx in ascending_idx:
            if idx != (len(bestval_model_list) - 1):
                os.remove(bestval_model_list[idx])

    # ========================================================================
    # 第十一步: 自动运行推理测试
    # ========================================================================
    if run_test:
        fusion_method = opt.fusion_method
        
        # 根据是否使用噪声选择不同的推理脚本
        if 'noise_setting' in hypes and hypes['noise_setting']['add_noise']:
            # 带噪声推理 (测试鲁棒性)
            cmd = f"python opencood/tools/inference_w_noise.py --model_dir {saved_path} --fusion_method {fusion_method}"
        else:
            # 标准推理
            cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        print(f"Running command: {cmd}")
        os.system(cmd)


# ============================================================================
# 程序入口
# ============================================================================
if __name__ == '__main__':
    main()