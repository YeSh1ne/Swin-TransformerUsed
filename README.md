# Swin-Transformer CIFAR100 完整实战

## 项目介绍
基于 Swin-Tiny Transformer 完成 CIFAR100 图像分类任务。
完整复现两种训练方案：
1. 模型**从零随机初始化**全程训练
2. 加载 ImageNet 预训练权重**微调训练**
同时自定义开发命令行推理脚本，完成图片分类预测全流程。

## 运行环境
- Framework: PyTorch
- Model: swin_tiny_patch4_window7_224
- Device: AutoDL 云服务器
- Dataset: CIFAR100（100 分类）

## 数据集加载方式
### 1. 数据来源
依托 `torchvision.datasets` 内置数据集：
- CIFAR100 / CIFAR10
- 首次运行代码会**自动下载**数据集至 `./data`，无需手动配置、无需上传数据集至仓库

### 2. 核心加载逻辑
所有数据读取、预处理、数据增强统一封装在 `build_dataset()`：
- 根据配置自动切换 CIFAR10 / CIFAR100
- 绑定对应数据集专属 `mean / std` 归一化
- 统一缩放至模型输入尺寸 `224×224`
- 训练集：随机水平翻转增强
- 测试集：仅标准化+尺寸缩放
- 解耦设计：后续更换**自定义私有数据集**，仅修改该函数即可

## 方案一：从零训练（无预训练）
### 训练特点
- 模型权重完全随机初始化，全程从头学习
- 训练轮数：90+ epoch

### 实验问题
1. 收敛上限低，训练 90 轮准确率仅约 **70%**
2. 数据集原生测试图可正确分类，但**置信度极低（30% 左右）**
3. 泛化能力弱，无法识别现实高清图片
4. 小数据集+深层模型从头训练，特征提取能力不足

### 训练命令
```bash
python -m torch.distributed.launch --nproc_per_node=1 main.py \
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
--local_rank 0 \
--opts \
DATA.DATASET cifar100 \
TRAIN.EPOCHS 120 \
MODEL.NUM_CLASSES 100
