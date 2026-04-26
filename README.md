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
  
## 执行文件
### 1.参数文件
默认参数C_设置在  根目录的config.py中, 里面有很多默认设置参数, 有注释, 可以查看有什么需要修改的参数
要是想按照模型来覆盖对应的参数可以看./configs目录下的模型文件, 里面有.yaml文件, 用来覆盖默认参数
也可以在命令行修改参数 --opt <参数名> <值>, 可以在默认参数查看参数名
### 2.数据文件
放在data/build.py下, 这里是加载数据集的位置, 自建数据集模式需要按照官方的数据集目录结构存放(可以查看official_readme.md), 这里直接采用官方CIFAR100, 通过内置函数自行处理
### 3.训练文件
根目录的main.py文件, 用来跑训练代码
训练代码下文有写
### 4.推理文件
根目录的predict文件, 执行只需python predict --img <推理图片路径> 即可

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
```
python -m torch.distributed.launch --nproc_per_node=1 main.py \
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
--local_rank 0 \
--opts \
DATA.DATASET cifar100 \
TRAIN.EPOCHS 120 \
MODEL.NUM_CLASSES 100
```

## 方案二：基于预训练权重的 CIFAR100 分类训练（最优方案）
### 一、方案核心逻辑
依托 ImageNet 预训练权重，通过微调适配 CIFAR100 分类任务，解决从零训练精度低、泛化能力弱的问题，兼顾训练效率与模型性能。

### 二、预训练加载原理
1.  核心逻辑：复用预训练模型的通用视觉特征，仅重新学习 CIFAR100 分类规则，无需从零训练模型主干网络。
2.  具体操作：
    - 加载 ImageNet 预训练权重（含通用视觉特征提取能力），自动舍弃原分类头（适配 1000 类分类的部分）。
    - 重新初始化 CIFAR100 对应的分类头（100 类），保留主干网络（特征提取层）全部参数。
    - 通过微调训练，让模型快速适配 CIFAR100 分类任务，无需重新学习基础视觉特征。

### 三、训练配置
1.  模型配置：使用 Swin 模型，输入尺寸 224×224，适配 CIFAR100 数据集。
2.  训练参数：
    - 训练轮数：无需过多迭代，40 轮即可满足精度要求
    - 类别数量：100 类（CIFAR100 标准分类）
    - 预训练权重路径：本地预训练文件（如 `swin_tiny_patch4_window7_224.pth`），无需额外手动下载。

### 四、训练命令（直接复制运行）
```
python -m torch.distributed.launch --nproc_per_node=1 main.py \
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
--local_rank 0 \
--opts \
DATA.DATASET=cifar100 \
MODEL.NUM_CLASSES=100 \
MODEL.PRETRAINED=./swin_tiny_patch4_window7_224.pth \
TRAIN.EPOCHS=40 \
SAVE.FREQ=10
```
