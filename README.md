Swin-Transformer-CIFAR100 完整实战记录
项目简介
本项目基于 Swin-Tiny Transformer，在 CIFAR100 数据集上完成图像分类完整实验。
包含：从零开始训练、预训练权重微调、自定义预测脚本、命令行推理、数据集加载改造全流程。
一、环境与依赖
运行环境：AutoDL 云服务器 + Conda
框架：PyTorch
模型：swin_tiny_patch4_window7_224
任务：CIFAR100 100 分类
二、数据集加载逻辑
1. 数据来源
使用 torchvision.datasets 内置数据集：
CIFAR100 / CIFAR10
代码首次运行会自动下载至 ./data 目录，无需手动准备数据集
2. 核心加载函数
所有数据读取、预处理、增强逻辑统一封装在：
python
运行
build_dataset(is_train, config)
执行流程：
根据配置自动区分 CIFAR10 / CIFAR100
加载对应数据集专属均值、方差
统一缩放至模型输入尺寸 224×224
训练集添加随机水平翻转数据增强
测试集仅做标准化与缩放
返回数据集实例 + 分类类别数
拓展：后续如需训练自定义私有数据集，仅需修改该函数，
替换为 ImageFolder 文件夹读取格式即可，主干代码无需改动。
三、第一阶段：纯从零训练（无预训练）
1. 训练方式
完全随机初始化模型权重，不加载任何预训练参数，直接在 CIFAR100 迭代训练。
2. 遇到的问题
训练 60～90 Epoch，最终精度仅 70% 左右，瓶颈明显
推理原生 CIFAR100 测试图，分类正确但置信度极低（30% 左右）
无法识别日常高清图片，域外图片全部错分
模型底层视觉特征提取能力弱，泛化能力差
3. 原因总结
100 分类任务难度大，单纯小数据集从头训练上限极低
模型从零学习边缘、纹理、物体结构，特征表达能力不足
CIFAR100 原图为 32×32 低分辨率小图，域内拟合、域外失效
四、第二阶段：引入 ImageNet 预训练权重微调
1. 预训练加载原理
官方 Swin 预训练权重基于 ImageNet 1000 分类训练
加载时自动舍弃最后一层分类头（head）
主干 Backbone 全部参数迁移复制
针对 CIFAR100 100 类，重新随机初始化分类层
冻结少量底层 + 整体微调，快速适配新任务
核心逻辑：
复用通用视觉特征，仅重新学习专属分类规则
2. 初始微调现象
刚加载预训练权重、未训练时，初始测试精度仅 47.3%
原因：旧 1000 类分类头被舍弃，新分类头完全随机初始化
随着 Epoch 增加，精度会快速线性上涨
3. 最终效果
仅微调 40 Epoch
测试集准确率大幅提升至 85%+
单类别预测置信度提升至 80%～95%
支持高清实拍图片、网络图片正常识别，泛化能力大幅增强
五、训练命令说明
1. 从零开始训练
bash
运行
python -m torch.distributed.launch --nproc_per_node=1 main.py \
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
--local_rank 0 \
--opts \
DATA.DATASET cifar100 \
TRAIN.EPOCHS 120 \
MODEL.NUM_CLASSES 100
2. 预训练权重微调
指定本地预训练权重路径，启用微调：
bash
运行
python -m torch.distributed.launch --nproc_per_node=1 main.py \
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml \
--local_rank 0 \
--opts \
DATA.DATASET cifar100 \
TRAIN.EPOCHS 40 \
MODEL.NUM_CLASSES 100 \
MODEL.PRETRAINED ./swin_tiny_patch4_window7_224.pth
六、自定义推理预测脚本 predict.py
1. 功能改造
支持 argparse 命令行传参
配置默认图片路径 test.png，不传参自动使用默认图
可自由指定任意图片路径推理
输出：类别名称、精准置信度、模型加载状态
2. 使用方式
使用默认图片
bash
运行
python predict.py
手动指定任意图片
bash
运行
python predict.py --img test2.png
python predict.py --img xxx.jpg
3. 权重加载
通过配置文件 + 本地训练好的 ckpt_epoch_x.pth 断点权重，
直接加载完整模型做评估推理。
七、关键知识点总结
小数据集分类任务，预训练微调 >> 全程从零训练
分类任务置信度和类别数量强相关：100 分类天然置信度偏低
域内图（数据集原图）& 域外图（日常照片）泛化差距由特征提取能力决定
项目解耦设计：数据集、模型、配置、推理完全分离，易于二次开发
如需改造自己的数据集，仅需修改 build_dataset 即可快速迁移
八、项目文件结构（开源整洁版）
plaintext
Swin-Transformer-CIFAR100/
├── configs/            # Swin 配置文件
├── models/             # 模型结构定义
├── output/             # 训练权重保存目录
├── data/               # 自动下载 CIFAR100 数据集
├── main.py             # 训练入口
├── predict.py          # 自定义推理脚本
├── utils.py            # 工具函数、预训练加载
└── README.md           # 项目全过程文档
