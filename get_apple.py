import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image

# 加载 CIFAR100 测试集
transform = T.Compose([T.Resize((32, 32))])
test_dataset = datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)

# 找 label=0 的苹果图（CIFAR100 中 0=apple）
for img, label in test_dataset:
    if label == 0:
        img.save("test.png")
        print("✅ 已保存 CIFAR100 原生苹果图为 test.png")
        break