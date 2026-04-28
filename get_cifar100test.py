import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
import os

# 创建 test 文件夹（自动建）
os.makedirs("./test", exist_ok=True)

# 预处理（只Resize，不破坏原图）
transform = T.Compose([T.Resize((32, 32))])

# 加载 CIFAR100 测试集
test_dataset = datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)

# CIFAR100 类别名称（用于给图片命名）
CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# ===================== 核心：批量保存测试图 =====================
save_count = 0
max_save = 20  # 你想保存多少张？自己改（默认20张）

for idx, (img, label) in enumerate(test_dataset):
    if save_count >= max_save:
        break
    
    # 文件名：类别名_编号.png
    class_name = CLASSES[label]
    img_path = f"./test/{class_name}_{idx}.png"
    
    # 保存图片
    img.save(img_path)
    print(f"✅ 已保存: {img_path}")
    save_count += 1

print(f"\n🎉 全部保存完成！共 {save_count} 张图片到 ./test 文件夹")