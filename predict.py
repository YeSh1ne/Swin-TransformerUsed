import torch
import torchvision.transforms as T
from PIL import Image
from models import build_model
from config import _C
import argparse
import os

# ==================== 固定配置 ====================
# CONFIG_PATH = "configs/swin/swin_tiny_patch4_window7_224.yaml"
CONFIG_PATH = "configs/swinv2/swinv2_tiny_patch4_window16_256.yaml"
# CHECKPOINT_PATH = "output/swin_tiny_patch4_window7_224/cifar100-pretrain/ckpt_epoch_39.pth"
CHECKPOINT_PATH = "output/swinv2_tiny_patch4_window16_256/default/ckpt_epoch_39.pth"
# ====================================================

# CIFAR100 中文类别
CLASSES = [
    '苹果', '金鱼', '婴儿', '熊', '河狸', '床', '蜜蜂', '甲虫',
    '自行车', '瓶子', '碗', '男孩', '桥梁', '公交车', '蝴蝶', '骆驼',
    '罐头', '城堡', '毛毛虫', '牛', '椅子', '黑猩猩', '时钟',
    '云朵', '蟑螂', '沙发', '螃蟹', '鳄鱼', '杯子', '恐龙',
    '海豚', '大象', '比目鱼', '森林', '狐狸', '女孩', '仓鼠',
    '房屋', '袋鼠', '键盘', '台灯', '割草机', '豹子', '狮子',
    '蜥蜴', '龙虾', '男人', '枫树', '摩托车', '山脉', '老鼠',
    '蘑菇', '橡树', '橙子', '兰花', '水獭', '棕榈树', '梨子',
    '皮卡卡车', '松树', '平原', '盘子', '罂粟花', '豪猪',
    '负鼠', '兔子', '浣熊', '鳐鱼', '道路', '火箭', '玫瑰',
    '大海', '海豹', '鲨鱼', '鼩鼱', '臭鼬', '摩天大楼', '蜗牛', '蛇',
    '蜘蛛', '松鼠', '有轨电车', '向日葵', '甜椒', '桌子',
    '坦克', '电话', '电视', '老虎', '拖拉机', '火车', '鳟鱼',
    '郁金香', '乌龟', '衣柜', '鲸鱼', '柳树', '狼', '女人', '蠕虫'
]

def load_model():
    config = _C.clone()
    config.merge_from_file(CONFIG_PATH)
    config.defrost()
    config.MODEL.NUM_CLASSES = 100
    config.LOCAL_RANK = 0
    config.freeze()

    model = build_model(config)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()
    return model

def predict_image(model, image_path):
    transform = T.Compose([
        # T.Resize((224, 224)),
        # v2版本用256 * 256
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(img)
        prob = torch.softmax(output, dim=1)
        top_prob, top_class = prob.topk(1, dim=1)

    return CLASSES[top_class.item()], top_prob.item() * 100

# ==================== 批量推理 ====================
def batch_predict(model, img_dir):
    # 支持的图片格式
    img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG']
    img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[-1] in img_exts]
    #平均置信度
    avgconf = 0
    cnt = 0
    if not os.path.exists(img_dir):
        print(f"❌ 目录不存在: {img_dir}")
        return
    if len(img_files) == 0:
        print(f"❌ 在 {img_dir} 中未找到图片")
        return

    print(f"\n✅ 找到 {len(img_files)} 张图片，开始批量推理...\n")
    print("=" * 70)
    print(f"{'文件名':<32} {'类别':<16} {'置信度'}")
    print("=" * 70)

    for filename in img_files:
        img_path = os.path.join(img_dir, filename)
        label, conf = predict_image(model, img_path)
        print(f"{filename:<32} {label:<16} {conf:.2f}%")
        avgconf += conf
        cnt += 1
        
    return avgconf, cnt
# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin Transformer 批量推理")
    parser.add_argument("--img_dir", default="./test", help="批量推理图片文件夹，默认 ./test")
    args = parser.parse_args()

    print("🔹 加载模型...")
    model = load_model()
    avgconf = 0
    print("🔹 开始批量推理...")
    avgconf, cnt = batch_predict(model, args.img_dir)
    print("\n")
    print(f"平均置信度为{avgconf/cnt:.2f}%")
    print("\n🎉 全部推理完成！")