import torch
import torchvision.transforms as T
from PIL import Image
from models import build_model
from config import _C
import argparse  # 新增：命令行解析

# ==================== 固定配置 ====================
CONFIG_PATH = "configs/swin/swin_tiny_patch4_window7_224.yaml"
CHECKPOINT_PATH = "output/swin_tiny_patch4_window7_224/default/ckpt_epoch_39.pth"
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
        T.Resize((224, 224)),
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

if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default="test.png", help="图片路径")
    args = parser.parse_args()

    print("🔹 加载模型...")
    model = load_model()

    print("🔹 预测中...")
    label, conf = predict_image(model, args.img)

    print("\n✅ 预测成功！")
    print(f"🖼️ 图片: {args.img}")
    print(f"🚀 类别: {label}")
    print(f"🎯 置信度: {conf:.2f}%")