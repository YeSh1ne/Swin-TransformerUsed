import torch
import torchvision.transforms as T
from PIL import Image
from models import build_model
from config import _C

# ==================== 配置 ====================
CONFIG_PATH = "configs/swin/swin_tiny_patch4_window7_224.yaml"
CHECKPOINT_PATH = "output/swin_tiny_patch4_window7_224/default/ckpt_epoch_55.pth"
IMAGE_PATH = "test.png"
# ==============================================

CLASSES = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

def load_model():
    # ✅ 直接用底层 _C，跳过所有 argparse 坑！
    config = _C.clone()
    config.merge_from_file(CONFIG_PATH)
    config.defrost()
    config.MODEL.NUM_CLASSES = 10
    config.LOCAL_RANK = 0  # 手动加，避免报错
    config.freeze()

    model = build_model(config)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()
    return model

def predict_image(model):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(IMAGE_PATH).convert("RGB")
    img = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(img)
        prob = torch.softmax(output, dim=1)
        top_prob, top_class = prob.topk(1, dim=1)

    return CLASSES[top_class.item()], top_prob.item() * 100

if __name__ == "__main__":
    print("🔹 加载模型...")
    model = load_model()

    print("🔹 预测中...")
    label, conf = predict_image(model)

    print("\n✅ 预测成功！")
    print(f"🖼️ 图片: {IMAGE_PATH}")
    print(f"🚀 类别: {label}")
    print(f"🎯 置信度: {conf:.2f}%")