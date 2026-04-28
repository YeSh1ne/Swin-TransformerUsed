import os
import torch
import torchvision.transforms as T
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from models import build_model
from config import _C

# ==================== 配置 ====================
CONFIG_PATH = "configs/swin/swin_tiny_patch4_window7_224.yaml"
CHECKPOINT_PATH = "output/swin_tiny_patch4_window7_224/cifar100-pretrain/ckpt_epoch_39.pth"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CLASSES = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
# CIFAR100
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
# ==================== 加载模型（全局只加载一次） ====================
print("🔹 正在加载模型...")
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
print("✅ 模型加载完成！")

# ==================== 预测函数 ====================
def predict_image(image_path):
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

    return CLASSES[top_class.item()], round(top_prob.item() * 100, 2)

# ==================== Flask 网站 ====================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 网页前端（自带上传界面）
HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Swin Transformer 图片分类</title>
    <style>
        body{font-family:Arial; max-width:800px; margin:20px auto; padding:20px; text-align:center;}
        .box{border:2px dashed #777; padding:40px; border-radius:10px; background:#fafafa;}
        button{padding:10px 20px; font-size:16px; background:#007bff; color:white; border:none; border-radius:5px; cursor:pointer}
        .result{margin-top:30px; font-size:20px; color:green}
    </style>
</head>
<body>
    <h1>🚀 Swin Transformer 图片分类</h1>
    <div class="box">
        <h3>上传图片（飞机/汽车/猫/狗等）</h3>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <br><br>
            <button type="submit">开始预测</button>
        </form>
    </div>

    {% if result %}
    <div class="result">
        <h3>✅ 预测结果</h3>
        <p>类别：{{ result.label }}</p>
        <p>置信度：{{ result.confidence }}%</p>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        label, conf = predict_image(filepath)
        return render_template_string(HTML, result={"label": label, "confidence": conf})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)