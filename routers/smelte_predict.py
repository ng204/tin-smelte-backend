import torchvision
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from middlewares.custom_response import CustomRoute
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from typing import Dict, Any

from utils.auth import oauth2_scheme

router = APIRouter(prefix="/smelte", tags=["锡品质识别"], dependencies=[Depends(oauth2_scheme)])

router.route_class = CustomRoute

# 配置参数（需与训练时一致）
MODEL_PATH = "best_model.pth"
N_CLASSES = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 类别名称映射（必须与训练时的顺序一致）
CLASS_NAMES = ["含锡35%-50%", "含锡35%以下", "含锡50%以上"]  # 根据实际类别修改

# 预处理转换（与测试集相同）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 加载模型（单例模式）
def load_model() -> torch.nn.Module:
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# 初始化模型实例
model = load_model()


async def validate_image(file: UploadFile):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "只允许上传图片文件（JPEG/PNG）")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(400, f"无法读取图片文件: {str(e)}")


def predict(image: Image.Image) -> Dict[str, Any]:
    try:
        # 预处理
        tensor = test_transform(image).unsqueeze(0).to(DEVICE)

        # 推理
        with torch.no_grad():
            outputs = model(tensor)
            _, preds = torch.max(outputs, 1)

        class_idx = preds.item()
        return {
            "class_index": class_idx,
            "class_name": CLASS_NAMES[class_idx],
            "confidence": torch.nn.functional.softmax(outputs, dim=1)[0][class_idx].item()
        }
    except Exception as e:
        raise HTTPException(500, f"模型推理失败: {str(e)}")


@router.post("/predict", response_model=Dict[str, Any])
async def smelter_predict(file: UploadFile = File(...)):
    """
    锡品质预测接口

    参数：
    - file: 上传的图片文件（支持JPG/PNG格式）

    返回：
    - filename: 文件名
    - class_index: 类别索引
    - class_name: 类别名称
    - confidence: 置信度
    """
    try:
        # 验证并读取图片
        image = await validate_image(file)

        # 执行预测
        result = predict(image)

        return {
            "filename": file.filename,
            **result
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, f"服务器内部错误: {str(e)}")
