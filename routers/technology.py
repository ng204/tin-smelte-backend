from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from PIL import Image
import io
import base64
import torch
import torchvision.transforms as transforms
import logging
import os
from typing import Optional, List, Dict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 复用smelte_predict.py的类别名和模型加载逻辑
CLASS_NAMES = ["含锡35%-50%", "含锡35%以下", "含锡50%以上"]
MODEL_PATH = "best_model.pth"
N_CLASSES = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置PyTorch性能优化
if DEVICE.type == 'cpu':
    # CPU优化
    torch.set_num_threads(4)  # 使用4个线程
    torch.set_num_interop_threads(2)
    logger.info("CPU推理优化已启用：4线程")
else:
    # GPU优化
    torch.backends.cudnn.benchmark = True
    logger.info("GPU推理优化已启用")

def load_model():
    """加载预训练的锡品位识别模型"""
    try:
        import torchvision
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
        
        # 检查模型文件是否存在
        if not os.path.exists(MODEL_PATH):
            logger.error(f"模型文件不存在: {MODEL_PATH}")
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
            
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        
        # 设置为推理模式以提高性能
        if hasattr(torch, 'jit') and DEVICE.type == 'cpu':
            # 在CPU上使用JIT编译可以提速
            try:
                dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
                model = torch.jit.trace(model, dummy_input)
                logger.info("模型已JIT编译优化")
            except Exception as e:
                logger.warning(f"JIT编译失败，使用普通模式: {e}")
        
        logger.info(f"模型加载成功，使用设备: {DEVICE}")
        return model
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

# 全局模型实例
model = None

def get_model():
    """获取模型实例（单例模式）"""
    global model
    if model is None:
        model = load_model()
    return model

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 降低分辨率从224到128，速度提升4倍
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

router = APIRouter(prefix='/technology', tags=["工艺流程"])

class ImageRecognitionRequest(BaseModel):
    image: str  # base64编码的图片
    type: str = "CNN"  # 识别类型

class ImageRecognitionResponse(BaseModel):
    description: str
    result: str
    confidence: float
    accuracy: Optional[float] = None  # 可选
    success: bool = True
    message: str = "识别成功"

@router.post("/recognize", response_model=ImageRecognitionResponse)
async def recognize_image(request: ImageRecognitionRequest):
    """
    锡品位图像识别API
    
    Args:
        request: 包含base64编码图片的请求
        
    Returns:
        ImageRecognitionResponse: 识别结果
    """
    try:
        logger.info("开始处理图像识别请求")
        
        # 验证输入
        if not request.image:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="图片数据不能为空"
            )
        
        # 解码base64图片
        try:
            if ',' in request.image:
                image_data = base64.b64decode(request.image.split(',')[1])
            else:
                image_data = base64.b64decode(request.image)
        except Exception as e:
            logger.error(f"Base64解码失败: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="图片格式错误，请检查base64编码"
            )
        
        # 打开和处理图片
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            logger.error(f"图片打开失败: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无法打开图片文件"
            )
        
        # 获取模型并进行预测
        try:
            model = get_model()
            
            # 图像预处理
            tensor = test_transform(image).unsqueeze(0).to(DEVICE)
            
            # 使用torch.inference_mode()替代torch.no_grad()以提高性能
            with torch.inference_mode():
                outputs = model(tensor)
                # 使用argmax替代max以提高速度
                preds = torch.argmax(outputs, dim=1)
                
            class_idx = preds.item()
            class_name = CLASS_NAMES[class_idx]
            
            # 计算置信度（使用更快的方法）
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][class_idx].item()
            
            # 没有真实标签，精度为None
            accuracy = None
            
            logger.info(f"识别成功: {class_name}, 置信度: {confidence:.4f}")
            
            return ImageRecognitionResponse(
                description=class_name,
                result=f"识别结果：{class_name}",
                confidence=round(confidence, 4),
                accuracy=accuracy,
                success=True,
                message="识别成功（精度仅供参考，实际以置信度为准）"
            )
            
        except Exception as e:
            logger.error(f"模型预测失败: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"模型预测失败: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图像识别过程中发生未知错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"图像识别失败: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(DEVICE),
            "classes": CLASS_NAMES
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

class ProcessQueryRequest(BaseModel):
    """工艺流程查询请求"""
    grade: str  # 品位信息，如"含锡35%-50%"
    question: Optional[str] = None  # 可选的具体问题

class ProcessQueryResponse(BaseModel):
    """工艺流程查询响应"""
    answer: str
    grade: str
    success: bool = True

@router.post("/process_query", response_model=ProcessQueryResponse)
async def process_query(request: ProcessQueryRequest):
    """
    工艺流程查询接口 - 基于品位信息查询工艺流程
    
    这个接口不使用LLM模型，而是基于预定义的工艺流程知识库返回结果
    
    Args:
        request: 包含品位信息的请求
        
    Returns:
        ProcessQueryResponse: 工艺流程信息
    """
    try:
        logger.info(f"工艺流程查询: 品位={request.grade}, 问题={request.question}")
        
        grade = request.grade.strip()
        logger.info(f"处理后的品位: '{grade}'")
        
        # 基于品位的工艺流程知识库
        process_knowledge = {
            "含锡35%-50%": """
【中品位锡精矿冶炼工艺流程】

1. 原料准备阶段
   - 锡精矿接收与检验
   - 品位分析：含锡35%-50%
   - 配料计算与混合

2. 预处理阶段
   - 破碎：将锡精矿破碎至合适粒度
   - 磨矿：细磨至200目以下
   - 焙烧：氧化焙烧，温度600-800℃，去除硫、砷等杂质

3. 还原熔炼阶段
   - 设备：反射炉或电炉
   - 温度：1200-1300℃
   - 还原剂：焦炭或煤粉
   - 熔剂：石灰石、萤石
   - 产物：粗锡（含锡85-95%）+ 炉渣

4. 粗锡精炼阶段
   - 火法精炼：
     * 除铁：加热至450℃，铁以氧化物形式浮出
     * 除铜：加热至600℃，加硫磺，铜以硫化物形式去除
     * 除砷锑：加热至800℃，通入空气氧化
   - 电解精炼：
     * 阳极：粗锡
     * 阴极：纯锡板
     * 电解液：氟硅酸-硫酸体系
     * 产物：精锡（纯度99.9%以上）

5. 产品处理
   - 铸锭：将精锡铸成标准锡锭
   - 质量检验：化学分析、物理性能测试
   - 包装入库

关键控制参数：
- 焙烧温度：600-800℃
- 熔炼温度：1200-1300℃
- 还原剂配比：焦炭用量为精矿量的15-20%
- 熔剂用量：石灰石5-8%，萤石2-3%
- 精炼温度：450-800℃（分阶段）
- 锡回收率：≥95%

注意事项：
- 严格控制焙烧温度，避免锡氧化损失
- 熔炼过程保持还原气氛
- 精炼过程分阶段控温，确保杂质充分去除
- 做好烟气收尘，回收有价金属
            """,
            
            "含锡35%以下": """
【低品位锡精矿冶炼工艺流程】

1. 原料准备阶段
   - 锡精矿接收与检验
   - 品位分析：含锡35%以下
   - 富集处理：重选或浮选提高品位至35%以上

2. 预富集阶段（重要）
   - 重选：摇床、螺旋溜槽
   - 浮选：使用捕收剂富集锡矿物
   - 目标：将品位提升至40%以上

3. 预处理阶段
   - 破碎与磨矿：细度要求更高，达到300目
   - 强化焙烧：
     * 温度：700-850℃
     * 时间：2-3小时
     * 目的：充分氧化杂质，提高后续回收率

4. 还原熔炼阶段
   - 设备：电炉（推荐）
   - 温度：1250-1350℃（略高于中品位）
   - 还原剂：焦炭用量增加至20-25%
   - 熔剂：石灰石8-10%，萤石3-5%
   - 分段熔炼：
     * 第一段：粗选，得到含锡60-70%的粗锡
     * 第二段：精选，进一步提高锡品位

5. 粗锡精炼阶段
   - 火法精炼（同中品位流程）
   - 电解精炼（同中品位流程）

6. 炉渣处理（重要）
   - 炉渣含锡较高（5-10%）
   - 返回熔炼或单独处理回收
   - 贫化处理：将炉渣锡含量降至1%以下

关键控制参数：
- 富集后品位：≥40%
- 焙烧温度：700-850℃
- 熔炼温度：1250-1350℃
- 还原剂配比：焦炭20-25%
- 熔剂用量：石灰石8-10%，萤石3-5%
- 锡回收率：≥90%（低于中高品位）

注意事项：
- 必须进行预富集，否则经济性差
- 焙烧时间要充分，确保杂质氧化完全
- 熔炼温度要适当提高，保证锡充分还原
- 炉渣必须返回处理，提高总回收率
- 能耗较高，注意成本控制
            """,
            
            "含锡50%以上": """
【高品位锡精矿冶炼工艺流程】

1. 原料准备阶段
   - 锡精矿接收与检验
   - 品位分析：含锡50%以上
   - 简单配料（可直接入炉）

2. 预处理阶段（简化）
   - 破碎：粗破即可
   - 轻度焙烧：
     * 温度：500-700℃
     * 时间：1-1.5小时
     * 目的：去除部分硫，保留锡矿物

3. 直接还原熔炼阶段
   - 设备：反射炉或电炉
   - 温度：1150-1250℃（较低温度即可）
   - 还原剂：焦炭用量仅需10-15%
   - 熔剂：石灰石3-5%，萤石1-2%
   - 一次熔炼即可得到高品位粗锡（含锡90-95%）

4. 简化精炼阶段
   - 火法精炼：
     * 除铁：400℃
     * 除铜：550℃
     * 除砷锑：750℃
   - 电解精炼（可选）：
     * 对于要求极高纯度的产品才需要
     * 一般火法精炼即可达到99.5%以上

5. 产品处理
   - 直接铸锭
   - 质量检验
   - 包装入库

关键控制参数：
- 焙烧温度：500-700℃（轻度）
- 熔炼温度：1150-1250℃
- 还原剂配比：焦炭10-15%
- 熔剂用量：石灰石3-5%，萤石1-2%
- 锡回收率：≥98%

工艺优势：
- 流程短，能耗低
- 还原剂用量少
- 熔炼温度低
- 回收率高
- 炉渣含锡低（<2%）
- 经济效益好

注意事项：
- 避免过度焙烧造成锡氧化损失
- 控制熔炼温度，防止锡挥发
- 精炼温度要精确控制
- 高品位原料要充分利用其优势，简化流程
            """
        }
        
        # 根据品位返回对应的工艺流程
        # 先进行精确匹配
        answer = process_knowledge.get(grade, "")
        
        if answer:
            logger.info(f"精确匹配成功: {grade}")
        else:
            logger.info(f"精确匹配失败，尝试模糊匹配: {grade}")
            # 如果没有精确匹配，进行模糊匹配
            grade_lower = grade.lower().replace(" ", "").replace("　", "")
            logger.info(f"标准化后的品位: '{grade_lower}'")
            
            # 匹配中品位 (35%-50%)
            if ("35%" in grade_lower or "35" in grade_lower) and ("50%" in grade_lower or "50" in grade_lower):
                if "以下" not in grade_lower and "以上" not in grade_lower:
                    answer = process_knowledge["含锡35%-50%"]
                    grade = "含锡35%-50%"
                    logger.info(f"模糊匹配成功: 中品位 (35%-50%)")
            
            # 匹配低品位 (35%以下)
            elif "35%" in grade_lower and "以下" in grade_lower:
                answer = process_knowledge["含锡35%以下"]
                grade = "含锡35%以下"
                logger.info(f"模糊匹配成功: 低品位 (35%以下)")
            elif "35" in grade_lower and "以下" in grade_lower:
                answer = process_knowledge["含锡35%以下"]
                grade = "含锡35%以下"
                logger.info(f"模糊匹配成功: 低品位 (35%以下)")
            
            # 匹配高品位 (50%以上)
            elif "50%" in grade_lower and "以上" in grade_lower:
                answer = process_knowledge["含锡50%以上"]
                grade = "含锡50%以上"
                logger.info(f"模糊匹配成功: 高品位 (50%以上)")
            elif "50" in grade_lower and "以上" in grade_lower:
                answer = process_knowledge["含锡50%以上"]
                grade = "含锡50%以上"
                logger.info(f"模糊匹配成功: 高品位 (50%以上)")
            
            # 如果还是没有匹配到，返回通用说明
            if not answer:
                logger.warning(f"无法匹配品位: {grade}")
                answer = f"抱歉，暂无 {grade} 品位的详细工艺流程信息。\n\n一般来说，锡冶炼的基本流程包括：\n1. 原料准备\n2. 预处理（破碎、磨矿、焙烧）\n3. 还原熔炼\n4. 粗锡精炼\n5. 产品处理\n\n具体参数需要根据实际品位调整。"
        
        logger.info(f"返回工艺流程信息，品位: {grade}, 长度: {len(answer)}")
        
        return ProcessQueryResponse(
            answer=answer,
            grade=grade,
            success=True
        )
        
    except Exception as e:
        logger.error(f"工艺流程查询失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"工艺流程查询失败: {str(e)}"
        ) 


class AIChatRequest(BaseModel):
    """AI对话请求"""
    message: str  # 用户消息
    history: Optional[list] = []  # 历史对话记录，格式：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

class AIChatResponse(BaseModel):
    """AI对话响应"""
    reply: str
    success: bool = True
    use_llm: bool = False  # 是否使用了LLM模型

@router.post("/ai_chat", response_model=AIChatResponse)
async def ai_chat(request: AIChatRequest):
    """
    AI助手对话接口 - 优先使用qwen2:7b模型，失败时使用预定义回复
    
    Args:
        request: 包含用户消息和历史记录的请求
        
    Returns:
        AIChatResponse: AI回复
    """
    try:
        logger.info(f"AI对话请求: {request.message[:50]}...")
        
        # 首先尝试使用qwen2:7b模型
        try:
            import httpx
            from openai import OpenAI
            from config import OLLAMA_HOST, LLM_MODEL
            
            # 构建对话消息
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的锡冶炼工艺助手，名叫锡冶炼小助手。你熟悉各种锡冶炼工艺流程、设备操作、故障排查和工艺优化。请用专业、友好的语气回答用户关于锡冶炼的问题。"
                }
            ]
            
            # 添加历史对话
            if request.history:
                messages.extend(request.history)
            
            # 添加当前消息
            messages.append({
                "role": "user",
                "content": request.message
            })
            
            # 准备base_url候选列表
            ollama_host = OLLAMA_HOST.rstrip("/")
            base_url_candidates = []
            
            # 添加主要URL
            if ollama_host.endswith("/v1"):
                base_url_candidates.append(ollama_host)
            else:
                base_url_candidates.append(ollama_host + "/v1")
            
            # 添加localhost变体（127.0.0.1 -> localhost）
            if "127.0.0.1" in ollama_host:
                localhost_url = ollama_host.replace("127.0.0.1", "localhost")
                if not localhost_url.endswith("/v1"):
                    localhost_url += "/v1"
                base_url_candidates.append(localhost_url)
            
            logger.info(f"尝试连接Ollama: {base_url_candidates}")
            
            # 健康检查函数
            def probe_ollama(base_url: str) -> bool:
                """检查Ollama服务是否健康"""
                root = base_url[:-3] if base_url.endswith("/v1") else base_url
                try:
                    with httpx.Client(trust_env=False, timeout=5.0) as client:
                        resp = client.get(f"{root}/api/tags")
                        if resp.status_code == 200:
                            logger.info(f"Ollama健康检查通过: {root}")
                            return True
                        logger.warning(f"Ollama健康检查失败 {root}: HTTP {resp.status_code}")
                        return False
                except Exception as e:
                    logger.warning(f"Ollama健康检查失败 {root}: {e}")
                    return False
            
            # 筛选健康的URL
            healthy_urls = []
            for url in base_url_candidates:
                if probe_ollama(url):
                    healthy_urls.append(url)
            
            if not healthy_urls:
                logger.warning("所有Ollama URL都不可用，使用预定义回复")
                reply = generate_fallback_reply(request.message)
                return AIChatResponse(
                    reply=reply,
                    success=True,
                    use_llm=False
                )
            
            # 尝试调用健康的URL
            for base_url in healthy_urls:
                try:
                    logger.info(f"调用LLM: {base_url}, 模型: {LLM_MODEL}")
                    
                    # 创建自定义httpx客户端，设置更长的超时和禁用代理
                    http_client = httpx.Client(
                        trust_env=False,  # 禁用环境变量中的代理设置
                        timeout=600.0     # 10分钟超时
                    )
                    
                    client = OpenAI(
                        api_key="ollama",
                        base_url=base_url,
                        http_client=http_client
                    )
                    
                    response = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2000  # 增加最大token数以获得更详细的回复
                    )
                    
                    reply = response.choices[0].message.content
                    logger.info(f"LLM回复成功，长度: {len(reply)}, URL: {base_url}")
                    
                    # 关闭http客户端
                    http_client.close()
                    
                    return AIChatResponse(
                        reply=reply,
                        success=True,
                        use_llm=True
                    )
                    
                except Exception as e:
                    logger.warning(f"LLM调用失败 ({base_url}): {str(e)}")
                    try:
                        http_client.close()
                    except:
                        pass
                    continue
            
            # 所有URL都失败，使用预定义回复
            logger.warning("所有LLM调用尝试都失败，使用预定义回复")
            reply = generate_fallback_reply(request.message)
            
            return AIChatResponse(
                reply=reply,
                success=True,
                use_llm=False
            )
            
        except Exception as llm_error:
            logger.warning(f"LLM调用异常: {str(llm_error)}，使用预定义回复")
            
            # LLM失败时使用预定义回复
            reply = generate_fallback_reply(request.message)
            
            return AIChatResponse(
                reply=reply,
                success=True,
                use_llm=False
            )
            
    except Exception as e:
        logger.error(f"AI对话失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI对话失败: {str(e)}"
        )

def generate_fallback_reply(message: str) -> str:
    """
    生成预定义回复（当LLM不可用时）
    
    Args:
        message: 用户消息
        
    Returns:
        str: 预定义回复
    """
    message_lower = message.lower()
    
    # 自我介绍
    if "自我介绍" in message or "你是谁" in message or "介绍" in message:
        return """你好！我是锡冶炼小助手，一个专注于锡冶炼工艺的智能助手。

我可以帮助您：
1. 解答锡冶炼工艺流程相关问题
2. 提供设备操作和维护建议
3. 协助进行故障诊断和排查
4. 分享工艺优化经验和建议
5. 解释相关技术参数和控制要点

如果您有任何关于锡冶炼的问题，欢迎随时向我提问！"""
    
    # 温度相关
    elif "温度" in message:
        return """锡冶炼过程中温度控制非常重要：

1. 焙烧阶段：
   - 低品位：700-850℃
   - 中品位：600-800℃
   - 高品位：500-700℃

2. 熔炼阶段：
   - 低品位：1250-1350℃
   - 中品位：1200-1300℃
   - 高品位：1150-1250℃

3. 精炼阶段：
   - 除铁：400-450℃
   - 除铜：550-600℃
   - 除砷锑：750-800℃

温度控制要精确，过高会导致锡氧化挥发，过低会影响还原效果。"""
    
    # 设备相关
    elif "设备" in message or "炉" in message:
        return """锡冶炼主要设备包括：

1. 焙烧设备：
   - 回转窑
   - 流化床焙烧炉
   - 多膛炉

2. 熔炼设备：
   - 反射炉：适用于中高品位
   - 电炉：适用于所有品位，能耗较高
   - 鼓风炉：传统工艺

3. 精炼设备：
   - 精炼锅（火法精炼）
   - 电解槽（电解精炼）

4. 辅助设备：
   - 破碎机、磨矿机
   - 除尘设备
   - 铸锭机

选择设备要考虑原料品位、产能要求、能耗成本等因素。"""
    
    # 故障排查
    elif "故障" in message or "问题" in message or "异常" in message:
        return """常见故障及处理方法：

1. 温度异常：
   - 检查燃料供给
   - 检查炉体密封
   - 调整配风比例

2. 回收率低：
   - 检查还原剂用量
   - 优化熔炼温度
   - 处理炉渣含锡

3. 产品质量不达标：
   - 加强原料检验
   - 优化精炼工艺
   - 严格控制杂质

4. 设备故障：
   - 定期维护保养
   - 及时更换易损件
   - 做好设备巡检

遇到具体问题可以详细描述，我会提供更针对性的建议。"""
    
    # 优化相关
    elif "优化" in message or "提高" in message or "改进" in message:
        return """工艺优化建议：

1. 原料优化：
   - 合理配料，稳定品位
   - 预富集处理低品位原料
   - 严格控制杂质含量

2. 工艺参数优化：
   - 精确控制焙烧温度和时间
   - 优化还原剂和熔剂配比
   - 改进精炼工艺流程

3. 能耗优化：
   - 回收利用余热
   - 优化燃料结构
   - 提高设备热效率

4. 环保优化：
   - 加强烟气收尘
   - 回收有价金属
   - 处理废水废渣

5. 管理优化：
   - 加强员工培训
   - 完善操作规程
   - 建立质量管理体系

持续改进是提高效益的关键。"""
    
    # 默认回复
    else:
        return """感谢您的提问！我是锡冶炼小助手。

我可以帮您解答关于锡冶炼的各类问题，包括：
- 工艺流程和参数
- 设备选型和操作
- 故障诊断和处理
- 工艺优化建议
- 质量控制要点

请告诉我您具体想了解哪方面的内容，我会尽力为您提供专业的解答。"""
