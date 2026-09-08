# -*- coding: UTF-8 -*-
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from pydantic import BaseModel
import re
from openai import OpenAI
import os
import json
import logging
import httpx
import tempfile

# 尝试导入文件处理库，如果失败则提供友好的错误提示
try:
    import fitz  # PyMuPDF for PDF processing
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyMuPDF (fitz) 未安装，PDF文件处理功能将不可用。请运行: pip install PyMuPDF")

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    logging.warning("python-docx 未安装，Word文档处理功能将不可用。请运行: pip install python-docx")

from database import get_session
from middlewares.custom_response import CustomRoute
from utils.auth import oauth2_scheme
import config
from py2neo import Graph
from datetime import datetime

"""
使用本地 Ollama(OpenAI 兼容接口) 调用大模型抽取三元组。
确保本机已启动 Ollama 并提供 http://localhost:11434/v1 接口，以及存在可用模型（默认 qwen3.5:2b）。
"""

router = APIRouter(prefix="/triple_construction", tags=["三元组构造"])
router.route_class = CustomRoute

# Pydantic models
class TextTripleRequest(BaseModel):
    text: str

class FileTripleRequest(BaseModel):
    filename: str
    text: str
    triples: List[Dict[str, str]]

class TripleItem(BaseModel):
    subject: str
    predicate: str
    object: str

class TextTripleResponse(BaseModel):
    text: str
    triples: List[TripleItem]

class FileTripleResponse(BaseModel):
    filename: str
    text: str
    triples: List[TripleItem]

class SaveTriplesRequest(BaseModel):
    triples: List[Dict[str, Any]]

class SaveTriplesResponse(BaseModel):
    status: str
    message: str
    savedTriples: List[Dict[str, Any]]

def call_llm_to_triples(text_prompt: str) -> List[Dict[str, str]]:
    """调用本地 Ollama(OpenAI 兼容) 抽取三元组，失败则尝试 /api/generate 兜底。
    返回 [{subject,predicate,object}, ...]
    """
    base_url_env = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    base_url_candidates = [
        base_url_env,
        base_url_env.replace("127.0.0.1", "localhost"),
    ]
    # 如果设置了 OLLAMA_HOST（如 http://127.0.0.1:11434），拼接 /v1
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if ollama_host and not ollama_host.endswith("/v1"):
        base_url_candidates.append(ollama_host + "/v1")
    model_name = os.getenv("LLM_MODEL", "qwen2:7b")
    if ":" not in model_name:
        model_name = model_name + ":latest"
    # 逐个 base_url 尝试
    def build_client(url: str) -> OpenAI:
        # 禁用系统代理，避免本地地址被代理导致 502
        # 增加超时时间到120秒，因为模型推理可能需要较长时间
        http_client = httpx.Client(trust_env=False, timeout=120.0)
        return OpenAI(
            api_key="ollama", 
            base_url=url, 
            http_client=http_client,
            max_retries=2  # 最多重试2次
        )

    def derive_root(url: str) -> str:
        return url[:-3] if url.endswith("/v1") else url

    def probe_ollama(root: str) -> bool:
        try:
            with httpx.Client(trust_env=False, timeout=5.0) as c:
                # 优先检查 /api/version
                r1 = c.get(f"{root}/api/version")
                if r1.status_code == 200:
                    return True
                # 其次检查 /api/tags
                r2 = c.get(f"{root}/api/tags")
                if r2.status_code == 200:
                    return True
                # 最后检查 /v1/models
                r3 = c.get(f"{root}/v1/models")
                return r3.status_code == 200
        except Exception:
            return False

    # 根据文本长度控制最大三元组数量（短句严控不超过2个）
    max_triples = 2 if len(text_prompt) <= 40 else 12

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个自然语言处理专家，严格从给定文本中抽取三元组。"
                "要求：1) 只从原文逐字抽取，实体与关系词必须逐字出现在原文；"
                "2) 不要臆造、不要引入原文未出现的概念；"
                f"3) 最多输出不超过{max_triples}个三元组；"
                "4) 建议使用原文中的关系词（如：通过、位于、包括、包含、属于、是、由…组成、使用、采用、导致、影响 等）；"
                "5) 若无法确认则不要输出。"
                "允许的输出格式：(实体1,关系,实体2) 或 [实体1,关系,实体2]，并尽量贴近原词形。"
                "示例1：输入：云南位于中国。输出：(云南,位于,中国)"
                "示例2：输入：有色金属包括锡。输出：(有色金属,包括,锡)"
                "示例3：输入：锡冶炼包含顶吹炉熔炼，电解精炼。输出：(锡冶炼,包含,顶吹炉熔炼)；(锡冶炼,包含,电解精炼)"
            ),
        },
        {"role": "user", "content": text_prompt},
    ]

    def ask_openai(messages: list) -> str:
        try:
            last_err = None
            # 仅对健康地址尝试
            healthy_urls = []
            for u in base_url_candidates:
                root = derive_root(u)
                if probe_ollama(root):
                    healthy_urls.append(u)
                else:
                    logging.getLogger("uvicorn").warning(f"LLM 健康检查失败: {root}")
            for url in healthy_urls:
                try:
                    c = build_client(url)
                    resp = c.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False,
                        temperature=0.0,
                    )
                    return resp.choices[0].message.content or ""
                except Exception as e:
                    last_err = e
                    logging.getLogger("uvicorn").warning(f"LLM /v1 调用失败({url}): {e}")
            if last_err:
                raise last_err
        except Exception as e:
            logging.getLogger("uvicorn").error(f"LLM 调用失败: {e}")
            return ""

    # /api/generate 兜底（Ollama 原生）
    def ask_generate(prompt: str) -> str:
        try:
            # 推导根地址
            # 取首个候选，或回退到默认
            bu = base_url_candidates[0] if base_url_candidates else base_url_env
            if bu.endswith("/v1"):
                root = bu[:-3]
            else:
                root = bu
            # 健康检查，不健康则直接放弃 generate
            if not probe_ollama(root):
                logging.getLogger("uvicorn").warning(f"LLM generate 跳过，健康检查失败: {root}")
                return ""
            url = f"{root}/api/generate"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            }
            # 增加超时时间到120秒
            with httpx.Client(timeout=120.0, trust_env=False) as client_http:
                r = client_http.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                return data.get("response", "") or ""
        except Exception as e:
            logging.getLogger("uvicorn").warning(f"LLM generate 兜底失败: {e}")
            return ""

    content = ask_openai(messages)
    if not content:
        # 构造 generate 提示
        gen_prompt = (
            "请从以下文本中抽取所有三元组，严格仅输出 JSON 数组，"
            "每个元素包含 subject、predicate、object 三个字段，不要任何额外文字。\n"
            "文本：\n" + text_prompt
        )
        content = ask_generate(gen_prompt)

    def parse_output(text: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        if not text:
            return results

        # 尝试 JSON 解析
        try:
            maybe_json = text.strip()
            # 提取可能包裹在代码块内的 JSON
            if maybe_json.startswith("```"):
                maybe_json = re.sub(r"^```[a-zA-Z]*", "", maybe_json).strip()
                if maybe_json.endswith("```"):
                    maybe_json = maybe_json[:-3]
            data = json.loads(maybe_json)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        s = item.get("subject") or item.get("实体1")
                        p = item.get("predicate") or item.get("关系")
                        o = item.get("object") or item.get("实体2")
                        if s and p and o:
                            results.append({
                                "subject": str(s).strip(),
                                "predicate": str(p).strip(),
                                "object": str(o).strip(),
                            })
                if results:
                    return results
        except Exception:
            pass

        # 匹配 (a,b,c) 或 （a，b，c）
        pattern_round = r"[\(（]\s*([^,，）)]+?)\s*[,，]\s*([^,，）)]+?)\s*[,，]\s*([^,，）)]+?)\s*[\)）]"
        for s, p, o in re.findall(pattern_round, text):
            results.append({
                "subject": s.strip().strip("'\" "),
                "predicate": p.strip().strip("'\" "),
                "object": o.strip().strip("'\" "),
            })

        # 匹配 [a,b,c] 或 【a，b，c】
        pattern_square = r"[\[【]\s*([^,，】\]]+?)\s*[,，]\s*([^,，】\]]+?)\s*[,，]\s*([^,，】\]]+?)\s*[\]】]"
        for s, p, o in re.findall(pattern_square, text):
            results.append({
                "subject": s.strip().strip("'\" "),
                "predicate": p.strip().strip("'\" "),
                "object": o.strip().strip("'\" "),
            })

        # 退化处理：用分号分隔，再逗号/顿号分隔
        if not results:
            parts = re.split(r"[；;\n]+", text)
            for part in parts:
                m = re.search(r"([^,，、]+)\s*[,，、]\s*([^,，、]+)\s*[,，、]\s*([^,，、]+)", part)
                if m:
                    s, p, o = m.groups()
                    results.append({
                        "subject": s.strip().strip("'\" "),
                        "predicate": p.strip().strip("'\" "),
                        "object": o.strip().strip("'\" "),
                    })
        return results

    def filter_triples_by_text_order(src: str, triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        src = src or ""
        filtered: List[Dict[str, str]] = []
        for t in triples:
            s = (t.get("subject") or "").strip()
            p = (t.get("predicate") or "").strip()
            o = (t.get("object") or "").strip()
            if not s or not p or not o:
                continue
            i1 = src.find(s)
            if i1 == -1:
                continue
            i2 = src.find(p, i1 + len(s))
            if i2 == -1:
                continue
            i3 = src.find(o, i2 + len(p))
            if i3 == -1:
                continue
            # 按原文顺序出现，避免重排导致的“看似合理但非逐字匹配”
            if i1 <= i2 <= i3:
                filtered.append({
                    "subject": s,
                    "predicate": p,
                    "object": o,
                })
        # 去重
        dedup = []
        seen = set()
        for it in filtered:
            key = (it["subject"], it["predicate"], it["object"])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(it)
        return dedup[:max_triples]

    triples = parse_output(content)
    triples = filter_triples_by_text_order(text_prompt, triples)
    if triples:
        logging.getLogger("uvicorn").info(f"LLM提取到 {len(triples)} 个三元组")
        return triples

    # 若首次失败，尝试强制 JSON 输出
    messages_json = [
        {
            "role": "system",
            "content": (
                "请从文本中抽取所有三元组，并仅输出 JSON 数组，"
                "每个元素包含 subject、predicate、object 三个字段，不要输出其他多余文本。"
            ),
        },
        {"role": "user", "content": text_prompt},
    ]
    content2 = ask_openai(messages_json)
    if not content2:
        # 再次尝试 generate 兜底
        gen_prompt_json = (
            "仅输出 JSON 数组，每个元素包含 subject、predicate、object 三个字段。\n"
            "文本：\n" + text_prompt
        )
        content2 = ask_generate(gen_prompt_json)
    triples2 = parse_output(content2)
    triples2 = filter_triples_by_text_order(text_prompt, triples2)
    if triples2:
        logging.getLogger("uvicorn").info(f"LLM(JSON)提取到 {len(triples2)} 个三元组")
        return triples2

    logging.getLogger("uvicorn").warning("LLM未能提取到三元组")
    return []


def extract_triples_rule_based(text: str) -> List[Dict[str, str]]:
    """基于规则的简易三元组抽取（中文优先），在 LLM 不可用时兜底。"""
    results: List[Dict[str, str]] = []
    if not text:
        return results

    # 常见关系关键词
    relation_patterns = [
        ("包括", r"(.+?)(?:包括|包含)[：:，,\s]*(.+)"),
        ("包含", r"(.+?)(?:包含)[：:，,\s]*(.+)"),
        ("位于", r"(.+?)位于(.+?)"),
        ("属于", r"(.+?)属于(.+?)"),
        ("是", r"(.+?)是(.+?)"),
        ("为", r"(.+?)为(.+?)"),
        ("含有", r"(.+?)含有(.+?)"),
        ("组成", r"(.+?)由(.+?)组成"),
        ("相关", r"(.+?)[与和及以及、]\s*(.+?)相关"),
        ("导致", r"(.+?)导致(.+?)"),
        ("影响", r"(.+?)影响(.+?)"),
        ("提高", r"(.+?)提高(.+?)"),
        ("降低", r"(.+?)降低(.+?)"),
        ("采用", r"(.+?)采用(.+?)"),
        ("使用", r"(.+?)使用(.+?)"),
        ("来源于", r"(.+?)来源于(.+?)"),
    ]

    # 逐行处理，降低误匹配
    lines = re.split(r"[\n。！？!?]+", text)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        matched = False
        for rel, pat in relation_patterns:
            m = re.search(pat, line)
            if not m:
                continue
            matched = True
            subj = m.group(1).strip(" '；;，,。\t\r\n")
            objs_raw = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
            if rel in ("包括", "包含") and objs_raw:
                # 按常见分隔符拆分多个客体
                objs = re.split(r"[、，,和及以及;；\s]+", objs_raw)
                for obj in objs:
                    obj = obj.strip(" '；;，,。\t\r\n")
                    if obj:
                        results.append({
                            "subject": subj,
                            "predicate": rel,
                            "object": obj,
                        })
            else:
                obj = objs_raw.strip(" '；;，,。\t\r\n")
                if subj and obj:
                    results.append({
                        "subject": subj,
                        "predicate": rel,
                        "object": obj,
                    })
            break

        # 若未匹配包含类，再尝试三段式“X，Y，Z”结构
        if not matched:
            m3 = re.search(r"([^,，、]+)[,，、]\s*([^,，、]+)[,，、]\s*([^,，、]+)", line)
            if m3:
                s, p, o = m3.groups()
                results.append({
                    "subject": s.strip(),
                    "predicate": p.strip(),
                    "object": o.strip(),
                })

    # 去重
    dedup = []
    seen = set()
    for it in results:
        key = (it["subject"], it["predicate"], it["object"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)
    if dedup:
        return dedup

    # 最弱兜底：在整段文本中找两个较长的中文词片段，形成“相关”关系
    # 仅作为兜底，避免完全空
    tokens = [t for t in re.split(r"[\s，,。；;、:：()（）\[\]【】]+", text) if t]
    tokens = sorted(set(tokens), key=lambda x: len(x), reverse=True)
    # 过滤常见停用词
    stopwords = {"的", "是", "在", "与", "和", "及", "以及", "对", "为", "为止", "通过", "进行", "一种"}
    strong = [t for t in tokens if len(t) >= 2 and t not in stopwords]
    if len(strong) >= 2:
        return [{"subject": strong[0][:12], "predicate": "相关", "object": strong[1][:12]}]
    return []

# 从文本构造三元组
@router.post("/text", response_model=TextTripleResponse)
async def construct_triples_from_text(
    request: TextTripleRequest
):
    """从文本构造三元组"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="文本内容不能为空")

        # 调用LLM提取三元组，失败则规则兜底
        raw_triples = call_llm_to_triples(request.text)
        if not raw_triples:
            raw_triples = extract_triples_rule_based(request.text)

        # 转换为标准格式
        triples = [
            TripleItem(
                subject=triple.get("subject", ""),
                predicate=triple.get("predicate", ""),
                object=triple.get("object", "")
            )
            for triple in raw_triples
        ]

        return TextTripleResponse(
            text=request.text,
            triples=triples
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文本处理失败: {str(e)}")

# 从文件构造三元组
@router.post("/file", response_model=FileTripleResponse)
async def construct_triples_from_file(
    file: UploadFile = File(...)
):
    """从文件构造三元组"""
    try:
        # 验证文件类型
        allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型 {file_extension}。支持的格式: {', '.join(allowed_extensions)}"
            )

        # 检查依赖
        if file_extension == '.pdf' and not PDF_SUPPORT:
            raise HTTPException(
                status_code=500,
                detail="PDF处理功能不可用。服务器缺少 PyMuPDF 库，请联系管理员安装: pip install PyMuPDF"
            )
        
        if file_extension in ['.doc', '.docx'] and not DOCX_SUPPORT:
            raise HTTPException(
                status_code=500,
                detail="Word文档处理功能不可用。服务器缺少 python-docx 库，请联系管理员安装: pip install python-docx"
            )

        # 读取文件内容
        file_content = await file.read()
        text_content = ""

        if file_extension == '.pdf':
            # 处理PDF文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            try:
                with fitz.open(temp_file_path) as doc:
                    for page in doc:
                        text_content += page.get_text()
            finally:
                os.unlink(temp_file_path)

        elif file_extension in ['.doc', '.docx']:
            # 处理Word文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            try:
                doc = Document(temp_file_path)
                text_content = "\n".join([para.text for para in doc.paragraphs])
            finally:
                os.unlink(temp_file_path)

        elif file_extension == '.txt':
            # 处理文本文件
            try:
                text_content = file_content.decode("utf-8")
            except UnicodeDecodeError:
                # 尝试其他编码
                try:
                    text_content = file_content.decode("gbk")
                except UnicodeDecodeError:
                    text_content = file_content.decode("latin-1")

        if not text_content.strip():
            raise HTTPException(status_code=400, detail="文件中未找到可处理的文本内容")

        # 调用LLM提取三元组，失败则规则兜底
        raw_triples = call_llm_to_triples(text_content)
        if not raw_triples:
            raw_triples = extract_triples_rule_based(text_content)

        # 转换为标准格式
        triples = [
            TripleItem(
                subject=triple.get("subject", ""),
                predicate=triple.get("predicate", ""),
                object=triple.get("object", "")
            )
            for triple in raw_triples
        ]

        return FileTripleResponse(
            filename=file.filename,
            text=text_content[:1000] + "..." if len(text_content) > 1000 else text_content,
            triples=triples
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger('uvicorn').exception(f'文件处理失败: {e}')
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")

# 保存三元组到知识图谱（写入 Neo4j）
@router.post("/save_with_categories", response_model=SaveTriplesResponse)
async def save_triples_with_categories(
    request: SaveTriplesRequest
):
    """保存带有类别信息的三元组到 Neo4j。返回每条三元组的保存状态与创建的节点/关系 id。"""
    try:
        if not request.triples:
            raise HTTPException(status_code=400, detail="没有提供三元组数据")

        logging.getLogger('uvicorn').info(f"save_with_categories 请求: {request.triples}")

        # 连接 Neo4j
        try:
            g = Graph(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
        except Exception as e:
            logging.getLogger('uvicorn').exception(f'连接 Neo4j 失败: {e}')
            raise HTTPException(status_code=500, detail=f"连接 Neo4j 失败: {e}")
        now = datetime.now().strftime("%Y-%m-%d")

        saved_triples: List[Dict[str, Any]] = []

        for item in request.triples:
            e1 = (item.get('entity1') or '').strip()
            e2 = (item.get('entity2') or '').strip()
            rel = (item.get('relation') or '').strip()
            c1 = (item.get('entity1Category') or 'Other').strip() or 'Other'
            c2 = (item.get('entity2Category') or 'Other').strip() or 'Other'

            if not e1 or not e2 or not rel:
                saved_triples.append({
                    'entity1': e1,
                    'entity1Category': c1,
                    'relation': rel,
                    'entity2': e2,
                    'entity2Category': c2,
                    'status': 'skipped',
                    'reason': '缺少实体或关系'
                })
                continue

            # 清理 label/relation 中的反引号，并做严格合法化，避免 Cypher 错误或注入问题
            def normalize_label(s: str) -> str:
                s = (s or '').replace('`', '').strip()
                # 将连续非法字符替换为下划线，允许中文、字母、数字和下划线
                s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", s)
                # 避免空 label
                if not s:
                    return 'Other'
                # 限制长度
                if len(s) > 64:
                    s = s[:64]
                return s

            def normalize_relation(s: str) -> str:
                s = (s or '').replace('`', '').strip()
                s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", s)
                if not s:
                    return 'RELATED'
                if len(s) > 64:
                    s = s[:64]
                return s

            safe_c1 = normalize_label(c1)
            safe_c2 = normalize_label(c2)
            safe_rel = normalize_relation(rel)

            try:
                # 先获取当前最大的 sortIndex
                max_index_query = "MATCH (n) WHERE n.sortIndex IS NOT NULL RETURN max(n.sortIndex) AS max_index"
                max_index_result = g.run(max_index_query).evaluate()
                next_index = (int(max_index_result) + 1) if max_index_result is not None else 1000000
                
                # MERGE 节点并设置更新时间和 sortIndex
                q1 = f"""
                MERGE (a:`{safe_c1}` {{name: $e1}}) 
                ON CREATE SET a.updatedAt=$now, a.createdAt=$now, a.sortIndex=$idx1
                ON MATCH SET a.updatedAt=$now
                RETURN id(a) as id
                """
                r1 = g.run(q1, e1=e1, now=now, idx1=next_index).data()
                nid1 = r1[0]['id'] if r1 else None
                
                # 为第二个节点获取新的 sortIndex（如果是新创建的）
                next_index2 = next_index + 1

                q2 = f"""
                MERGE (b:`{safe_c2}` {{name: $e2}}) 
                ON CREATE SET b.updatedAt=$now, b.createdAt=$now, b.sortIndex=$idx2
                ON MATCH SET b.updatedAt=$now
                RETURN id(b) as id
                """
                r2 = g.run(q2, e2=e2, now=now, idx2=next_index2).data()
                nid2 = r2[0]['id'] if r2 else None

                # MERGE 关系并设置创建时间
                qrel = (
                    f"MATCH (a:`{safe_c1}` {{name: $e1}}),(b:`{safe_c2}` {{name: $e2}}) "
                    f"MERGE (a)-[r:`{safe_rel}`]->(b) SET r.createdAt=$now RETURN id(r) as id"
                )
                rr = g.run(qrel, e1=e1, e2=e2, now=now).data()
                rid = rr[0]['id'] if rr else None

                saved_triples.append({
                    'entity1': e1,
                    'entity1Category': c1,
                    'relation': rel,
                    'entity2': e2,
                    'entity2Category': c2,
                    'status': 'saved',
                    'node1_id': nid1,
                    'node2_id': nid2,
                    'rel_id': rid
                })
            except Exception as e:
                logging.getLogger('uvicorn').exception(f'保存三元组到 Neo4j 失败: {e}')
                saved_triples.append({
                    'entity1': e1,
                    'entity1Category': c1,
                    'relation': rel,
                    'entity2': e2,
                    'entity2Category': c2,
                    'status': 'error',
                    'reason': str(e)
                })

        return SaveTriplesResponse(
            status="success",
            message=f"处理完成，共 {len(saved_triples)} 条三元组",
            savedTriples=saved_triples
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存三元组失败: {str(e)}")


# ==================== 语音识别接口 ====================

class VoiceTripleResponse(BaseModel):
    text: str
    triples: Optional[List[Dict[str, str]]] = None

# 从语音构造三元组
@router.post("/voice", response_model=VoiceTripleResponse)
async def construct_triples_from_voice(
    audio: UploadFile = File(...)
):
    """
    从上传的音频文件中识别语音，提取文本并构造三元组
    
    - audio: 音频文件（支持 WAV、MP3、OGG、FLAC 等格式）
    - 返回识别的文本和提取的三元组列表
    """
    try:
        # 导入语音识别库
        import speech_recognition as sr
    except ImportError:
        raise HTTPException(
            status_code=500, 
            detail="speech_recognition 未安装，请运行: pip install SpeechRecognition"
        )
    
    if not audio:
        raise HTTPException(status_code=400, detail="缺少音频文件")
    
    # 检查文件格式
    file_ext = os.path.splitext(audio.filename)[1].lower()
    if file_ext not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm']:
        raise HTTPException(status_code=400, detail="不支持的音频格式，请上传 WAV、MP3、OGG、FLAC、M4A 或 WEBM 格式")
    
    recognizer = sr.Recognizer()
    tmp_original_path = None
    tmp_wav_path = None
    
    try:
        # 创建临时文件保存上传的音频（保持原始格式）
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_original_path = tmp_file.name
        
        logging.getLogger('uvicorn').info(f"上传的音频文件: {audio.filename}, 大小: {len(content)} 字节")
        
        # 如果不是 WAV 格式，需要转换
        if file_ext != '.wav':
            # 检查是否支持音频转换
            try:
                from pydub import AudioSegment
                from pydub.utils import which
                HAS_PYDUB = True
            except ImportError:
                raise HTTPException(
                    status_code=400, 
                    detail="当前不支持 MP3 等格式，请上传 WAV 格式音频文件，或联系管理员安装 pydub 库"
                )
            
            # 尝试找到 ffmpeg
            import subprocess
            import shutil
            
            # 方法1: 检查系统 PATH 中的 ffmpeg
            ffmpeg_path = shutil.which('ffmpeg')
            
            # 方法2: 检查常见安装位置（如果方法1失败）
            if not ffmpeg_path:
                common_paths = [
                    r'D:\APP\chorme_download\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe',
                    r'C:\ffmpeg\bin\ffmpeg.exe',
                    r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                    r'D:\ffmpeg\bin\ffmpeg.exe',
                    r'D:\APP\ffmpeg\bin\ffmpeg.exe',
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        ffmpeg_path = path
                        break
            
            if ffmpeg_path:
                logging.getLogger('uvicorn').info(f"找到 ffmpeg: {ffmpeg_path}")
                # 设置 pydub 使用的 ffmpeg 路径
                AudioSegment.converter = ffmpeg_path
                
                # 查找 ffprobe（在同一目录下）
                ffprobe_path = ffmpeg_path.replace('ffmpeg.exe', 'ffprobe.exe')
                if os.path.exists(ffprobe_path):
                    logging.getLogger('uvicorn').info(f"找到 ffprobe: {ffprobe_path}")
                    AudioSegment.ffprobe = ffprobe_path
                else:
                    logging.getLogger('uvicorn').warning(f"未找到 ffprobe: {ffprobe_path}")
                    # ffprobe 不是必需的，可以继续尝试
            else:
                logging.getLogger('uvicorn').warning("未找到 ffmpeg")
                raise HTTPException(
                    status_code=400, 
                    detail="系统未安装 ffmpeg，无法转换音频格式。请上传 WAV 格式音频文件。\n\n" +
                           "如需支持 MP3 等格式，请：\n" +
                           "1. 下载 ffmpeg: https://github.com/BtbN/FFmpeg-Builds/releases\n" +
                           "2. 解压后将 bin 目录添加到系统 PATH\n" +
                           "3. 重启后端服务"
                )
            
            # 尝试转换音频格式 - 使用 subprocess 直接调用 ffmpeg（更可靠）
            try:
                logging.getLogger('uvicorn').info(f"转换音频格式: {file_ext} -> .wav (使用 ffmpeg)")
                
                # 生成输出文件路径
                tmp_wav_path = tmp_original_path.replace(file_ext, '.wav')
                
                # 直接调用 ffmpeg 命令行转换
                import subprocess
                cmd = [
                    ffmpeg_path,
                    '-i', tmp_original_path,  # 输入文件
                    '-ar', '16000',           # 采样率 16000Hz
                    '-ac', '1',               # 单声道
                    '-y',                     # 覆盖输出文件
                    tmp_wav_path              # 输出文件
                ]
                
                logging.getLogger('uvicorn').info(f"执行命令: {' '.join(cmd)}")
                
                # 增加超时时间到120秒，并移除可能导致挂起的标志
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120,
                    shell=False
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr.decode('utf-8', errors='ignore')
                    logging.getLogger('uvicorn').error(f"ffmpeg 转换失败: {error_msg}")
                    raise Exception(f"ffmpeg 返回错误: {error_msg[:500]}")
                
                # 记录 ffmpeg 输出信息
                stderr_output = result.stderr.decode('utf-8', errors='ignore')
                if stderr_output:
                    logging.getLogger('uvicorn').info(f"ffmpeg 输出: {stderr_output[:200]}")
                
                # 检查输出文件是否存在
                if not os.path.exists(tmp_wav_path):
                    raise Exception("转换后的 WAV 文件未生成")
                
                file_size = os.path.getsize(tmp_wav_path)
                logging.getLogger('uvicorn').info(f"音频转换成功: {tmp_wav_path}, 大小: {file_size} 字节")
                
            except subprocess.TimeoutExpired:
                logging.getLogger('uvicorn').error("音频转换超时")
                raise HTTPException(
                    status_code=400, 
                    detail="音频转换超时，文件可能太大或格式复杂"
                )
            except Exception as conv_err:
                logging.getLogger('uvicorn').error(f"音频转换失败: {conv_err}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"音频格式转换失败: {str(conv_err)}。建议上传 WAV 格式音频文件"
                )
        else:
            tmp_wav_path = tmp_original_path
        
        try:
            # 使用 speech_recognition 读取音频文件
            with sr.AudioFile(tmp_wav_path) as source:
                # 调整识别器以适应环境噪音
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
            
            logging.getLogger('uvicorn').info("音频数据读取成功，开始识别...")
            
            # 尝试多种语音识别方式（按优先级）
            text = None
            last_error = None
            
            # 方式1: 尝试使用 Whisper（OpenAI 本地模型）
            try:
                import whisper
                logging.getLogger('uvicorn').info("尝试使用 Whisper 本地模型识别...")
                model = whisper.load_model("base")  # 可选: tiny, base, small, medium, large
                result = model.transcribe(tmp_wav_path, language="zh")
                text = result["text"]
                logging.getLogger('uvicorn').info(f"Whisper 识别成功: {text}")
            except ImportError:
                logging.getLogger('uvicorn').warning("Whisper 未安装，跳过。安装命令: pip install openai-whisper")
            except Exception as e:
                last_error = e
                logging.getLogger('uvicorn').warning(f"Whisper 识别失败: {e}")
            
            # 方式2: 如果 Whisper 失败，尝试使用 Vosk（离线识别）
            if not text:
                try:
                    from vosk import Model, KaldiRecognizer
                    import wave
                    logging.getLogger('uvicorn').info("尝试使用 Vosk 离线识别...")
                    
                    # Vosk 模型路径（需要预先下载）
                    model_path = os.getenv("VOSK_MODEL_PATH", "model")
                    if os.path.exists(model_path):
                        model = Model(model_path)
                        wf = wave.open(tmp_wav_path, "rb")
                        rec = KaldiRecognizer(model, wf.getframerate())
                        rec.SetWords(True)
                        
                        result_text = ""
                        while True:
                            data = wf.readframes(4000)
                            if len(data) == 0:
                                break
                            if rec.AcceptWaveform(data):
                                result = json.loads(rec.Result())
                                result_text += result.get("text", "") + " "
                        
                        final_result = json.loads(rec.FinalResult())
                        result_text += final_result.get("text", "")
                        text = result_text.strip()
                        logging.getLogger('uvicorn').info(f"Vosk 识别成功: {text}")
                    else:
                        logging.getLogger('uvicorn').warning(f"Vosk 模型未找到: {model_path}")
                except ImportError:
                    logging.getLogger('uvicorn').warning("Vosk 未安装，跳过。安装命令: pip install vosk")
                except Exception as e:
                    last_error = e
                    logging.getLogger('uvicorn').warning(f"Vosk 识别失败: {e}")
            
            # 方式3: 最后尝试 Google 语音识别（需要网络）
            if not text:
                try:
                    logging.getLogger('uvicorn').info("尝试使用 Google 语音识别...")
                    text = recognizer.recognize_google(audio_data, language="zh-CN")
                    logging.getLogger('uvicorn').info(f"Google 识别成功: {text}")
                except sr.UnknownValueError:
                    logging.getLogger('uvicorn').warning("Google 无法识别语音内容")
                except sr.RequestError as e:
                    last_error = e
                    logging.getLogger('uvicorn').warning(f"Google 识别请求失败: {e}")
            
            # 如果所有方法都失败
            if not text or not text.strip():
                error_msg = "所有语音识别方法均失败。\n\n"
                error_msg += "建议:\n"
                error_msg += "1. 安装 Whisper 本地模型: pip install openai-whisper\n"
                error_msg += "2. 或安装 Vosk: pip install vosk 并下载中文模型\n"
                error_msg += "3. 或配置网络代理以访问 Google 语音识别服务\n"
                if last_error:
                    error_msg += f"\n最后错误: {str(last_error)}"
                raise HTTPException(status_code=500, detail=error_msg)
            
            logging.getLogger('uvicorn').info(f"最终识别结果: {text}")
            
            # 返回识别的文本，前端会再次调用 /text 接口来提取三元组
            return VoiceTripleResponse(text=text)
            
        finally:
            # 删除临时文件
            for path in [tmp_original_path, tmp_wav_path]:
                if path:
                    try:
                        os.unlink(path)
                    except Exception:
                        pass
    
    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="无法识别语音，请确保音频清晰且包含中文内容")
    except sr.RequestError as e:
        logging.getLogger('uvicorn').error(f"语音识别请求错误: {e}")
        raise HTTPException(status_code=500, detail=f"语音识别服务请求失败: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger('uvicorn').exception(f"语音处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"语音处理失败: {str(e)}")
