# -*- coding: UTF-8 -*-
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

import jieba
from fastapi import APIRouter, Depends, HTTPException
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
from pydantic import BaseModel
from sqlmodel import Session, select

import config
from database import get_session
from middlewares.custom_response import CustomRoute

from models.knowledge_graph import Wnode, Qa
from utils.auth import oauth2_scheme
from openai import OpenAI
import httpx
import logging
import json

router = APIRouter(prefix="/graph", tags=["知识图谱"])
router.route_class = CustomRoute

# 复用 Neo4j 连接，避免每次请求创建 Graph
from threading import Lock
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
_graph_instance: Graph | None = None
_graph_lock = Lock()

def get_graph() -> Graph:
    global _graph_instance
    if _graph_instance is None:
        with _graph_lock:
            if _graph_instance is None:
                # 动态拼接连接池参数，缓解高并发下连接池耗尽
                max_pool = int(os.getenv("NEO4J_MAX_POOL_SIZE", "200") or 200)
                acquire_timeout = os.getenv("NEO4J_ACQUIRE_TIMEOUT", "60s")
                try:
                    parsed = urlparse(config.NEO4J_URI)
                    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
                    # 仅在未设置时追加，避免覆盖已有显式配置
                    q.setdefault("max_connection_pool_size", str(max_pool))
                    q.setdefault("connection_acquisition_timeout", acquire_timeout)
                    new_uri = urlunparse((
                        parsed.scheme,
                        parsed.netloc,
                        parsed.path,
                        parsed.params,
                        urlencode(q),
                        parsed.fragment,
                    ))
                except Exception:
                    new_uri = config.NEO4J_URI
                _graph_instance = Graph(new_uri, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    return _graph_instance

# Pydantic models for Neo4j management
class NodeData(BaseModel):
    id: Optional[int] = None
    name: str
    category: str
    properties: Optional[Dict[str, Any]] = {}

class RelationshipData(BaseModel):
    startNode: str
    endNode: str
    type: str
    startLabel: Optional[str] = None
    endLabel: Optional[str] = None

class CreateNodeWithRelationRequest(BaseModel):
    """一次性创建新节点并与已存在或待创建的原节点建立关系。

    - newCategory/newName: 新节点的类别与名称
    - origCategory/origName: 原节点的类别与名称（若不存在将创建）
    - relationType: 两者之间的关系类型
    - newProperties: 可选，新节点的属性字典
    - origProperties: 可选，原节点的属性字典（当原节点不存在且需创建时使用）
    """
    newCategory: str
    newName: str
    origCategory: str
    origName: str
    relationType: str
    direction: Optional[str] = "new_to_orig"  # new_to_orig 或 orig_to_new
    newProperties: Optional[Dict[str, Any]] = {}
    origProperties: Optional[Dict[str, Any]] = {}

class Neo4jStats(BaseModel):
    entityCount: int
    relationCount: int
    todayUpdate: int
    dataSize: float

# 旧的 color 映射仅提供一个默认黑色，已弃用；改为在 get_data 内按类别分配安全调色板
color = {
    "other": "#1677ff"
}

# 简易类别映射，用于将英文类别转换为中文展示
CATEGORY_MAP = {
    "ElectricFurnace": "电炉",
    "Person": "人物",
    "Company": "公司",
    "Technology": "技术",
    "Product": "产品",
    "Process": "工艺",
}


def get_words_path():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    words_path = os.path.join(cur_dir, '../data/words.txt')
    return words_path


def get_str_by_dict(mydict):
    """将节点属性字典转换为显示字符串，过滤掉内部使用的属性"""
    last = ""
    # 需要过滤掉的内部属性列表
    internal_props = {'sortIndex', 'sortindex', 'updatedAt', 'createdAt', 'layoutX', 'layoutY'}
    
    for key in mydict:
        # 跳过内部属性
        if key in internal_props:
            continue
        last = str(key) + ":" + str(mydict[key]) + "<br>" + last
    return last


def _build_llm_clients():
    """构建本地 LLM 调用所需的候选地址与工具。"""
    base_url_env = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    base_url_candidates = [
        base_url_env,
        base_url_env.replace("127.0.0.1", "localhost"),
    ]
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if ollama_host and not ollama_host.endswith("/v1"):
        base_url_candidates.append(ollama_host + "/v1")
    model_name = os.getenv("LLM_MODEL", "qwen2:7b")
    if ":" not in model_name:
        model_name = model_name + ":latest"

    def build_client(url: str) -> OpenAI:
        http_client = httpx.Client(trust_env=False, timeout=600.0)
        return OpenAI(api_key="ollama", base_url=url, http_client=http_client)

    def derive_root(url: str) -> str:
        return url[:-3] if url.endswith("/v1") else url

    def probe_ollama(root: str) -> bool:
        try:
            with httpx.Client(trust_env=False, timeout=5.0) as c:
                r1 = c.get(f"{root}/api/version")
                if r1.status_code == 200:
                    return True
                r2 = c.get(f"{root}/api/tags")
                if r2.status_code == 200:
                    return True
                r3 = c.get(f"{root}/v1/models")
                return r3.status_code == 200
        except Exception:
            return False

    return base_url_candidates, model_name, build_client, derive_root, probe_ollama


def _qa_llm_answer(question: str, context: str) -> str:
    """调用 LLM 生成问答回复。优先 /v1/chat/completions，失败则尝试 /api/generate。"""
    base_url_candidates, model_name, build_client, derive_root, probe_ollama = _build_llm_clients()

    system_prompt = (
        "你是锡冶炼领域的问答助手。请用地道中文、简洁、准确地回答问题。\n"
        "已知背景（来自知识图谱或检索）仅供参考：\n" + (context or "无") + "\n"
        "要求：\n"
        "1) 严格限制在锡冶炼/有色冶金领域，禁止输出与领域无关的内容；\n"
        "2) 若背景可直接回答，请直接给出明确结论；\n"
        "3) 若背景不足，不得编造事实；可给出需要哪些信息或建议查询方向；\n"
        "4) 若用户仅寒暄或问题与领域无关，请简短礼貌地引导其提出与锡冶炼相关的问题；\n"
        "5) 回答长度不超过120字。\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # 首选 /v1/chat/completions
    try:
        healthy_urls = []
        last_err = None
        for u in base_url_candidates:
            root = derive_root(u)
            if probe_ollama(root):
                healthy_urls.append(u)
            else:
                logging.getLogger("uvicorn").warning(f"LLM 健康检查失败: {root}")
        for url in healthy_urls:
            try:
                client = build_client(url)
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=False,
                    temperature=0.2,
                )
                content = resp.choices[0].message.content or ""
                if content:
                    return content.strip()
            except Exception as e:
                last_err = e
                logging.getLogger("uvicorn").warning(f"LLM /v1 调用失败({url}): {e}")
        if last_err:
            raise last_err
    except Exception as e:
        logging.getLogger("uvicorn").error(f"LLM 调用失败: {e}")

    # 兜底 /api/generate
    try:
        bu = base_url_candidates[0] if base_url_candidates else os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
        root = derive_root(bu)
        if not probe_ollama(root):
            logging.getLogger("uvicorn").warning(f"LLM generate 跳过，健康检查失败: {root}")
            return ""
        url = f"{root}/api/generate"
        prompt = system_prompt + "\n用户问题：" + question + "\n请直接回答。"
        payload = {"model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
        with httpx.Client(timeout=600.0, trust_env=False) as c:
            r = c.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return (data.get("response", "") or "").strip()
    except Exception as e:
        logging.getLogger("uvicorn").warning(f"LLM generate 兜底失败: {e}")
        return ""


@router.get("/getdata/")
def get_data(start: str = "", relation: str = "", end: str = "", start_id: Optional[int] = None, end_id: Optional[int] = None, start_label: str = "", end_label: str = "", limit: int = 200):
    """
    知识图谱数据获取接口

    参数：
    - start: 起始节点名称（可选）
    - relation: 关系类型（可选）
    - end: 结束节点名称（可选）
    - start_id: 起始节点 id（可选，优先于 start）
    - end_id: 结束节点 id（可选，优先于 end）
    - start_label/end_label: 对起点/终点增加标签过滤（可选）

    返回：
    - datas: 节点数据
    - links: 连接数据
    - legend_data: 图例数据
    - categories: 节点类别
    """
    g = get_graph()
    datas = []
    links = []
    node_cache = set()
    link_cache = set()
    categories = []
    legend_data = []
    # 按类别动态分配颜色，禁用黑色，首选蓝色
    palette = [
        "#1677ff", "#fa8c16", "#52c41a", "#ff4d4f", "#722ed1", "#13c2c2", "#eb2f96",
        "#2f54eb", "#a0d911", "#fa541c"
    ]
    label_to_color: Dict[str, str] = {}
    def get_color_for(label_name: str) -> str:
        key = label_name or "其他"
        if key in label_to_color:
            return label_to_color[key]
        color_val = palette[len(label_to_color) % len(palette)]
        label_to_color[key] = color_val
        return color_val
    # 限制返回行数，避免一次性返回过大结果集
    limit = max(1, min(int(limit or 200), 1000))
    
    # 检查是否查询单个节点（按实体查询）
    is_single_node_query = False
    if (start or start_id is not None) and not end and not end_id and not relation:
        is_single_node_query = True
    
    if is_single_node_query:
        # 单节点查询：先查询该节点及其关系，如果没有关系则只返回该节点
        where_parts: List[str] = []
        params: Dict[str, Any] = {}
        
        if start_id is not None:
            where_parts.append("id(n) = $start_id")
            params["start_id"] = int(start_id)
        elif start:
            where_parts.append("coalesce(n.name, n.nodename) = $start_name")
            params["start_name"] = start
        if start_label:
            where_parts.append("$start_label IN labels(n)")
            params["start_label"] = start_label
            
        where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        
        # 先尝试查询有关系的情况
        cypher_with_rel = f"MATCH (n)-[r]->(b){where_clause} RETURN n, type(r) as relType, b LIMIT $limit"
        params["limit"] = limit
        
        try:
            nodes_data_all = g.run(cypher_with_rel, **params).data()
        except Exception:
            nodes_data_all = []
        
        # 如果没有找到任何关系，查询该节点本身
        if not nodes_data_all:
            cypher_single = f"MATCH (n){where_clause} RETURN n LIMIT 1"
            try:
                single_node_data = g.run(cypher_single, **params).data()
                if single_node_data:
                    n_node = single_node_data[0].get('n')
                    if n_node:
                        start_dict = dict(n_node)
                        start_name = start_dict.get('name') or start_dict.get('nodename') or ''
                        
                        try:
                            start_label = list(n_node.labels)[0] if getattr(n_node, 'labels', None) else start_dict.get('type') or start_dict.get('category') or 'Unknown'
                        except Exception:
                            start_label = start_dict.get('type') or start_dict.get('category') or 'Unknown'
                        
                        start_label_disp = CATEGORY_MAP.get(start_label, start_label)
                        
                        try:
                            start_id = getattr(n_node, 'identity', None)
                        except Exception:
                            start_id = None
                        
                        start_node_key = f"{start_name}|{start_label_disp}|{start_id}"
                        if start_name and start_node_key not in node_cache:
                            start_color = get_color_for(start_label_disp)
                            datas.append({
                                "name": start_name,
                                "id": start_id,
                                "attr": start_dict,
                                "color": start_color,
                                "des": get_str_by_dict(start_dict),
                                "category": start_label_disp,
                                "labels": list(getattr(n_node, 'labels', []) or [])
                            })
                            node_cache.add(start_node_key)
                            
                            if start_label_disp not in legend_data:
                                legend_data.append(start_label_disp)
                                categories.append({"name": start_label_disp})
            except Exception as e:
                logging.getLogger('uvicorn').exception(f'查询单节点时出错: {e}')
    else:
        # 原有的关系查询逻辑
        # 构造 MATCH 子句
        mr = "r" if not relation else f"r:`{relation}`"
        where_parts: List[str] = []
        params: Dict[str, Any] = {"limit": limit}
        # 起点过滤：优先 id 其次 name
        if start_id is not None:
            where_parts.append("id(n) = $start_id")
            params["start_id"] = int(start_id)
        elif start:
            where_parts.append("coalesce(n.name, n.nodename) = $start_name")
            params["start_name"] = start
        if start_label:
            where_parts.append("$start_label IN labels(n)")
            params["start_label"] = start_label
        # 终点过滤
        if end_id is not None:
            where_parts.append("id(b) = $end_id")
            params["end_id"] = int(end_id)
        elif end:
            where_parts.append("coalesce(b.name, b.nodename) = $end_name")
            params["end_name"] = end
        if end_label:
            where_parts.append("$end_label IN labels(b)")
            params["end_label"] = end_label
        where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
        # 使用通用安全写法：先不限定 r 的标签，必要时用 type(r) 进行过滤
        base_match = f"MATCH (n)-[r]->(b){where_clause} "
        if relation:
            cypher = base_match + "WHERE " + (" AND ".join([p for p in []])) + (" AND " if where_clause else " WHERE ") + "type(r) = $relType RETURN n, type(r) as relType, b LIMIT $limit"
            # 调整：如果已有 where_clause，则已经包含 WHERE，拼接 AND type(r) = $relType
            if where_clause:
                cypher = f"MATCH (n)-[r]->(b){where_clause} AND type(r) = $relType RETURN n, type(r) as relType, b LIMIT $limit"
            else:
                cypher = f"MATCH (n)-[r]->(b) WHERE type(r) = $relType RETURN n, type(r) as relType, b LIMIT $limit"
            params["relType"] = relation
        else:
            cypher = base_match + "RETURN n, type(r) as relType, b LIMIT $limit"

        try:
            nodes_data_all = g.run(cypher, **params).data()
        except Exception:
            # 回退策略：忽略 relation 过滤
            fallback = f"MATCH (n)-[r]->(b){where_clause} RETURN n, type(r) as relType, b LIMIT $limit"
            nodes_data_all = g.run(fallback, **params).data()
    
    # 处理查询结果
    for nodes_relations in nodes_data_all:
        try:
            n_node = nodes_relations.get('n')
            # 使用 cypher 返回的 relType 优先
            rel_type_from_cypher = nodes_relations.get('relType')
            b_node = nodes_relations.get('b')
            # 如果没有 relType，则尝试从 r 字段回退（兼容老逻辑）
            r_rel = nodes_relations.get('r') if 'r' in nodes_relations else None
            if not (n_node and (rel_type_from_cypher is not None or r_rel is not None) and b_node):
                continue

            start_dict = dict(n_node)
            end_dict = dict(b_node)
            start_name = start_dict.get('name') or start_dict.get('nodename') or ''
            end_name = end_dict.get('name') or end_dict.get('nodename') or ''

            # relation type 优先使用 cypher 中的 relType（字符串），否则回退到 r_rel 的属性或表示
            try:
                if rel_type_from_cypher is not None:
                    relation = str(rel_type_from_cypher)
                else:
                    relation = getattr(r_rel, 'type', None) or (r_rel.__class__.__name__ if r_rel is not None else '')
            except Exception:
                relation = str(r_rel)

            # labels
            try:
                start_label = list(n_node.labels)[0] if getattr(n_node, 'labels', None) else start_dict.get('type') or start_dict.get('category') or 'Unknown'
            except Exception:
                start_label = start_dict.get('type') or start_dict.get('category') or 'Unknown'
            try:
                end_label = list(b_node.labels)[0] if getattr(b_node, 'labels', None) else end_dict.get('type') or end_dict.get('category') or 'Unknown'
            except Exception:
                end_label = end_dict.get('type') or end_dict.get('category') or 'Unknown'

            # map display category
            start_label_disp = CATEGORY_MAP.get(start_label, start_label)
            end_label_disp = CATEGORY_MAP.get(end_label, end_label)

            # ids
            try:
                start_id = getattr(n_node, 'identity', None)
            except Exception:
                start_id = None
            try:
                end_id = getattr(b_node, 'identity', None)
            except Exception:
                end_id = None

            # add start node (unique by name+label+id)
            start_node_key = f"{start_name}|{start_label_disp}|{start_id}"
            if start_name and start_node_key not in node_cache:
                start_color = get_color_for(start_label_disp)
                datas.append({
                    "name": start_name,
                    "id": start_id,
                    "attr": start_dict,
                    "color": start_color,
                    "des": get_str_by_dict(start_dict),
                    "category": start_label_disp,
                    "labels": list(getattr(n_node, 'labels', []) or [])
                })
                node_cache.add(start_node_key)

            # add end node (unique by name+label+id)
            end_node_key = f"{end_name}|{end_label_disp}|{end_id}"
            if end_name and end_node_key not in node_cache:
                end_color = get_color_for(end_label_disp)
                datas.append({
                    "name": end_name,
                    "id": end_id,
                    "attr": end_dict,
                    "color": end_color,
                    "des": get_str_by_dict(end_dict),
                    "category": end_label_disp,
                    "labels": list(getattr(b_node, 'labels', []) or [])
                })
                node_cache.add(end_node_key)

            # legend & categories
            if start_label_disp not in legend_data:
                legend_data.append(start_label_disp)
                categories.append({"name": start_label_disp})
            if end_label_disp not in legend_data:
                legend_data.append(end_label_disp)
                categories.append({"name": end_label_disp})

            # add link，确保 relation 为字符串
            try:
                rel_name = str(relation)
            except Exception:
                rel_name = ''
            cache_relation = f"{start_name}|{start_label_disp}|{start_id}__{end_name}|{end_label_disp}|{end_id}__{rel_name}"
            if cache_relation not in link_cache:
                links.append({
                    "source": start_name,
                    "target": end_name,
                    "name": rel_name,
                    "sourceLabel": start_label_disp,
                    "targetLabel": end_label_disp,
                    "sourceId": start_id,
                    "targetId": end_id,
                })
                link_cache.add(cache_relation)
        except Exception as e:
            logging.getLogger('uvicorn').exception(f'处理 getdata 时出错: {e}')
            continue
    return {"datas": datas, "links": links, "legend_data": legend_data, "categories": categories}


@router.post('/save_layout')
def save_layout(payload: Dict[str, Any]):
    """保存节点布局：接收 {nodes: [{name, x, y}, ...]}，把 layoutX/layoutY 写入对应节点属性"""
    try:
        nodes = payload.get('nodes') if isinstance(payload, dict) else None
        if not nodes:
            raise HTTPException(status_code=400, detail='nodes 必填')
        g = get_graph()
        for n in nodes:
            name = n.get('name')
            x = n.get('x')
            y = n.get('y')
            if not name:
                continue
            # 使用参数化 Cypher 更新属性
            try:
                g.run("MATCH (m {name:$name}) SET m.layoutX=$x, m.layoutY=$y RETURN m", name=name, x=float(x) if x is not None else None, y=float(y) if y is not None else None)
            except Exception:
                # 尝试更宽松的更新（字符串转换）
                try:
                    g.run("MATCH (m {name:$name}) SET m.layoutX=$x, m.layoutY=$y RETURN m", name=name, x=str(x), y=str(y))
                except Exception as e:
                    logging.getLogger('uvicorn').exception(f'保存布局到 Neo4j 失败 for {name}: {e}')
        return {"status": "success", "message": "布局已保存"}
    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger('uvicorn').exception(f'save_layout error: {e}')
        raise HTTPException(status_code=500, detail=f"保存失败: {e}")


@router.get("/init_words/")
def init_words(session: Session = Depends(get_session)):
    """
    知识图谱词表初始化接口
    """
    words_path = get_words_path()
    g = get_graph()
    with open(words_path, "w", encoding="utf-8") as ci:
        sql = "MATCH (n) RETURN n"
        nodes_data_all = g.run(sql).data()
        for nodes_relations in nodes_data_all:
            name = dict(nodes_relations['n'])['name']
            start_label = str(nodes_relations['n'].labels).replace(":", "")
            ci.write(str(name) + "\n")
            # 保存词表到数据库
            node = session.scalars(select(Wnode).where(Wnode.nodename == name).where(Wnode.type == start_label)).first()
            if not node:
                node = Wnode()
            node.nodename = name
            node.type = start_label
            session.add(node)
            session.commit()
            session.refresh(node)
    return {"message": "词表初始化成功！！"}


# 控制是否启用 LLM 兜底（默认启用，性能测试时可通过环境变量关闭）
KG_WENDA_USE_LLM = str(os.getenv("KG_WENDA_USE_LLM", "true")).lower() in {"1", "true", "yes"}
_jieba_loaded = False

@router.get("/wenda/{question}")
def wenda(question, session: Session = Depends(get_session)):
    """
    知识图谱问答接口 - 基于完整知识图谱的智能问答

    参数：
    - question: 问题

    返回：
    - answer: 答案
    - knowledge_count: 使用的知识三元组数量
    """
    g = get_graph()
    final_answer = ""
    
    if not question or not question.strip():
        return {"answer": "请提出一个问题。", "knowledge_count": 0}
    
    question = question.strip()
    
    # 1) 过滤寒暄/无关问题
    trivial_greetings = {"你好", "您好", "hi", "hello", "在吗", "在不", "哈喽"}
    if question.lower() in trivial_greetings:
        return {"answer": "你好！请提一个与锡冶炼相关的问题（如工艺、设备、流程等）。", "knowledge_count": 0}

    # 2) 读取知识图谱作为知识库（优化：只读取最相关的部分）
    knowledge_context = []
    has_relevant_knowledge = False  # 标记是否找到相关知识
    
    try:
        # 提取问题中的关键词
        import jieba
        keywords = list(jieba.cut(question))
        keywords = [k for k in keywords if len(k) > 1]  # 过滤单字
        
        logging.getLogger('uvicorn').info(f"提取的关键词: {keywords}")
        
        if keywords:
            # 尝试查询包含关键词的节点及其关系
            keyword_pattern = "|".join(keywords[:5])  # 最多使用5个关键词
            cypher_query = f"""
            MATCH (n)-[r]->(m) 
            WHERE n.name =~ '(?i).*({keyword_pattern}).*' 
               OR m.name =~ '(?i).*({keyword_pattern}).*'
            RETURN n, type(r) as relType, m 
            LIMIT 50
            """
            relationships = g.run(cypher_query).data()
            
            logging.getLogger('uvicorn').info(f"关键词查询结果数量: {len(relationships)}")
            
            # 如果找到了相关的知识
            if len(relationships) >= 3:  # 至少3个三元组才认为有相关知识
                has_relevant_knowledge = True
                for rel_data in relationships:
                    try:
                        start_node = rel_data.get('n')
                        end_node = rel_data.get('m')
                        relation_type = rel_data.get('relType')
                        
                        if not (start_node and end_node and relation_type):
                            continue
                        
                        start_dict = dict(start_node)
                        end_dict = dict(end_node)
                        start_name = start_dict.get('name') or start_dict.get('nodename') or '未知实体'
                        end_name = end_dict.get('name') or end_dict.get('nodename') or '未知实体'
                        
                        triple = f"({start_name}, {relation_type}, {end_name})"
                        knowledge_context.append(triple)
                    except Exception as inner_e:
                        logging.getLogger('uvicorn').warning(f"解析三元组失败: {str(inner_e)}")
                        continue
            else:
                logging.getLogger('uvicorn').info("关键词查询结果太少，判定为知识图谱中无相关信息")
        else:
            logging.getLogger('uvicorn').info("未提取到有效关键词")
                
    except Exception as e:
        logging.getLogger('uvicorn').error(f"读取知识图谱失败: {str(e)}")
        return {"answer": "抱歉，读取知识图谱时出现错误，请稍后重试。", "knowledge_count": 0}
    
    # 如果知识图谱中没有相关信息，直接使用Ollama回答
    if not has_relevant_knowledge or not knowledge_context:
        logging.getLogger('uvicorn').info("知识图谱中未找到相关信息，使用Ollama直接回答")
        
        if KG_WENDA_USE_LLM:
            try:
                base_url_candidates, model_name, build_client, derive_root, probe_ollama = _build_llm_clients()
                
                # 使用Ollama直接回答（不依赖知识图谱）
                direct_system_prompt = """你是一个AI助手。请直接回答用户的问题。

要求：
1) 回答要准确、简洁
2) 如果不确定，请明确说明
3) 回答不超过200字
4) 在回答开头先说："这个问题不在图谱回答范围，但可以使用大模型进行回答。"然后再给出答案"""

                direct_messages = [
                    {"role": "system", "content": direct_system_prompt},
                    {"role": "user", "content": question}
                ]
                
                # 尝试调用 LLM
                healthy_urls = []
                for u in base_url_candidates:
                    root = derive_root(u)
                    if probe_ollama(root):
                        healthy_urls.append(u)
                
                final_answer = ""
                for url in healthy_urls:
                    try:
                        client = build_client(url)
                        resp = client.chat.completions.create(
                            model=model_name,
                            messages=direct_messages,
                            stream=False,
                            temperature=0.3,
                            timeout=600,
                        )
                        content = resp.choices[0].message.content or ""
                        if content:
                            # 清理LLM响应中的特殊标记
                            content = content.strip()
                            content = content.replace('<|endoftext|>', '')
                            content = content.replace('<|im_start|>', '')
                            content = content.replace('<|im_end|>', '')
                            
                            # 移除换行符中的多余空格
                            content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
                            
                            final_answer = content
                            logging.getLogger("uvicorn").info("使用Ollama直接回答成功")
                            break
                    except Exception as e:
                        logging.getLogger("uvicorn").warning(f"Ollama直接回答失败({url}): {e}")
                
                if not final_answer:
                    final_answer = "抱歉，我暂时无法回答这个问题。"
                
                # 保存问答记录
                if final_answer:
                    try:
                        qa_list = session.scalars(select(Qa).where(Qa.question == question)).all()
                        for qa_item in qa_list:
                            session.delete(qa_item)
                        
                        qa = Qa()
                        qa.question = question
                        qa.answer = final_answer
                        session.add(qa)
                        session.commit()
                        session.refresh(qa)
                    except Exception as e:
                        logging.getLogger('uvicorn').warning(f"保存问答记录失败: {e}")
                        try:
                            session.rollback()
                        except Exception:
                            pass
                
                logging.getLogger('uvicorn').info(f"问题: {question}")
                logging.getLogger('uvicorn').info(f"答案: {final_answer}")
                logging.getLogger('uvicorn').info(f"答案长度: {len(final_answer)} 字符")
                logging.getLogger('uvicorn').info(f"知识三元组数量: 0 (直接回答)")
                
                try:
                    session.close()
                except Exception:
                    pass
                
                result = {
                    "answer": final_answer,
                    "knowledge_count": 0
                }
                
                logging.getLogger('uvicorn').info(f"准备返回结果: {result}")
                
                return result
                
            except Exception as e:
                logging.getLogger("uvicorn").error(f"Ollama直接回答失败: {e}")
                return {"answer": "抱歉，处理问题时出现错误，请稍后重试。", "knowledge_count": 0}
        else:
            return {"answer": "LLM 服务未启用，无法回答问题。", "knowledge_count": 0}
    
    # 如果找到了相关知识，继续使用知识图谱回答
    MAX_TRIPLES = 50
    knowledge_context = knowledge_context[:MAX_TRIPLES]
    knowledge_str = "\n".join(knowledge_context)
    
    logging.getLogger('uvicorn').info(f"加载知识图谱三元组数量: {len(knowledge_context)}")
    
    # 3) 使用 LLM 基于完整知识图谱回答问题
    if KG_WENDA_USE_LLM:
        try:
            base_url_candidates, model_name, build_client, derive_root, probe_ollama = _build_llm_clients()
            
            # 基于知识图谱回答
            system_prompt = """你是一个知识图谱问答助手。我会给你一些知识图谱三元组，请你基于这些信息回答问题。

要求：
1) 只使用我提供的知识图谱信息回答
2) 不要编造不存在的知识
3) 直接给出答案，不要重复问题
4) 不要输出"用户问题："、"知识图谱："等提示词
5) 回答简洁明了，不超过150字"""

            # 将知识图谱和问题组合成用户消息
            user_message = f"""知识图谱三元组：
{knowledge_str}

问题：{question}

请直接回答问题，不要重复问题和知识图谱内容。"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # 尝试调用 LLM
            healthy_urls = []
            last_err = None
            for u in base_url_candidates:
                root = derive_root(u)
                if probe_ollama(root):
                    healthy_urls.append(u)
                else:
                    logging.getLogger("uvicorn").warning(f"LLM 健康检查失败: {root}")
            
            final_answer = ""
            for url in healthy_urls:
                try:
                    client = build_client(url)
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False,
                        temperature=0.0,
                        timeout=600,
                    )
                    content = resp.choices[0].message.content or ""
                    if content:
                        # 清理LLM响应中的特殊标记和多余内容
                        content = content.strip()
                        
                        # 移除特殊标记
                        content = content.replace('<|endoftext|>', '')
                        content = content.replace('<|im_start|>', '')
                        content = content.replace('<|im_end|>', '')
                        
                        # 移除可能出现的提示词重复
                        if '用户问题：' in content:
                            # 只保留第一个答案部分
                            content = content.split('用户问题：')[0].strip()
                        if '知识图谱：' in content:
                            content = content.split('知识图谱：')[0].strip()
                        
                        # 移除换行符中的多余空格
                        content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
                        
                        final_answer = content
                        break
                except Exception as e:
                    last_err = e
                    logging.getLogger("uvicorn").warning(f"LLM 调用失败({url}): {e}")
            
            if not final_answer and last_err:
                logging.getLogger("uvicorn").error(f"所有 LLM 调用均失败: {last_err}")
                final_answer = "抱歉，AI 服务暂时不可用，请稍后重试。"
                
        except Exception as e:
            logging.getLogger("uvicorn").error(f"LLM 问答失败: {e}")
            final_answer = "抱歉，处理问题时出现错误，请稍后重试。"
    else:
        final_answer = "LLM 服务未启用，无法回答问题。"
    
    # 4) 保存问答记录到数据库
    if final_answer:
        try:
            # 删除旧的相同问题记录
            qa_list = session.scalars(select(Qa).where(Qa.question == question)).all()
            for qa_item in qa_list:
                session.delete(qa_item)
            
            # 保存新记录
            qa = Qa()
            qa.question = question
            qa.answer = final_answer
            session.add(qa)
            session.commit()
            session.refresh(qa)
        except Exception as e:
            logging.getLogger('uvicorn').warning(f"保存问答记录失败: {e}")
            # 回滚失败的事务，但不影响返回结果
            try:
                session.rollback()
            except Exception:
                pass
    
    logging.getLogger('uvicorn').info(f"问题: {question}")
    logging.getLogger('uvicorn').info(f"答案: {final_answer}")
    logging.getLogger('uvicorn').info(f"答案长度: {len(final_answer)} 字符")
    logging.getLogger('uvicorn').info(f"知识三元组数量: {len(knowledge_context)}")
    
    # 确保返回前关闭数据库会话
    try:
        session.close()
    except Exception:
        pass
    
    result = {
        "answer": final_answer,
        "knowledge_count": len(knowledge_context)
    }
    
    logging.getLogger('uvicorn').info(f"准备返回结果: {result}")
    
    return result

# ==================== Neo4j 管理接口 ====================

# 获取统计信息
@router.get("/neo4j/stats", response_model=Neo4jStats)
def get_neo4j_stats():
    """获取Neo4j数据库统计信息"""
    try:
        g = get_graph()

        # 获取节点数量
        entity_count = g.run("MATCH (n) RETURN count(n) as count").data()[0]['count']

        # 获取关系数量
        relation_count = g.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']

        # 计算今日更新（获取今天更新的节点数量）
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        today_update_query = "MATCH (n) WHERE n.updatedAt = $today RETURN count(n) as count"
        try:
            today_update = g.run(today_update_query, today=today).data()[0]['count']
        except Exception:
            today_update = 0

        # 计算数据库大小（基于节点和关系数量的估算）
        # 假设每个节点平均占用 1KB，每个关系平均占用 0.5KB
        estimated_size_kb = (entity_count * 1.0) + (relation_count * 0.5)
        data_size = round(estimated_size_kb / 1024, 2)  # 转换为 MB
        
        # 如果没有数据，显示为 0
        if entity_count == 0 and relation_count == 0:
            data_size = 0.0

        return Neo4jStats(
            entityCount=entity_count,
            relationCount=relation_count,
            todayUpdate=today_update,
            dataSize=data_size
        )

    except Exception as e:
        logging.getLogger('uvicorn').exception(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

# 获取节点列表
@router.get("/neo4j/nodes")
def get_nodes(page: int = 1, size: int = 10):
    """获取节点列表"""
    try:
        g = get_graph()
        page = max(1, int(page or 1))
        # 允许更大的限制以支持"查询所有"功能，最大50000条
        size = max(1, min(50000, int(size or 10)))
        skip = (page - 1) * size
        count_q = "MATCH (n) RETURN count(n) as total"
        total = g.run(count_q).evaluate() or 0
        data_q = """
        MATCH (n)
        RETURN n
        ORDER BY coalesce(n.sortIndex, 999999) ASC, id(n) ASC
        SKIP $skip LIMIT $limit
        """
        rows = g.run(data_q, skip=skip, limit=size).data()
        paginated_nodes = []
        
        # 动态分配显示 ID：从 (page-1)*size + 1 开始
        display_id = skip + 1
        
        for row in rows:
            node = row.get('n')
            if node is None:
                continue
            node_data = dict(node)
            try:
                # 使用动态计算的显示 ID，而不是存储的 customId
                node_data['id'] = display_id
                display_id += 1
            except Exception:
                node_data['id'] = None
            try:
                raw_label = list(node.labels)[0] if getattr(node, 'labels', None) else node_data.get('type') or node_data.get('category') or 'Unknown'
            except Exception:
                raw_label = node_data.get('type') or node_data.get('category') or 'Unknown'
            node_data['category'] = CATEGORY_MAP.get(str(raw_label), str(raw_label))
            
            # 确保 updatedAt 字段存在且格式正确（只显示日期）
            updated_at = node_data.get('updatedAt') or node_data.get('updated_at') or node_data.get('createdAt') or node_data.get('created_at')
            if updated_at:
                # 如果是字符串，提取日期部分
                if isinstance(updated_at, str):
                    # 提取日期部分（YYYY-MM-DD）
                    try:
                        # 尝试解析并格式化为只显示日期
                        if ' ' in updated_at:
                            node_data['updatedAt'] = updated_at.split(' ')[0]
                        else:
                            node_data['updatedAt'] = updated_at[:10] if len(updated_at) >= 10 else updated_at
                    except Exception:
                        node_data['updatedAt'] = datetime.now().strftime("%Y-%m-%d")
                else:
                    try:
                        node_data['updatedAt'] = str(updated_at)[:10]
                    except Exception:
                        node_data['updatedAt'] = datetime.now().strftime("%Y-%m-%d")
            else:
                node_data['updatedAt'] = datetime.now().strftime("%Y-%m-%d")
            
            paginated_nodes.append(node_data)
        return {"nodes": paginated_nodes, "total": int(total), "page": page, "size": size}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取节点列表失败: {str(e)}")

# 获取关系列表
@router.get("/neo4j/relations")
def get_relations(page: int = 1, size: int = 10):
    """获取关系列表"""
    try:
        g = get_graph()
        page = max(1, int(page or 1))
        # 允许更大的限制以支持"查询所有"功能，最大50000条
        size = max(1, min(50000, int(size or 10)))
        skip = (page - 1) * size
        count_q = "MATCH ()-[r]->() RETURN count(r) as total"
        total = g.run(count_q).evaluate() or 0
        data_q = "MATCH ()-[r]->() RETURN r ORDER BY id(r) SKIP $skip LIMIT $limit"
        rows = g.run(data_q, skip=skip, limit=size).data()
        paginated_relations = []
        for row in rows:
            relation = row.get('r')
            if relation is None:
                continue
            try:
                start_name = relation.start_node.get('name') or str(relation.start_node)
            except Exception:
                start_name = str(getattr(relation, 'start_node', ''))
            try:
                end_name = relation.end_node.get('name') or str(relation.end_node)
            except Exception:
                end_name = str(getattr(relation, 'end_node', ''))
            relation_data = {
                "id": getattr(relation, 'identity', None),
                "startNode": start_name,
                "endNode": end_name,
                "type": type(relation).__name__
            }
            paginated_relations.append(relation_data)
        return {"relations": paginated_relations, "total": int(total), "page": page, "size": size}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取关系列表失败: {str(e)}")

# 添加节点
@router.post("/neo4j/add_node")
def add_node(node_data: NodeData):
    """添加节点"""
    try:
        g = get_graph()

        # 查询当前最大的 sortIndex
        max_sort_result = g.run("""
            MATCH (n)
            WHERE n.sortIndex IS NOT NULL
            RETURN max(n.sortIndex) AS max_index
        """).evaluate()

        # 确定下一个索引值
        # 如果数据库为空，从 1 开始
        node_count = g.run("MATCH (n) RETURN count(n) AS total").evaluate() or 0
        
        if node_count == 0:
            next_sort_index = 1
        else:
            next_sort_index = (int(max_sort_result) + 1) if max_sort_result is not None else 1

        # 创建节点
        node = Node(node_data.category)
        node['name'] = node_data.name

        # 合并属性
        for k, v in (node_data.properties or {}).items():
            node[k] = v

        # 写入排序字段
        node['sortIndex'] = next_sort_index

        # 时间字段（只保存日期）
        node['createdAt'] = datetime.now().strftime("%Y-%m-%d")
        node['updatedAt'] = datetime.now().strftime("%Y-%m-%d")

        g.create(node)

        return {
            "message": "节点创建成功",
            "sortIndex": next_sort_index
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加节点失败: {str(e)}")

# 删除节点
@router.delete("/neo4j/delete_node")
def delete_node(label: str, name: str):
    """删除节点"""
    try:
        g = get_graph()

        # 使用 coalesce 兼容 name/nodename；不直接拼接 label，防止中文/特殊字符引起语法错误
        query = (
            "MATCH (n) "
            "WHERE coalesce(n.name, n.nodename) = $name "
            "AND ($label = '' OR $label IN labels(n)) "
            "DETACH DELETE n"
        )
        g.run(query, name=name, label=label or '')

        return {"message": "节点删除成功"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除节点失败: {str(e)}")

# 深度删除：删除整个Neo4j数据库中的所有数据
@router.delete("/neo4j/delete_node_full")
def delete_node_full(label: str = "", name: str = ""):
    """删除整个Neo4j数据库中的所有节点和关系。
    
    注意：该操作将清空整个数据库，请谨慎使用！
    参数 label 和 name 仅为保持接口兼容性，实际不使用。
    """
    try:
        g = get_graph()

        # 删除所有节点和关系
        # DETACH DELETE 会先删除关系再删除节点
        query = "MATCH (n) DETACH DELETE n"
        g.run(query)

        # 验证删除结果
        count_query = "MATCH (n) RETURN count(n) as total"
        remaining = g.run(count_query).evaluate() or 0
        
        if remaining > 0:
            raise Exception(f"删除后仍有 {remaining} 个节点残留")

        return {"message": "数据库中所有节点和关系已成功删除"}
    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger('uvicorn').exception(f"删除全部数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除全部失败: {str(e)}")

# 更新节点
@router.put("/neo4j/update_node")
def update_node(label: str, name: str, properties: Dict[str, Any]):
    """更新节点（完全替换属性，不在properties中的非系统属性将被删除）"""
    try:
        g = get_graph()

        # 参数检查
        if not label or not name:
            raise HTTPException(status_code=400, detail="label 与 name 为必填")

        # 先检查是否存在（兼容 name/nodename）
        query_check = f"MATCH (n:{label}) WHERE coalesce(n.name, n.nodename) = $name RETURN n LIMIT 1"
        result = g.run(query_check, name=name).data()
        if not result:
            raise HTTPException(status_code=404, detail="不存在该实体")
        
        # 获取现有节点的所有属性
        existing_node = result[0]['n']
        existing_props = dict(existing_node)
        
        # 定义系统属性，这些属性不能被删除
        system_keys = {'id', 'name', 'nodename', 'type', 'category', 'updatedAt', 'createdAt', 'created_at', 'mysql_id'}
        
        # 清洗新属性键，避免反引号破坏语法
        safe_props: Dict[str, Any] = {}
        for k, v in (properties or {}).items():
            if not k:
                continue
            key = str(k).replace("`", "").strip()
            if not key or key in system_keys:
                continue
            safe_props[key] = v

        # 找出需要删除的属性（存在于旧节点但不在新属性中的非系统属性）
        props_to_delete = [k for k in existing_props.keys() if k not in system_keys and k not in safe_props]
        
        # 构建Cypher语句
        query_parts = []
        params = {"name": name}
        
        # 1. 删除不再需要的属性
        if props_to_delete:
            for prop in props_to_delete:
                query_parts.append(f"REMOVE n.`{prop}`")
        
        # 2. 设置新的属性
        set_items = []
        for k, v in safe_props.items():
            param_key = k.replace(".", "_")  # 避免参数名中的点号
            set_items.append(f"n.`{k}`=${param_key}")
            params[param_key] = v
        
        # 3. 总是更新更新时间（只保存日期）
        updated_at = datetime.now().strftime("%Y-%m-%d")
        set_items.append("n.updatedAt=$updatedAt")
        params["updatedAt"] = updated_at
        
        # 合并所有操作
        if set_items:
            query_parts.append("SET " + ", ".join(set_items))
        
        if not query_parts:
            # 如果没有任何操作，至少更新时间
            query_parts.append("SET n.updatedAt=$updatedAt")
        
        query_update = (
            f"MATCH (n:{label}) WHERE coalesce(n.name, n.nodename) = $name "
            + " ".join(query_parts) + 
            " RETURN n"
        )
        
        g.run(query_update, **params)

        return {"message": "节点更新成功"}

    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger('uvicorn').exception(f"更新节点失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新节点失败: {str(e)}")

# 查询节点
@router.post("/neo4j/query_node")
def query_node(query_data: Dict[str, Any]):
    """查询节点，支持精确字段匹配和模糊查询(q 字段)

    请求体示例：
      { "label": "Node", "properties": {"name": "xxx"} }
    或：
      { "label": "Node", "q": "关键字" }
    """
    try:
        g = get_graph()

        label = query_data.get('label', '')
        properties = query_data.get('properties', {}) or {}
        q = query_data.get('q')

        if q:
            # 模糊匹配 name 或 nodename 字段，忽略大小写
            where_clause = "(toLower(n.name) CONTAINS toLower($q) OR toLower(coalesce(n.nodename,'')) CONTAINS toLower($q))"
            params = {'q': q}
        elif properties:
            where_clause = " AND ".join([f"n.{k}=${k}" for k in properties])
            params = properties
        else:
            where_clause = "1=1"
            params = {}

        # 如果未指定 label 或 label 是通配（如 Node/空字符串），则不在 MATCH 中指定标签
        if not label or label.lower() == 'node':
            query = f"MATCH (n) WHERE {where_clause} RETURN n"
        else:
            query = f"MATCH (n:{label}) WHERE {where_clause} RETURN n"

        result = g.run(query, **params)
        nodes: List[Dict[str, Any]] = []
        for record in result:
            try:
                n = record["n"]
                d = dict(n)
                # 提取 Neo4j 标签作为类别；若无则回退到节点属性中的 type/category
                try:
                    raw_label = list(n.labels)[0] if getattr(n, 'labels', None) else d.get('type') or d.get('category') or ''
                except Exception:
                    raw_label = d.get('type') or d.get('category') or ''
                display_label = CATEGORY_MAP.get(str(raw_label), str(raw_label)) if raw_label else ''
                d['category'] = display_label
                # 附带 labels/id 以便前端备用
                try:
                    d['labels'] = list(n.labels) if getattr(n, 'labels', None) else []
                except Exception:
                    d['labels'] = []
                try:
                    d['id'] = getattr(n, 'identity', None)
                except Exception:
                    pass
                nodes.append(d)
            except Exception:
                # 忽略异常条目，继续返回其他节点
                continue

        return {"nodes": nodes}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询节点失败: {str(e)}")

# 添加关系
@router.post("/neo4j/add_relationship")
def add_relationship(relation_data: RelationshipData):
    """添加关系"""
    try:
        g = get_graph()

        start_name = (relation_data.startNode or "").strip()
        end_name = (relation_data.endNode or "").strip()
        rel_type = (relation_data.type or "").strip()
        if not start_name or not end_name or not rel_type:
            raise HTTPException(status_code=400, detail="startNode、endNode、type 为必填")

        # 轻量清洗，去除反引号，防止拼接破坏语法
        def sanitize_label(label: Optional[str]) -> Optional[str]:
            if not label:
                return None
            v = str(label).strip()
            if not v:
                return None
            return v.replace("`", "")

        s_label = sanitize_label(relation_data.startLabel)
        e_label = sanitize_label(relation_data.endLabel)
        rel_quota = rel_type.replace("`", "")

        try:
            # 统一用 coalesce(a.name, a.nodename) 进行匹配，兼容不同属性名
            if s_label and e_label:
                query = (
                    f"MATCH (a:`{s_label}`), (b:`{e_label}`) "
                    f"WHERE coalesce(a.name, a.nodename) = $startName AND coalesce(b.name, b.nodename) = $endName "
                    f"MERGE (a)-[r:`{rel_quota}`]->(b) RETURN r"
                )
                g.run(query, startName=start_name, endName=end_name)
            else:
                query = (
                    f"MATCH (a), (b) "
                    f"WHERE coalesce(a.name, a.nodename) = $startName AND coalesce(b.name, b.nodename) = $endName "
                    f"MERGE (a)-[r:`{rel_quota}`]->(b) RETURN r"
                )
                g.run(query, startName=start_name, endName=end_name)
        except Exception as inner:
            # 回退到无标签匹配，避免因标签不一致导致的失败
            query = (
                f"MATCH (a), (b) "
                f"WHERE coalesce(a.name, a.nodename) = $startName AND coalesce(b.name, b.nodename) = $endName "
                f"MERGE (a)-[r:`{rel_quota}`]->(b) RETURN r"
            )
            g.run(query, startName=start_name, endName=end_name)

        return {"message": "关系创建成功"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加关系失败: {str(e)}")

# 一次性创建新节点并与原节点建立关系
@router.post("/neo4j/create_node_with_relation")
def create_node_with_relation(payload: CreateNodeWithRelationRequest):
    """创建新节点（若不存在），同时确保原节点存在，并建立二者之间的指定关系。

    前端应传入：newCategory/newName、origCategory/origName、relationType，可选属性 newProperties、origProperties。
    若原节点不存在，将按 origCategory 与 origProperties 创建。
    """
    try:
        g = get_graph()

        # 获取当前最大的 sortIndex
        node_count = g.run("MATCH (n) RETURN count(n) AS total").evaluate() or 0
        
        if node_count == 0:
            next_sort_index = 1
        else:
            max_sort_result = g.run("MATCH (n) WHERE n.sortIndex IS NOT NULL RETURN max(n.sortIndex) AS max_index").evaluate()
            next_sort_index = (int(max_sort_result) + 1) if max_sort_result is not None else 1

        # 1) 确保原节点存在（优先根据 coalesce(name, nodename) 匹配）
        match_orig = (
            f"MATCH (o:{payload.origCategory}) "
            f"WHERE coalesce(o.name, o.nodename) = $origName RETURN o"
        )
        res_orig = g.run(match_orig, origName=payload.origName).data()
        if not res_orig:
            # 创建原节点（统一写入 name，便于后续一致匹配）
            orig_props = dict(payload.origProperties or {})
            orig_node = Node(payload.origCategory, name=payload.origName, **orig_props)
            orig_node['updatedAt'] = datetime.now().strftime("%Y-%m-%d")
            orig_node['createdAt'] = datetime.now().strftime("%Y-%m-%d")
            orig_node['sortIndex'] = next_sort_index
            g.create(orig_node)
            next_sort_index += 1

        # 2) 确保新节点存在（一般为创建），同样用 coalesce 匹配
        match_new = (
            f"MATCH (n:{payload.newCategory}) "
            f"WHERE coalesce(n.name, n.nodename) = $newName RETURN n"
        )
        res_new = g.run(match_new, newName=payload.newName).data()
        if not res_new:
            new_props = dict(payload.newProperties or {})
            new_node = Node(payload.newCategory, name=payload.newName, **new_props)
            new_node['updatedAt'] = datetime.now().strftime("%Y-%m-%d")
            new_node['createdAt'] = datetime.now().strftime("%Y-%m-%d")
            new_node['sortIndex'] = next_sort_index
            g.create(new_node)
            next_sort_index += 1

        # 3) 建立关系（使用 MERGE 防重复），支持方向选择
        direction = (payload.direction or "new_to_orig").strip()
        if direction not in ("new_to_orig", "orig_to_new"):
            direction = "new_to_orig"
        # 为防止 name/nodename 差异，统一用 coalesce 匹配再 MERGE 关系
        rel_type = (payload.relationType or "").replace("`", "")
        if direction == "new_to_orig":
            rel_cypher = (
                f"MATCH (o:{payload.origCategory}), (n:{payload.newCategory}) "
                f"WHERE coalesce(o.name, o.nodename) = $origName AND coalesce(n.name, n.nodename) = $newName "
                f"MERGE (n)-[r:`{rel_type}`]->(o) RETURN r"
            )
        else:
            rel_cypher = (
                f"MATCH (o:{payload.origCategory}), (n:{payload.newCategory}) "
                f"WHERE coalesce(o.name, o.nodename) = $origName AND coalesce(n.name, n.nodename) = $newName "
                f"MERGE (o)-[r:`{rel_type}`]->(n) RETURN r"
            )

        g.run(rel_cypher, origName=payload.origName, newName=payload.newName)

        return {"message": "节点与关系创建成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建节点并建立关系失败: {str(e)}")

# 删除关系
@router.delete("/neo4j/delete_relationship")
def delete_relationship(start_node: str, end_node: str, relation_type: str):
    """删除关系"""
    try:
        g = get_graph()

        query = """
        MATCH (a {name: $startName})-[r:`""" + relation_type + """`]->(b {name: $endName})
        DELETE r
        """

        g.run(query, startName=start_node, endName=end_node)

        return {"message": "关系删除成功"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除关系失败: {str(e)}")


# 获取关键流程图谱数据（按实体类别查询）
@router.get("/key-process/{process_id}")
def get_key_process_data(process_id: str):
    """
    获取关键流程的知识图谱数据（按实体类别查询）
    
    参数:
    - process_id: 流程ID，对应实体类别名称
      - rotary-kiln: 回转窑焙烧
      - fluidized-bed: 沸腾炉系统
      - smoke-furnace: 烟化炉系统
      - refining: 精炼系统
      - electric-furnace: 电炉
      - top-blow: 顶吹炉熔炉
      - parameters: 参数
    
    返回:
    - nodes: 节点列表
    - links: 关系列表
    """
    try:
        g = get_graph()
        
        # 流程ID到实体类别的映射
        process_to_category_map = {
            'rotary-kiln': '回转窑焙烧',
            'fluidized-bed': '沸腾炉系统',
            'smoke-furnace': '烟化炉系统',
            'refining': '精炼系统',
            'electric-furnace': '电炉',
            'top-blow': '顶吹炉熔炉',
            'parameters': '参数'
        }
        
        label = process_to_category_map.get(process_id, '')
        
        if not label:
            logging.getLogger('uvicorn').warning(f'未知的流程ID: {process_id}')
            return generate_sample_process_data(process_id)
        
        logging.getLogger('uvicorn').info(f'查询流程类别: {label}')
        
        # 1) 查询该类别的所有节点
        node_query = """
        MATCH (n)
        WHERE $label IN labels(n) OR n.type = $label OR n.category = $label
        RETURN n
        LIMIT 500
        """
        
        try:
            node_results = g.run(node_query, label=label).data()
            logging.getLogger('uvicorn').info(f'找到 {len(node_results)} 个节点（类别: {label}）')
        except Exception as e:
            logging.getLogger('uvicorn').error(f'查询节点失败: {e}')
            node_results = []
        
        if not node_results:
            logging.getLogger('uvicorn').warning(f'未找到类别 {label} 的节点，返回示例数据')
            return generate_sample_process_data(process_id)
        
        # 提取节点名称
        node_names = []
        nodes_dict = {}
        
        # 颜色映射
        category_colors = {
            '回转窑焙烧': '#1890ff',
            '沸腾炉系统': '#52c41a',
            '烟化炉系统': '#faad14',
            '精炼系统': '#722ed1',
            '电炉': '#eb2f96',
            '顶吹炉熔炉': '#13c2c2',
            '参数': '#fa8c16',
            '设备': '#1890ff',
            '工艺': '#52c41a',
            '物料': '#faad14',
            '产品': '#eb2f96',
            '原料': '#13c2c2',
            '温度': '#fa8c16',
            '压力': '#2f54eb',
            '实体': '#1677ff',
        }
        
        for item in node_results:
            try:
                n_node = item.get('n')
                if not n_node:
                    continue
                
                n_dict = dict(n_node)
                n_name = n_dict.get('name') or n_dict.get('nodename') or ''
                
                if not n_name:
                    continue
                
                node_names.append(n_name)
                
                # 获取节点类别
                n_category = label  # 默认使用查询的类别
                try:
                    if hasattr(n_node, 'labels') and n_node.labels:
                        n_category = list(n_node.labels)[0]
                except:
                    pass
                
                if not n_category:
                    n_category = n_dict.get('type') or n_dict.get('category') or label
                
                n_category_zh = CATEGORY_MAP.get(n_category, n_category)
                
                if n_name not in nodes_dict:
                    nodes_dict[n_name] = {
                        'name': n_name,
                        'category': n_category_zh,
                        'type': n_category_zh,
                        'color': category_colors.get(n_category_zh, category_colors.get(label, '#1677ff'))
                    }
                    
            except Exception as e:
                logging.getLogger('uvicorn').warning(f'处理节点时出错: {e}')
                continue
        
        logging.getLogger('uvicorn').info(f'处理后节点数量: {len(nodes_dict)}')
        
        # 2) 查询这些节点之间的所有关系（出边和入边）
        if not node_names:
            logging.getLogger('uvicorn').warning('没有有效的节点名称')
            return generate_sample_process_data(process_id)
        
        links = []
        link_cache = set()
        
        # 分批查询关系（避免参数过多）
        batch_size = 50
        for i in range(0, len(node_names), batch_size):
            batch_names = node_names[i:i+batch_size]
            
            # 查询出边
            out_rel_query = """
            MATCH (n)-[r]->(m)
            WHERE n.name IN $names OR n.nodename IN $names
            RETURN n, type(r) as relType, m
            LIMIT 1000
            """
            
            # 查询入边
            in_rel_query = """
            MATCH (m)-[r]->(n)
            WHERE n.name IN $names OR n.nodename IN $names
            RETURN m as n, type(r) as relType, n as m
            LIMIT 1000
            """
            
            try:
                out_results = g.run(out_rel_query, names=batch_names).data()
                in_results = g.run(in_rel_query, names=batch_names).data()
                all_rel_results = out_results + in_results
                
                logging.getLogger('uvicorn').info(f'批次 {i//batch_size + 1}: 找到 {len(all_rel_results)} 个关系')
                
                for rel_item in all_rel_results:
                    try:
                        n_node = rel_item.get('n')
                        m_node = rel_item.get('m')
                        rel_type = rel_item.get('relType')
                        
                        if not (n_node and m_node and rel_type):
                            continue
                        
                        n_dict = dict(n_node)
                        m_dict = dict(m_node)
                        
                        n_name = n_dict.get('name') or n_dict.get('nodename') or ''
                        m_name = m_dict.get('name') or m_dict.get('nodename') or ''
                        
                        if not (n_name and m_name):
                            continue
                        
                        # 确保两个节点都添加到nodes_dict中
                        for node, node_obj in [(n_name, n_node), (m_name, m_node)]:
                            if node not in nodes_dict:
                                try:
                                    node_dict = dict(node_obj)
                                    node_category = label
                                    try:
                                        if hasattr(node_obj, 'labels') and node_obj.labels:
                                            node_category = list(node_obj.labels)[0]
                                    except:
                                        pass
                                    
                                    if not node_category:
                                        node_category = node_dict.get('type') or node_dict.get('category') or '实体'
                                    
                                    node_category_zh = CATEGORY_MAP.get(node_category, node_category)
                                    
                                    nodes_dict[node] = {
                                        'name': node,
                                        'category': node_category_zh,
                                        'type': node_category_zh,
                                        'color': category_colors.get(node_category_zh, '#1677ff')
                                    }
                                except:
                                    pass
                        
                        # 添加关系（去重）
                        link_key = f"{n_name}|{m_name}|{rel_type}"
                        if link_key not in link_cache:
                            links.append({
                                'source': n_name,
                                'target': m_name,
                                'name': str(rel_type),
                                'type': str(rel_type),
                                'relation': str(rel_type)
                            })
                            link_cache.add(link_key)
                            
                    except Exception as e:
                        logging.getLogger('uvicorn').warning(f'处理关系时出错: {e}')
                        continue
                        
            except Exception as e:
                logging.getLogger('uvicorn').error(f'查询关系失败: {e}')
                continue
        
        nodes = list(nodes_dict.values())
        
        logging.getLogger('uvicorn').info(f'最终结果: {len(nodes)} 个节点, {len(links)} 个关系')
        
        # 如果没有找到任何关系，但有节点，仍然返回节点
        if not nodes:
            logging.getLogger('uvicorn').warning('未找到任何节点，返回示例数据')
            return generate_sample_process_data(process_id)
        
        return {
            'nodes': nodes,
            'links': links,
            'datas': nodes  # 兼容前端的 datas 字段
        }
        
    except Exception as e:
        logging.getLogger('uvicorn').error(f'获取关键流程图谱数据失败: {e}', exc_info=True)
        # 返回示例数据作为兜底
        return generate_sample_process_data(process_id)


def generate_sample_process_data(process_id: str):
    """生成示例流程数据（当数据库中没有数据时使用）"""
    process_data_map = {
        'rotary-kiln': {
            'nodes': [
                {'name': '回转窑', 'category': '设备', 'color': '#1890ff'},
                {'name': '锡精矿', 'category': '原料', 'color': '#13c2c2'},
                {'name': '焙烧', 'category': '工艺', 'color': '#52c41a'},
                {'name': '氧化锡', 'category': '产品', 'color': '#eb2f96'},
                {'name': '温度控制', 'category': '参数', 'color': '#722ed1'},
            ],
            'links': [
                {'source': '锡精矿', 'target': '回转窑', 'name': '投入'},
                {'source': '回转窑', 'target': '焙烧', 'name': '进行'},
                {'source': '焙烧', 'target': '氧化锡', 'name': '产出'},
                {'source': '温度控制', 'target': '焙烧', 'name': '调节'},
            ]
        },
        'fluidized-bed': {
            'nodes': [
                {'name': '沸腾炉', 'category': '设备', 'color': '#1890ff'},
                {'name': '气体分布', 'category': '工艺', 'color': '#52c41a'},
                {'name': '物料悬浮', 'category': '工艺', 'color': '#52c41a'},
                {'name': '热交换', 'category': '工艺', 'color': '#52c41a'},
                {'name': '压力系统', 'category': '参数', 'color': '#722ed1'},
            ],
            'links': [
                {'source': '沸腾炉', 'target': '气体分布', 'name': '控制'},
                {'source': '气体分布', 'target': '物料悬浮', 'name': '形成'},
                {'source': '物料悬浮', 'target': '热交换', 'name': '进行'},
                {'source': '压力系统', 'target': '气体分布', 'name': '调节'},
            ]
        },
        'smoke-furnace': {
            'nodes': [
                {'name': '烟化炉', 'category': '设备', 'color': '#1890ff'},
                {'name': '炉渣', 'category': '原料', 'color': '#13c2c2'},
                {'name': '挥发', 'category': '工艺', 'color': '#52c41a'},
                {'name': '烟尘', 'category': '产品', 'color': '#eb2f96'},
                {'name': '温度', 'category': '参数', 'color': '#722ed1'},
            ],
            'links': [
                {'source': '炉渣', 'target': '烟化炉', 'name': '投入'},
                {'source': '烟化炉', 'target': '挥发', 'name': '进行'},
                {'source': '挥发', 'target': '烟尘', 'name': '产生'},
                {'source': '温度', 'target': '挥发', 'name': '控制'},
            ]
        },
        'refining': {
            'nodes': [
                {'name': '精炼炉', 'category': '设备', 'color': '#1890ff'},
                {'name': '粗锡', 'category': '原料', 'color': '#13c2c2'},
                {'name': '除杂', 'category': '工艺', 'color': '#52c41a'},
                {'name': '精锡', 'category': '产品', 'color': '#eb2f96'},
                {'name': '温度', 'category': '参数', 'color': '#722ed1'},
                {'name': '时间', 'category': '参数', 'color': '#722ed1'},
            ],
            'links': [
                {'source': '粗锡', 'target': '精炼炉', 'name': '投入'},
                {'source': '精炼炉', 'target': '除杂', 'name': '进行'},
                {'source': '除杂', 'target': '精锡', 'name': '产出'},
                {'source': '温度', 'target': '除杂', 'name': '控制'},
                {'source': '时间', 'target': '除杂', 'name': '控制'},
            ]
        },
        'electric-furnace': {
            'nodes': [
                {'name': '电炉', 'category': '设备', 'color': '#1890ff'},
                {'name': '氧化锡', 'category': '原料', 'color': '#13c2c2'},
                {'name': '还原熔炼', 'category': '工艺', 'color': '#52c41a'},
                {'name': '粗锡', 'category': '产品', 'color': '#eb2f96'},
                {'name': '电极', 'category': '设备', 'color': '#1890ff'},
                {'name': '功率', 'category': '参数', 'color': '#722ed1'},
            ],
            'links': [
                {'source': '氧化锡', 'target': '电炉', 'name': '投入'},
                {'source': '电炉', 'target': '还原熔炼', 'name': '进行'},
                {'source': '还原熔炼', 'target': '粗锡', 'name': '产出'},
                {'source': '电极', 'target': '电炉', 'name': '组成'},
                {'source': '功率', 'target': '还原熔炼', 'name': '控制'},
            ]
        },
        'top-blow': {
            'nodes': [
                {'name': '顶吹炉', 'category': '设备', 'color': '#1890ff'},
                {'name': '锡精矿', 'category': '原料', 'color': '#13c2c2'},
                {'name': '氧气', 'category': '原料', 'color': '#13c2c2'},
                {'name': '熔炼', 'category': '工艺', 'color': '#52c41a'},
                {'name': '粗锡', 'category': '产品', 'color': '#eb2f96'},
                {'name': '炉渣', 'category': '产品', 'color': '#eb2f96'},
            ],
            'links': [
                {'source': '锡精矿', 'target': '顶吹炉', 'name': '投入'},
                {'source': '氧气', 'target': '顶吹炉', 'name': '注入'},
                {'source': '顶吹炉', 'target': '熔炼', 'name': '进行'},
                {'source': '熔炼', 'target': '粗锡', 'name': '产出'},
                {'source': '熔炼', 'target': '炉渣', 'name': '产出'},
            ]
        },
        'parameters': {
            'nodes': [
                {'name': '温度', 'category': '参数', 'color': '#fa8c16'},
                {'name': '压力', 'category': '参数', 'color': '#fa8c16'},
                {'name': '流量', 'category': '参数', 'color': '#fa8c16'},
                {'name': '浓度', 'category': '参数', 'color': '#fa8c16'},
                {'name': '时间', 'category': '参数', 'color': '#fa8c16'},
                {'name': '功率', 'category': '参数', 'color': '#fa8c16'},
            ],
            'links': [
                {'source': '温度', 'target': '压力', 'name': '影响'},
                {'source': '流量', 'target': '浓度', 'name': '影响'},
                {'source': '时间', 'target': '温度', 'name': '影响'},
                {'source': '功率', 'target': '温度', 'name': '影响'},
            ]
        }
    }
    
    data = process_data_map.get(process_id, process_data_map['rotary-kiln'])
    return {
        'nodes': data['nodes'],
        'links': data['links'],
        'datas': data['nodes']
    }
