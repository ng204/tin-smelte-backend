# -*- coding: UTF-8 -*-
import os
import shutil
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from database import get_session
from middlewares.custom_response import CustomRoute
from utils.auth import oauth2_scheme
from jwt import decode as jwt_decode, PyJWTError
import config
import json
import logging

# 创建必要的目录
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)
META_FILE = UPLOAD_DIR / ".uploads_meta.json"


def load_meta() -> dict:
    if not META_FILE.exists():
        return {}
    try:
        with open(META_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_meta(meta: dict):
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def get_token_info_from_request(request: Request):
    """从请求中解析 Authorization header，返回 (username, role) 或 (None,None)"""
    auth = request.headers.get('authorization') or request.headers.get('Authorization')
    if not auth:
        return None, None
    parts = auth.split()
    if len(parts) != 2:
        return None, None
    token = parts[1]
    try:
        payload = jwt_decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        sub = payload.get('sub')
        if not sub:
            return None, None
        # 支持形如 "admin:username" 或 "user:username" 的 sub
        if isinstance(sub, str) and ':' in sub:
            role, uname = sub.split(':', 1)
            return uname, role
        return sub, None
    except PyJWTError:
        return None, None

router = APIRouter(prefix="/files", tags=["文件共享"])
router.route_class = CustomRoute

# Pydantic models
class FileInfo(BaseModel):
    name: str
    description: Optional[str] = None
    uploader: str
    uploadTime: str
    size: int
    shareScope: str

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: Optional[str] = None
    description: Optional[str] = None
    share_scope: Optional[str] = None

class SearchResponse(BaseModel):
    status: str
    keyword: str
    count: int
    files: List[FileInfo]

# 文件上传接口
@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    description: str = "",
    shareScope: str = "all",
):
    """上传文件接口"""
    try:
        # 验证文件类型
        allowed_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.txt']
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail="不支持的文件类型")

        # 验证文件大小 (50MB)
        max_size = 50 * 1024 * 1024
        file_content = await file.read()
        if len(file_content) > max_size:
            raise HTTPException(status_code=400, detail="文件大小超过50MB限制")

        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / unique_filename

        # 保存文件
        with open(file_path, "wb") as f:
            f.write(file_content)

        # 记录上传者信息到 meta（始终写入，确保前端能显示上传者）
        meta = load_meta()
        # 解析当前请求 token，从中获取用户名与角色
        uname, urole = get_token_info_from_request(request)
        if uname:
            uploader = f"{uname} ({urole or 'user'})"
        else:
            uploader = 'anonymous'

        # 保存 username 与 role 分离字段，便于权限判断
        username_only = uname or 'anonymous'
        role_only = urole or 'user' if uname else 'anonymous'
        meta[unique_filename] = {
            'uploader': f"{username_only} ({role_only})",
            'username': username_only,
            'role': role_only,
            'description': description,
            'shareScope': shareScope,
            'uploadTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'size': len(file_content)
        }
        save_meta(meta)

        # 返回成功响应
        return UploadResponse(
            status="success",
            message="文件上传成功",
            filename=unique_filename,
            description=description,
            share_scope=shareScope
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

# 获取文件列表接口
@router.get("/", response_model=List[FileInfo])
async def get_files():
    """获取文件列表"""
    try:
        files = []
        meta = load_meta()
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                m = meta.get(file_path.name, {})
                files.append(FileInfo(
                    name=file_path.name,
                    description=m.get('description', ''),
                    uploader=m.get('uploader', 'admin'),
                    uploadTime=m.get('uploadTime', datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")),
                    size=m.get('size', stat.st_size),
                    shareScope=m.get('shareScope', 'all')
                ))

        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

# 文件搜索接口
@router.get("/search", response_model=SearchResponse)
async def search_files(
    keyword: str = Query(..., description="搜索关键词"),
):
    """搜索文件"""
    try:
        if not keyword.strip():
            raise HTTPException(status_code=400, detail="搜索关键词不能为空")

        matched_files = []
        keyword_lower = keyword.lower()
        logger = logging.getLogger("uvicorn")
        logger.debug(f"文件搜索: keyword={keyword}")

        meta = load_meta()
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                filename = file_path.name
                # 支持模糊匹配：若文件名包含下划线前缀（时间戳），也匹配真实文件名部分
                stripped_name = filename.split('_', 1)[1] if '_' in filename else filename
                matched = (keyword_lower in filename.lower() or keyword_lower in stripped_name.lower())
                logger.debug(f"检查文件: {filename}, stripped={stripped_name}, matched={matched}")
                if matched:
                    stat = file_path.stat()
                    m = meta.get(filename, {})
                    matched_files.append(FileInfo(
                        name=filename,
                        description=m.get('description', ''),
                        uploader=m.get('uploader', 'admin'),
                        uploadTime=m.get('uploadTime', datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")),
                        size=m.get('size', stat.st_size),
                        shareScope=m.get('shareScope', 'all')
                    ))

        return SearchResponse(
            status="success",
            keyword=keyword,
            count=len(matched_files),
            files=matched_files
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件搜索失败: {str(e)}")

# 文件下载接口
@router.get("/download/{filename}")
async def download_file(filename: str):
    """下载文件"""
    try:
        # URL解码
        from urllib.parse import unquote
        filename = unquote(filename)

        # 检查是否是标准文档文件
        docs_files = {
            "企业标准.docx": DOCS_DIR / "企业标准.docx",
            "编制说明.docx": DOCS_DIR / "编制说明.docx"
        }

        if filename in docs_files:
            file_path = docs_files[filename]
        else:
            file_path = UPLOAD_DIR / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件下载失败: {str(e)}")

# 文件删除接口
@router.delete("/{filename}")
async def delete_file(
    filename: str,
    request: Request
):
    """删除文件"""
    try:
        from urllib.parse import unquote
        filename = unquote(filename)

        # 权限检查：仅允许管理员删除任意文件，普通用户只能删除自己上传的文件
        from urllib.parse import unquote
        filename = unquote(filename)

        file_path = UPLOAD_DIR / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")

        # 读取 meta
        meta = load_meta()
        info = meta.get(filename, {})
        file_owner = info.get('uploader')  # e.g. 'alice (user)'

        # 解析当前请求用户与角色
        cur_user, cur_role = get_token_info_from_request(request)
        if not cur_user:
            raise HTTPException(status_code=401, detail="需要登录后才可删除文件")

        # 管理员可以删除任意文件
        if cur_role and str(cur_role).lower().startswith('admin'):
            allowed = True
        else:
            # 普通用户仅能删除自己上传的文件；meta 中的 uploader 可能含有角色后缀
            allowed = False
            if file_owner:
                # 比较用户名前缀
                if isinstance(file_owner, str) and file_owner.startswith(cur_user):
                    allowed = True

        if not allowed:
            raise HTTPException(status_code=403, detail="没有删除该文件的权限")

        # 执行删除并更新 meta
        file_path.unlink()
        if filename in meta:
            del meta[filename]
            save_meta(meta)

        return {"status": "success", "message": "文件删除成功"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件删除失败: {str(e)}")

# 格式化文件大小的辅助函数
def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1

    return f"{size:.1f} {size_names[i]}"
