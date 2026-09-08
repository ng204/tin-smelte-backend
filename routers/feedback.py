# -*- coding: UTF-8 -*-
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import Session, select

from database import get_session
from middlewares.custom_response import CustomRoute
from utils.auth import oauth2_scheme
from models.feedback import Feedback
from models.user import User
from models.admin import Admin
from jwt import decode as jwt_decode, PyJWTError
import config
import logging

router = APIRouter(prefix="/feedback", tags=["用户反馈"])
router.route_class = CustomRoute

# Pydantic models
class FeedbackRequest(BaseModel):
    feedback_type: str
    title: str
    content: str
    contact_info: Optional[str] = None
    is_anonymous: bool = False
    # 可选的前端辅助字段（后端优先使用 token 解析结果）
    user_name: Optional[str] = None
    user_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: Optional[int] = None


class FeedbackItem(BaseModel):
    id: int
    feedbackType: str
    title: str
    content: str
    contactInfo: Optional[str] = None
    isAnonymous: bool
    userId: Optional[str] = None
    userName: Optional[str] = None
    status: str
    createdAt: Optional[str] = None


# 提交反馈接口
@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    request: Request,
    session: Session = Depends(get_session)
):
    """提交用户反馈并写入数据库"""
    try:
        # 验证反馈类型
        valid_types = ['bug', 'suggestion', 'data', 'ui', 'other']
        if feedback.feedback_type not in valid_types:
            raise HTTPException(status_code=400, detail="无效的反馈类型")

        # 验证必填字段
        if not feedback.title.strip():
            raise HTTPException(status_code=400, detail="反馈标题不能为空")

        if not feedback.content.strip():
            raise HTTPException(status_code=400, detail="反馈内容不能为空")

        # 尝试从 Authorization header 提取 token, 解析出 username 并对应到 User/Admin 表
        user_id = None
        user_name = None
        try:
            auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
            token = None
            if auth_header and auth_header.lower().startswith("bearer "):
                token = auth_header.split(" ", 1)[1]
            if token:
                payload = jwt_decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
                sub = payload.get("sub")
                if sub:
                    role = 'user'
                    username = sub
                    if ':' in sub:
                        parts = sub.split(':', 1)
                        role, username = parts[0], parts[1]

                    if role == 'admin':
                        a = session.scalars(select(Admin).where(Admin.username == username)).first()
                        if a:
                            user_id = str(a.id)
                            user_name = a.username or getattr(a, 'real_name', None)
                    else:
                        u = session.scalars(select(User).where(User.username == username)).first()
                        if u:
                            user_id = str(u.id)
                            user_name = u.username or getattr(u, 'real_name', None)
        except PyJWTError:
            # token 无效则视为匿名提交，继续保存
            logging.getLogger("uvicorn").info("解析反馈提交者 token 失败，作为匿名处理")

        # 后端优先使用 token 解析出的 user 信息；若无，则使用前端提交的辅助字段
        if not user_name and getattr(feedback, 'user_name', None):
            user_name = feedback.user_name
        if not user_id and getattr(feedback, 'user_id', None):
            user_id = feedback.user_id

        # 保存到数据库，显式设置创建时间
        fb = Feedback(
            feedback_type=feedback.feedback_type,
            title=feedback.title,
            content=feedback.content,
            contact_info=feedback.contact_info,
            is_anonymous=feedback.is_anonymous,
            user_id=user_id,
            user_name=user_name,
            status='pending',
            created_at=datetime.now(),
        )
        session.add(fb)
        session.commit()
        session.refresh(fb)

        src = f"user_id={fb.user_id} user_name={fb.user_name}" if fb.user_id or fb.user_name else "anonymous"
        logging.getLogger("uvicorn").info(f"收到反馈: id={fb.id} type={fb.feedback_type} title={fb.title} from={src}")

        return FeedbackResponse(status="success", message="感谢您的反馈，我们会尽快处理！", feedback_id=fb.id)

    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger("uvicorn").exception("提交反馈失败")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")


# 获取反馈列表接口（管理员功能）
@router.get("/", response_model=List[FeedbackItem])
async def get_feedback_list(
    status_filter: Optional[str] = None,
    type_filter: Optional[str] = None,
    page: int = 1,
    size: int = 20,
    session: Session = Depends(get_session)
):
    """从数据库中查询反馈，支持状态/类型过滤与分页"""
    try:
        stmt = select(Feedback)
        results = session.exec(stmt).all()

        # 过滤
        if status_filter:
            results = [r for r in results if r.status == status_filter]
        if type_filter:
            results = [r for r in results if r.feedback_type == type_filter]

        total = len(results)
        start = (page - 1) * size
        end = start + size
        page_items = results[start:end]

        out = []
        for r in page_items:
            out.append({
                "id": r.id,
                "feedbackType": r.feedback_type,
                "title": r.title,
                "content": r.content,
                "contactInfo": r.contact_info,
                "isAnonymous": r.is_anonymous,
                "userId": r.user_id,
                "userName": r.user_name,
                "status": r.status,
                "createdAt": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None,
            })

        logging.getLogger("uvicorn").info(f"查询反馈，共 {total} 条，返回 {len(out)} 条 page={page} size={size}")
        return out

    except Exception as e:
        logging.getLogger("uvicorn").exception("查询反馈失败")
        raise HTTPException(status_code=500, detail=f"获取反馈列表失败: {str(e)}")


# 更新反馈状态接口（管理员功能）
@router.put("/{feedback_id}/status")
async def update_feedback_status(
    feedback_id: int,
    status: str,
    session: Session = Depends(get_session)
):
    """更新反馈状态并保存"""
    try:
        valid_statuses = ['pending', 'in_review', 'resolved']
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail="无效的状态值")

        fb = session.get(Feedback, feedback_id)
        if not fb:
            raise HTTPException(status_code=404, detail="反馈不存在")
        fb.status = status
        fb.updated_at = datetime.now()
        session.add(fb)
        session.commit()
        session.refresh(fb)

        logging.getLogger("uvicorn").info(f"更新反馈状态 id={feedback_id} -> {status}")
        return {"status": "success", "message": "反馈状态已更新"}

    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger("uvicorn").exception("更新反馈状态失败")
        raise HTTPException(status_code=500, detail=f"更新反馈状态失败: {str(e)}")


# 删除反馈接口（管理员功能）
@router.delete("/{feedback_id}")
async def delete_feedback(
    feedback_id: int,
    session: Session = Depends(get_session)
):
    """删除反馈（管理员功能）"""
    try:
        fb = session.get(Feedback, feedback_id)
        if not fb:
            raise HTTPException(status_code=404, detail="反馈不存在")
        session.delete(fb)
        session.commit()
        logging.getLogger("uvicorn").info(f"删除反馈 id={feedback_id}")
        return {"status": "success", "message": "反馈已删除"}

    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger("uvicorn").exception("删除反馈失败")
        raise HTTPException(status_code=500, detail=f"删除反馈失败: {str(e)}")


# 回复反馈接口（管理员功能）
@router.post("/{feedback_id}/reply")
async def reply_to_feedback(
    feedback_id: int,
    reply_content: str,
    session: Session = Depends(get_session)
):
    """回复反馈并保存回复内容"""
    try:
        if not reply_content.strip():
            raise HTTPException(status_code=400, detail="回复内容不能为空")

        fb = session.get(Feedback, feedback_id)
        if not fb:
            raise HTTPException(status_code=404, detail="反馈不存在")
        fb.reply = reply_content
        fb.replied_at = datetime.now()
        fb.updated_at = datetime.now()
        session.add(fb)
        session.commit()
        session.refresh(fb)

        logging.getLogger("uvicorn").info(f"回复反馈 id={feedback_id}")
        return {"status": "success", "message": "回复已提交"}

    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger("uvicorn").exception("回复反馈失败")
        raise HTTPException(status_code=500, detail=f"回复反馈失败: {str(e)}")
