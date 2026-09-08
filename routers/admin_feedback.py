from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from database import get_session
from models.feedback import Feedback
import logging

router = APIRouter()


@router.get("/admin/feedback")
async def admin_list_feedback(
    status: Optional[str] = None,
    type: Optional[str] = None,
    q: Optional[str] = None,
    page: int = 1,
    size: int = 20,
    session: Session = Depends(get_session)
):
    """管理员查询反馈，返回数组在 data 字段中以兼容前端解析"""
    try:
        stmt = select(Feedback)
        items = session.exec(stmt).all()

        if status:
            items = [i for i in items if i.status == status]
        if type:
            items = [i for i in items if i.feedback_type == type]
        if q:
            q_l = q.lower()
            items = [i for i in items if q_l in (i.title or '').lower() or q_l in (i.content or '').lower()]

        total = len(items)
        start = (page - 1) * size
        end = start + size
        page_items = items[start:end]

        out = []
        for r in page_items:
            # 返回同时包含 snake_case 和 camelCase 字段，方便前端兼容
            created_at = r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None
            out.append({
                "id": r.id,
                "feedback_type": r.feedback_type,
                "title": r.title,
                "content": r.content,
                "contact_info": r.contact_info,
                "is_anonymous": r.is_anonymous,
                "user_id": r.user_id,
                "user_name": r.user_name,
                "status": r.status,
                "created_at": created_at,
                # camelCase
                "feedbackType": r.feedback_type,
                "contactInfo": r.contact_info,
                "isAnonymous": r.is_anonymous,
                "userId": r.user_id,
                "userName": r.user_name,
                "statusText": r.status,
                "createdAt": created_at,
            })

        logging.getLogger("uvicorn").info(f"管理员查询反馈，共 {total} 条，返回 {len(out)} 条 page={page} size={size}")
        return {"status": "success", "data": out, "total": total}

    except Exception as e:
        logging.getLogger("uvicorn").exception("管理员查询反馈失败")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/admin/feedback/{feedback_id}")
async def admin_delete_feedback(feedback_id: int, session: Session = Depends(get_session)):
    try:
        fb = session.get(Feedback, feedback_id)
        if not fb:
            raise HTTPException(status_code=404, detail="反馈不存在")
        session.delete(fb)
        session.commit()
        logging.getLogger("uvicorn").info(f"管理员删除反馈 id={feedback_id}")
        return {"status": "success", "message": "反馈已删除"}
    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger("uvicorn").exception("管理员删除反馈失败")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/admin/feedback/{feedback_id}/status")
async def admin_update_status(feedback_id: int, status: str, session: Session = Depends(get_session)):
    try:
        fb = session.get(Feedback, feedback_id)
        if not fb:
            raise HTTPException(status_code=404, detail="反馈不存在")
        fb.status = status
        fb.updated_at = datetime.now()
        session.add(fb)
        session.commit()
        session.refresh(fb)
        logging.getLogger("uvicorn").info(f"管理员更新反馈状态 id={feedback_id} -> {status}")
        return {"status": "success", "message": "反馈状态已更新"}
    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger("uvicorn").exception("管理员更新反馈失败")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/feedback/{feedback_id}/reply")
async def admin_reply_feedback(feedback_id: int, reply_content: str, session: Session = Depends(get_session)):
    try:
        fb = session.get(Feedback, feedback_id)
        if not fb:
            raise HTTPException(status_code=404, detail="反馈不存在")
        fb.reply = reply_content
        fb.replied_at = datetime.now()
        fb.updated_at = datetime.now()
        session.add(fb)
        session.commit()
        session.refresh(fb)
        logging.getLogger("uvicorn").info(f"管理员回复反馈 id={feedback_id}")
        return {"status": "success", "message": "回复已提交"}
    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger("uvicorn").exception("管理员回复失败")
        raise HTTPException(status_code=500, detail=str(e))


