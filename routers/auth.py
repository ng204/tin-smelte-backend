from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
import re
import logging
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select
from sqlalchemy import func

from database import get_session
from middlewares.custom_response import CustomRoute
from models.user import User, UserLogin, UserCreate
from models.admin import Admin
from utils.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
)

router = APIRouter(prefix='/auth', tags=["登陆注册"])
# 密码复杂度校验：8-16位，至少包含字母、数字和符号
_PW_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)(?=.*[^A-Za-z0-9])[A-Za-z\d\W_]{8,16}$")

def _validate_password_complexity_or_raise(pw: str):
    if not isinstance(pw, str) or not _PW_RE.match(pw or ""):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="密码需为8-16位，且同时包含字母、数字和符号"
        )


# 统一返回添加
router.route_class = CustomRoute


@router.post("/register")
def register(user_data: UserCreate, session: Session = Depends(get_session)):
    # 允许请求体中携带 role 字段（'user' | 'admin'），不在模型中但可从 dict 获取
    payload = user_data.model_dump(by_alias=True)
    role_raw = payload.get('role', 'user')
    role = str(role_raw or 'user').strip().lower()

    # 兼容中文或其他写法
    admin_alias = {"admin", "administrator", "root", "管理员"}

    # 统一检查：两个表都不能重名
    exists_user = session.scalars(select(User).where(User.username == user_data.username)).first()
    exists_admin = session.scalars(select(Admin).where(Admin.username == user_data.username)).first()
    if exists_user or exists_admin:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="该账号已被注册使用")

    # 校验密码复杂度
    _validate_password_complexity_or_raise(user_data.password)

    if role in admin_alias:
        existing = session.scalars(select(Admin).where(Admin.username == user_data.username)).first()
        if existing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="管理员已存在")
        hashed_password = get_password_hash(user_data.password)
        admin = Admin(
            username=user_data.username,
            real_name=user_data.realName,
            password=hashed_password,
            role='admin',
        )
        session.add(admin)
        session.commit()
        session.refresh(admin)
        logging.getLogger("uvicorn").info(f"管理员注册成功：{admin.username}，信息已加入 Admin 表")
        return {"message": "注册成功，信息已加入 Admin 表", "username": admin.username}

    # 默认注册到普通用户表

    hashed_password = get_password_hash(user_data.password)
    user = User(
        username=user_data.username,
        real_name=user_data.realName,
        password=hashed_password,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    logging.getLogger("uvicorn").info(f"用户注册成功：{user.username}，信息已加入 User 表")
    return {"message": "注册成功，信息已加入 User 表", "username": user.username}


@router.post("/login")
def login(
        login_data: UserLogin,
        session: Session = Depends(get_session)
):
    payload = login_data.model_dump(by_alias=True)
    role = str(payload.get('role') or '').strip().lower()

    # 兼容中文
    admin_alias = {"admin", "administrator", "root", "管理员"}

    if role in admin_alias:
        admin = session.scalars(select(Admin).where(Admin.username == login_data.username)).first()
        if not admin:
            logging.getLogger("uvicorn").info(f"登录失败：Admin 表中不存在管理员信息，username={login_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Admin表中不存在管理员信息",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if admin.is_disabled:
            logging.getLogger("uvicorn").info(f"登录失败：Admin 已被禁用，username={login_data.username}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="该账号已被封禁",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not verify_password(login_data.password, admin.password):
            logging.getLogger("uvicorn").warning(f"登录失败：管理员密码错误，username={login_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="管理员密码错误",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token = create_access_token(data={"sub": f"admin:{admin.username}"})
        logging.getLogger("uvicorn").info(f"登录成功：Admin 表存在管理员信息，username={admin.username}")
        return {"access_token": access_token, "token_type": "bearer", "role": "admin"}

    # 默认查询普通用户
    user = session.scalars(select(User).where(User.username == login_data.username)).first()
    if not user:
        logging.getLogger("uvicorn").info(f"登录失败：User 表中不存在用户信息，username={login_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User表中不存在用户信息",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # 管理员账号不可被禁用，仅检查普通用户状态
    if not str(login_data.model_dump().get('role','user')).lower().startswith('admin'):
        if getattr(user, 'is_disabled', False) or getattr(user, 'status', '') == 'disabled':
            logging.getLogger("uvicorn").info(f"登录失败：User 已被禁用，username={login_data.username}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="该账号已被封禁",
                headers={"WWW-Authenticate": "Bearer"},
            )
    if not verify_password(login_data.password, user.password):
        logging.getLogger("uvicorn").warning(f"登录失败：用户密码错误，username={login_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": f"user:{user.username}"})
    logging.getLogger("uvicorn").info(f"登录成功：User 表存在用户信息，username={user.username}")
    return {"access_token": access_token, "token_type": "bearer", "role": "user"}


@router.post("/login-swagger")
def login_swagger(
        login_data: OAuth2PasswordRequestForm = Depends(),
        session: Session = Depends(get_session)
):
    user = session.scalars(select(User).where(User.username == login_data.username)).first()

    if not user or not verify_password(login_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users")
def list_users(page: int = 1, size: int = 10, q: str | None = None, session: Session = Depends(get_session)):
    """列出用户（数据库层分页 + 模糊查询），返回结构：{status, data: {users, total, page, size}}"""
    # 约束分页参数，避免过大 size 造成压力
    page = max(1, int(page or 1))
    size = max(1, min(100, int(size or 10)))

    filters = []
    if q:
        # 使用 ILIKE/LIKE 模糊查询
        filters.append(User.username.ilike(f"%{q}%"))

    # 计算总数
    count_stmt = select(func.count()).select_from(User)
    for cond in filters:
        count_stmt = count_stmt.where(cond)
    total = session.exec(count_stmt).one()

    # 查询分页数据
    data_stmt = select(User)
    for cond in filters:
        data_stmt = data_stmt.where(cond)
    data_stmt = data_stmt.offset((page - 1) * size).limit(size)

    page_users = session.scalars(data_stmt).all()
    result = []
    for u in page_users:
        enabled = not (getattr(u, 'is_disabled', False) or str(getattr(u, 'status', '') or '').lower() == 'disabled')
        result.append({
            "id": u.id,
            "username": u.username,
            "realName": u.real_name,
            "created_at": u.created_at,
            "updated_at": u.updated_at,
            "status": getattr(u, 'status', None),
            "is_disabled": getattr(u, 'is_disabled', None),
            "enabled": enabled,
        })
    logging.getLogger("uvicorn").info(f"查询 User 表，共 {total} 条，返回 {len(result)} 条，page={page} size={size} q={q}")
    return {"status": "success", "data": {"users": result, "total": int(total), "page": page, "size": size}}


@router.delete('/users/{username}')
def delete_user(username: str, session: Session = Depends(get_session)):
    """删除用户（同时支持删除 Admin 表中的管理员）"""
    # 尝't remove super-admin safety here; simple deletion
    u = session.scalars(select(User).where(User.username == username)).first()
    if u:
        session.delete(u)
        session.commit()
        logging.getLogger("uvicorn").info(f"删除 User: {username}")
        return {"status": "success", "message": "用户已删除"}

    a = session.scalars(select(Admin).where(Admin.username == username)).first()
    if a:
        session.delete(a)
        session.commit()
        logging.getLogger("uvicorn").info(f"删除 Admin: {username}")
        return {"status": "success", "message": "管理员已删除"}

    raise HTTPException(status_code=404, detail="用户不存在")


@router.get("/admins")
def list_admins(session: Session = Depends(get_session)):
    admins = session.scalars(select(Admin)).all()
    result = [
        {
            "id": a.id,
            "username": a.username,
            "realName": a.real_name,
            "created_at": a.created_at,
            "updated_at": a.updated_at,
        }
        for a in admins
    ]
    logging.getLogger("uvicorn").info(f"查询 Admin 表，共 {len(result)} 条")
    return result


@router.put('/users/{username}')
def update_user(username: str, payload: dict, session: Session = Depends(get_session)):
    """更新用户信息（支持更新 real_name、password、role）"""
    # 尝试在 User 表中更新
    u = session.scalars(select(User).where(User.username == username)).first()
    if u:
        # 更新 real name
        if 'realName' in payload:
            u.real_name = payload.get('realName')
        # 更新密码（如果提供）
        if payload.get('password'):
            u.password = get_password_hash(payload.get('password'))
        # 更新禁用状态
        if 'is_disabled' in payload:
            u.is_disabled = bool(payload.get('is_disabled'))
        # 更新状态字段（normal/disabled）
        if 'status' in payload:
            u.status = str(payload.get('status') or '').strip()
        session.add(u)
        session.commit()
        session.refresh(u)
        logging.getLogger('uvicorn').info(f'更新 User: {username}')
        return {"status": "success", "message": "用户已更新"}

    # 若在 User 中未找到，尝试在 Admin 表中更新
    a = session.scalars(select(Admin).where(Admin.username == username)).first()
    if a:
        if 'realName' in payload:
            a.real_name = payload.get('realName')
        if payload.get('password'):
            a.password = get_password_hash(payload.get('password'))
        session.add(a)
        session.commit()
        session.refresh(a)
        logging.getLogger('uvicorn').info(f'更新 Admin: {username}')
        return {"status": "success", "message": "管理员已更新"}

    raise HTTPException(status_code=404, detail='用户不存在')


# ===== 忘记密码 / 重置密码 =====
class VerifyPayload(BaseModel):
    username: str
    role: str | None = None


class ResetPasswordPayload(BaseModel):
    username: str
    newPassword: str
    role: str | None = None


@router.post('/verify_username')
def verify_username(payload: VerifyPayload, session: Session = Depends(get_session)):
    """验证用户名是否存在且允许重置。支持 role 指定 'user' 或 'admin'。"""
    role = str(payload.role or '').strip().lower()
    admin_alias = {"admin", "administrator", "root", "管理员"}

    if role in admin_alias:
        admin = session.scalars(select(Admin).where(Admin.username == payload.username)).first()
        if not admin:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="管理员不存在")
        return {"status": "success", "role": "admin"}

    # 默认验证普通用户
    user = session.scalars(select(User).where(User.username == payload.username)).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    if getattr(user, 'is_disabled', False) or getattr(user, 'status', '') == 'disabled':
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="该账号已被封禁，无法重置")
    return {"status": "success", "role": "user"}


def _validate_password_complexity(pw: str):
    """测试阶段：不做任何复杂度校验。"""
    return


@router.post('/reset_password')
def reset_password(payload: ResetPasswordPayload, session: Session = Depends(get_session)):
    """直接重置指定用户名的密码（演示环境，无邮件验证码）。"""
    role = str(payload.role or '').strip().lower()
    # 复杂度校验
    _validate_password_complexity_or_raise(payload.newPassword)

    # 管理员与普通用户分别处理
    admin_alias = {"admin", "administrator", "root", "管理员"}
    if role in admin_alias:
        admin = session.scalars(select(Admin).where(Admin.username == payload.username)).first()
        if not admin:
            raise HTTPException(status_code=404, detail="管理员不存在")
        admin.password = get_password_hash(payload.newPassword)
        session.add(admin)
        session.commit()
        session.refresh(admin)
        return {"status": "success", "message": "管理员密码重置成功"}

    user = session.scalars(select(User).where(User.username == payload.username)).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if getattr(user, 'is_disabled', False) or getattr(user, 'status', '') == 'disabled':
        raise HTTPException(status_code=403, detail="该账号已被封禁，无法重置")
    user.password = get_password_hash(payload.newPassword)
    session.add(user)
    session.commit()
    session.refresh(user)
    return {"status": "success", "message": "密码重置成功"}


# ========== 会话维护：刷新与退出 ==========
@router.post('/refresh')
async def refresh_token():
    """简化实现：直接返回 200，前端据此维持会话。
    可按需改为校验 refreshToken 并签发新 accessToken。
    """
    return {"status": "success"}


@router.post('/logout')
async def logout():
    """简化实现：无状态 JWT，后端不需要做服务器端销毁。
    返回 200 让前端清理本地令牌并跳转登录页。
    """
    return {"status": "success"}

