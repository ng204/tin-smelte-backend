from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import Optional
from jwt import decode as jwt_decode, PyJWTError
import logging

import config
from database import get_session
from models.user import User
from models.admin import Admin
from utils.auth import oauth2_scheme

router = APIRouter()


@router.get('/user/info')
async def user_info(token: Optional[str] = Depends(oauth2_scheme), session: Session = Depends(get_session)):
    if not token:
        raise HTTPException(status_code=401, detail='Missing token')
    try:
        payload = jwt_decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
    except PyJWTError:
        raise HTTPException(status_code=401, detail='Invalid token')

    sub = payload.get('sub')
    if not sub:
        raise HTTPException(status_code=401, detail='Invalid token payload')

    role = 'user'
    username = sub
    if ':' in sub:
        parts = sub.split(':', 1)
        role, username = parts[0], parts[1]

    # fetch from DB
    if role == 'admin':
        a = session.scalars(select(Admin).where(Admin.username == username)).first()
        if not a:
            logging.getLogger('uvicorn').info(f'admin not found: {username}')
            raise HTTPException(status_code=404, detail='Admin not found')
        return {"username": a.username, "realName": a.real_name, "roles": ["admin"], "homePath": "/admin-management"}

    u = session.scalars(select(User).where(User.username == username)).first()
    if not u:
        logging.getLogger('uvicorn').info(f'user not found: {username}')
        raise HTTPException(status_code=404, detail='User not found')

    return {"username": u.username, "realName": u.real_name, "roles": ["user"], "homePath": "/"}










