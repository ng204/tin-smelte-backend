from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select, func

from middlewares.custom_response import CustomRoute
from models.inbound import Inbound, InboundBase, InboundRead, InboundPagination
from database import get_session
from utils.auth import oauth2_scheme

router = APIRouter(prefix="/inbound", tags=["物料入库"], dependencies=[Depends(oauth2_scheme)])

router.route_class = CustomRoute


# 增
@router.post("")
async def create_inbound(
        inbound: InboundBase,
        session: Session = Depends(get_session)
):
    db_inbound = Inbound.model_validate(inbound)
    session.add(db_inbound)
    session.commit()
    session.refresh(db_inbound)
    return None


# 列表
@router.get("", response_model=InboundPagination)
async def read_inbounds(
        size: int = Query(0, ge=0),  # 从0开始,
        count: int = Query(20, le=100),  # 最多每页100条
        session: Session = Depends(get_session)
):
    # 获取总数
    total = session.scalars(select(func.count(Inbound.id))).first()
    # 获取分页数据
    result = session.scalars(select(Inbound).offset(size).limit(count))
    return {"total": total, "items": result.all()}


# 查
@router.get("/{inbound_id}", response_model=InboundRead)
async def read_inbound(
        inbound_id: int,
        session: Session = Depends(get_session)
):
    inbound = session.get(Inbound, inbound_id)
    if not inbound:
        raise HTTPException(status_code=404, detail="物料记录未找到")
    return inbound


# 改
@router.patch("/{inbound_id}")
async def update_inbound(
        inbound_id: int,
        inbound: InboundBase,
        session: Session = Depends(get_session)
):
    db_inbound = session.get(Inbound, inbound_id)
    if not db_inbound:
        raise HTTPException(status_code=404, detail="物料记录未找到")

    inbound_data = inbound.model_dump(exclude_unset=True)
    for key, value in inbound_data.items():
        setattr(db_inbound, key, value)

    session.add(db_inbound)
    session.commit()
    session.refresh(db_inbound)
    return None


# 删
@router.delete("/{inbound_id}")
async def delete_inbound(
        inbound_id: int,
        session: Session = Depends(get_session)
):
    inbound = session.get(Inbound, inbound_id)
    if not inbound:
        raise HTTPException(status_code=404, detail="物料记录未找到")
    session.delete(inbound)
    session.commit()
    return None
