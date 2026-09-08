from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, func

from middlewares.custom_response import CustomRoute
from models.outbound import Outbound, OutboundBase, OutboundRead, OutboundPagination
from database import get_session
from utils.auth import oauth2_scheme

router = APIRouter(prefix="/outbound", tags=["物料出库"], dependencies=[Depends(oauth2_scheme)])

router.route_class = CustomRoute


# 增
@router.post("")
async def create_outbound(
        outbound: OutboundBase,
        session: Session = Depends(get_session)
):
    db_outbound = Outbound.model_validate(outbound)
    session.add(db_outbound)
    session.commit()
    session.refresh(db_outbound)
    return None


# 列表
@router.get("", response_model=OutboundPagination)
async def read_outbounds(
        size: int = 0,
        count: int = 20,
        session: Session = Depends(get_session)
):
    # 获取总数
    total = session.scalars(select(func.count(Outbound.id))).one()
    # 获取分页数据
    result = session.scalars(select(Outbound).offset(size).limit(count))
    return {"total": total, "items": result.all()}


# 查
@router.get("/{outbound_id}", response_model=OutboundRead)
async def read_outbound(
        outbound_id: int,
        session: Session = Depends(get_session)
):
    outbound = session.get(Outbound, outbound_id)
    if not outbound:
        raise HTTPException(status_code=404, detail="物料记录未找到")
    return outbound


# 改
@router.patch("/{outbound_id}")
async def update_outbound(
        outbound_id: int,
        outbound: OutboundBase,
        session: Session = Depends(get_session)
):
    db_outbound = session.get(Outbound, outbound_id)
    if not db_outbound:
        raise HTTPException(status_code=404, detail="物料记录未找到")

    outbound_data = outbound.model_dump(exclude_unset=True)
    for key, value in outbound_data.items():
        setattr(db_outbound, key, value)

    session.add(db_outbound)
    session.commit()
    session.refresh(db_outbound)
    return None


# 删
@router.delete("/{outbound_id}")
async def delete_outbound(
        outbound_id: int,
        session: Session = Depends(get_session)
):
    outbound = session.get(Outbound, outbound_id)
    if not outbound:
        raise HTTPException(status_code=404, detail="物料记录未找到")
    session.delete(outbound)
    session.commit()
    return None
