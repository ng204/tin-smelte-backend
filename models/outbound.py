from datetime import datetime
from sqlmodel import SQLModel, Field as SqlField
from pydantic import BaseModel, Field, ConfigDict

from models.base import FormattedDateTime

outbound_example_data = {
    "name": "乙锡",
    "number": "20230801-001",
    "operator": "张三",
    "weight": "5000",
    "source": "XX供应商",
    "receptionUnit": "第一仓库",
    "cost": "25000.00"
}


class OutboundBase(BaseModel):
    name: str = Field(description="物料名称")
    number: str = Field(description="物料批号")
    operator: str = Field(description="操作人")
    weight: str = Field(description="重量（kg）")
    source: str = Field(description="来源")
    reception_unit: str = Field(
        description="接收单位",
        alias="receptionUnit",
    )
    cost: str = Field(description="成本（元）")

    # 统一使用ConfigDict配置
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": outbound_example_data
        }
    )


class Outbound(SQLModel, OutboundBase, table=True):
    id: int = SqlField(default=None, primary_key=True, index=True, unique=True, description="id")
    time: datetime = SqlField(default_factory=datetime.now, description="出库时间")


# class OutboundCreate(OutboundBase):
#     model_config = ConfigDict(
#         json_schema_extra={
#             "examples": OutboundBase.model_config["json_schema_extra"]["examples"]
#         }
#     )
#
#
# class OutboundUpdate(OutboundBase):
#     model_config = ConfigDict(
#         json_schema_extra={
#             "examples": OutboundBase.model_config["json_schema_extra"]["examples"]
#         }
#     )


class OutboundRead(OutboundBase):
    id: int = Field(description="id")
    time: FormattedDateTime = Field(description="出库时间")
    model_config = ConfigDict(
        populate_by_name=True,
        # from_attributes=True,
        json_schema_extra={
            "example": {**{"id": 1, "time": "2023-08-01 10:00:00"}, **outbound_example_data}
        }
    )


# 分页响应模型
class OutboundPagination(BaseModel):
    total: int
    items: list[OutboundRead]
