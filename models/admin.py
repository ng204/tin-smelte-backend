from datetime import datetime
from pydantic import ConfigDict, BaseModel, Field as PydanticField
from sqlmodel import SQLModel, Field


class Admin(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True, description="管理员唯一ID（自动递增）")

    username: str = Field(index=True, unique=True, min_length=3, max_length=20, description="用户名")

    real_name: str = Field(max_length=20, alias="realName", description="昵称")

    password: str = Field(min_length=6, description="密码（存储哈希值，非明文）")

    # 若数据库存在非空且无默认的 role 列，这里提供模型字段与默认值
    role: str = Field(default="admin", description="角色标识")

    created_at: datetime = Field(default_factory=datetime.now, description='创建时间')
    updated_at: datetime = Field(default_factory=datetime.now, description='更新时间')
    is_disabled: bool = Field(default=False, description='管理员是否被禁用')

    model_config = ConfigDict(populate_by_name=True)


class AdminCreate(BaseModel):
    username: str = PydanticField(description='用户名')
    realName: str = PydanticField(description='昵称')
    password: str = PydanticField(description='密码（明文）')
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "username": "admin01",
                "realName": "管理员01",
                "password": "123456",
            }
        },
    )


