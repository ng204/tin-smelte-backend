from datetime import datetime
from pydantic import ConfigDict, BaseModel
from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
    id: int = Field(
        default=None,
        primary_key=True,
        description="用户唯一ID（自动递增）"
    )

    username: str = Field(
        index=True,  # 添加索引提高查询效率
        unique=True,  # 用户名唯一性约束
        min_length=3,  # 最小长度限制
        max_length=20,  # 最大长度限制
        description="用户名"
    )

    real_name: str = Field(
        max_length=20,
        alias="realName",
        description="昵称"
    )

    password: str = Field(
        min_length=6,  # 密码最小长度
        description="密码（存储哈希值，非明文）"
    )

    created_at: datetime = Field(default_factory=datetime.now, description='创建时间')
    updated_at: datetime = Field(default_factory=datetime.now, description='更新时间')
    is_disabled: bool = Field(default=False, description='用户是否被禁用')
    status: str = Field(default="normal", description='账号状态 normal/disabled')

    model_config = ConfigDict(
        populate_by_name=True,  # 允许别名访问
    )


# 注册请求模型
class UserCreate(BaseModel):
    username: str = Field(description='用户名')
    realName: str = Field(description='昵称')
    password: str = Field(description='密码（明文）')
    # 注册时可选角色，用于区分写入 Admin 还是 User 表
    role: str | None = None
    model_config = ConfigDict(
        populate_by_name=True,
        # from_attributes=True,
        json_schema_extra={
            "example": {
                "username": "test",
                "realName": "测试用户",
                "password": "test",
                "role": "user",
            }
        }
    )


# 登录请求模型
class UserLogin(BaseModel):
    username: str = Field(description='用户名')
    password: str = Field(description='密码（明文）')
    role: str | None = None
    model_config = ConfigDict(
        populate_by_name=True,
        # from_attributes=True,
        json_schema_extra={
            "example": {
                "username": "test",
                "password": "test",
                "role": "user",
            }
        }
    )
