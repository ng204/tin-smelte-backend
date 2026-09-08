from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field


class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    feedback_type: str = Field(index=True, description="反馈类型")
    title: str = Field(description="标题")
    content: str = Field(description="内容")
    contact_info: Optional[str] = Field(default=None, description="联系方式")
    is_anonymous: bool = Field(default=False, description="是否匿名")
    user_id: Optional[str] = Field(default=None, description="提交用户 ID")
    user_name: Optional[str] = Field(default=None, description="提交用户名")
    status: str = Field(default="pending", description="处理状态 pending/in_review/resolved")
    reply: Optional[str] = Field(default=None, description="管理员回复")
    replied_at: Optional[datetime] = Field(default=None, description="回复时间")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")










