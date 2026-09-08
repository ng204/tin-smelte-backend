from datetime import datetime
from typing import Annotated, Optional, Any
from pydantic import BeforeValidator, BaseModel

# 自定义类型：自动转换datetime为格式化字符串
FormattedDateTime = Annotated[
    str,
    BeforeValidator(lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt, datetime) else dt)
]


class BaseResponse(BaseModel):
    code: int
    message: str
    data: Optional[Any]
