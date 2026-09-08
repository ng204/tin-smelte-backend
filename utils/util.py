from datetime import datetime


# 格式化 datetime 对象为字符串的函数
def format_datetime(dt: datetime):
    return dt.strftime("%Y-%m-%d %H:%M:%S")
