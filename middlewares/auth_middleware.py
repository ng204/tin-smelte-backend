from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from utils.auth import get_current_user

EXCLUDED_PATHS = [
    "/docs", "/redoc", "/openapi.json",
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/refresh-token",
    "/api/auth/login-swagger",
    "/api/technology/recognize",  # 添加技术API到排除路径
    "/technology/recognize",  # 添加不带前缀的路径
    "/",  # 根路径
    "/health",  # 健康检查
    # 新增的集成功能API路径
    "/api/files",
    "/api/feedback",
    "/api/graph",
    "/api/triple_construction",
    "/api/qa"
]


async def auth_middleware(request: Request, call_next):
    
    # 跳过公开路径
    is_excluded = any(
        request.url.path.startswith(_path.rstrip('/')) or
        request.url.path == _path or
        (request.url.path + '/').startswith(_path) or
        (request.url.path.rstrip('/') + '/').startswith(_path)
        for _path in EXCLUDED_PATHS
    )
    
    if is_excluded:
        return await call_next(request)


    # 提取Token
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "code": 401,
                "message": "Missing or invalid authentication token",
                "data": None
            }
        )

    token = auth_header.split(" ")[1]

    # 验证Token
    try:
        user = await get_current_user(token)  # 替换为实际的验证逻辑
        request.state.user = user  # 将用户信息存入请求上下文
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "code": e.status_code,
                "message": e.detail,
                "data": None
            }
        )

    return await call_next(request)
