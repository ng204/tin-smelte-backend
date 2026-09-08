import json
from fastapi import Request
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse


class CustomRoute(APIRoute):
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request):
            response = await original_route_handler(request)

            if request.url.path == "/api/auth/login-swagger":
                return response

            if isinstance(response, JSONResponse):
                # 处理原始响应内容
                original_content = None
                if response.body:
                    try:
                        decoded_body = response.body.decode("utf-8")
                        original_content = json.loads(decoded_body) if decoded_body.strip() else None
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        original_content = None

                # 判断状态码是否成功（2xx范围）
                is_success = 200 <= response.status_code < 300

                wrapped_content = {
                    "code": response.status_code,
                    "message": "success" if is_success else "error",
                    "data": original_content
                }

                # 过滤掉原始 Content-Length 头
                headers = {
                    k: v for k, v in response.headers.items()
                    if k.lower() != "content-length"
                }

                return JSONResponse(
                    content=wrapped_content,
                    status_code=response.status_code,
                    headers=headers
                )

            return response

        return custom_route_handler
