from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from routers import technology, data_analysis, real_time_monitoring, data_services
from routers.telemetry import router as telemetry_router
from routers.param_optimization import router as param_opt_router
from routers.param_optimization import bootstrap_on_startup
from routers.param_optimization import get_state as paramopt_state
from routers.param_optimization import optimize as paramopt_optimize
from routers.param_optimization import refresh as paramopt_refresh
from routers.telemetry import telemetry_start
from routers.temperature_sender import router as temp_sender_router
from routers.temperature_config import router as temp_config_router
from database import create_db_and_tables
from middlewares.auth_middleware import auth_middleware
from routers.auth import router as auth_router
from routers.inbound import router as inbound_router
from routers.outbound import router as outbound_router
from routers.smelte_predict import router as smelte_decide_router
from routers.knowledge_graph import router as know_router
from routers.file_sharing import router as file_router
from routers.feedback import router as feedback_router
from routers.admin_feedback import router as admin_feedback_router
from routers.triple_construction import router as triple_router
from routers.smart_qa import router as qa_router
from routers.user import router as user_router
from routers.warehouse import router as warehouse_router
from routers.refining_optimization import router as refining_opt_router
from routers.crystallizer_control import router as crystallizer_router

async def startup():
    create_db_and_tables()
    # 启动TCP遥测服务（默认 0.0.0.0:8080，转发到本机 8001 实时监控接口）
    try:
        telemetry_start(host="0.0.0.0", port=8080)
        print("Telemetry TCP server started on 0.0.0.0:8080")
    except Exception as e:
        print(f"Failed to start telemetry TCP server: {e}")
    # 预热参数优化：运行一次 diaoyong2.py 获取初始状态
    try:
        bootstrap_on_startup()
        print("Param optimization bootstrap completed")
    except Exception as e:
        print(f"Param optimization bootstrap failed: {e}")

async def shutdown():
    print("Application shutdown")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()

app = FastAPI(
    title="锡冶炼智能分析系统",
    description="基于机器学习的锡冶炼工艺优化和数据分析系统",
    version="1.0.0",
    lifespan=lifespan
)

# 中间件
app.add_middleware(
    CORSMiddleware,
    # 在开发环境允许前端开发服务器的地址，避免使用通配符 * 与 allow_credentials=True 同时出现
    allow_origins=["http://localhost:5666"],
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
    expose_headers=["X-Custom-Header"]  # 暴露自定义头给浏览器
)
# app.middleware("http")(auth_middleware)  # 临时禁用认证中间件

# 注册所有路由
app.include_router(technology.router, prefix="/api")
app.include_router(data_analysis.router)
app.include_router(real_time_monitoring.router, prefix="/api")
app.include_router(data_services.router)
app.include_router(auth_router, prefix="/api")
app.include_router(inbound_router, prefix="/api")
app.include_router(outbound_router, prefix="/api")
app.include_router(smelte_decide_router, prefix="/api")
app.include_router(know_router, prefix="/api")
app.include_router(file_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")
app.include_router(admin_feedback_router, prefix="/api")
app.include_router(triple_router, prefix="/api")
app.include_router(qa_router, prefix="/api")
app.include_router(user_router, prefix="/api")
app.include_router(warehouse_router, prefix="/api")
app.include_router(refining_opt_router, prefix="/api")
app.include_router(crystallizer_router, prefix="/api")
app.include_router(telemetry_router)
app.include_router(param_opt_router, prefix="/api")
app.include_router(param_opt_router)
app.include_router(temp_sender_router)
app.include_router(temp_config_router)

# 显式注册一次，避免某些环境下 include_router 未生效时导致 404
try:
    existing_paths = {getattr(r, 'path', '') for r in app.routes}
    if '/api/param-opt/state' not in existing_paths:
        app.add_api_route('/api/param-opt/state', paramopt_state, methods=['GET'])
    if '/api/param-opt/optimize' not in existing_paths:
        app.add_api_route('/api/param-opt/optimize', paramopt_optimize, methods=['POST'])
    if '/api/param-opt/refresh' not in existing_paths:
        app.add_api_route('/api/param-opt/refresh', paramopt_refresh, methods=['POST'])
    if '/param-opt/state' not in existing_paths:
        app.add_api_route('/param-opt/state', paramopt_state, methods=['GET'])
    if '/param-opt/optimize' not in existing_paths:
        app.add_api_route('/param-opt/optimize', paramopt_optimize, methods=['POST'])
    if '/param-opt/refresh' not in existing_paths:
        app.add_api_route('/param-opt/refresh', paramopt_refresh, methods=['POST'])
except Exception as _e:
    print('[param-opt] add_api_route fallback failed:', _e)

@app.get("/")
async def root():
    return {
        "message": "锡冶炼智能分析系统API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "锡冶炼智能分析系统"
    }

@app.get("/routes")
async def list_routes():
    paths = []
    for r in app.routes:
        try:
            paths.append(getattr(r, 'path', str(r)))
        except Exception:
            continue
    return {"routes": paths}

if __name__ == "__main__":
    # 监听 0.0.0.0，确保同时支持 IPv4 本地回环 (127.0.0.1) 和容器/外部访问
    # 设置超时时间为600秒（10分钟），以支持LLM长时间处理
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        timeout_keep_alive=600,  # 保持连接超时
        timeout_graceful_shutdown=600  # 优雅关闭超时
    )
