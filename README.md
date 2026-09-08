# tin-smelte-backend｜锡冶炼工业流程智能辅助生产系统（后端）

为 tin-smelte 前端提供业务 API，覆盖账号管理、知识图谱、知识抽取与问答、锡品位识别、温度监测、工艺优化、仓储物流和文件共享等功能。

## 一、项目架构

### 1. 服务组成

主入口是 main.py，负责注册路由、配置跨域、创建数据库表、启动 TCP 遥测接收和预热参数优化模块。项目采用单个 FastAPI 应用配合业务路由的结构，部分实时状态和任务保存在进程内存中，本地运行采用单进程即可。

### 2. 技术栈与目录

| 层次 | 技术 / 用途 |
| --- | --- |
| Web API | FastAPI、Uvicorn、Pydantic |
| 关系数据 | SQLModel、SQLAlchemy、PyMySQL；支持配置 SQLite |
| 账号与令牌 | PyJWT、Passlib、Argon2 |
| 知识图谱 | Neo4j、py2neo、jieba |
| 大模型调用 | Ollama 的兼容接口、本地模型 |
| 图像与模型推理 | PyTorch、torchvision、Pillow、OpenCV |
| 工艺优化与数据分析 | NumPy、Pandas、scikit-learn、Gym、XGBoost |
| 文档解析 | PyMuPDF、python-docx |

~~~text
tin-smelte-backend/
├── main.py                      # FastAPI 入口与启动生命周期
├── config.py                    # .env 加载及数据库、图谱、模型服务配置
├── database.py                  # 数据库引擎、会话和自动建表
├── requirements.txt             # Python 依赖清单
├── routers/                     # 按功能拆分的 API
├── models/                      # SQLModel 数据表与数据结构
├── middlewares/                 # 响应封装与认证中间件
├── utils/                       # 认证等工具
├── parameter_optimization/      # 熔炼优化环境、模型、标准化器和 Excel 数据
├── best_model.pth               # 锡品位识别权重
├── data/                        # 业务数据、轨迹文件；可存放 SQLite 数据库
├── dataset/                     # 本地数据集资源
├── uploads/                     # 上传文件与元数据
├── docs/                        # 共享文档资源
├── warehouse_data/              # 仓储布局与状态 JSON
├── robotic-warehouse/           # 仓储仿真相关代码，独立于 API 启动入口
├── scripts/                     # 图数据库检查等辅助脚本
├── Dockerfile                   # 容器构建参考配置
└── docker-compose.yml           # 多服务部署参考配置
~~~

## 二、运行环境与使用方法

### 1. 环境要求

| 项目 | 说明 |
| --- | --- |
| Python | 建议使用 3.12；当前 NumPy、SciPy 等锁定依赖要求至少 3.11，旧文档中的 3.8+ 不适用 |
| 关系数据库 | 可用 MySQL，或通过 DATABASE_URL 使用 SQLite 完成本地基础运行 |
| 识别模型 | 根目录必须存在与当前三分类 ResNet50 匹配的 best_model.pth |
| Neo4j | 知识图谱查询、管理和图谱问答需要 |
| Ollama | 大模型三元组抽取和增强问答需要，默认模型为 qwen2:7b |
| GPU | 图像识别可使用 CPU；部分优化权重的设备兼容性取决于训练和保存方式 |
| 端口 | HTTP 8001、TCP 遥测 8080；Neo4j Bolt 默认 7687、Ollama 默认 11434 |

Redis、PostgreSQL、RabbitMQ、Prometheus 和 Grafana 出现在已有 Compose 参考文件中，但不属于本 README 基础启动流程的必需服务。

### 2. 创建环境并安装依赖

以下命令均在项目根目录执行。

Windows PowerShell：

~~~powershell
cd tin-smelte-backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install gym xgboost
python -m pip check
~~~

Linux / macOS：

~~~bash
cd tin-smelte-backend
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install gym xgboost
python -m pip check
~~~

[parameter_optimization/env.py](parameter_optimization/env.py) 在应用导入时使用 gym 和 xgboost，但当前 requirements.txt 尚未列出这两个包，因此需要额外安装。它们的版本尚未锁定；恢复既有优化模型时，应使用与训练环境兼容的版本。

如果 PowerShell 不允许激活脚本，可直接使用 .\.venv\Scripts\python.exe 替代后续命令中的 python。

### 3. 配置数据库与环境变量

在后端根目录新建 **.env**。下面是本地基础运行的示例，替换其中的占位值：

~~~dotenv
# 本地数据库；路径相对于启动命令所在目录
DATABASE_URL=sqlite:///./data/monitoring.db

# JWT 配置
SECRET_KEY=replace-with-a-random-local-secret
ACCESS_TOKEN_EXPIRE_MINUTES=60

# 图谱功能需要
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=replace-with-your-neo4j-password

# 大模型抽取和增强问答需要
LLM_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_HOST=http://127.0.0.1:11434
LLM_MODEL=qwen2:7b
KG_WENDA_USE_LLM=true

# 无设备时的监测演示开关；接入设备时设置 false
SIMULATION_MODE=true
RTM_BASE_URL=http://127.0.0.1:8001
~~~

确保数据库父目录存在：

~~~bash
python -c "from pathlib import Path; Path('data').mkdir(exist_ok=True)"
~~~

[config.py](config.py) 会加载根目录 .env，且已有进程环境变量优先。下文启动命令另外使用 --env-file .env，确保在导入实时监测等模块之前就加载 SIMULATION_MODE 等变量。JWT 算法在代码中固定为 HS256。

**使用 MySQL：** 删除或注释上述 DATABASE_URL，配置以下变量；同时确认启动终端没有遗留的 DATABASE_URL 环境变量，因为它的优先级更高。

~~~dotenv
MYSQL_SERVER=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=tin_smelte
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
~~~

先在 MySQL 中创建数据库并为所用账号授予访问权限：

~~~sql
CREATE DATABASE IF NOT EXISTS tin_smelte CHARACTER SET utf8mb4;
~~~

应用启动时调用 SQLModel 的 create_all 创建缺失的表，不会自动创建 MySQL 数据库，也不会迁移已有表结构。SQLite 适合本地基础功能验证；切换数据库不会自动迁移账号或业务数据。

### 4. 准备模型和数据文件

**启动必需：根目录的 best_model.pth。** [routers/smelte_predict.py](routers/smelte_predict.py) 在导入时加载该文件，因此缺失或不匹配会导致整个应用启动失败。模型为三分类 ResNet50，类别顺序为“含锡 35%～50%”“含锡 35% 以下”“含锡 50% 以上”。

**熔炼参数优化所需：** 保持下列文件及路径完整，模型、标准化器和数据应来自同一套训练资源：

~~~text
parameter_optimization/
├── 2024years.xlsx
├── final_actor.pth
├── state_predictor2.pkl
├── state_scaler_X2.pkl
├── state_scaler_y2.pkl
├── xgboost_model.json
├── scaler_X.pkl
├── scaler_y.pkl
├── sn_feature_names.pkl
└── saved_models/
    ├── co_predictor.pth
    └── 2scaler.pkl
~~~

启动时会预热优化模块；缺少这些优化资源时可能记录预热失败，相关接口将无法正常返回优化结果。Excel 字段需与 parameter_optimization/env.py 中的状态列和动作列一致。

**精炼轨迹回放所需：** data/trajectory_run_3_start_2345 (3).csv，需包含 timestep、temp_1～temp_5、action_1～action_5 和 phase 列。

程序源码不会自动生成上述训练权重和工艺数据，取得项目后应先检查资源是否齐全。

### 5. 配置知识图谱与本地大模型（按需）

启动 Neo4j，将 .env 中的地址、用户名和密码设置为实际连接信息。新图数据库需要通过前端“知识图谱管理”添加节点和关系，或通过“知识抽取”生成三元组后保存。连接成功不代表已经存在业务图谱数据。

安装并启动 Ollama，准备与 LLM_MODEL 一致的模型。沿用当前默认配置时，可执行：

~~~bash
ollama pull qwen2:7b
ollama list
~~~

如果 Ollama 服务尚未运行，在单独终端执行 ollama serve。替换模型时同步修改 LLM_MODEL，然后重启后端。KG_WENDA_USE_LLM=false 可关闭图谱问答中的大模型增强，但不会关闭三元组抽取对模型服务的调用。

语音抽取属于额外能力，当前依赖清单没有包含其完整依赖。需要使用时补充：

~~~bash
python -m pip install SpeechRecognition pydub
~~~

非 WAV 音频转换还需安装 FFmpeg 并加入 PATH；代码另有 Whisper、Vosk 识别分支，使用对应分支时需单独安装包和模型，Vosk 模型路径通过 VOSK_MODEL_PATH 配置。

### 6. 启动与验证

保持当前目录为后端根目录，在已激活的虚拟环境中运行：

~~~bash
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --env-file .env --timeout-keep-alive 600
~~~

开发时可追加 --reload；重载会重新加载模型并尝试启动遥测。python main.py 同样使用 8001，但排查环境变量加载顺序时优先使用上面的显式命令。

| 地址 | 用途 |
| --- | --- |
| http://localhost:8001/health | 基础 HTTP 健康检查 |
| http://localhost:8001/docs | Swagger UI，请求参数与响应结构 |
| http://localhost:8001/redoc | ReDoc 接口文档 |
| http://localhost:8001/routes | 查看实际注册的路由 |
| http://localhost:8001/api/technology/health | 图像识别模型状态 |
| http://localhost:8001/telemetry/status | TCP 遥测接收状态 |

/health 成功只说明应用可以响应，不会验证所有图谱、模型和设备功能。首次启动需要加载权重并预热优化模块，耗时取决于本机资源。

前端默认使用 http://localhost:5666，其 /api 代理指向后端 8001。后端当前 CORS 允许的来源是 http://localhost:5666，浏览器直接跨域访问其他来源时需调整 [main.py](main.py) 中的配置。

### 7. 首次注册与接口调用

前端注册页可创建普通用户或管理员。也可在 Swagger UI 调用 POST /api/auth/register，请求体示例：

~~~json
{
  "username": "demo_user",
  "realName": "演示用户",
  "password": "TinDemo#2026",
  "role": "user"
}
~~~

该账号仅为请求示例。注册成功后，调用 POST /api/auth/login，提交相同的 username、password 和 role；从返回数据中取得 access_token，在需要令牌的请求中设置：

~~~http
Authorization: Bearer <access_token>
~~~

注册密码需为 8～16 位，同时包含字母、数字和符号。部分接口使用 CustomRoute 包装响应，另一些直接返回业务结构，具体格式以各接口实际响应为准。

当前认证实现面向演示：全局认证中间件在 main.py 中被注释，注册允许选择管理员，重置密码没有邮件或验证码验证。公开部署前需要完善这些现有认证逻辑。

### 8. 接入温度数据

无设备时可通过启动前加载 SIMULATION_MODE=true 查看模拟监测数据。接入设备时改为 false 并重启，然后选择以下方式：

- **HTTP 上报：** 调用 POST /api/real-time-monitoring/update-data 或 POST /api/real-time-monitoring/factory-data，支持单点及工厂数据接入。
- **TCP 上报：** 将设备连接到后端机器的 8080 端口，发送以换行符结尾的文本，如下所示（末尾需发送真实换行）：

  ~~~text
  uart1 data: 749
  ~~~

程序提取最后一个数值；包含 uart4 的行作为顶吹炉温度，其他行作为结晶机温度。

TCP 接收线程通过 RTM_BASE_URL 转发到实时监测 HTTP 接口，默认目标为本机 8001。变更 HTTP 端口时，应同步修改该变量和前端代理。当前启动逻辑默认监听 TCP 8080，修改监听地址或端口需调整启动配置或调用遥测管理接口。

实时数据还可通过 ws://localhost:8001/api/real-time-monitoring/ws 订阅。实时历史缓存在进程内存中，不能作为重启后仍可查询的持久化生产历史库。

### 9. 常见问题与部署配置

| 现象 | 排查方法 |
| --- | --- |
| No module named gym/xgboost | 在当前虚拟环境补装依赖，确认启动所用 Python 一致 |
| best_model.pth 不存在或模型结构不匹配 | 从后端根目录启动，检查权重和三分类 ResNet50 结构 |
| 数据库连接或建表失败 | 检查 DATABASE_URL 优先级、MySQL 数据库及权限，或 SQLite 父目录 |
| 优化预热失败 | 检查完整模型资源、Excel 列名及训练时的库版本；部分权重加载未指定 CPU 映射，CPU 环境需注意权重设备兼容性 |
| 图谱为空或连接失败 | 检查 Neo4j 服务、认证信息和业务节点、关系是否已经导入 |
| 大模型请求失败或超时 | 检查 Ollama 服务地址、模型名称及本机内存，必要时调整模型与调用超时 |
| 模拟模式未生效 | 使用带 --env-file .env 的启动命令，修改变量后重启 |
| /api/data-services/... 返回 404 | 当前真实路由是 /data-services/...；数据分析同样使用 /data-analysis/...，没有 /api 前缀 |
| 温度不更新 | 检查 TCP 8080、换行结束符、RTM_BASE_URL 和实际数据时间戳 |

仓库中的 Dockerfile、docker-compose.yml 和部分旧启动脚本使用 8000，与本文的前后端联调端口 8001 不同。采用容器配置前，需要统一 HTTP 端口、健康检查、遥测回传地址与前端代理，补齐上文说明的依赖和模型挂载。容器中 127.0.0.1 指向容器自身，数据库、Neo4j、Ollama 地址也需要按部署网络调整。

## 三、功能说明

下表列出主要接口分组；完整方法、字段和返回值以启动后的 /docs 为准。

| 模块 | 主要接口路径 | 功能与当前实现 |
| --- | --- | --- |
| 账号管理 | /api/auth/*、/api/user/info | 注册、登录、用户与管理员查询、账号状态管理、资料修改、密码重置 |
| 锡品位识别 | /api/technology/recognize、/api/smelte/predict | 分别接收 Base64 图片或上传文件，使用 ResNet50 返回含锡类别与置信度 |
| 工艺流程规划 | /api/technology/process_query、/api/technology/ai_chat | 按含锡品位查询流程，并提供辅助问答接口 |
| 知识图谱 | /api/graph/* | 图谱查询、布局保存、节点与关系维护、关键流程查询及图谱问答 |
| 知识抽取 | /api/triple_construction/* | 从文本、文档或语音内容提取三元组，按类别写入图谱 |
| 独立问答示例 | /api/qa/* | 单条、批量问答及历史入口，目前采用内置关键词和示例知识；前端智能问答页面实际调用 /api/graph/wenda/{question} |
| 实时监测 | /api/real-time-monitoring/* | 当前数据、内存历史、单点与批量更新、工厂上报、数据质量、系统状态和 WebSocket 推送 |
| 温度接入与发送 | /telemetry/*、/api/temperature-sender/*、/api/temperature-config/* | TCP 遥测管理、模拟温度 UDP 发送、温区及波动参数配置 |
| 熔炼参数优化 | /api/param-opt/* | 状态查询、动作建议、结果刷新；基于本地数据和模型，也注册了 /param-opt/* 别名 |
| 精炼轨迹回放 | /api/refining-optimization/* | 读取既有 CSV 返回温度和动作轨迹，当前接口不执行在线训练 |
| 结晶机温控 | /api/crystallizer/* | 温区配置、实时温度、启动停止、运行状态和传感器信息，当前包含内存模拟实现 |
| 仓储物流 | /api/warehouse/* | 仓库信息、货架更新、矩阵查询、重置、调度方案生成与执行，使用本地 JSON 存储 |
| 物料出入库 | /api/inbound、/api/outbound | 出入库记录分页查询、详情、新增、更新和删除 |
| 文件共享 | /api/files/* | 上传、列表、搜索、下载和删除，文件及元数据保存在本地目录 |
| 用户反馈 | /api/feedback/*、/api/admin/feedback* | 提交反馈、查看、状态更新、回复和删除 |
| 数据分析 | /data-analysis/* | 统计、趋势、相关性分析和数据概览，目前使用生成的示例数据 |
| 数据服务 | /data-services/* | 数据处理、验证、转换、导入导出及规则查询 |

推荐使用顺序是：完成基础启动与账号注册，再通过前端验证文件或仓储功能；随后准备 Neo4j 图谱与 Ollama 模型，体验知识抽取、问答和流程规划；最后按需接入设备并准备工艺优化资源。
