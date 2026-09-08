import os
from dotenv import load_dotenv
from pathlib import Path

# 加载当前目录下的 .env（避免被 IDE 的工作目录影响）
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

# 数据库配置
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "your_password")
MYSQL_SERVER = os.getenv("MYSQL_SERVER", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "tin_smelte")

# JWT 配置
SECRET_KEY = os.getenv("SECRET_KEY", "U3JmP4gX2k7vwZVoYh7UEWcc1DEXFJ61")  # 生产环境应使用环境变量
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Neo4j 配置
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")

# LLM 配置（Ollama）
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2:7b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
KG_WENDA_USE_LLM = os.getenv("KG_WENDA_USE_LLM", "true").lower() in {"1", "true", "yes"}  # 默认启用 LLM
