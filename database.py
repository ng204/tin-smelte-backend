import os
from sqlmodel import SQLModel, create_engine, Session

import config


def _build_engine():
    """根据环境变量/配置构建数据库引擎。
    优先使用 DATABASE_URL；否则使用 MySQL 配置；
    如果检测到 MySQL 主机是 docker 名称（mysql/db）且本机运行，回退到 SQLite。
    """
    # 1) 优先使用 DATABASE_URL
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        if database_url.startswith("sqlite"):
            return create_engine(
                database_url,
                echo=False,
                connect_args={"check_same_thread": False},
            )
        # 非 SQLite，启用连接池参数以支持高并发
        return create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=50,
            max_overflow=50,
            pool_timeout=30,
        )

    # 2) 使用 MySQL 配置
    mysql_server = str(config.MYSQL_SERVER).strip()
    docker_like_hosts = {"mysql", "db"}
    if mysql_server in docker_like_hosts:
        # 回退到本地 SQLite，避免本机直跑时解析不到 docker 网络主机名
        db_path = os.path.join("data", "monitoring.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )

    # 默认使用 MySQL，并启用较大的连接池以提升并发能力
    return create_engine(
        f"mysql+pymysql://{config.MYSQL_USER}:{config.MYSQL_PASSWORD}@{config.MYSQL_SERVER}:{config.MYSQL_PORT}/{config.MYSQL_DATABASE}",
        echo=False,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=50,
        max_overflow=50,
        pool_timeout=30,
    )


# 连接数据库（惰性初始化方便测试替换）
engine = _build_engine()


# 创建数据库表
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
