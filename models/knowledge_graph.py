# -*- coding: UTF-8 -*-
from sqlmodel import SQLModel, Field as SqlField


class Wnode(SQLModel, table=True):
    id: int = SqlField(default=None, primary_key=True, description="id")
    nodename: str = SqlField(default="", description="节点名", max_length=100)
    type: str = SqlField(default="", description="类型", max_length=100)


class Qa(SQLModel, table=True):
    id: int = SqlField(default=None, primary_key=True, description="id")
    question: str = SqlField(default="", description="问题", max_length=1000)
    answer: str = SqlField(default="", description="答案", max_length=1000)
