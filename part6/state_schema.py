from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypedDict, List
from typing_extensions import Annotated
import operator
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="用于 Web 研究的搜索查询列表。"
    )
    rationale: str = Field(
        description="简要解释为什么这些查询与研究主题相关。"
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="提供的摘要是否足以回答用户的问题。"
    )
    knowledge_gap: str = Field(
        description="对缺少或需要澄清的信息的描述。"
    )
    follow_up_queries: List[str] = Field(
        description="解决知识差距的后续查询列表。"
    )



class OverallState(TypedDict):
    topic: str # 研究主题
    search_query: Annotated[list, operator.add] # 搜索列表
    web_research_result: Annotated[list, operator.add] # 搜索摘要结果
    research_loop_count: int # 当前循环次数
    max_research_loops: int # 最大循环次数
    is_sufficient: bool # 是否足够研究
    knowledge_gap: str # 当前搜索后，还缺少的信息
    follow_up_queries: List[str] # 接下来需要搜索的问题
    final_answer: str


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: List[str]
    research_loop_count: int
    number_of_ran_queries: int


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    search_query: list[Query]


class WebSearchState(TypedDict):
    search_query: str