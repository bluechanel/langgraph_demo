from dotenv import load_dotenv
# 加载环境变量
load_dotenv(verbose=True)
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.graph import START, END
from state_schema import SearchQueryList, Reflection, OverallState,QueryGenerationState,ReflectionState,WebSearchState
from prompts import get_current_date,query_writer_instructions,reflection_instructions,answer_instructions,web_searcher_prompt

# 此处定义你自己的模型
llm = ChatOpenAI(model="qwen3_32")
# 此处定义搜索工具
tool = TavilySearch(max_results=2)

# Nodes
def generate_query(state: OverallState) -> QueryGenerationState:
    """分解用户问题"""
    structured_llm = llm.with_structured_output(SearchQueryList)
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=state["topic"]
    )
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: OverallState):
    """ """
    return [
        Send("web_research", {"search_query": search_query})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState) -> OverallState:
    """创建一个搜索Agent"""
    formatted_prompt = web_searcher_prompt.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )
    agent = create_react_agent(model=llm,tools=[tool],)
    response = agent.invoke({"messages": [{"role": "user", "content": formatted_prompt}]})
    return {"web_research_result": [response["messages"][-1].content]}


def reflection(state: OverallState) -> ReflectionState:
    """反思节点，分析摘要内容，并确定是否需要进一步搜索"""
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=state["topic"],
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }

def evaluate_research(
    state: OverallState,
) -> OverallState:
    """路由函数，非节点"""
    max_research_loops = state.get("max_research_loops")

    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send("web_research",{"search_query": follow_up_query,},)
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]

def finalize_answer(state: OverallState):
    """生成最终回答"""
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=state["topic"],
        summaries="\n---\n\n".join(state["web_research_result"]),
    )
    result = llm.invoke(formatted_prompt)
    return {"final_answer": result.content,}


grpah = StateGraph(OverallState)
grpah.add_node("generate_query", generate_query)
grpah.add_node("web_research", web_research)
grpah.add_node("reflection", reflection)
grpah.add_node("finalize_answer", finalize_answer)
grpah.add_edge(START, "generate_query")
grpah.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
grpah.add_edge("web_research", "reflection")
grpah.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
grpah.add_edge("finalize_answer", END)
app = grpah.compile()