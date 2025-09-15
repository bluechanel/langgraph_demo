import json
from langchain_core.messages import ToolMessage
from langgraph.constants import END
from typing import Annotated, Type, Optional
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from pydantic import SecretStr, BaseModel, Field
from langchain_core.tools import BaseTool
from tavily import TavilyClient, AsyncTavilyClient
from dotenv import load_dotenv
# 加载环境变量
load_dotenv(verbose=True)

# 自定义工具部分
class TavilySearchInput(BaseModel):
    query: str = Field(description=("搜索查询"))

class TavilySearchTool(BaseTool):
    name: str = "tavily_search"
    description: str = """一个针对全面、准确和可信的结果进行了优化的搜索引擎。
当需要回答有关时事的问题时很有用。
输入应该是搜索查询。"""
    args_schema: Type[BaseModel] = TavilySearchInput
    # return_direct: bool = True

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> int:
        client = TavilyClient()
        search_r = client.search(query=query, max_results=2)
        return search_r

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> int:
        client = AsyncTavilyClient()
        search_r = await client.search(query=query, max_results=2)
        return search_r

# 自定义的工具执行节点
class ToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            print(f'正在执行工具 {tool_call["name"]}，参数 {tool_call["args"]}')
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            print(f'工具{tool_call["name"]}, 执行结果{json.dumps(tool_result, ensure_ascii=False)}')
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# 此处定义你自己的模型
llm = ChatOpenAI(model="qwen3_32")
# 定义工具
tool = TavilySearchTool()
tools = [tool]
# 工具绑定到模型
llm_with_tools = llm.bind_tools(tools)
# 定义图状态
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 此处维护完整的消息历史
graph = StateGraph(State)
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 工具路由
def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# 使用LangGraph提供的工具节点
tool_node = ToolNode(tools=tools)
graph.add_node("chatbot", chatbot)
# 添加工具节点
graph.add_node("tools", tool_node)
# 添加工具条件边
graph.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph.add_edge("tools", "chatbot")
graph.add_edge(START, "chatbot")
app = graph.compile()

if __name__ == "__main__":
    messages = []
    while True:
        user_input = input("👨‍💻: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break
        messages.append({"role": "user", "content": user_input})
        response = app.invoke({"messages": messages})
        messages = response["messages"]
        print(f'🤖: {response["messages"][-1].content}')