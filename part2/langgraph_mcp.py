"""
先启动 mcp_server.py
"""
from dotenv import load_dotenv
# 加载环境变量
load_dotenv(verbose=True)
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from pydantic import SecretStr

# 此处定义你自己的模型
llm = ChatOpenAI(model="qwen3_32")
# 配置MCP Server
client = MultiServerMCPClient(
    {
        "search": {
            "url": "http://localhost:8000/mcp/",
            "transport": "streamable_http",
        }
    }
)

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 此处维护完整的消息历史
graph = StateGraph(State)

async def main():
    tools = await client.get_tools()
    # 工具绑定到模型
    llm_with_tools = llm.bind_tools(tools)
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    graph = StateGraph(State)
    graph.add_node(chatbot)
    graph.add_node(ToolNode(tools))
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph.add_edge("tools", "chatbot")
    app = graph.compile()
    messages = []
    while True:
        user_input = input("👨‍💻: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break
        messages.append({"role": "user", "content": user_input})
        response = await app.ainvoke({"messages": messages})
        messages = response["messages"]
        print(f'🤖: {response["messages"][-1].content}')

asyncio.run(main())