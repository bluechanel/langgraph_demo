from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from pydantic import SecretStr
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr
from dotenv import load_dotenv
# 加载环境变量
load_dotenv(verbose=True)

# 此处定义你自己的模型
llm = ChatOpenAI(model="qwen3_32")

# 定义工具
tool = TavilySearch(max_results=2)
tools = [tool]
# 工具绑定到模型
llm_with_tools = llm.bind_tools(tools)

# 定义图状态
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 此处维护完整的消息历史

graph = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
# 使用LangGraph提供的工具节点
tool_node = ToolNode(tools=tools)

graph.add_node("chatbot", chatbot)
# 添加工具节点
graph.add_node("tools", tool_node)
# 添加工具 条件分支
graph.add_conditional_edges(
    "chatbot",
    tools_condition,
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