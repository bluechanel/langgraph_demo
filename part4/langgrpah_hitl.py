from dotenv import load_dotenv
# 加载环境变量
load_dotenv(verbose=True)
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
memory = InMemorySaver()
# 此处定义你自己的模型
llm = ChatOpenAI(model="qwen3_32")

# 定义人机交互工具
@tool
def human_assistance(query: str) -> str:
    """请求人类帮助，模型在不确定时会调用此工具"""
    human_response = interrupt({"query": query})
    return human_response["data"]

tools = [human_assistance]
llm_with_tools = llm.bind_tools(tools)

# 定义图状态
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 此处维护完整的消息历史

graph = StateGraph(State)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
tool_node = ToolNode(tools=tools)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph.add_edge("tools", "chatbot")
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    thread_id = "multi_tool_hitl_demo"
    pending_interrupt = False
    # 修改此处对话代码，区分普通对话和模型请求人类帮助
    while True:
        if not pending_interrupt:
            user_input = input("👨‍💻: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break
            response = app.invoke(
                {"messages": {"role": "user", "content": user_input}},
                config={"configurable": {"thread_id": thread_id}},
            )
        else:
            # 人类输入，恢复中断
            human_input = input("👨‍💻(人工介入): ")
            human_command = Command(resume={"data": human_input})
            response = app.invoke(
                human_command,
                config={"configurable": {"thread_id": thread_id}},
            )

        # 判断中断还是正常输出
        if "__interrupt__" in response:
            pending_interrupt = True
            query = response["__interrupt__"][0].value["query"]
            print(f"🤖 <human_assistance> 请求人工帮助: {query}")
        else:
            pending_interrupt = False
            print(f'🤖: {response["messages"][-1].content}')