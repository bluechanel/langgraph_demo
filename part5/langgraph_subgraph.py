from dotenv import load_dotenv
# 加载环境变量
load_dotenv(verbose=True)
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.constants import END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# 此处定义你自己的模型
llm = ChatOpenAI(model="qwen3_32")
memory = InMemorySaver()
# 定义图状态
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 此处维护完整的消息历史
    mem0_user_id: str
    preference: str
# 子图， 使用与父图同样的状态
child_graph = StateGraph(State)
def summarize_memory(state: State):
    """总结过往的历史聊天"""
    summary_prompt = [
        {
            "role": "system",
            "content": "你是一个总结助手，请用简明扼要的语言总结对话历史。",
        },
        {"role": "user", "content": str(state["messages"])},
    ]
    result = llm.invoke(summary_prompt)
    return {"preference": result.content}
child_graph.add_node("summarize", summarize_memory)
child_graph.add_edge("summarize", END)
child_graph.add_edge(START, "summarize")
child_graph_compile = child_graph.compile()

# 父图
parent_graph = StateGraph(State)
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
parent_graph.add_node("chatbot", chatbot)
# 添加子图节点
parent_graph.add_node("child_graph", child_graph_compile)
parent_graph.add_edge("chatbot", "child_graph")
parent_graph.add_edge("child_graph", END)
parent_graph.add_edge(START, "chatbot")
app = parent_graph.compile(checkpointer=memory)

if __name__ == "__main__":
    while True:
        user_input = input("👨‍💻: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break
        response = app.invoke(
            {"messages": {"role": "user", "content": user_input}},
            config={"configurable": {"thread_id": "1"}},
        )
        messages = response["messages"]
        print(f'🤖: {response["messages"][-1].content}')
        print("--" * 10)
        print(f'🔅历史聊天记录总结: {response["preference"]}')