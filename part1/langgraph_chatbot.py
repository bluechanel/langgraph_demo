from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import SecretStr
from dotenv import load_dotenv
# 加载环境变量
load_dotenv(verbose=True)

# 此处定义你自己的模型
llm = ChatOpenAI(model="qwen3_32")

# 定义图状态
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 此处维护完整的消息历史

graph = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

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