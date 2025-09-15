from dotenv import load_dotenv
# 加载环境变量
load_dotenv(verbose=True)
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition
from mem0 import Memory
from langchain_openai import OpenAIEmbeddings

openai_embeddings = OpenAIEmbeddings(
    model="gte-large-zh",
    check_embedding_ctx_length=False
)
# 此处定义你自己的模型
llm = ChatOpenAI(model="qwen3_32")

# 定义工具
tool = TavilySearch(max_results=2)
tools = [tool]
# 工具绑定到模型
llm_with_tools = llm.bind_tools(tools)
# 创建Mem0记忆配置
config = {
    "vector_store": {
        "provider": "milvus",
        "config": {
            "collection_name": "mem0",
            "embedding_model_dims": "1024",
            "url": os.getenv("MILVUS_URL"),
            "token": os.getenv("MILVUS_TOKEN"),
        },
    },
    "embedder": {
        "provider": "langchain",
        "config": {
            "model": openai_embeddings,
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "qwen3_32",
            "temperature": 0.2,
            "max_tokens": 20000,
        },
    },
}
memory = Memory.from_config(config)

# 定义图状态
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 此处维护完整的消息历史
    mem0_user_id: str # 增加用户id

graph = StateGraph(State)

def chatbot(state: State):
    messages = state["messages"]
    user_id = state["mem0_user_id"]

    # 检索记忆
    memories = memory.search(messages[-1].content, user_id=user_id)

    context = "Relevant information from previous conversations:\n"
    for m in memories.get("results", []):
        context += f"- {m['memory']}\n"

    system_message = {"role":"system", "content": f"""You are a helpful customer support assistant. Use the provided context to personalize your responses and remember user preferences and past interactions.
    {context}"""}

    full_messages = [system_message] + messages
    response = llm_with_tools.invoke(full_messages)

    # 存储记忆
    memory.add(
        f"User: {messages[-1].content}\nAssistant: {response.content}", user_id=user_id
    )
    return {"messages": [response]}

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
    while True:
        user_input = input("👨‍💻: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break
        response = app.invoke({"messages": {"role": "user", "content": user_input}, "mem0_user_id":"mem0_wiley"})
        messages = response["messages"]
        print(f'🤖: {response["messages"][-1].content}')