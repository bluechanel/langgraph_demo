from dotenv import load_dotenv
# 加载环境变量
load_dotenv(verbose=True)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
# 此处定义你自己的模型
llm = ChatOpenAI(model="qwen3_32")
memory = InMemorySaver()
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"已成功预订住宿 {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"成功预订了航班从{from_airport}到{to_airport}."
flight_assistant = create_react_agent(
    model=llm,
    tools=[book_flight],
    prompt="你是航班预订助理",
    name="flight_assistant"
)
hotel_assistant = create_react_agent(
    model=llm,
    tools=[book_hotel],
    prompt="你是酒店预订助理",
    name="hotel_assistant"
)
app = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=llm,
    prompt="你管理一个酒店预订助理和一个航班预订助理。将工作分配给他们，完成用户给出的任务，不要询问更多信息。"
).compile()

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
        for m in response["messages"]:
            print(m.pretty_print())