from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama  # U
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b    

@tool
def multiply(a: int, b: int) -> int:    
    """Multiply two numbers."""
    return a * b

tools = [add, subtract, multiply]
model_name = os.getenv("OLLAMA_MODEL", "mistral")
model = ChatOllama(model=model_name, temperature=0).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_promt = SystemMessage(
        content = """You are my AI assistant. For ANY arithmetic operation, you MUST use the provided tools.
        Do not calculate results directly. Use a separate tool call for each operation.
        Process operations sequentially, one at a time.
        After using tools, summarize the results in a clear format."""
    )
    response = model.invoke([system_promt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1] 
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)


graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        #edge: node
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


# inputs = {"messages": [("user", "Add 10 + 20 and then multiply the result by 5.Also tell me a astronomy joke please." )]}
inputs = {"messages": [("user", "Add 10 + 20. then multiply with 5.now tell me a astronomy joke please." )]}
print_stream(app.stream(inputs, stream_mode="values"))
