# This script demonstrates a simple conversational AI agent using the Mistral model via Ollama, integrated with LangGraph for managing dialogue state. It loads environment settings using .env, processes user input through a local model, and outputs structured responses in a clean terminal interface. Ideal for understanding how to combine local LLMs with stateful agent logic in Python.
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

# Load model name from .env
load_dotenv()
model_name = os.getenv("OLLAMA_MODEL", "mistral")

# Define state


class AgentState(TypedDict):
    messages: List[HumanMessage]


# Initialize model
llm = ChatOllama(model=model_name)

# Node logic


def process(state: AgentState) -> AgentState:
    print("\nThinking...\n")
    response = llm.invoke(state["messages"])
    print("AI Response:")
    print("------------")
    print(response.content)
    print("------------\n")
    return state


# LangGraph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Intro
print("=" * 50)
print("        🤖 Chat Agent Powered by Ollama")
print(f"        Model in Use: {model_name}")
print("        Type 'exit' to quit the chat.")
print("=" * 50)

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("\n👋 Goodbye! See you next time.")
        break

    agent.invoke({"messages": [HumanMessage(content=user_input)]})
