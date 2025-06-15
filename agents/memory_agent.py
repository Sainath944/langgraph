# here the main thing is to create a memoryy for the agent and gonna use different types of messages like the ai messaes ad the human message
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os

model_name = os.getenv("OLLAMA_MODEL", "mistral")

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatOllama(model=model_name)


def process(state: AgentState) -> AgentState:
    """this node will do solve the request input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI Response:{response.content}\n")

    print("------------\n")
    print("current state:", state)
    print("------------\n")
    return state


# LangGraph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


conversation_history = []

user_input = input("You: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})

    # print(f"\nAI Response:\n{result['messages'][-1].content}\n")
    conversation_history = result["messages"]
    user_input = input("You: ")
print("\n👋 Goodbye! See you next time.")


with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("\nEnd of Conversation Log.\n")

print("Your conversation has been logged to 'logging.txt'.")
# This script demonstrates a simple conversational AI agent using the Mistral model via Ollama, integrated with LangGraph for managing dialogue state. It loads environment settings using .env, processes user input through a local model, and outputs structured responses in a clean terminal interface. Ideal for understanding how to combine local LLMs with stateful agent logic in Python. 
