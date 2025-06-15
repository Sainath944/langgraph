from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama  # U
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os
load_dotenv()

#global variable to store the document content
document_content = ""   

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] 


@tool
def update(content: str) -> str:
    """updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has updated successfully! The current content is:\n{document_content}"

@tool 
def save(filename: str) -> str:
    """
    saves the current document content to a file with the given filename and finishes the process
    
    args:
        filename (Str): the name of the file to save the document content to a text file
    """

    global document_content
    if not filename.endswith('.txt'):
        filename += '.txt'

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\nDocument saved successfully to {filename}.\n")
        return f"Document saved successfully to {filename}."
    except Exception as e:
        return f"An error occurred while saving the document: {str(e)}"
    

tools = [update, save]
model_name = os.getenv("OLLAMA_MODEL", "mistral")
model = ChatOllama(model=model_name, temperature=0).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a Document.What would you like to create?."
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\n What would you like to do with the document?")
        print(f"\n User Input: {user_input}\n")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)


    print(f"AI Response: {response.content}\n")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}\n")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState):
    """determine if we should continue or end the conversation"""
    messages = state["messages"]

    if not messages:
        return "continue"
    
    #this checks the most recent toolmessage whether it resulting from the save or not 
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"




def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools=tools))


graph.set_entry_point("agent")

graph.add_edge("agent", "tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()   


def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()