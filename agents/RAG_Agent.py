# from dotenv import load_dotenv
# import os
# from langgraph.graph import StateGraph, END
# from typing import TypedDict, Annotated, Sequence
# from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
# from operator import add as add_messages
# from langchain_community.chat_models import ChatOllama
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_ollama import ChatOllama  # ✅ NEW IMPORT
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ NEW IMPORT
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

# Load .env variables if any
load_dotenv()

# Initialize Mistral from Ollama
llm = ChatOllama(
    model="mistral", temperature=0
)

# Use HuggingFace sentence-transformers for embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load your PDF
pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF loaded successfully. Total pages: {len(pages)}")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunk the PDF text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
pages_split = text_splitter.split_documents(pages)

# Set up vector store
persist_directory = r"./chroma_store"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"Chroma vector store created.")
except Exception as e:
    print(f"Vector store error: {e}")
    raise

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """
    Retrieves information from the Stock Market PDF using semantic similarity.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant info found in the PDF."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(results)

tools = [retriever_tool]
llm = llm.bind_tools(tools)

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Decide whether to continue the LangGraph cycle
def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0

# System prompt
system_prompt = """
You are an intelligent AI assistant for answering questions about the Stock Market Performance 2024 PDF.
Use ONLY the retriever tool to pull information from the document. Cite the content you retrieve.
Never hallucinate. If you don’t know, just say so.
"""

tools_dict = {tool.name: tool for tool in tools}

# LLM agent
def call_llm(state: AgentState) -> AgentState:
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(messages)
    return {'messages': [response]}

# Tool executor
def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        tool_name = t['name']
        query = t['args'].get('query', '')
        print(f"Invoking Tool: {tool_name} | Query: {query}")
        if tool_name not in tools_dict:
            result = "Invalid tool selected."
        else:
            result = tools_dict[tool_name].invoke(query)
        results.append(ToolMessage(tool_call_id=t['id'], name=tool_name, content=str(result)))
    print("Tools done. Returning to LLM.")
    return {'messages': results}

# Define LangGraph
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

# Compile agent
rag_agent = graph.compile()

# Terminal loop
def running_agent():
    print("\n=== STOCK MARKET PDF QA ===")
    while True:
        user_input = input("\nYour question (or type 'exit'): ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

running_agent()
