# LangGraph Project Documentation

## Overview
This project demonstrates various implementations using LangGraph, a framework for building stateful applications with LLMs. It includes multiple examples ranging from basic operations to complex RAG implementations.

## Project Structure

### Graph Examples
1. **helloworld_agent.ipynb**
   - Basic implementation of LangGraph
   - Demonstrates how to create a simple greeting agent
   - Shows fundamental graph structure with a single node

2. **operations.ipynb**
   - Performs mathematical operations on lists
   - Supports addition and multiplication
   - Shows how to handle multiple inputs and operations

3. **multiple_inputs.ipynb**
   - Demonstrates handling multiple input values
   - Calculates total sum of input values
   - Shows state management with multiple inputs

4. **conditional_edge.ipynb**
   - Implements conditional routing in graph
   - Shows addition and subtraction operations
   - Demonstrates decision-making in graph structure

### Agent Implementations

1. **RAG_Agent.py**
   - Retrieval-Augmented Generation implementation
   - Uses PDF document for knowledge base
   - Implements vector store using Chroma
   - Uses HuggingFace embeddings

2. **Drafter.py**
   - Document manipulation agent
   - Supports document updates and saving
   - Implements tool-based document management

3. **ReAct.py**
   - Implements ReAct pattern for arithmetic operations
   - Uses multiple tools for calculations
   - Shows sequential operation processing

4. **memory_agent.py**
   - Implements conversation memory
   - Maintains chat history
   - Logs conversations to file

5. **agent_bot.py**
   - Simple conversational AI implementation
   - Uses Ollama for chat responses
   - Clean terminal interface

### Exercises
1. **add_edge_p.ipynb**
   - Practice exercise for edge addition
   - Demonstrates multi-node graph structure
   - Shows basic conversation flow

2. **compliment_agenttt.ipynb**
   - Simple compliment generator
   - Shows basic node implementation
   - Demonstrates string manipulation

## Key Features of LangGraph
- State Management
- Graph-based Flow Control
- Tool Integration
- Conditional Routing
- Memory Management
- Multiple Input Handling

## Setup and Requirements
1. Install required packages:
   ```bash
   pip install langgraph
   pip install IPython
   pip install langchain-community
   pip install langchain-core
   ```

2. For RAG implementation:
   - Requires PDF documents
   - HuggingFace embeddings
   - Chroma vector store

## Usage
Each notebook and Python file can be run independently. The examples progress from basic to more complex implementations of LangGraph functionality.

## Notes
- Make sure to have proper environment variables set up when required
- Some implementations require specific model availability (e.g., Mistral via Ollama)
- Vector store operations require sufficient disk space
