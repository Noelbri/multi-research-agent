#!/usr/bin/env python3

# Simple test to verify the agent setup
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_experimental.tools import PythonREPLTool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_groq import ChatGroq
    from pydantic import BaseModel, Field
    from typing import Literal
    from langchain_core.messages import HumanMessage
    from dotenv import load_dotenv
    import functools
    import operator
    from typing import Sequence
    from typing_extensions import TypedDict
    from langchain_core.messages import BaseMessage
    from langgraph.graph import END, StateGraph, START
    from langgraph.prebuilt import create_react_agent
    
    print("✅ All imports successful!")
    
    # Test environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if groq_api_key:
        print("✅ GROQ_API_KEY found")
    else:
        print("❌ GROQ_API_KEY not found")
        
    if tavily_api_key:
        print("✅ TAVILY_API_KEY found")
    else:
        print("❌ TAVILY_API_KEY not found")
        
    print("\n🎉 Agent setup test completed successfully!")
    print("You can now run 'python3 agent.py' to start the multi-agent system.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
