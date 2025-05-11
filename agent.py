# -*- coding: utf-8 -*-
"""
LangGraph Agent Definition.
This file contains the setup for the LangChain/LangGraph agent,
including tool definitions, LLM initialization, and workflow compilation.
Uses the standard MessagesState for handling message types.
"""

#import environment file
import os
from dotenv import load_dotenv

# Modules for structuring text
from typing import Annotated
from typing_extensions import TypedDict
from typing import List, Union, Callable
from pydantic import BaseModel, Field

# LangGraph modules for defining graphs
from langgraph.graph import MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Modeules for Messages
from langchain_core.messages import AIMessage, HumanMessage

# Module for setting up OpenAI
from langchain_openai import ChatOpenAI

# Module for setting up Google Gen AI
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai.types import (Tool, GenerateContentConfig, 
    GoogleSearch, ThinkingConfig, ToolCodeExecution)

# Modules for creating ReAct agents with Supervisor architecture
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# Import the modules for saving memory
from langgraph.checkpoint.memory import MemorySaver

from langchain.tools import tool as LangChainTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from prompt import get_enhanced_supervisor_prompt

# Load the .env file
load_dotenv()

def get_workflow_app():
    """
    Initialize and return the LangGraph workflow application.
    This function handles all the setup for the agents and workflow.
    
    Returns:
        The compiled LangGraph workflow application.
    """
    # --- LangChain/LangGraph Setup ---

    # Initialize OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize Gemini
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Warning: GEMINI_API_KEY environment variable not set.")
        print("Please set the GEMINI_API_KEY environment variable in your .env file")
        print("You can get an API key from Google AI Studio: https://makersuite.google.com/")

    # Initialize Tavily Search
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY environment variable not set.")
        print("Please set the TAVILY_API_KEY environment variable in your .env file")

    # Initialize GPT
    gpt = ChatOpenAI(
        model_name="gpt-4.1-mini-2025-04-14",
        temperature=0.2,
        top_p=0.95
    )

    # Initialize Gemini LLM
    gemini = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash-001',  # Same model as main LLM
        google_api_key=gemini_api_key,
        temperature=0.01,
        max_output_tokens=2048,
        model_kwargs={ 
            "tools": [Tool(code_execution=ToolCodeExecution())],
            "tool_config": {"function_calling_config": {"mode": "AUTO"}}
        }
    )

    # initiatilize Tavily Search:
    tavily_search = TavilySearchResults(api_key=tavily_api_key)

    # Define Wikipedia Tool with explicit name
    api_wrapper = WikipediaAPIWrapper(top_k_results=1)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Set the name explicitly to match what the frontend is expecting
    wikipedia_tool.name = "wikipedia_query_run"
    wikipedia_tool.description = "Searches Wikipedia for information about a given topic."

    # Print the tool name for debugging
    print(f"Registered Wikipedia tool with name: {wikipedia_tool.name}")

    search_agent = create_react_agent(
        model=gpt,
        tools=[tavily_search],
        name="search_agent",
        prompt="""
        You are an expert researcher with access to Tavily Web Search.
        Search for the user's query and summarize the results.
        When searching the web: Use the tavily_search_results tool with a specific query parameter.
        1. After receiving tool results, analyze them and provide a clear, concise summary.
        2. Only call a tool once for a query unless you explicitly need more information.
        3. Always provide an actual response when you have enough information.
        """
    )

    code_agent = create_react_agent(
        model=gemini,
        tools=[],
        name="code_agent",
        prompt="""
        You are an expert coder.
        Write and execute code to solve user queries and complex tasks.
        When executing code: Use your built-in code execution capabilities with proper code formatting.
        1. After receiving results, analyze them and provide a clear, concise summary.
        2. Only call a tool once for a query unless you explicitly need more information.
        3. Always provide an actual response when you have enough information.
        """
    )

    wiki_agent = create_react_agent(
        model=gpt,
        tools=[wikipedia_tool],
        name="wiki_agent",
        prompt="""
        You are an expert on Wikipedia.
        Search Wikipedia for the user's query and summarize the results.
        For Wikipedia searches: Use the wikipedia_query_run tool.
        EXAMPLE: "I'll search Wikipedia for that information" followed by using wikipedia_query_run(query="your search term")
        Always use the proper format when calling tools. Do not create invalid tool calls.
        1. After receiving tool results, analyze them and provide a clear, concise summary.
        2. Only call a tool once for a query unless you explicitly need more information.
        3. Always provide an actual response when you have enough information.
        """
    )

    # Create supervisor workflow
    workflow = create_supervisor(
        [search_agent, code_agent, wiki_agent],
        model=gpt,
        prompt=get_enhanced_supervisor_prompt(),
        # output_mode="last_message",
        output_mode="full_history",
        parallel_tool_calls=False
    )

    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory
    )
    
    return app

# Function for testing the workflow with a single user
def test_workflow():
    app = get_workflow_app()
    
    # Set up a streaming function for a single user
    def stream_memory_responses(user_input: str):
        config = {"configurable": {"thread_id": "single_session_memory"}}

        # Stream the events in the graph
        for event in app.stream({"messages": [("user", user_input)]}, config):

            # Return the agent's last response
            for value in event.values():
                if "messages" in value and value["messages"]:
                    print("Agent:", value["messages"])
    
    # Test the workflow with a few example queries
    print("Testing workflow with example queries...")
    stream_memory_responses("What is the Colosseum?")
    stream_memory_responses("what is the factorial of 8?")
    stream_memory_responses("search for flights from ny to instabul during the last week of Dec")