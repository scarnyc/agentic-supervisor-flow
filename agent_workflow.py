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

from langchain.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun

# Load the .env file
load_dotenv()

# --- LangChain/LangGraph Setup ---

# Initialize LLM
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("Warning: GEMINI_API_KEY environment variable not set.")
    print("Please set the GEMINI_API_KEY environment variable in your .env file")
    print("You can get an API key from Google AI Studio: https://makersuite.google.com/")

# Use a generally available model name
model_name = "gemini-2.5-flash-preview-04-17"

# Initialize the LLM with proper configuration
reasoning_llm = ChatGoogleGenerativeAI(
    model=model_name,
    google_api_key=gemini_api_key,
    temperature=0.2,
    top_p=0.95,
    top_k=30,
    max_output_tokens=8192,
    convert_system_message_to_human=False,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=500)
    )
 )

# Define Wikipedia Tool with explicit name
api_wrapper = WikipediaAPIWrapper(top_k_results=1)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Set the name explicitly to match what the frontend is expecting
wikipedia_tool.name = "wikipedia_query_run"
wikipedia_tool.description = "Searches Wikipedia for information about a given topic."

# Print the tool name for debugging
print(f"Registered Wikipedia tool with name: {wikipedia_tool.name}")

# Create LLM with Google Web Search
search_llm = ChatGoogleGenerativeAI(
	model=model_name,  # Same model as main LLM
	google_api_key=gemini_api_key,
	temperature=0.01,   # Lower temperature for search
	max_output_tokens=2048,
	model_kwargs={ 
	    "tools": [types.Tool(google_search=types.GoogleSearch())],
	    "tool_config": {"function_calling_config": {"mode": "AUTO"}}
	}
)

code_llm = ChatGoogleGenerativeAI(
    model=model_name,  # Same model as main LLM
    google_api_key=gemini_api_key,
    temperature=0.01,
    max_output_tokens=2048,
    model_kwargs={ 
        "tools": [types.Tool(code_execution=types.ToolCodeExecution())],
        "tool_config": {"function_calling_config": {"mode": "AUTO"}}
    }
)

search_agent = create_react_agent(
    model=search_llm,
    tools=[],
    name="search_agent",
    prompt="""
    You are an expert researcher with access to Google Web Search.
    Search for the user's query and summarize the results.
    When searching the web: Use the google_search tool with a specific query parameter.
    EXAMPLE: For flight info, use google_search(query="flights from New York to Istanbul December 2025")
    1. After receiving tool results, analyze them and provide a clear, concise summary.
    2. Only call a tool once for a query unless you explicitly need more information.
    3. Always provide an actual response when you have enough information.
    """
)

code_agent = create_react_agent(
    model=code_llm,
    tools=[],
    name="code_agent",
    prompt="""
    You are an expert coder.
    Write and execute code to solve user queries and complex tasks.
    When executing code: Use the code_execution tool with proper code formatting.
    EXAMPLE: code_execution(code="print('Hello world')")
    1. After receiving tool results, analyze them and provide a clear, concise summary.
    2. Only call a tool once for a query unless you explicitly need more information.
    3. Always provide an actual response when you have enough information.
    """
)

wiki_agent = create_react_agent(
    model=reasoning_llm,
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
    prompt=("""
        You are a word-class team supervisor managing a team of agents, including:
        - search_agent (uses Google web search for real-time information), 
        - a code_agent (writes and executes code to solve problems),
        - a wiki_agent (searches Wikipedia)

        HOW TO USE TOOLS:
        - For questions about real-time information (flights, prices, schedules, or current events), respond with: "Let me search that for you" and ALWAYS delegate to the search_agent first.
        - For computational and math problems, delegate to the code_agent
        - For general knowledge questions that don't require real-time information, delegate to the wiki_agent.


        *Very Important* always pass the relevant context from the prior agent to the next agent, including 
        user prompts, Google web search results, Wikipedia info, and code output.
        1. After receiving results, analyze them and provide a clear, concise summary.
        2. Only call an agent once for a query unless you explicitly need more information.
        3. Always provide an actual response when you have enough information.
    """),
    output_mode="last_message"
    # output_mode="full_history"
)

memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory
)



# Set up a streaming function for a single user
def stream_memory_responses(user_input: str):
    config = {"configurable": {"thread_id": "single_session_memory"}}

    # Stream the events in the graph
    for event in graph.stream({"messages": [("user", user_input)]}, config):

        # Return the agent's last response
        for value in event.values():
            if "messages" in value and value["messages"]:
                print("Agent:", value["messages"])

stream_memory_responses("What is the Colosseum?")
stream_memory_responses("search for flights from ny to instabul during the last week of Dec")