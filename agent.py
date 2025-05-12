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
from typing import Annotated, Dict, List, Union, Callable, Any, Tuple
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import re
from functools import partial

# LangGraph modules for defining graphs
from langgraph.graph import MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Modules for Messages
from langchain_core.messages import AIMessage, HumanMessage

# Module for setting up OpenAI
from langchain_openai import ChatOpenAI

# Module for setting up Google Gen AI
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai.types import (Tool, GenerateContentConfig, 
    ThinkingConfig, ToolCodeExecution)

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

def process_citations(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process the response from the search agent to ensure proper citation formatting.
    
    Args:
        response: The response dictionary from the search agent
        
    Returns:
        The processed response with properly formatted citations
    """
    if "messages" not in response:
        return response
    
    processed_messages = []
    
    for message in response["messages"]:
        if isinstance(message, AIMessage) or (isinstance(message, tuple) and message[0] == "assistant"):
            # Extract the content
            content = message.content if isinstance(message, AIMessage) else message[1]
            
            # Process citations
            content = format_citations(content)
            
            # Recreate the message with the updated content
            if isinstance(message, AIMessage):
                processed_message = AIMessage(content=content)
            else:
                processed_message = ("assistant", content)
            
            processed_messages.append(processed_message)
        else:
            processed_messages.append(message)
    
    response["messages"] = processed_messages
    return response

def format_citations(content: str) -> str:
    """
    Format citations in the content string.
    
    Args:
        content: The text content to process
        
    Returns:
        The content with properly formatted citations
    """
    # Check if content already has properly formatted citations
    if "<cite index=" in content:
        return content
        
    # Extract information that should be cited
    search_results_pattern = r"Based on the search results:(.*?)(?:\n\n|$)"
    search_results_match = re.search(search_results_pattern, content, re.DOTALL)
    
    if not search_results_match:
        return content
        
    search_results_text = search_results_match.group(1).strip()
    
    # Replace with properly cited content
    cited_text = f"<cite index=\"0-1\">{search_results_text}</cite>"
    
    # Replace the original text with cited text
    content = content.replace(search_results_text, cited_text)
    
    # Also look for statements that reference sources
    source_patterns = [
        r"According to (.*?), (.*?)\.",
        r"(.*?) reports that (.*?)\.",
        r"As mentioned in (.*?), (.*?)\."
    ]
    
    for i, pattern in enumerate(source_patterns):
        for match in re.finditer(pattern, content, re.DOTALL):
            full_match = match.group(0)
            if "<cite" not in full_match:  # Avoid double-citing
                cited_match = f"<cite index=\"{i+1}-1\">{full_match}</cite>"
                content = content.replace(full_match, cited_match)
    
    # Extract and format URLs
    url_pattern = r"(https?://[^\s]+)"
    urls = re.findall(url_pattern, content)
    
    if urls:
        sources_text = "\n\nSources:\n"
        for i, url in enumerate(urls):
            sources_text += f"[{i+1}] {url}\n"
        
        # Remove the URLs from the main content
        for url in urls:
            content = content.replace(url, "")
        
        # Add the sources section
        content += sources_text
    
    return content

def post_process_workflow_output(output, post_processors=None):
    """Apply post-processing functions to workflow outputs"""
    if not post_processors:
        return output
        
    processed_output = output
    for processor in post_processors:
        processed_output = processor(processed_output)
    return processed_output

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

    # Enhanced search agent with better citation handling
    search_agent = create_react_agent(
        model=gpt,
        tools=[tavily_search],
        name="search_agent",
        prompt="""
        You are an expert researcher with access to Tavily Web Search.
        Your role is to search the web for accurate information and present it with proper citations.
        
        When searching the web:
        1. Use the tavily_search_results tool with a specific query parameter
        2. Analyze the search results thoroughly
        3. Provide a clear, concise summary with proper citations
        4. Include the source URLs in your response
        
        CITATION FORMATTING INSTRUCTIONS (CRITICAL):
        - Every claim based on search results MUST be wrapped in citation tags: <cite index="SOURCE_INDEX">your text here</cite>
        - The SOURCE_INDEX should be formatted as "result_number-sentence_number" (e.g., "0-1" for first result, first sentence)
        - For each search result you reference, include the URL at the end of your response
        - If using multiple sources, use different index numbers
        - If you're unsure about information, indicate this clearly
        - Never fabricate citations or sources
        
        Example of properly cited response:
        
        <cite index="0-1">The average temperature in New York in December is 35°F (1.7°C).</cite> <cite index="1-1">Snowfall is common, with the city receiving approximately 4.8 inches of snow during the month.</cite>
        
        Sources:
        [1] weather.com/new-york-climate
        [2] nyc.gov/winter-statistics
        
        Format your response carefully following these instructions. This is critical for providing trustworthy information.
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
        output_mode="full_history",
        parallel_tool_calls=False
    )

    # Store original methods
    original_invoke = workflow.invoke
    original_stream = workflow.stream

    # Override the invoke method with post-processing
    def invoke_with_post_processing(input_data, config=None):
        result = original_invoke(input_data, config)
        return post_process_workflow_output(result, [process_citations])

    # Override the stream method with post-processing
    def stream_with_post_processing(input_data, config=None):
        for event in original_stream(input_data, config):
            yield post_process_workflow_output(event, [process_citations])

    # Replace the methods
    workflow.invoke = invoke_with_post_processing
    workflow.stream = stream_with_post_processing

    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory
    )
    
    return app
