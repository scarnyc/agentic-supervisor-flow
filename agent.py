# -*- coding: utf-8 -*-
"""
LangGraph Agent Definition.
This file contains the setup for the LangChain/LangGraph agent,
including tool definitions, LLM initialization, and workflow compilation.
Uses the standard MessagesState for handling message types.
"""

#import environment file
import os
import sys
from dotenv import load_dotenv
import logging

# Modules for structuring text
from typing import Annotated, Dict, List, Union, Callable, Any, Tuple, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import re
import json
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
try:
    from google import genai
    from langchain_google_gemini import ChatGoogleGenerativeAI
    from google.genai.types import (Tool, GenerateContentConfig, 
        ThinkingConfig, ToolCodeExecution)
    gemini_available = True
except ImportError:
    gemini_available = False
    logging.warning("Google Gemini modules not available. Code agent will not be functional.")

# Modules for creating ReAct agents with Supervisor architecture
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# Import the modules for saving memory
from langgraph.checkpoint.memory import MemorySaver

# Tools
try:
    from langchain.tools import tool as LangChainTool
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
    from langchain_community.tools.tavily_search.tool import TavilySearchResults
    tools_available = True
except ImportError:
    tools_available = False
    logging.warning("LangChain tools not available. Search and Wiki agents will not be functional.")

# Local imports
try:
    from prompt import get_enhanced_supervisor_prompt
except ImportError:
    # Fallback supervisor prompt if import fails
    def get_enhanced_supervisor_prompt():
        return """
        You are an expert AI assistant coordinating with multiple specialized agents to help answer the user's query.
        Based on the user's query, decide which of the following agents to use:
        
        1. search_agent: For retrieving information from the web
        2. wiki_agent: For searching Wikipedia
        3. code_agent: For writing and executing code
        
        For each user query, think about which agent would be most appropriate to handle it.
        Only use one agent at a time, and only when necessary.
        """

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the .env file
load_dotenv()

# Class to track Tavily search results
class TavilyResultTracker:
    def __init__(self):
        self.last_results = {}
    
    def store_result(self, session_id, results):
        try:
            self.last_results[session_id] = results
        except Exception as e:
            logger.error(f"Error storing Tavily results: {e}")
            self.last_results[session_id] = []
    
    def get_result(self, session_id):
        return self.last_results.get(session_id, [])

# Create a global instance of the tracker
tavily_tracker = TavilyResultTracker()

def extract_urls_from_tavily_result(content: str) -> List[str]:
    """
    Extract URLs from Tavily search results content.
    
    Args:
        content: The content string from the agent
        
    Returns:
        List of extracted URLs or empty list if none found
    """
    if not content:
        return []
        
    try:
        # Look for tool output sections that contain Tavily results
        tavily_pattern = r"Action: tavily_search_results\s+Action Input: .*?\s+Observation: (.*?)(?:\n\nThought:|$)"
        tavily_matches = re.findall(tavily_pattern, content, re.DOTALL)
        
        urls = []
        for match in tavily_matches:
            try:
                # Try to parse the JSON content from the observation
                data = json.loads(match.strip())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'url' in item:
                            urls.append(item['url'])
                elif isinstance(data, dict) and 'results' in data:
                    for item in data['results']:
                        if isinstance(item, dict) and 'url' in item:
                            urls.append(item['url'])
            except json.JSONDecodeError:
                # Fallback to regex if JSON parsing fails
                url_pattern = r'https?://[^\s"\')]+' 
                found_urls = re.findall(url_pattern, match)
                urls.extend(found_urls)
            except Exception as e:
                logger.error(f"Error extracting URLs from Tavily result: {e}")
        
        return urls
    except Exception as e:
        logger.error(f"Error in extract_urls_from_tavily_result: {e}")
        return []

def format_citations(content: str) -> str:
    """
    Format citations in the content string.
    
    Args:
        content: The text content to process
        
    Returns:
        The content with properly formatted citations
    """
    if not content:
        return content
        
    try:
        # Check if content already has properly formatted citations and sources
        if "<cite index=" in content and "Sources:" in content:
            return content
        
        # Extract URLs from the content (if they exist in Tavily results)
        urls = extract_urls_from_tavily_result(content)
        
        # Extract information that should be cited
        search_results_pattern = r"Based on (?:the|my) search (?:results|findings):(.*?)(?:\n\n|$)"
        search_results_match = re.search(search_results_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if search_results_match:
            search_results_text = search_results_match.group(1).strip()
            
            # Replace with properly cited content - cite the whole paragraph for now
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
        
        # Add source URLs if we found any
        if urls:
            if "Sources:" not in content:
                sources_text = "\n\nSources:\n"
                for i, url in enumerate(urls):
                    sources_text += f"[{i+1}] {url}\n"
                
                # Add the sources section
                content += sources_text
        
        return content
    except Exception as e:
        logger.error(f"Error in format_citations: {e}")
        return content  # Return original content if processing fails

def process_citations(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process the response from the search agent to ensure proper citation formatting.
    
    Args:
        response: The response dictionary from the search agent
        
    Returns:
        The processed response with properly formatted citations
    """
    if not isinstance(response, dict) or "messages" not in response:
        return response
    
    try:
        processed_messages = []
        
        for message in response["messages"]:
            if isinstance(message, AIMessage) or (isinstance(message, tuple) and len(message) > 1 and message[0] == "assistant"):
                # Extract the content
                content = message.content if isinstance(message, AIMessage) else message[1]
                
                # Process citations
                content = format_citations(content)
                
                # Transform agent transfer messages
                agent_transfer_patterns = [
                    (r"Successfully transferred to search_agent", "Using Web Search Tool..."),
                    (r"transferred to search_agent", "Using Web Search Tool..."),
                    (r"Successfully transferred to wiki_agent", "Using Wikipedia Tool..."),
                    (r"transferred to wiki_agent", "Using Wikipedia Tool..."),
                    (r"Successfully transferred to code_agent", "Using Code Execution Tool..."),
                    (r"transferred to code_agent", "Using Code Execution Tool..."),
                    (r"Successfully transferred back to supervisor", "Thinking..."),
                    (r"transferred back to supervisor", "Thinking..."),
                    (r"^search_agent$", "Using Web Search Tool..."),
                    (r"^wiki_agent$", "Using Wikipedia Tool..."),
                    (r"^code_agent$", "Using Code Execution Tool..."),
                    (r"^supervisor$", "Thinking...")
                ]
                
                for pattern, replacement in agent_transfer_patterns:
                    content = re.sub(pattern, replacement, content)
                
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
    except Exception as e:
        logger.error(f"Error in process_citations: {e}")
        return response  # Return original response if processing fails

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
        error_msg = "OPENAI_API_KEY environment variable not set"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Initialize Gemini
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.warning("GEMINI_API_KEY environment variable not set.")
        logger.warning("Please set the GEMINI_API_KEY environment variable in your .env file")
        logger.warning("You can get an API key from Google AI Studio: https://makersuite.google.com/")

    # Initialize Tavily Search
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.warning("TAVILY_API_KEY environment variable not set.")
        logger.warning("Please set the TAVILY_API_KEY environment variable in your .env file")

    # Initialize GPT
    try:
        gpt = ChatOpenAI(
            model_name="gpt-4.1-mini-2025-04-14",
            temperature=0.2,
            top_p=0.95
        )
        logger.info("Successfully initialized GPT model")
    except Exception as e:
        logger.error(f"Failed to initialize GPT model: {e}")
        gpt = None

    # Initialize Gemini LLM
    gemini = None
    if gemini_available and gemini_api_key:
        try:
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
            logger.info("Successfully initialized Gemini model")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")

    # Initialize tools
    tavily_search = None
    wikipedia_tool = None

    if tools_available:
        # Initiate Tavily Search with enhanced configuration
        if tavily_api_key:
            try:
                tavily_search = TavilySearchResults(
                    api_key=tavily_api_key,
                    k=5,  # Number of results
                    include_raw_content=True,  # Include the raw content
                    include_images=False,
                    include_answer=True,
                    max_results=5,
                    search_depth="advanced"
                )
                logger.info("Successfully initialized Tavily Search")
            except Exception as e:
                logger.error(f"Failed to initialize Tavily Search: {e}")
        
        # Define Wikipedia Tool with explicit name
        try:
            api_wrapper = WikipediaAPIWrapper(top_k_results=1)
            wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

            # Set the name explicitly to match what the frontend is expecting
            wikipedia_tool.name = "wikipedia_query_run"
            wikipedia_tool.description = "Searches Wikipedia for information about a given topic."
            
            logger.info(f"Registered Wikipedia tool with name: {wikipedia_tool.name}")
        except Exception as e:
            logger.error(f"Failed to initialize Wikipedia tool: {e}")

    # Create agents only if required components are available
    agents = []
    
    # Enhanced search agent with better citation handling
    if gpt and tavily_search:
        try:
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
                3. IMPORTANT: For each source you use, INCLUDE THE SOURCE URL in your response
                4. Format your response with proper citations using <cite> tags
                
                CITATION FORMATTING INSTRUCTIONS (CRITICAL):
                - Every claim based on search results MUST be wrapped in citation tags: <cite index="SOURCE_INDEX">your text here</cite>
                - The SOURCE_INDEX should be formatted as "result_number-sentence_number" (e.g., "0-1" for first result, first sentence)
                - For each search result you reference, include a numbered source list at the end with URLs, like:
                
                Sources:
                [1] https://example.com/page1
                [2] https://example.com/page2
                
                Example of a properly formatted response:
                
                <cite index="0-1">The average temperature in New York in December is 35°F (1.7°C).</cite> <cite index="1-1">Snowfall is common during this month.</cite>
                
                Sources:
                [1] https://weather.com/new-york-climate
                [2] https://nyc.gov/winter-statistics
                
                Format your response carefully following these instructions. This is critical for providing trustworthy information.
                """
            )
            agents.append(search_agent)
            logger.info("Successfully created search agent")
        except Exception as e:
            logger.error(f"Failed to create search agent: {e}")
    
    if gemini:
        try:
            code_agent = create_react_agent(
                model=gemini,
                tools=[],  # Code agent uses built-in code execution from Gemini
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
            agents.append(code_agent)
            logger.info("Successfully created code agent")
        except Exception as e:
            logger.error(f"Failed to create code agent: {e}")
    
    if gpt and wikipedia_tool:
        try:
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
            agents.append(wiki_agent)
            logger.info("Successfully created wiki agent")
        except Exception as e:
            logger.error(f"Failed to create wiki agent: {e}")
    
    # Check if we have at least one agent
    if not agents:
        error_msg = "No agents could be created. Check API keys and dependencies."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Create supervisor workflow
    try:
        workflow = create_supervisor(
            agents,
            model=gpt,
            prompt=get_enhanced_supervisor_prompt(),
            output_mode="full_history",
            parallel_tool_calls=False
        )
        logger.info("Successfully created supervisor workflow")
    except Exception as e:
        logger.error(f"Failed to create supervisor workflow: {e}")
        raise RuntimeError(f"Failed to create supervisor workflow: {e}")

    # Create memory saver and compile the workflow
    try:
        memory = MemorySaver()
        app = workflow.compile(
            checkpointer=memory
        )
        logger.info("Successfully compiled workflow")
    except Exception as e:
        logger.error(f"Failed to compile workflow: {e}")
        raise RuntimeError(f"Failed to compile workflow: {e}")
    
    # Now wrap the compiled app with our post-processing
    original_invoke = app.invoke
    original_stream = app.stream
    
    # Create wrapped versions of invoke and stream that apply post-processing
    def invoke_with_post_processing(input_data, config=None):
        try:
            result = original_invoke(input_data, config)
            return process_citations(result)
        except Exception as e:
            logger.error(f"Error in invoke_with_post_processing: {e}")
            # Return a fallback response
            return {"messages": [("assistant", "Sorry, I encountered an error processing your request.")]}
    
    def stream_with_post_processing(input_data, config=None):
        try:
            for event in original_stream(input_data, config):
                try:
                    yield process_citations(event)
                except Exception as e:
                    logger.error(f"Error processing stream event: {e}")
                    yield {"messages": [("assistant", "Sorry, there was an error processing part of the response.")]}
        except Exception as e:
            logger.error(f"Error in stream_with_post_processing: {e}")
            yield {"messages": [("assistant", "Sorry, I encountered an error processing your request.")]}
    
    # Replace the methods on the compiled app
    app.invoke = invoke_with_post_processing
    app.stream = stream_with_post_processing
    
    return app