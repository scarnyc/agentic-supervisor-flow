# -*- coding: utf-8 -*-
"""
LangGraph Agent Definition.
This file contains the setup for the LangChain/LangGraph agent,
including tool definitions, LLM initialization, and workflow compilation.
Uses the standard MessagesState for handling message types.
"""

#import environment file
import os
import logging
from dotenv import load_dotenv

# Modules for structuring text
from typing import Annotated, Dict, List, Union, Callable, Any, Tuple
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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Module for setting up OpenAI
from langchain_openai import ChatOpenAI

# Import modules with error handling
logger = logging.getLogger(__name__)

# Check if Gemini is available, if not, log a warning
gemini_available = True
try:
    # Module for setting up Google Gen AI
    from google import genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from google.genai.types import (Tool, GenerateContentConfig, 
        GoogleSearch, ThinkingConfig, ToolCodeExecution)
except ImportError:
    logger.warning("Google Gemini modules not available. Code agent will not be functional.")
    gemini_available = False
    
# Modules for creating ReAct agents with Supervisor architecture
try:
    from langgraph_supervisor import create_supervisor
    from langgraph.prebuilt import create_react_agent
except ImportError:
    logger.error("LangGraph Supervisor not available. Please install with: pip install langgraph-supervisor")
    raise

# Import the modules for saving memory
from langgraph.checkpoint.memory import MemorySaver

from langchain.tools import tool as LangChainTool
try:
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.tools import WikipediaQueryRun
except ImportError:
    logger.error("Wikipedia tools not available. Please install with: pip install langchain-community")
    raise

# Try to import Tavily search
tavily_available = True
try:
    from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
    from langchain_community.tools.tavily_search.tool import TavilySearchResults
except ImportError:
    logger.warning("Tavily search not available. Web search will be disabled.")
    tavily_available = False

# Import the prompt template
try:
    from prompt import get_enhanced_supervisor_prompt
except ImportError:
    # Define a fallback prompt if the import fails
    def get_enhanced_supervisor_prompt():
        return """
        You are a helpful AI assistant that coordinates between different specialized agents.
        Based on the user's query, decide which agent to call:
        - search_agent: For questions about current events, facts, or information that might change over time
        - code_agent: For writing, explaining, or executing code
        - wiki_agent: For questions about general knowledge, historical facts, or academic topics
        
        Current date: {{current_date}}
        
        Begin by analyzing the user's request, then call the most appropriate specialized agent.
        Always provide a final, comprehensive response based on the information from the specialized agent.
        """

# Load the .env file
load_dotenv()

# Class to track Tavily search results
class TavilyResultTracker:
    def __init__(self):
        self.last_results = {}
    
    def store_result(self, session_id, results):
        self.last_results[session_id] = results
    
    def get_result(self, session_id):
        return self.last_results.get(session_id, [])

# Create a global instance of the tracker
tavily_tracker = TavilyResultTracker()

def extract_urls_from_tavily_result(content):
    """
    Extract URLs from Tavily search results content.
    
    Args:
        content: The content string from the agent
        
    Returns:
        List of extracted URLs or empty list if none found
    """
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
        except:
            # Fallback to regex if JSON parsing fails
            url_pattern = r'https?://[^\s"\')]+' 
            found_urls = re.findall(url_pattern, match)
            urls.extend(found_urls)
    
    return urls

def format_citations(content: str) -> str:
    """
    Format citations in the content string.
    
    Args:
        content: The text content to process
        
    Returns:
        The content with properly formatted citations
    """
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

def transform_agent_messages(content: str) -> str:
    """
    Transform technical agent transfer messages into user-friendly alternatives.
    
    Args:
        content: The text content to process
        
    Returns:
        The content with user-friendly agent transfer messages
    """
    # Define patterns for agent transfer messages
    patterns = [
        (r"Transferring to search agent", "🔍 Searching the web for relevant information..."),
        (r"Transferring to wiki agent", "📚 Looking up information in Wikipedia..."),
        (r"Transferring to code agent", "💻 Setting up code execution environment..."),
        (r"I'll delegate this to the search agent", "🔍 Searching the web for relevant information..."),
        (r"I'll delegate this to the wiki agent", "📚 Looking up information in Wikipedia..."),
        (r"I'll delegate this to the code agent", "💻 Setting up code execution environment..."),
        (r"I'll use the search agent", "🔍 Searching the web for relevant information..."),
        (r"I'll use the wiki agent", "📚 Looking up information in Wikipedia..."),
        (r"I'll use the code agent", "💻 Setting up code execution environment...")
    ]
    
    # Replace each pattern
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
    
    return content

def process_citations(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process the response from the search agent to ensure proper citation formatting
    and transform technical messages into user-friendly ones.
    
    Args:
        response: The response dictionary from the search agent
        
    Returns:
        The processed response with properly formatted citations and user-friendly messages
    """
    if "messages" not in response:
        return response
    
    processed_messages = []
    
    for message in response["messages"]:
        if isinstance(message, AIMessage) or (isinstance(message, tuple) and message[0] == "assistant"):
            # Extract the content
            content = message.content if isinstance(message, AIMessage) else message[1]
            
            # Transform agent messages
            content = transform_agent_messages(content)
            
            # Process citations
            content = format_citations(content)
            
            # Recreate the message with the updated content
            if isinstance(message, AIMessage):
                processed_message = AIMessage(content=content)
            else:
                processed_message = ("assistant", content)
            
            processed_messages.append(processed_message)
        elif isinstance(message, ToolMessage) or (isinstance(message, tuple) and message[0] == "tool"):
            # Extract tool message content for processing if needed
            content = message.content if isinstance(message, ToolMessage) else message[1]
            
            # Add the processed tool message
            if isinstance(message, ToolMessage):
                processed_messages.append(message)
            else:
                processed_messages.append(message)
        else:
            # Pass through other message types unchanged
            processed_messages.append(message)
    
    response["messages"] = processed_messages
    return response

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

    # Initialize Gemini with error handling
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini = None
    
    if gemini_available and gemini_api_key:
        try:
            # Initialize Gemini LLM
            gemini = ChatGoogleGenerativeAI(
                model='gemini-2.0-flash-001',
                google_api_key=gemini_api_key,
                temperature=0.1,
                max_output_tokens=2048,
                model_kwargs={ 
                    "tools": [Tool(code_execution=ToolCodeExecution())],
                    "tool_config": {"function_calling_config": {"mode": "AUTO"}}
                }
            )
            logger.info("Successfully initialized Gemini model")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            gemini_available = False
    else:
        logger.warning("Gemini not available. Code agent will use OpenAI GPT instead.")

    # Initialize GPT
    try:
        gpt = ChatOpenAI(
            model_name="gpt-4.1-mini-2025-04-14",
            temperature=0.2,
            top_p=0.95,
            request_timeout=60  # 60 second timeout
        )
        logger.info("Successfully initialized GPT model")
    except Exception as e:
        logger.error(f"Failed to initialize GPT: {e}")
        raise ValueError("Unable to initialize the GPT model")

    # Initialize Tavily Search with error handling
    tavily_search = None
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if tavily_available and tavily_api_key:
        try:
            tavily_search = TavilySearchResults(
                api_key=tavily_api_key,
                k=5,  # Number of results
                include_raw_content=True,
                include_images=False,
                include_answer=True,
                max_results=5,
                search_depth="advanced"
            )
            logger.info("Successfully initialized Tavily Search")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily Search: {e}")
            tavily_available = False
    else:
        logger.warning("Tavily Search not available. Web search will be disabled.")

    # Define Wikipedia Tool with explicit name and error handling
    wikipedia_tool = None
    try:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1)
        wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
        wikipedia_tool.name = "wikipedia_query_run"
        wikipedia_tool.description = "Searches Wikipedia for information about a given topic."
        logger.info(f"Registered Wikipedia tool with name: {wikipedia_tool.name}")
    except Exception as e:
        logger.error(f"Failed to initialize Wikipedia tool: {e}")
        raise ValueError("Unable to initialize the Wikipedia tool")

    # Define agents with proper error handling
    search_agent = None
    if tavily_available and tavily_search:
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
            logger.info("Successfully created search agent")
        except Exception as e:
            logger.error(f"Failed to create search agent: {e}")
            search_agent = None

    # Create code agent using either Gemini or GPT as fallback
    try:
        if gemini_available and gemini:
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
                4. If you encounter any issues with code execution, gracefully explain the error and suggest fixes.
                5. Always wrap code blocks in triple backticks with the language specified.
                """
            )
        else:
            # Use GPT as fallback for code agent
            code_agent = create_react_agent(
                model=gpt,
                tools=[],
                name="code_agent",
                prompt="""
                You are an expert coder.
                Your role is to write clear, well-documented code to solve user queries.
                For coding tasks:
                1. Carefully analyze the request and break it down into manageable parts
                2. Write easy-to-understand code with comments explaining key sections
                3. Provide explanations of how the code works
                4. Include usage examples where appropriate
                5. Always wrap code blocks in triple backticks with the language specified
                
                Note: While you don't have direct code execution capability, you provide high-quality code that users can run themselves.
                """
            )
        logger.info("Successfully created code agent")
    except Exception as e:
        logger.error(f"Failed to create code agent: {e}")
        # Create a minimal code agent that just writes code
        code_agent = create_react_agent(
            model=gpt,
            tools=[],
            name="code_agent",
            prompt="You are a coding assistant. Write clear, well-documented code to solve user queries."
        )

    # Create wiki agent with error handling
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
        logger.info("Successfully created wiki agent")
    except Exception as e:
        logger.error(f"Failed to create wiki agent: {e}")
        raise ValueError("Unable to create wiki agent")

    # Create supervisor workflow with error handling
    try:
        # Prepare the list of available agents
        available_agents = []
        if search_agent:
            available_agents.append(search_agent)
        if code_agent:
            available_agents.append(code_agent)
        if wiki_agent:
            available_agents.append(wiki_agent)
        
        workflow = create_supervisor(
            available_agents,
            model=gpt,
            prompt=get_enhanced_supervisor_prompt(),
            output_mode="full_history",
            parallel_tool_calls=False
        )
        logger.info("Successfully created supervisor workflow")
    except Exception as e:
        logger.error(f"Failed to create supervisor workflow: {e}")
        raise ValueError("Unable to create supervisor workflow")

    # Create memory saver and compile the workflow
    try:
        memory = MemorySaver()
        app = workflow.compile(
            checkpointer=memory
        )
        logger.info("Successfully compiled workflow")
    except Exception as e:
        logger.error(f"Failed to compile workflow: {e}")
        raise ValueError("Unable to compile workflow")
    
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
            return {
                "messages": [
                    ("assistant", "I'm sorry, I encountered an error processing your request. Please try again or ask a different question.")
                ]
            }
    
    def stream_with_post_processing(input_data, config=None):
        try:
            for event in original_stream(input_data, config):
                yield process_citations(event)
        except Exception as e:
            logger.error(f"Error in stream_with_post_processing: {e}")
            # Yield a fallback response
            yield {
                "messages": [
                    ("assistant", "I'm sorry, I encountered an error processing your request. Please try again or ask a different question.")
                ]
            }
    
    # Replace the methods on the compiled app
    app.invoke = invoke_with_post_processing
    app.stream = stream_with_post_processing
    
    return app