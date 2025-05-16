# agent.py
"""
LangGraph Agent Definition.
This file contains the setup for the LangChain/LangGraph agent,
including tool definitions, LLM initialization, and workflow compilation.
Uses the standard MessagesState for handling message types.
"""

# Import environment file
import os
import time
import logging
from dotenv import load_dotenv

# Modules for structuring text
# from typing import Dict, Any

# Module for setting up Anthropic's Claude
from langchain_anthropic import ChatAnthropic

# Modules for creating ReAct agents with Supervisor architecture
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import custom Agent tools
from agent_tools.search_tools import create_tavily_search_tool, process_citations_for_response
from agent_tools.wiki_tools import create_wikipedia_tool
from agent_tools.code_tools import get_code_tools

# Import the prompts
from prompts import (get_supervisor_prompt, get_code_prompt, 
    get_search_prompt, get_wiki_prompt)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load the .env file
load_dotenv()


def get_workflow_app():
    """
    Initialize and return the LangGraph workflow application.
    This function handles all the setup for the agents and workflow.

    Returns:
        The compiled LangGraph workflow application.
    """
    # Initialize Anthropic/Claude
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY environment variable not set.")
        logger.warning(
            "Please set the ANTHROPIC_API_KEY environment variable in your .env file"
        )

    # Initialize Tavily Search
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.warning("TAVILY_API_KEY environment variable not set.")
        logger.warning(
            "Please set the TAVILY_API_KEY environment variable in your .env file"
        )

    # Initialize Claude LLM
    if anthropic_api_key:
        try:
            claude = ChatAnthropic(
                model_name="claude-3-7-sonnet-latest",
                anthropic_api_key=anthropic_api_key,
                max_tokens=3000,  # Adjust as needed for the final answer
                thinking={
                    "type": "enabled",
                    "budget_tokens": 1024  # Adjust the token budget for the thinking process
                })
            logger.info("Successfully initialized Claude model")
        except Exception as e:
            logger.error(f"Failed to initialize Claude model: {e}")
            raise RuntimeError(f"Failed to initialize Claude model: {e}")
    else:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Initialize tools
    tavily_search = None
    wikipedia_tool = None

    # Initialize Tavily Search with enhanced configuration
    if tavily_api_key:
        tavily_search = create_tavily_search_tool(tavily_api_key)

    # Initialize Wikipedia Tool
    wikipedia_tool = create_wikipedia_tool()

    # Initialize code tools
    code_tools = get_code_tools()

    # Create agents only if required components are available
    agents = []

    # Enhanced search agent with better citation handling
    if claude and tavily_search:
        try:
            search_agent = create_react_agent(
                model=claude,
                tools=[tavily_search],
                name="search_agent",
                prompt=get_search_prompt())
            agents.append(search_agent)
            logger.info("Successfully created search agent")
        except Exception as e:
            logger.error(f"Failed to create search agent: {e}")

    if claude and code_tools:
        try:
            code_agent = create_react_agent(
                model=claude,
                tools=code_tools,  # Code agent uses built-in code execution
                name="code_agent",
                prompt=get_code_prompt())
            agents.append(code_agent)
            logger.info("Successfully created code agent")
        except Exception as e:
            logger.error(f"Failed to create code agent: {e}")

    if claude and wikipedia_tool:
        try:
            wiki_agent = create_react_agent(
                model=claude,
                tools=[wikipedia_tool],
                name="wiki_agent",
                prompt=get_wiki_prompt())
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
        workflow = create_supervisor(agents,
                                     model=claude,
                                     prompt=get_supervisor_prompt(),
                                     output_mode="full_history",
                                     parallel_tool_calls=False)
        logger.info("Successfully created supervisor workflow")
    except Exception as e:
        logger.error(f"Failed to create supervisor workflow: {e}")
        raise RuntimeError(f"Failed to create supervisor workflow: {e}")

    # Create memory saver and compile the workflow
    try:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        logger.info("Successfully compiled workflow")
    except Exception as e:
        logger.error(f"Failed to compile workflow: {e}")
        raise RuntimeError(f"Failed to compile workflow: {e}")

    # Now wrap the compiled app with our post-processing
    original_invoke = app.invoke
    original_stream = app.stream

    # Create wrapped versions of invoke and stream that apply post-processing
    def invoke_with_post_processing(input_data, config=None, **kwargs):
        """
        Wrapper for the original invoke method that applies post-processing to results.
        Falls back to a generic error message instead of hardcoded examples.
        """
        try:
            result = original_invoke(input_data, config, **kwargs)
            return process_citations(result)
        except Exception as e:
            logger.error(f"Error in invoke_with_post_processing: {e}")

            # Instead of hardcoding a specific response, create a generic error message
            error_message = f"I encountered an error while processing your request. This might be due to a temporary issue with one of my tools. Please try rephrasing your question or try again later."

            # Return a proper error response format
            return {
                "messages": [("assistant", error_message)]
            }

    # Modify the stream_with_post_processing function to accept additional keywords
    def stream_with_post_processing(input_data, config=None, **kwargs):
        """
        Wrapper for the original stream method that applies post-processing to results.
        Falls back to generic error messages instead of hardcoded examples.
        """
        try:
            # Pass any additional kwargs to the original_stream function
            for event in original_stream(input_data, config, **kwargs):
                try:
                    yield process_citations(event)
                except Exception as e:
                    logger.error(f"Error processing stream event: {e}")

                    # Generic error for processing failures
                    yield {
                        "messages": [
                            ("assistant", "I encountered an error while processing part of the response. Let me try to continue with what I can provide.")
                        ]
                    }
        except Exception as e:
            logger.error(f"Error in stream_with_post_processing: {e}")

            # Generic streaming error
            yield {
                "messages": [
                    ("assistant", "I encountered an error while retrieving information. This might be due to a temporary issue with one of my tools. Please try rephrasing your question or try again later.")
                ]
            }

    # Replace the methods on the compiled app
    app.invoke = invoke_with_post_processing
    app.stream = stream_with_post_processing

    return app