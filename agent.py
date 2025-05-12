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
from typing import Dict, Any
import re
import json
import builtins
import contextlib
import io

# Modules for Messages
from langchain_core.messages import AIMessage

# Module for setting up OpenAI
from langchain_openai import ChatOpenAI

# Module for setting up Google Gen AI
from langchain_google_genai import ChatGoogleGenerativeAI

# Module for setting up Anthropic's Claude
from langchain_anthropic import ChatAnthropic

# Modules for creating ReAct agents with Supervisor architecture
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# Import Langchain modules
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from prompt import get_enhanced_supervisor_prompt

# Import CodeAct related modules
from langchain_sandbox import PyodideSandboxTool
from langgraph_codeact import create_codeact

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
    search_results_match = re.search(search_results_pattern, content,
                                     re.DOTALL | re.IGNORECASE)

    if search_results_match:
        search_results_text = search_results_match.group(1).strip()

        # Replace with properly cited content - cite the whole paragraph for now
        cited_text = f"<cite index=\"0-1\">{search_results_text}</cite>"

        # Replace the original text with cited text
        content = content.replace(search_results_text, cited_text)

    # Also look for statements that reference sources
    source_patterns = [
        r"According to (.*?), (.*?)\.", r"(.*?) reports that (.*?)\.",
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


def process_citations(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process the response from agents to ensure proper citation formatting.
    
    Args:
        response: The response dictionary from the agent
        
    Returns:
        The processed response with properly formatted citations
    """
    if "messages" not in response:
        return response

    processed_messages = []

    for message in response["messages"]:
        # Handle different message types
        if isinstance(message, AIMessage) or (hasattr(message, 'type')
                                              and message.type == 'ai'):
            # AI message from LLM
            content = message.content
            content = format_citations(content)
            if isinstance(message, AIMessage):
                processed_message = AIMessage(content=content)
            else:
                # If it's not a standard AIMessage, preserve the original with updated content
                message.content = content
                processed_message = message
            processed_messages.append(processed_message)
        elif hasattr(message, 'type') and message.type == 'tool':
            # Tool message (from code execution, etc.)
            # Skip processing if it's an internal handoff
            if (hasattr(message, 'response_metadata')
                    and ("__handoff_destination" in message.response_metadata
                         or "__is_handoff_back" in message.response_metadata)):
                processed_messages.append(message)
                continue

            # For code execution results, format the content
            content = message.content
            content = format_citations(content)

            # Preserve the original message but update its content
            message.content = content
            processed_messages.append(message)
        elif isinstance(message, tuple) and len(message) > 1:
            # Handle tuple format (role, content)
            role, content = message[0], message[1]
            content = format_citations(content)
            processed_messages.append((role, content))
        else:
            # Keep any other message types as-is
            processed_messages.append(message)

    response["messages"] = processed_messages
    return response


def eval(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    # Store original keys before execution
    original_keys = set(_locals.keys())

    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(code, builtins.__dict__, _locals)
        result = f.getvalue()
        if not result:
            result = "<code ran, no output printed to stdout>"
    except Exception as e:
        result = f"Error during execution: {repr(e)}"

    # Determine new variables created during execution
    new_keys = set(_locals.keys()) - original_keys
    new_vars = {key: _locals[key] for key in new_keys}
    return result, new_vars


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
        print(
            "Please set the GEMINI_API_KEY environment variable in your .env file"
        )
        print(
            "You can get an API key from Google AI Studio: https://makersuite.google.com/"
        )

    # Initialize Anthropic/Claude
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        print("Warning: ANTHROPIC_API_KEY environment variable not set.")
        print(
            "Please set the ANTHROPIC_API_KEY environment variable in your .env file"
        )
        print(
            "You can get an API key from Anthropic: https://console.anthropic.com/"
        )

    # Initialize Tavily Search
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY environment variable not set.")
        print(
            "Please set the TAVILY_API_KEY environment variable in your .env file"
        )

    # Initialize GPT
    gpt = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14",
                     temperature=0.2,
                     top_p=0.95)

    # Initialize Gemini LLM
    gemini = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash-001',  # Same model as main LLM
        google_api_key=gemini_api_key,
        temperature=0.01,
        max_output_tokens=2048)

    # Initialize Claude LLM specifically for code execution
    claude = ChatAnthropic(model_name="claude-3-7-sonnet-20240229",
                           anthropic_api_key=anthropic_api_key,
                           temperature=0.01,
                           max_tokens=4096)

    # Initiate Tavily Search with enhanced configuration:
    tavily_search = TavilySearchResults(
        api_key=tavily_api_key,
        k=5,  # Number of results
        include_raw_content=True,  # Include the raw content
        include_images=False,
        include_answer=True,
        max_results=5,
        search_depth="advanced")

    # Define Wikipedia Tool with explicit name
    api_wrapper = WikipediaAPIWrapper(top_k_results=1)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Set the name explicitly to match what the frontend is expecting
    wikipedia_tool.name = "wikipedia_query_run"
    wikipedia_tool.description = "Searches Wikipedia for information about a given topic."

    # Print the tool name for debugging
    print(f"Registered Wikipedia tool with name: {wikipedia_tool.name}")

    # Enhanced search agent with better citation handling
    search_agent = create_react_agent(model=gemini,
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
        """)

    code_agent = create_codeact(model=claude,
                                tools=[PyodideSandboxTool()],
                                eval_fn=eval,
                                prompt="""
        You are an expert AI code assistant powered by Claude 3.7 Sonnet.
        
        Your role is to handle code requests from users through the CodeAct system.
        
        When receiving code requests:
        1. Acknowledge the code request briefly
        2. Mention that you're using Claude 3.7 Sonnet's code execution capabilities
        3. Pass the request to the underlying system
        
        Example response:
        "I'll handle your code request using Claude 3.7 Sonnet's code execution capabilities..."
        
        Do not attempt to write or execute code yourself - this will be handled automatically.
        """)

    wiki_agent = create_react_agent(model=gpt,
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
        """)

    # Create supervisor workflow
    workflow = create_supervisor([search_agent, code_agent, wiki_agent],
                                 model=gpt,
                                 prompt=get_enhanced_supervisor_prompt(),
                                 output_mode="full_history",
                                 parallel_tool_calls=False)

    # Create memory saver and compile the workflow
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # Create a state store for CodeAct sessions
    codeact_states = {}

    # Now wrap the compiled app with our post-processing
    original_invoke = app.invoke
    original_stream = app.stream

    # Create wrapped versions of invoke and stream that apply post-processing and handle CodeAct
    def invoke_with_post_processing(input_data, config=None):
        # Extract session ID from config
        session_id = config.get("configurable", {}).get(
            "thread_id", "default") if config else "default"

        # Check if this is a code execution request
        if isinstance(input_data, dict) and "messages" in input_data:
            last_message = input_data["messages"][-1] if input_data[
                "messages"] else None

            # Check if the last message contains code keywords
            is_code_request = False
            if last_message:
                content = last_message[1] if isinstance(
                    last_message, tuple) else last_message.content
                code_indicators = [
                    "write a code", "create a program", "write code",
                    "code that", "write me a", "coding example",
                    "code example", "python script", "function that",
                    "class that"
                ]

                is_code_request = any(indicator in content.lower()
                                      for indicator in code_indicators)

                # For code requests, get a preliminary result to see if code_agent is triggered
                if is_code_request:
                    result = original_invoke(input_data, config)

                    # Check if code_agent was activated
                    if any("code_agent" in str(val)
                           for val in result.values()):
                        # This is a confirmed code request that triggered code_agent
                        # Process with CodeAct
                        if session_id not in codeact_states:
                            codeact_states[session_id] = {"messages": []}

                        # Add the user's message to CodeAct state
                        codeact_states[session_id]["messages"].append({
                            "role":
                            "user",
                            "content":
                            content
                        })

                        # Execute with CodeAct
                        codeact_result = create_codeact.invoke(
                            codeact_states[session_id])

                        # Update CodeAct state
                        codeact_states[session_id] = codeact_result

                        # Extract response
                        assistant_messages = [
                            msg for msg in codeact_result["messages"]
                            if isinstance(msg, dict)
                            and msg.get("role") == "assistant"
                        ]

                        if assistant_messages:
                            assistant_response = assistant_messages[-1][
                                "content"]

                            # Replace the code_agent result with the CodeAct result
                            for key in result:
                                if isinstance(
                                        result[key],
                                        dict) and "messages" in result[key]:
                                    messages = result[key]["messages"]
                                    for i, msg in enumerate(messages):
                                        if "code_agent" in str(msg):
                                            if isinstance(msg, AIMessage):
                                                result[key]["messages"][
                                                    i] = AIMessage(
                                                        content=
                                                        assistant_response)
                                            elif isinstance(
                                                    msg,
                                                    tuple) and len(msg) > 1:
                                                result[key]["messages"][i] = (
                                                    "assistant",
                                                    assistant_response)

                        return process_citations(result)

                    # If code_agent wasn't triggered, just return the result
                    return process_citations(result)

        # For non-code requests or code requests that don't trigger code_agent
        result = original_invoke(input_data, config)
        return process_citations(result)

    def stream_with_post_processing(input_data, config=None):
        # Extract session ID from config
        session_id = config.get("configurable", {}).get(
            "thread_id", "default") if config else "default"

        # Track if we've detected a code request
        code_request_detected = False
        code_agent_triggered = False
        user_message = ""

        # Check if this is a code execution request
        if isinstance(input_data, dict) and "messages" in input_data:
            last_message = input_data["messages"][-1] if input_data[
                "messages"] else None

            if last_message:
                user_message = last_message[1] if isinstance(
                    last_message, tuple) else last_message.content
                code_indicators = [
                    "write a code", "create a program", "write code",
                    "code that", "write me a", "coding example",
                    "code example", "python script", "function that",
                    "class that"
                ]

                code_request_detected = any(indicator in user_message.lower()
                                            for indicator in code_indicators)

        # Begin streaming
        accumulating_content = ""
        codeact_used = False

        for event in original_stream(input_data, config):
            processed_event = process_citations(event)

            # Check if code_agent is triggered during streaming
            if not code_agent_triggered and code_request_detected:
                for key, value in processed_event.items():
                    if isinstance(value, dict) and "messages" in value:
                        for msg in value["messages"]:
                            if "code_agent" in str(msg):
                                code_agent_triggered = True
                                break

            if code_agent_triggered and not codeact_used:
                # This is the first event after code agent was triggered
                codeact_used = True

                # Add the user's message to CodeAct state
                if session_id not in codeact_states:
                    codeact_states[session_id] = {"messages": []}

                codeact_states[session_id]["messages"].append({
                    "role":
                    "user",
                    "content":
                    user_message
                })

                # Execute with CodeAct - non-streaming for simplicity
                codeact_result = app.invoke(
                    codeact_states[session_id])

                # Update CodeAct state
                codeact_states[session_id] = codeact_result

                # Extract response
                assistant_messages = [
                    msg for msg in codeact_result["messages"]
                    if isinstance(msg, dict) and msg.get("role") == "assistant"
                ]

                if assistant_messages:
                    assistant_response = assistant_messages[-1]["content"]

                    # Create a modified event with the CodeAct result
                    modified_event = {}
                    for key, value in processed_event.items():
                        if isinstance(value, dict) and "messages" in value:
                            modified_messages = []
                            for msg in value["messages"]:
                                if "code_agent" in str(msg):
                                    if isinstance(msg, AIMessage):
                                        modified_messages.append(
                                            AIMessage(
                                                content=assistant_response))
                                    elif isinstance(msg,
                                                    tuple) and len(msg) > 1:
                                        modified_messages.append(
                                            ("assistant", assistant_response))
                                else:
                                    modified_messages.append(msg)

                            new_value = value.copy()
                            new_value["messages"] = modified_messages
                            modified_event[key] = new_value
                        else:
                            modified_event[key] = value

                    yield modified_event
                    continue

            # For regular events or if CodeAct wasn't involved
            yield processed_event

    # Replace the methods on the compiled app
    app.invoke = invoke_with_post_processing
    app.stream = stream_with_post_processing

    return app
