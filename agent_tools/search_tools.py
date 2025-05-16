# agent_tools/search_tools.py

import re
import logging
import json
from typing import Dict, Any, List
from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

class TavilyResultTracker:
    """Class to track Tavily search results."""

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


def process_search_results(results, max_tokens=2000):
    """
    Process and truncate search results to stay within token budget.

    Args:
        results: Raw search results from Tavily
        max_tokens: Maximum tokens to return

    Returns:
        Processed and truncated search results
    """
    processed_results = []
    estimated_tokens = 0

    # Handle different result formats
    if isinstance(results, dict) and 'results' in results:
        results_list = results['results']
    elif isinstance(results, list):
        results_list = results
    else:
        logger.warning(f"Unexpected results format: {type(results)}")
        return results  # Return as-is if format unknown

    # Sort results by relevance score if available
    if all('score' in r for r in results_list):
        results_list = sorted(results_list, key=lambda x: x.get('score', 0), reverse=True)

    for result in results_list:
        # Estimate tokens in this result (rough estimate: 1 token ≈ 4 chars)
        content = result.get('content', '')
        result_tokens = len(content) // 4

        # If adding this result would exceed our budget, truncate it
        if estimated_tokens + result_tokens > max_tokens:
            # Calculate how many tokens we can still add
            remaining_tokens = max_tokens - estimated_tokens
            # Truncate the content to fit (with a small buffer)
            truncated_length = max(0, remaining_tokens * 3)  # Using 3 not 4 for safety margin
            result['content'] = content[:truncated_length] + "..." if truncated_length > 0 else ""
            estimated_tokens = max_tokens
            processed_results.append(result)
            break

        estimated_tokens += result_tokens
        processed_results.append(result)

        # Stop if we've hit our token budget
        if estimated_tokens >= max_tokens:
            break

    logger.info(f"Processed search results: {len(processed_results)} items, ~{estimated_tokens} tokens")
    return processed_results


def extract_key_facts(search_results, max_facts=5):
    """
    Extract key facts from search results for better citation.

    Args:
        search_results: Processed search results
        max_facts: Maximum number of key facts to extract

    Returns:
        List of key facts with their sources
    """
    facts = []

    for i, result in enumerate(search_results):
        # Extract the source URL
        source_url = result.get('url', '')
        content = result.get('content', '')

        # Simple sentence splitting (could be improved with NLP)
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]

        # Take up to 2 key sentences from each result
        for j, sentence in enumerate(sentences[:2]):
            if len(facts) >= max_facts:
                break

            facts.append({
                'content': sentence,
                'source_index': f"{i}-{j}",
                'url': source_url
            })

    return facts


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

        return urls[:5]  # Limit to 5 URLs to save tokens
    except Exception as e:
        logger.error(f"Error in extract_urls_from_tavily_result: {e}")
        return []


def format_citations(content: str) -> str:
    """
    Format citations in the content string with token optimization.

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
        search_results_match = re.search(search_results_pattern, content,
                                     re.DOTALL | re.IGNORECASE)

        if search_results_match:
            search_results_text = search_results_match.group(1).strip()

            # Limit the cited text to a reasonable length
            if len(search_results_text) > 500:
                search_results_text = search_results_text[:500] + "..."

            # Replace with properly cited content - cite the whole paragraph for now
            cited_text = f"<cite index=\"0-1\">{search_results_text}</cite>"

            # Replace the original text with cited text
            content = content.replace(search_results_text, cited_text)

        # Also look for statements that reference sources - limit to a few patterns
        source_patterns = [
            r"According to (.*?), (.*?)\.", 
            r"(.*?) reports that (.*?)\.",
            r"As mentioned in (.*?), (.*?)\."
        ]

        for i, pattern in enumerate(source_patterns):
            # Limit to 2 matches per pattern to save tokens
            matches = 0
            for match in re.finditer(pattern, content, re.DOTALL):
                if matches >= 2:
                    break

                full_match = match.group(0)
                if "<cite" not in full_match:  # Avoid double-citing
                    # Limit citation length
                    if len(full_match) > 200:
                        full_match_short = full_match[:200] + "..."
                        content = content.replace(full_match, full_match_short)
                        full_match = full_match_short

                    cited_match = f"<cite index=\"{i+1}-1\">{full_match}</cite>"
                    content = content.replace(full_match, cited_match)
                    matches += 1

        # Add source URLs if we found any
        if urls:
            if "Sources:" not in content:
                sources_text = "\n\nSources:\n"
                for url in urls:
                    sources_text += f"{url}\n"

                # Add the sources section
                content += sources_text

        return content
    except Exception as e:
        logger.error(f"Error in format_citations: {e}")
        return content  # Return original content if processing fails


def create_tavily_search_tool(tavily_api_key):
    """
    Create the Tavily search tool with token management.

    Args:
        tavily_api_key: API key for Tavily

    Returns:
        Configured search tool or None if failed
    """
    try:
        from langchain_community.tools.tavily_search.tool import TavilySearchResults

        # Create a wrapper function around the TavilySearchResults
        def tavily_search_with_processing(*args, **kwargs):
            try:
                results = TavilySearchResults(
                    api_key=tavily_api_key,
                    k=5,
                    include_raw_content=True,
                    include_images=False,
                    include_answer=True,
                    max_results=5,
                    search_depth="advanced"
                )(*args, **kwargs)

                # Process and limit the size of results
                return process_search_results(results, max_tokens=2000)
            except Exception as e:
                logger.error(f"Error in Tavily search: {e}")
                # Return fallback results
                return [{
                    "url": "https://en.wikipedia.org/wiki/Search_error",
                    "content": "Search encountered an error. Please try again with a more specific query.",
                    "title": "Search Error"
                }]

        # Create the tool with our wrapped function
        search_tool = Tool(
            name="tavily_search_results",
            func=tavily_search_with_processing,
            description="Search the web for current information. Useful for questions about current events or trending topics."
        )

        logger.info("Successfully initialized Tavily Search with token management")
        return search_tool

    except Exception as e:
        logger.error(f"Failed to initialize Tavily Search: {e}")
        return None


def process_citations_for_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process citations in agent responses with token optimization.

    Args:
        response: The response dictionary from an agent

    Returns:
        Processed response with formatted citations
    """
    if not isinstance(response, dict) or "messages" not in response:
        return response

    try:
        from langchain_core.messages import AIMessage

        processed_messages = []
        sources_set = set()  # Use a set to avoid duplicate sources

        for message in response["messages"]:
            is_ai_message = isinstance(message, AIMessage)
            is_assistant_tuple = (isinstance(message, tuple) and len(message) > 1 and message[0] == "assistant")

            if is_ai_message or is_assistant_tuple:
                # Extract the content
                content = message.content if is_ai_message else message[1]

                # Skip processing for very short messages (often system messages)
                if len(content) < 50:
                    processed_messages.append(message)
                    continue

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

                # Process citations
                content = format_citations(content)

                # Extract source URLs - look for URLs in citation patterns or directly
                found_urls = re.findall(r'https?://[^\s\'"]+', content)
                for url in found_urls:
                    sources_set.add(url)

                # Create a limited sources section if needed
                if sources_set and "Sources:" not in content:
                    sources_text = "\n\nSources:\n"
                    for url in list(sources_set)[:5]:  # Limit to top 5 sources
                        sources_text += f"{url}\n"

                    content += sources_text

                # Ensure we're not losing substantive content when the supervisor
                # only returns the sources
                if content.strip().startswith("Sources:") and len(content.strip().split("\n")) <= 4:
                    for prev_message in response["messages"]:
                        # Look for a previous message with substantial content
                        prev_content = prev_message.content if isinstance(prev_message, AIMessage) else (prev_message[1] if is_assistant_tuple else "")

                        # If we find a substantive previous message that isn't just a source citation
                        if (prev_content and 
                            len(prev_content.strip()) > 100 and 
                            not prev_content.strip().startswith("Sources:") and
                            not "Using Web Search Tool" in prev_content and
                            not "Using Wikipedia Tool" in prev_content):

                            # Append sources to the substantive content
                            if "Sources:" not in prev_content:
                                prev_content += "\n\n" + content
                                content = prev_content
                            break

                # Recreate message with updated content
                if is_ai_message:
                    processed_message = AIMessage(content=content)
                else:
                    processed_message = ("assistant", content)

                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)

        response["messages"] = processed_messages
        return response
    except Exception as e:
        logger.error(f"Error in process_citations_for_response: {e}")
        return response  # Return original response if processing fails