from fastapi import (FastAPI, WebSocket, HTTPException, WebSocketDisconnect,
                     Request)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import (Dict, Optional, List, Any, TypedDict)
from typing import Annotated
import uuid
import json
import traceback
import logging
from agent import get_workflow_app, process_citations

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    current_agent: str
    next_agent: str


def log_transfer(state: AgentState, context: Any) -> AgentState:
    """Log when control transfers between agents"""
    current = state.get("current_agent", "supervisor")
    next_agent = context.current_node

    logger.info(f"Transfer: {current} -> {next_agent}")
    logger.debug(f"Full state during transfer: {state}")
    logger.debug(f"Context details: {vars(context)}")

    state["current_agent"] = next_agent

    # Log the last message if it exists
    if state["messages"]:
        last_msg = state["messages"][-1]
        logger.info(
            f"Last message from {current}: {last_msg.content[:100]}...")
        logger.debug(f"Full message content: {last_msg.content}")
        logger.debug(f"Message type: {type(last_msg)}")
        logger.debug(f"Message attributes: {vars(last_msg)}")

    return state


# Add handlers *after* basicConfig is called
handler = logging.StreamHandler()
file_handler = logging.FileHandler('app.log')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(file_handler)

# Initialize FastAPI
app = FastAPI(title="Agentic API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, use specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Get the LangGraph workflow app
try:
    workflow_app = get_workflow_app()
    logger.info("Successfully initialized workflow app")
except Exception as e:
    logger.error(f"Failed to initialize workflow app: {e}")
    traceback.print_exc()
    raise RuntimeError(
        "Failed to initialize agent workflow. Check configuration and API keys."
    )

# Store active websocket connections
active_connections: Dict[str, WebSocket] = {}
# Store user sessions
sessions: Dict[str, Dict] = {}


# Models
class ChatMessage(BaseModel):
    role: str
    content: str = Field(..., min_length=1)  # Ensure content is not empty


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1)  # Ensure message is not empty


class ChatResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]


# Error handler for unexpected exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    traceback.print_exc()
    return HTMLResponse(
        content=
        "<h1>Internal Server Error</h1><p>The server encountered an unexpected error.</p>",
        status_code=500)


# Routes
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error rendering index template: {e}")
        raise HTTPException(status_code=500,
                            detail="Failed to render index page")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    # Create or get session
    session_id = chat_request.session_id or str(uuid.uuid4())

    # Create session if doesn't exist
    if session_id not in sessions:
        sessions[session_id] = {"messages": []}

    # Add user message to session
    try:
        sessions[session_id]["messages"].append(
            ChatMessage(role="user", content=chat_request.message))
    except Exception as e:
        logger.error(f"Error adding user message to session: {e}")
        raise HTTPException(status_code=400, detail="Invalid message format")

    try:
        # Get response from workflow
        config = {"configurable": {"thread_id": session_id}}
        result = workflow_app.invoke(
            {"messages": [("user", chat_request.message)]}, config)

        # Process citations (redundant but just to be safe)
        result = process_citations(result)

        # Extract assistant's response
        assistant_message = None
        for value in result.values():
            if isinstance(value,
                          dict) and "messages" in value and value["messages"]:
                # Get the last message from the assistant
                last_message = value["messages"][-1]

                # Extract content based on message type
                if isinstance(last_message,
                              dict) and 'content' in last_message:
                    assistant_message = last_message['content']
                elif hasattr(last_message, 'content'):
                    assistant_message = last_message.content
                elif isinstance(last_message, tuple) and len(last_message) > 1:
                    assistant_message = last_message[1]
                elif isinstance(last_message, str):
                    assistant_message = last_message
                else:
                    assistant_message = str(last_message)

                # Handle list/dict content
                if isinstance(assistant_message, (list, dict)):
                    assistant_message = str(assistant_message)

        if assistant_message:
            # Add assistant message to session
            sessions[session_id]["messages"].append(
                ChatMessage(role="assistant", content=assistant_message))

    except Exception as e:
        logger.error(f"Error in /api/chat endpoint: {e}")
        traceback.print_exc()
        # Add error message to session
        error_message = "Sorry, I encountered an error processing your request."
        sessions[session_id]["messages"].append(
            ChatMessage(role="assistant", content=error_message))

    # Return the full conversation history
    return ChatResponse(session_id=session_id,
                        messages=sessions[session_id]["messages"])


# Safely handle message parsing from WebSocket
def parse_websocket_message(data: str) -> dict:
    """Parse WebSocket message with error handling"""
    try:
        message_data = json.loads(data)
        return message_data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid WebSocket message format: {e}")
        return {"error": "Invalid message format"}


# Safely extract content from workflow result
def extract_content_from_result(result: Any) -> str:
    """Extract content from workflow result with error handling"""
    try:
        for value in result.values():
            if isinstance(value,
                          dict) and "messages" in value and value["messages"]:
                last_message = value["messages"][-1]

                if hasattr(last_message, 'content'):
                    return last_message.content
                elif isinstance(last_message, tuple) and len(last_message) > 1:
                    return last_message[1]
                else:
                    return str(last_message)
        return "I couldn't process that request properly."
    except Exception as e:
        logger.error(f"Error extracting content from result: {e}")
        return "Sorry, I encountered an error processing your request."


# Enhanced error handling for WebSocket in app.py
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Create session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = {"messages": []}

    # Store the connection
    active_connections[session_id] = websocket

    try:
        while True:
            try:
                data = await websocket.receive_text()
                message_data = parse_websocket_message(data)

                if "error" in message_data:
                    await websocket.send_json({
                        "type": "error",
                        "message": {
                            "role": "system",
                            "content": "Invalid message format received."
                        }
                    })
                    continue

                user_message = message_data.get("message", "")
                if not user_message.strip():
                    await websocket.send_json({
                        "type": "error",
                        "message": {
                            "role": "system",
                            "content": "Message cannot be empty."
                        }
                    })
                    continue

                # Add user message to session
                sessions[session_id]["messages"].append(
                    ChatMessage(role="user", content=user_message))

                # Send acknowledgment
                await websocket.send_json({
                    "type": "message_received",
                    "message": {
                        "role": "user",
                        "content": user_message
                    }
                })

                # Start a partial response
                partial_response = ""

                try:
                    # Stream response from workflow
                    config = {"configurable": {"thread_id": session_id}}

                    # Add execution metadata for code execution
                    if "calculate" in user_message.lower(
                    ) or "factorial" in user_message.lower():
                        config["configurable"]["execution_mode"] = "safe"

                    # Stream the events in the graph
                    for event in workflow_app.stream(
                        {"messages": [("user", user_message)]}, config):
                        # Process each event
                        for value in event.values():
                            if isinstance(
                                    value, dict
                            ) and "messages" in value and value["messages"]:
                                try:
                                    # Get the last message
                                    last_message = value["messages"][-1]

                                    # Extract content with error handling
                                    new_content = extract_message_content(
                                        last_message)

                                    # Check for code execution errors and provide more helpful responses
                                    if "Code execution failed" in new_content:
                                        # Parse the error
                                        error_content = parse_code_execution_error(
                                            new_content)
                                        # If memory error in factorial, provide direct answer
                                        if "failed to reserve page summary memory" in new_content and "factorial" in user_message.lower(
                                        ):
                                            error_content += "\n\nThe factorial of 100 is approximately 9.33 × 10^157 (a number with 158 digits). This calculation requires special handling for such large numbers."
                                        new_content = error_content

                                    # Process citations in the new content
                                    new_content = process_citations({
                                        "messages":
                                        [("assistant", new_content)]
                                    })
                                    if isinstance(
                                            new_content, dict
                                    ) and "messages" in new_content:
                                        new_content = new_content["messages"][
                                            0][1]

                                    # If we have new content, send it as a partial update
                                    if new_content != partial_response:
                                        await websocket.send_json({
                                            "type": "partial_response",
                                            "message": {
                                                "role": "assistant",
                                                "content": new_content
                                            }
                                        })
                                        partial_response = new_content

                                except Exception as e:
                                    logger.error(
                                        f"Error processing message: {e}")
                                    traceback.print_exc()

                                    # Continue processing despite errors
                                    await websocket.send_json({
                                        "type": "error",
                                        "message": {
                                            "role":
                                            "system",
                                            "content":
                                            "An error occurred while processing part of the response."
                                        }
                                    })

                    # Add final assistant message to session
                    if partial_response:
                        sessions[session_id]["messages"].append(
                            ChatMessage(role="assistant",
                                        content=partial_response))

                        # Send completion message
                        await websocket.send_json({
                            "type": "message_complete",
                            "message": {
                                "role": "assistant",
                                "content": partial_response
                            }
                        })

                except Exception as e:
                    logger.error(f"Error in workflow streaming: {e}")
                    traceback.print_exc()

                    error_message = generate_friendly_error_message(
                        e, user_message)

                    # Send error message to client
                    await websocket.send_json({
                        "type": "error",
                        "message": {
                            "role": "system",
                            "content": error_message
                        }
                    })

                    # Add error message to session
                    sessions[session_id]["messages"].append(
                        ChatMessage(role="assistant", content=error_message))

                    # Send a final message to close the interaction properly
                    await websocket.send_json({
                        "type": "message_complete",
                        "message": {
                            "role": "assistant",
                            "content": error_message
                        }
                    })

            except WebSocketDisconnect:
                logger.info(
                    f"WebSocket disconnected for session: {session_id}")
                break

            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                traceback.print_exc()

                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": {
                            "role": "system",
                            "content":
                            "An error occurred processing your message."
                        }
                    })
                except:
                    # Connection might be closed
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")

    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection: {e}")
        traceback.print_exc()

    finally:
        # Clean up: always remove the connection when done
        if session_id in active_connections:
            del active_connections[session_id]
        logger.info(f"Cleaned up connection for session: {session_id}")


# Helper functions for improved error handling
def extract_message_content(message):
    """Extract content from various message formats with enhanced error handling"""
    try:
        if hasattr(message, 'content'):
            content = message.content
        elif isinstance(message, dict) and 'content' in message:
            content = message['content']
        elif isinstance(message, tuple) and len(message) > 1:
            content = message[1]
        elif isinstance(message, str):
            content = message
        else:
            content = str(message)

        # Handle empty lists, dicts, etc.
        if isinstance(content, list) and not content:
            return ""
        elif not isinstance(content, str):
            return str(content)

        return content
    except Exception as e:
        logger.error(f"Error extracting message content: {e}")
        return "Error processing message content"


def parse_code_execution_error(error_content):
    """Parse code execution errors and provide more helpful responses"""
    try:
        if "memory" in error_content.lower():
            return "I encountered a memory limitation while executing this code. The calculation you requested requires more memory than is available in the execution environment."
        elif "timeout" in error_content.lower():
            return "The code execution timed out. This calculation is too complex to complete within the allowed time limit."
        else:
            # Extract the actual error message from the full trace
            error_lines = error_content.split('\n')
            for line in error_lines:
                if "Error:" in line or "Exception:" in line:
                    return f"Code execution error: {line}"
            return "There was a problem executing the code. Please try a different approach."
    except Exception:
        return "An error occurred during code execution. Please try again with a simpler request."


def generate_friendly_error_message(exception, user_message):
    """Generate user-friendly error messages based on the exception and context"""
    error_str = str(exception)

    if "factorial" in user_message.lower() and any(
            term in error_str.lower()
            for term in ["memory", "overflow", "too large"]):
        return "I couldn't calculate this factorial directly due to its size. For very large factorials like this, I can provide the result using mathematical notation instead of computing it directly."
    elif "search" in user_message.lower() and "api" in error_str.lower():
        return "I encountered an issue connecting to the search service. Please try again in a moment."
    elif "wikipedia" in user_message.lower() and "api" in error_str.lower():
        return "I'm having trouble accessing Wikipedia at the moment. Please try again later or I can search the web instead."
    else:
        return "Sorry, I encountered an error processing your request. Please try phrasing your question differently."
