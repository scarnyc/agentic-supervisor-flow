from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
import uuid
import json
import traceback
import logging
from agent import get_workflow_app, process_citations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ])
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="CopilotKit API")

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
                if hasattr(last_message, 'content'):
                    # Handle ToolMessage or AIMessage objects
                    assistant_message = last_message.content
                elif isinstance(last_message, tuple) and len(last_message) > 1:
                    # Handle tuple format (role, content)
                    assistant_message = last_message[1]
                else:
                    # Fallback - try to convert to string
                    assistant_message = str(last_message)

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


# WebSocket endpoint for streaming responses
@app.websocket("/api/chat/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    # Create session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = {"messages": []}

    # Store the connection
    active_connections[session_id] = websocket

    try:
        while True:
            # Wait for messages from the client
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

                                    # Debug logging (with sensitive info redacted)
                                    logger.debug(
                                        f"Message type: {type(last_message)}")

                                    # Extract content based on message type
                                    if hasattr(last_message, 'content'):
                                        new_content = last_message.content
                                    elif isinstance(
                                            last_message,
                                            tuple) and len(last_message) > 1:
                                        new_content = last_message[1]
                                    else:
                                        new_content = str(last_message)

                                    # Ensure content is a string
                                    if not isinstance(new_content, str):
                                        new_content = str(new_content)

                                    # Process citations in the new content
                                    new_content = process_citations({
                                        "messages":
                                        [("assistant", new_content)]
                                    })
                                    if isinstance(
                                            new_content, dict
                                    ) and "messages" in new_content and new_content[
                                            "messages"]:
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
                                    # Error handling for message processing
                                    logger.error(
                                        f"Error processing message: {e}")
                                    traceback.print_exc()

                                    # Notify the client
                                    await websocket.send_json({
                                        "type": "error",
                                        "message": {
                                            "role":
                                            "system",
                                            "content":
                                            "An error occurred while processing the response."
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
                    # Error handling for workflow streaming
                    logger.error(f"Error in workflow streaming: {e}")
                    traceback.print_exc()

                    error_message = "Sorry, I encountered an error processing your request."

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
                # Handle disconnect within the message loop
                logger.info(
                    f"WebSocket disconnected for session: {session_id}")
                break

            except Exception as e:
                # Handle other exceptions within the message loop
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
        # Remove the connection when disconnected (outer exception handler)
        logger.info(f"WebSocket disconnected for session: {session_id}")

    except Exception as e:
        # Catch-all for any other exceptions in the WebSocket endpoint
        logger.error(f"Unexpected error in WebSocket connection: {e}")
        traceback.print_exc()

    finally:
        # Clean up: always remove the connection when done
        if session_id in active_connections:
            del active_connections[session_id]
        logger.info(f"Cleaned up connection for session: {session_id}")
