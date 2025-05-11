from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, Optional, List
import uuid
import json
import os
import traceback
from agent import get_workflow_app

# Import Request for type annotation
from fastapi import Request

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
workflow_app = get_workflow_app()

# Store active websocket connections
active_connections: Dict[str, WebSocket] = {}
# Store user sessions
sessions: Dict[str, Dict] = {}

# Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    messages: List[ChatMessage]

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    # Create or get session
    session_id = chat_request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = {"messages": []}
    
    # Add user message to session
    sessions[session_id]["messages"].append(
        ChatMessage(role="user", content=chat_request.message)
    )
    
    try:
        # Get response from workflow
        config = {"configurable": {"thread_id": session_id}}
        result = workflow_app.invoke(
            {"messages": [("user", chat_request.message)]}, 
            config
        )
        
        # Extract assistant's response
        assistant_message = None
        for value in result.values():
            if "messages" in value and value["messages"]:
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
                ChatMessage(role="assistant", content=assistant_message)
            )
    
    except Exception as e:
        print(f"Error in /api/chat endpoint: {e}")
        traceback.print_exc()
        # Add error message to session
        sessions[session_id]["messages"].append(
            ChatMessage(role="assistant", content="Sorry, I encountered an error processing your request.")
        )
    
    # Return the full conversation history
    return ChatResponse(
        session_id=session_id,
        messages=sessions[session_id]["messages"]
    )

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
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            # Add user message to session
            sessions[session_id]["messages"].append(
                ChatMessage(role="user", content=user_message)
            )
            
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
                    {"messages": [("user", user_message)]},
                    config
                ):
                    # Process each event
                    for value in event.values():
                        if "messages" in value and value["messages"]:
                            try:
                                # Get the last message
                                last_message = value["messages"][-1]
                                
                                # Debug logging
                                print(f"Message type: {type(last_message)}")
                                print(f"Message structure: {last_message}")
                                if hasattr(last_message, '__dict__'):
                                    print(f"Message attributes: {last_message.__dict__}")
                                
                                # Extract content based on message type
                                if hasattr(last_message, 'content'):
                                    # Handle ToolMessage or AIMessage objects
                                    new_content = last_message.content
                                elif isinstance(last_message, tuple) and len(last_message) > 1:
                                    # Handle tuple format (role, content)
                                    new_content = last_message[1]
                                else:
                                    # Fallback - try to convert to string
                                    new_content = str(last_message)
                                
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
                                print(f"Error processing message: {e}")
                                traceback.print_exc()
                                
                                # Notify the client
                                await websocket.send_json({
                                    "type": "error",
                                    "message": {
                                        "role": "system",
                                        "content": "An error occurred while processing the response."
                                    }
                                })
                
                # Add final assistant message to session
                if partial_response:
                    sessions[session_id]["messages"].append(
                        ChatMessage(role="assistant", content=partial_response)
                    )
                    
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
                print(f"Error in workflow streaming: {e}")
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
                    ChatMessage(role="assistant", content=error_message)
                )
                
                # Send a final message to close the interaction properly
                await websocket.send_json({
                    "type": "message_complete",
                    "message": {
                        "role": "assistant",
                        "content": error_message
                    }
                })
    
    except WebSocketDisconnect:
        # Remove the connection when disconnected
        print(f"WebSocket disconnected for session: {session_id}")
        if session_id in active_connections:
            del active_connections[session_id]
    
    except Exception as e:
        # Catch-all for any other exceptions
        print(f"Unexpected error in WebSocket connection: {e}")
        traceback.print_exc()
        
        # Try to send a final error message if possible
        try:
            await websocket.send_json({
                "type": "error",
                "message": {
                    "role": "system",
                    "content": "An unexpected error occurred with the connection."
                }
            })
        except:
            pass  # Connection might already be closed