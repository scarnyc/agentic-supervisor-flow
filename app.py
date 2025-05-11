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
    sessions[session_i