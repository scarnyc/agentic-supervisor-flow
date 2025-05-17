# AI Copilot Multi-Agent Workflow Implementation

This project implements a multi-agent workflow that acts as an AI Copilot, capable of using multiple tools and specialized agents to assist users with a wide range of tasks.

## Architecture Overview

The system uses a supervisor-based architecture with specialized agents:

1. **Search Agent (Claude 3.7)** - Web search capabilities using Tavily API
2. **Code Agent (Claude 3.7)** - Secure code execution with Python REPL
3. **Wiki Agent (Claude 3.7)** - Wikipedia knowledge access

The system is built using:
- **FastAPI** - For backend API and WebSockets
- **LangGraph** - For agent orchestration
- **LangGraph-Supervisor** - For agent coordination
- **Uvicorn** - ASGI web server

## Key Features

- **Reasoning** with Claude 3.7 Sonnet
- **Real-time streaming** responses via WebSockets
- **Citation formatting** for web search results
- **Secure code execution** via Python REPL
- **Dark mode interface** with enhanced code formatting
- **Source attribution** with proper citation formatting

## Installation

1. Clone or fork this repository
2. Run the setup script:
```bash
bash setup.sh
```

3. Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your_anthropic_key
TAVILY_API_KEY=your_tavily_key
```

4. Run the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 5000
```

## Requirements

- Python 3.12+
- FastAPI
- LangGraph & LangGraph-Supervisor
- Langchain
- Anthropic Python SDK
- Tavily Search API

## Security Features

- Secure code execution in Python REPL
- Environment variable protection
- WebSocket connection handling
- Input validation and sanitization

## UI Features

- Responsive dark mode interface
- Code syntax highlighting
- Citation formatting
- Real-time response streaming
- Enhanced message styling

## Architecture Diagram

```
User Query
    │
    ▼
┌───────────────┐
│ Supervisor    │
│ (Claude 3.7)  │
└───────────────┘
    │
    ├─────────────────┬─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│Search Agent │  │ Code Agent  │  │ Wiki Agent  │
│ (Claude 3.7)│  │(Claude 3.7) │  │(Claude 3.7) │
└─────────────┘  └─────────────┘  └─────────────┘
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│Tavily Search│  │Python REPL  │  │Wikipedia API│
└─────────────┘  └─────────────┘  └─────────────┘

```
## Roadmap

### v0
- Caching Results: Add a simple cache for commonly requested information to reduce API calls
- Progressive Enhancement: In the frontend, show typing indicators during tool transitions for a more natural feel
- Error Recovery: Implement automatic retries for temporary API failures
- User Feedback Loop: Add a thumbs up/down mechanism to collect feedback on answers

### v1
- Generative UI
- Support for Gemini 2.5 Pro Reasoning vs. standard responses
- Support for GPT 4.1 for writing
- Long Memory
  
### v2
- Human in the loop (stop and ask for input)
- Log-in screen with Google oAuth for sign-in
- MCP Servers
- File System
- App optimized for security, speed & efficiency

### v3
- Planning: research, generation, reflection
- RAG, Deep Research w/ Perplexity
- Upgraded web search with Google SerpAPI
- Persist user Chat history (UI)
- Evals, monitoring & logging
- Experiment with thinking budget / prompt caching
- Show thinking output

### V4
- Slack, LinkedIn, gmail, Nasa toolkit, Substack
- User-input OpenAI / Gemini API Key
- Security with Cloudflare
```
