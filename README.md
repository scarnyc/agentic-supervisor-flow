# AI Copilot Multi-Agent Workflow Implementation

This project implements a multi-agent workflow that acts as an AI Copilot, capable of using multiple tools and specialized agents to assist users with a wide range of tasks.

## Architecture Overview

The system uses a supervisor-based architecture with three specialized agents:

1. **Search Agent (Gemini 2.0 Flash)** - Web search capabilities using Tavily API
2. **Code Agent (Claude 3.7 Sonnet)** - Code execution in a secure sandbox using CodeAct + langchain-sandbox
3. **Wiki Agent (GPT-4.1)** - Wikipedia knowledge access

The system is built using:
- **FastAPI** - For backend API and WebSockets
- **LangGraph** - For agent orchestration
- **CodeAct** - For structured code execution with Claude
- **Langchain-Sandbox** - For secure code execution

## Key Features

- **Enhanced UI** with clear indicators for different agent activities
- **Secure code execution** in an isolated sandbox environment
- **Streaming responses** via WebSockets
- **Citation formatting** for web search results
- **Multiple LLM integration** (Gemini, Claude, and GPT)

## Installation and Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following API keys:
```
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_anthropic_key
TAVILY_API_KEY=your_tavily_key
```

4. Run the server:
```bash
python main.py
```

## Requirements

- Python 3.9+
- FastAPI
- LangGraph
- Langchain
- LangGraph-CodeAct
- Langchain-Sandbox
- Anthropic Python SDK
- Google GenAI Python SDK

## Security Considerations

The code execution is performed in a secure subprocess sandbox with:
- Timeout limitations (30 seconds)
- Read-only file system access
- Restricted package access
- Proper logging of executed code

## UI Features

- Dark mode interface
- Real-time streaming of responses
- Visual indicators for agent activities
- Enhanced code formatting with language detection
- Citation formatting for search results

## Agent Capabilities

1. **Search Agent**
   - Web search with citation formatting
   - Source URL extraction and formatting
   - Result summarization

2. **Code Agent**
   - Code execution in a secure sandbox
   - Support for multiple programming languages
   - Error handling and explanation

3. **Wiki Agent**
   - Wikipedia article search and retrieval
   - Content summarization
   - Educational content generation

## Architecture Diagram

```
User Query
    │
    ▼
┌─────────────┐
│ Supervisor  │
│   (GPT-4.1) │
└─────────────┘
    │
    ├─────────────────┬─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│Search Agent │  │ Code Agent  │  │ Wiki Agent  │
│  (Gemini)   │  │  (Claude)   │  │  (GPT-4.1)  │
└─────────────┘  └─────────────┘  └─────────────┘
    │                 │                 │
    ▼                 ▼                 ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│Tavily Search│  │CodeAct+     │  │Wikipedia API│
│             │  │Sandbox      │  │             │
└─────────────┘  └─────────────┘  └─────────────┘
```