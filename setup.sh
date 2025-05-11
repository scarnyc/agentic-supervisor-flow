#!/bin/bash

# Create directory structure
mkdir -p templates
mkdir -p static/css
mkdir -p static/js

# Create .env file with placeholders
cat > .env << EOL
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
EOL

# Create requirements.txt
cat > requirements.txt << EOL
fastapi
uvicorn
pydantic
python-dotenv
langchain
langchain-openai
langchain-google-genai
langchain-community
langgraph
langgraph-supervisor
jinja2
wikipedia
google-generativeai
EOL

echo "Directory structure created."
echo "Please run: pip install -r requirements.txt"
echo "Then update the .env file with your actual API keys."
echo ""
echo "To start the application, run one of the following:"
echo "1. python main.py                 # Run the server"
echo "2. python main.py --reload        # Run the server with auto-reload for development"
echo "3. python main.py --test          # Run only the workflow tests"
echo "4. python main.py --host 127.0.0.1 --port 5000  # Custom host/port"