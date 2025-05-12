#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for the CopilotKit application.
This file runs the FastAPI server and can also be used to test the agent workflow.
"""

import argparse
import uvicorn
import os
import sys
import logging
from dotenv import load_dotenv
from app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def check_environment():
    """
    Check if all required environment variables are set.
    Returns True if environment is valid, False otherwise.
    """
    required_vars = ['OPENAI_API_KEY']
    optional_vars = ['GEMINI_API_KEY', 'TAVILY_API_KEY']
    
    missing_required = [var for var in required_vars if not os.getenv(var)]
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    
    if missing_required:
        for var in missing_required:
            logger.error(f"Required environment variable {var} is not set")
        logger.error("Please set the required environment variables in your .env file")
        return False
        
    if missing_optional:
        for var in missing_optional:
            logger.warning(f"Optional environment variable {var} is not set")
            
            if var == 'GEMINI_API_KEY':
                logger.warning("Code execution agent will be disabled or fall back to OpenAI")
                # Set a flag to indicate Gemini is not available
                os.environ['GEMINI_AVAILABLE'] = 'false'
            elif var == 'TAVILY_API_KEY':
                logger.warning("Web search functionality will be limited")
    
    return True

def test_workflow():
    """
    Run a simple test of the workflow.
    """
    from agent import get_workflow_app
    
    workflow = get_workflow_app()
    
    # Test queries
    test_queries = [
        "What is quantum computing?",
        "Write a Python function to calculate the Fibonacci sequence.",
        "Who was Albert Einstein?"
    ]
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        try:
            # Invoke the workflow with the query
            config = {"configurable": {"thread_id": "test"}}
            result = workflow.invoke({"messages": [("user", query)]}, config)
            
            # Log the result
            logger.info(f"Result: {result}")
            logger.info("Test completed successfully")
        except Exception as e:
            logger.error(f"Test failed: {e}")

def main():
    """
    Parse command line arguments and run the application or tests.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if the environment is properly configured
    env_valid = check_environment()
    if not env_valid:
        logger.warning("Continuing despite environment issues...")
    
    parser = argparse.ArgumentParser(description='Run the CopilotKit application or tests')
    parser.add_argument('--test', action='store_true', help='Run agent workflow tests')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    if args.test:
        # Run the workflow tests
        logger.info("Running workflow tests...")
        test_workflow()
        logger.info("Workflow tests completed.")
    else:
        # Run the FastAPI server
        logger.info(f"Starting CopilotKit server on http://{args.host}:{args.port}")
        logger.info("Press Ctrl+C to exit")
        uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()