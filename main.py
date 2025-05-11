#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for the CopilotKit application.
This file runs the FastAPI server and can also be used to test the agent workflow.
"""

import argparse
import uvicorn
from app import app
from agent_workflow import test_workflow

def main():
    """
    Parse command line arguments and run the application or tests.
    """
    parser = argparse.ArgumentParser(description='Run the CopilotKit application or tests')
    parser.add_argument('--test', action='store_true', help='Run agent workflow tests')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    if args.test:
        # Run the workflow tests
        print("Running workflow tests...")
        test_workflow()
        print("Workflow tests completed.")
    else:
        # Run the FastAPI server
        print(f"Starting CopilotKit server on http://{args.host}:{args.port}")
        uvicorn.run("app:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()