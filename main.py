#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for the AI by Design Agentic application.
This file runs the FastAPI server and can also be used to test the agent workflow.
"""

import argparse
import uvicorn
import sys
import os
import logging
import traceback
from contextlib import contextmanager
from app import app

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add handlers
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


@contextmanager
def exception_handler():
    """Context manager to handle exceptions gracefully"""
    try:
        yield
    except KeyboardInterrupt:
        logger.info("Application terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)


def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        for var in missing_vars:
            logger.error(f"Required environment variable {var} is not set")
        logger.error(
            "Please set the required environment variables in your .env file")
        return False

    # Check optional variables
    optional_vars = {
        "GEMINI_API_KEY": "Google Gemini API for code agent",
        "OPENAI_API_KEY": "OPEN AI API for code agent",
        "TAVILY_API_KEY": "Tavily Search API for web search agent"
    }

    for var, purpose in optional_vars.items():
        if not os.getenv(var):
            logger.warning(
                f"Optional environment variable {var} is not set ({purpose})")

    return True


def test_workflow():
    """Run tests on the agent workflow"""
    try:
        from app import workflow_app

        # Test basic functionality
        logger.info("Testing basic agent functionality...")

        # Test query that should use the search agent
        test_queries = [
            "What is the latest news about AI?",
            "Who is the current CEO of Microsoft?",
            "Write a function to calculate fibonacci numbers in Python"
        ]

        for query in test_queries:
            logger.info(f"Testing query: {query}")
            try:
                result = workflow_app.invoke(
                    {"messages": [("user", query)]},
                    {"configurable": {
                        "thread_id": "test"
                    }})
                logger.info("Query processed successfully")

                # Check if we got a response
                has_response = False
                for value in result.values():
                    if "messages" in value and value["messages"]:
                        has_response = True
                        break

                if has_response:
                    logger.info("Test passed: Received a response")
                else:
                    logger.error("Test failed: No response received")

            except Exception as e:
                logger.error(f"Test failed for query '{query}': {e}")

        logger.info("Workflow tests completed.")
    except ImportError:
        logger.error("Could not import workflow_app from app module")
    except Exception as e:
        logger.error(f"Error during workflow testing: {e}")
        traceback.print_exc()


def main():
    """
    Parse command line arguments and run the application or tests.
    """
    with exception_handler():
        parser = argparse.ArgumentParser(
            description='Run the Agentic application or tests')
        parser.add_argument('--test',
                            action='store_true',
                            help='Run agent workflow tests')
        parser.add_argument('--host',
                            type=str,
                            default='0.0.0.0',
                            help='Host to run the server on')
        parser.add_argument('--port',
                            type=int,
                            default=5000,
                            help='Port to run the server on')
        parser.add_argument('--reload',
                            action='store_true',
                            help='Enable auto-reload for development')
        parser.add_argument(
            '--log-level',
            type=str,
            default='info',
            choices=['debug', 'info', 'warning', 'error', 'critical'],
            help='Set logging level')

        args = parser.parse_args()

        # Set log level
        log_levels = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        logging.getLogger().setLevel(log_levels[args.log_level])

        # Check environment before proceeding
        if not check_environment():
            logger.warning("Continuing despite environment issues...")

        if args.test:
            # Run the workflow tests
            logger.info("Running workflow tests...")
            test_workflow()
        else:
            # Run the FastAPI server
            try:
                logger.info(
                    f"Starting Agentic server on http://{args.host}:{args.port}"
                )
                logger.info("Press Ctrl+C to exit")

                # The key change: use app:app instead of main:app
                # This directly runs the FastAPI instance from app.py
                uvicorn.run("app:app",
                            host=args.host,
                            port=args.port,
                            reload=args.reload,
                            log_level=args.log_level)
            except Exception as e:
                logger.error(f"Failed to start server: {e}")
                traceback.print_exc()
                sys.exit(1)


if __name__ == "__main__":
    main()
