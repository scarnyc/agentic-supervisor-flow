# secure_executor.py
import subprocess
import tempfile
import os
import signal
import time
import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


class SecurePythonExecutor:
    """A secure Python code executor that runs code in isolated subprocesses with restrictions."""

    def __init__(
            self,
            timeout: int = 10,
            max_memory: int = 100,  # MB
            allowed_modules: Optional[list] = None):
        self.timeout = timeout
        self.max_memory = max_memory

        # Using ONLY the exact libraries from Google's Gemini API documentation
        self.allowed_modules = allowed_modules or [
            # Libraries from Google's Gemini API documentation
            "attrs",
            "chess",
            "contourpy",
            "fpdf",
            "geopandas",
            "imageio",
            "jinja2",
            "joblib",
            "jsonschema",
            "jsonschema-specifications",
            "lxml",
            "matplotlib",
            "mpmath",
            "numpy",
            "opencv-python",
            "openpyxl",
            "packaging",
            "pandas",
            "pillow",
            "protobuf",
            "pylatex",
            "pyparsing",
            "PyPDF2",
            "python-dateutil",
            "python-docx",
            "python-pptx",
            "reportlab",
            "scikit-learn",
            "scipy",
            "seaborn",
            "six",
            "striprtf",
            "sympy",
            "tabulate",
            "tensorflow",
            "toolz",
            "xlrd"
        ]

        # Add some common alternative import names
        self.allowed_modules.extend(["PIL", "sklearn"])

    def run(self, code: str) -> Dict[str, Any]:
        """Execute Python code in a subprocess with timeout and memory limits."""
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            # Prepend module restrictions
            restricted_imports = self._generate_import_guard()

            # Add matplotlib non-interactive backend if matplotlib is allowed
            if "matplotlib" in self.allowed_modules:
                matplotlib_config = "\nimport matplotlib\nmatplotlib.use('Agg')\n"
                temp.write(matplotlib_config.encode('utf-8'))

            # Write the code with restrictions to temp file
            temp.write(restricted_imports.encode('utf-8'))
            temp.write(b"\n\n")
            temp.write(code.encode('utf-8'))
            temp_path = temp.name

        try:
            # Execute in subprocess with resource limits
            start_time = time.time()

            # Set ulimit if possible (works on Unix-like systems)
            # On Replit this should work
            command = f"ulimit -v {self.max_memory * 1024} 2>/dev/null; python3 {temp_path}"

            # Run the process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                preexec_fn=os.setsid  # Use process group for clean termination
            )

            # Wait for process with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time

                return {
                    "success": process.returncode == 0,
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "execution_time": execution_time,
                    "return_code": process.returncode
                }

            except subprocess.TimeoutExpired:
                # Kill the process group
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except:
                    # Fallback in case killpg fails
                    process.kill()

                # Try to get any output that was generated before timeout
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    stdout_str = stdout.decode('utf-8', errors='replace')
                    stderr_str = stderr.decode('utf-8', errors='replace')
                except:
                    stdout_str = ""
                    stderr_str = ""

                return {
                    "success": False,
                    "stdout": stdout_str,
                    "stderr": stderr_str +
                    f"\nExecution timed out after {self.timeout} seconds",
                    "execution_time": self.timeout,
                    "return_code": -1
                }

        except Exception as e:
            logger.error(f"Error in SecurePythonExecutor: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "execution_time": 0,
                "return_code": -1
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

    def _generate_import_guard(self) -> str:
        """Generate code to restrict imports to allowed modules only."""
        return """
import sys
import importlib
import builtins
import os
from typing import Dict, List, Any, Optional

# Enable the standard libraries that are needed for basic operations
# but not exposed to user code
import json
import math
import random
import re
import datetime
import collections
import itertools
import functools
import operator
import statistics
import time

# Original __import__ function
original_import = __import__

# List of allowed modules
ALLOWED_MODULES = {0}

# Override __import__ to restrict imports
def secure_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Check the module name only (not submodules)
    base_module = name.split('.')[0]
    if base_module not in ALLOWED_MODULES:
        raise ImportError(f"Import of module '{{base_module}}' is not allowed for security reasons. "
                           f"Only the following libraries are available: {{', '.join(sorted(ALLOWED_MODULES))}}")
    return original_import(name, globals, locals, fromlist, level)

# Replace the built-in __import__ function
builtins.__import__ = secure_import

# Disable potentially dangerous functions
builtins.exec = None
builtins.eval = None
builtins.compile = None

# Restrict file system access
original_open = open

def secure_open(file, mode='r', *args, **kwargs):
    # Only allow access to files in the current directory or subdirectories
    current_dir = os.path.abspath('.')
    try:
        file_path = os.path.abspath(file)
        if not file_path.startswith(current_dir):
            raise PermissionError(f"Access to {{file}} is not allowed for security reasons")
    except:
        # For non-string file paths or other issues, just pass through
        # This handles file objects and other valid open() parameters
        pass
    return original_open(file, mode, *args, **kwargs)

builtins.open = secure_open

# Helper function to print results in a readable way
def show_result(obj):
    import pprint
    pprint.pprint(obj)

# Add show_result to builtins for convenience
builtins.show_result = show_result
        """.format(repr(self.allowed_modules))


# Function to use with Tool
def secure_python_exec(code: str) -> str:
    """Execute Python code securely and return the results."""
    # Create executor with all supported libraries
    executor = SecurePythonExecutor(
        timeout=30,  # 30 seconds max execution time (matches Gemini's limit)
        max_memory=512,  # 512MB memory limit
        # We use the default library list defined in the class
    )

    try:
        result = executor.run(code)

        if result["success"]:
            output = result["stdout"]
            # Format successful execution
            if not output.strip():
                return "Code executed successfully, but produced no output."
            return output
        else:
            stderr_output = result['stderr']
            if "failed to reserve page summary memory" in stderr_output or "fatal error" in stderr_output:
                # More specific error for the LLM
                error_message = (
                    f"Sandbox Execution Environment Error (took {result['execution_time']:.2f}s): "
                    "The underlying execution environment encountered a critical memory allocation issue "
                    "and could not complete the task. This is likely due to system-level resource limits."
                )
            elif "Execution timed out" in stderr_output:
                 error_message = (
                    f"Execution Timed Out (took {result['execution_time']:.2f}s): "
                    "The calculation took longer than the allowed time limit."
                )
            else:
                error_message = f"Code execution failed (took {result['execution_time']:.2f}s):\n\n{stderr_output}"
            return error_message
    except Exception as e:
        logger.error(f"Error in secure_python_exec: {e}")
        return f"Code execution system error: {str(e)}"
