#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    """
    Script to run the Django server with the AI Notebook application
    """
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_notebook.settings")
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed?"
        ) from exc
    
    # Default to runserver command if no arguments are provided
    if len(sys.argv) == 1:
        # Run Django on port 8001 to avoid conflict with vLLM API on 8000
        execute_from_command_line([sys.argv[0], "runserver", "0.0.0.0:8001"])
    else:
        execute_from_command_line(sys.argv)