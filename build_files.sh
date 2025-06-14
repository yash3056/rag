#!/bin/bash

# Build the project
echo "Creating staticfiles directory..."
uv sync 
uv run python manage.py collectstatic --noinput --clear
