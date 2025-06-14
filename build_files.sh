#!/bin/bash
# Verify uv installation
echo "Verifying python installation..."
python --version

# Build the project
echo "Installing dependencies and creating staticfiles directory..."
pip install -r requirements.txt
uv run python manage.py collectstatic --noinput --clear
