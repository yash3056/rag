#!/usr/bin/env python
import os
import sys
import django
from pathlib import Path
from huggingface_hub import snapshot_download
import os

def download_model_if_needed():
    """Download the OpenVINO model if not already present"""
    model_dir = "ov_model/"
    
    # Check if model directory exists and has files
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        print("Downloading OpenVINO model...")
        snapshot_download(
            repo_id="yash3056/ov-phi4-mini-reasoning",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        print("Model download completed.")
    else:
        print("OpenVINO model already exists.")


def collect_static_files():
    """
    Collect static files if the static directory doesn't exist or is empty
    This ensures static files are available for the application
    """
    print("Checking static files...")
    
    # Import settings after Django setup
    from django.conf import settings
    
    static_root = Path(settings.STATIC_ROOT)
    
    # Check if static files directory exists and has content
    if not static_root.exists() or not any(static_root.iterdir()):
        print("Static files not found. Collecting static files...")
        try:
            from django.core.management import call_command
            call_command('collectstatic', '--noinput')
            print("Static files collected successfully.")
        except Exception as e:
            print(f"Error collecting static files: {str(e)}")
            # Don't exit here as static files might not be critical for basic functionality
    else:
        print("Static files already exist.")

def initialize_database():
    """
    Initialize the database by running Django migrations
    This ensures all database tables are created before the application starts
    """
    print("Initializing database...")
    
    # Set up Django environment
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_notebook.settings")
    django.setup()
    
    # Check if database exists
    db_path = Path("db.sqlite3")
    is_new_db = not db_path.exists()
    
    # Run migrations
    try:
        from django.core.management import call_command
        call_command('makemigrations')
        call_command('migrate')
        print("Database migrations completed successfully.")
        
        if is_new_db:
            print("New database initialized!")
    except Exception as e:
        print(f"Error running migrations: {str(e)}")
        sys.exit(1)
    
    # Create projects directory if it doesn't exist
    from django.conf import settings
    projects_dir = settings.PROJECTS_BASE_DIR
    Path(projects_dir).mkdir(exist_ok=True)
    print(f"Ensured projects directory exists at: {projects_dir}")
    
    # Collect static files if needed
    collect_static_files()

if __name__ == "__main__":
    initialize_database()
    download_model_if_needed()
    # For local development
    print("Starting Django development server...")
    from django.core.management import execute_from_command_line
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_notebook.settings")
    execute_from_command_line([sys.argv[0], "runserver"] + sys.argv[1:])

