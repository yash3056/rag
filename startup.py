#!/usr/bin/env python
import os
import sys
import django
from pathlib import Path

# check if 

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

if __name__ == "__main__":
    initialize_database()
    # For local development
    # print("Starting Django development server...")
    # from django.core.management import execute_from_command_line
    # os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_notebook.settings")
    # execute_from_command_line([sys.argv[0], "runserver", "0.0.0.0:8000"] + sys.argv[1:])
    
    # For deployment on Render, use Gunicorn
    # Command: gunicorn ai_notebook.wsgi:application --bind 0.0.0.0:$PORT --workers 3
    print("Use 'gunicorn ai_notebook.wsgi:application --bind 0.0.0.0:$PORT --workers 3' to start the server in production.")