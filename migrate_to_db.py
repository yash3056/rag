import os
import json
import django
import sys
from datetime import datetime
import uuid

# Set up Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ai_notebook.settings")
django.setup()

from document_qa.models import Project, Source
from document_qa.document_qa import get_project_document_folder, count_project_sources

def migrate_projects_to_db():
    """Migrate projects from JSON file to database"""
    json_path = 'projects.json'
    
    # Check if JSON file exists
    if not os.path.exists(json_path):
        print(f"JSON file {json_path} not found. Nothing to migrate.")
        return
    
    # Load projects from JSON
    try:
        with open(json_path, 'r') as f:
            json_projects = json.load(f)
            print(f"Loaded {len(json_projects)} projects from JSON file.")
    except Exception as e:
        print(f"Error loading projects from JSON: {e}")
        return
    
    # Migrate each project
    for json_project in json_projects:
        # Check if project already exists in database
        try:
            project = Project.objects.get(id=uuid.UUID(json_project['id']))
            print(f"Project '{json_project['title']}' already exists in database.")
        except Project.DoesNotExist:
            # Create new project in database
            try:
                created_at = datetime.fromisoformat(json_project['created_at'])
                updated_at = datetime.fromisoformat(json_project['updated_at']) if json_project.get('updated_at') else None
                
                project = Project(
                    id=uuid.UUID(json_project['id']),
                    title=json_project['title'],
                    description=json_project.get('description', ''),
                    created_at=created_at,
                    updated_at=updated_at or created_at,
                    sources_count=json_project.get('sources_count', 0)
                )
                project.save()
                print(f"Created project '{project.title}' in database.")
                
                # Add sources
                document_folder = get_project_document_folder(str(project.id))
                if os.path.exists(document_folder):
                    for filename in os.listdir(document_folder):
                        if filename.lower().endswith(".pdf"):
                            file_path = os.path.join(document_folder, filename)
                            file_size = os.path.getsize(file_path)
                            source = Source(
                                project=project,
                                filename=filename,
                                size=file_size,
                                date_added=datetime.fromtimestamp(os.path.getctime(file_path))
                            )
                            source.save()
                            print(f"  - Added source '{filename}'")
            except Exception as e:
                print(f"Error migrating project {json_project['id']}: {e}")
    
    print("\nMigration completed.")
    print(f"Total projects in database: {Project.objects.count()}")
    print(f"Total sources in database: {Source.objects.count()}")
    
    # Create backup of the JSON file
    if os.path.exists(json_path):
        backup_path = f"{json_path}.bak"
        try:
            shutil.copy(json_path, backup_path)
            print(f"\nCreated backup of JSON file at {backup_path}")
            print("You can safely delete the original JSON file once you've verified the migration.")
        except Exception as e:
            print(f"Error creating backup: {e}")

if __name__ == "__main__":
    import shutil
    print("Starting migration from JSON to database...")
    migrate_projects_to_db()