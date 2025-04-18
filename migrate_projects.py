import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_notebook.settings')
django.setup()

from django.contrib.auth.models import User
from document_qa.models import Project

def migrate_projects_to_admin():
    """Migrate all existing projects to the admin user"""
    try:
        # Get admin user
        admin_user = User.objects.get(username='admin')
        
        # Get all projects without a user
        orphaned_projects = Project.objects.filter(user__isnull=True)
        count = orphaned_projects.count()
        
        print(f"Found {count} projects without a user. Migrating to admin user...")
        
        # Update projects to belong to admin
        for project in orphaned_projects:
            project.user = admin_user
            project.save()
            print(f"Migrated project '{project.title}' to admin user")
        
        print(f"Migration complete. {count} projects now belong to admin user.")
    except User.DoesNotExist:
        print("Error: Admin user not found. Please create an admin user first.")
    except Exception as e:
        print(f"Error during migration: {str(e)}")

if __name__ == "__main__":
    migrate_projects_to_admin()