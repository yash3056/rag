import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_notebook.settings')
django.setup()

from django.contrib.auth.models import User

def create_admin_if_not_exists():
    """Create the admin user with username 'admin' and password 'admin' if it doesn't exist"""
    try:
        # Try to get the admin user
        admin_user = User.objects.get(username='admin')
        print("Admin user already exists. Password unchanged.")
    except User.DoesNotExist:
        # Create a new admin user if it doesn't exist
        admin_user = User(username='admin', is_staff=True, is_superuser=True)
        admin_user.set_password('admin')
        admin_user.save()
        print("Created new admin user with username 'admin' and password 'admin'")

if __name__ == "__main__":
    create_admin_if_not_exists()