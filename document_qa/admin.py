"""
Django Admin Configuration for Document QA App

This file configures the Django admin interface to provide a web-based
administration panel for managing projects and document sources.

Access the admin interface at: http://localhost:8000/admin/
Default admin credentials: username='admin', password='admin'
"""

from django.contrib import admin
from .models import Project, Source

# Register your models here.

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    """
    Admin interface configuration for Project model
    
    Features:
    - Display project title, owner, document count, and timestamps
    - Filter projects by user and creation date
    - Search projects by title and description
    - Protect system fields from accidental modification
    """
    list_display = ('title', 'user', 'sources_count', 'created_at', 'updated_at')
    list_filter = ('user', 'created_at')  # Add filter sidebar for user and date
    search_fields = ('title', 'description')  # Enable search functionality
    readonly_fields = ('id', 'created_at', 'updated_at')  # Prevent editing system fields

@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    """
    Admin interface configuration for Source model
    
    Features:
    - Display document filename, associated project, file size, and upload date
    - Filter documents by project and upload date
    - Search documents by filename
    - Protect upload timestamp from modification
    """
    list_display = ('filename', 'project', 'size', 'date_added')
    list_filter = ('project', 'date_added')  # Add filter sidebar for project and date
    search_fields = ('filename',)  # Enable search by filename
    readonly_fields = ('date_added',)  # Prevent editing upload timestamp
