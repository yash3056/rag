"""
Django App Configuration for Document QA App

This file configures the document_qa Django app and defines its behavior
within the Django project. It's automatically loaded when Django starts.
"""

from django.apps import AppConfig


class DocumentQaConfig(AppConfig):
    """
    Configuration class for the document_qa Django app
    
    This class defines app-specific settings and is automatically
    discovered by Django when the app is listed in INSTALLED_APPS.
    """
    # Use BigAutoField for auto-generated primary keys (64-bit integers)
    # This is the modern Django standard for ID fields
    default_auto_field = 'django.db.models.BigAutoField'
    
    # The Python path to the app - must match the app directory name
    name = 'document_qa'
