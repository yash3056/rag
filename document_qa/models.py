from django.db import models
import uuid
from datetime import datetime
from django.contrib.auth.models import User

class Project(models.Model):
    """Django model for projects"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='projects', null=True)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    sources_count = models.IntegerField(default=0)
    
    def __str__(self):
        return self.title
    
    def to_dict(self):
        """Convert model instance to dictionary compatible with existing code"""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "sources_count": self.sources_count
        }

class Source(models.Model):
    """Django model for document sources"""
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='sources')
    filename = models.CharField(max_length=255)
    size = models.BigIntegerField(default=0)
    date_added = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.filename} ({self.project.title})"
    
    def to_dict(self):
        """Convert model instance to dictionary compatible with existing code"""
        return {
            "filename": self.filename,
            "size": self.size,
            "date_added": self.date_added.timestamp() if self.date_added else None
        }

# Note: Django will manage the database with models instead of using JSON file
# but we'll keep compatibility with the existing code
