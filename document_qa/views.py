import os
import json
import uuid
from datetime import datetime
from pathlib import Path

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import FileSystemStorage

from .models import Project, Source
from .document_qa import (
    DocumentQA, get_document_qa, get_project_folder, 
    get_project_document_folder, count_project_sources, rebuild_index
)

# Import model-specific functionality
from model import generate_response, load_model, process_query_with_context

# Helper to load projects from file for compatibility
def load_projects_from_file():
    try:
        projects_file = os.path.join(settings.BASE_DIR, settings.PROJECTS_FILE)
        if os.path.exists(projects_file):
            with open(projects_file, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading projects from file: {e}")
        return []

# Helper to save projects to file for compatibility
def save_projects_to_file(projects_data):
    try:
        projects_file = os.path.join(settings.BASE_DIR, settings.PROJECTS_FILE)
        with open(projects_file, 'w') as f:
            json.dump(projects_data, f, indent=2)
    except Exception as e:
        print(f"Error saving projects to file: {e}")

# API Views for Projects
@require_http_methods(["GET"])
def get_projects(request):
    """Get all projects"""
    # For now, we'll continue using the file-based approach for compatibility
    projects = load_projects_from_file()
    return JsonResponse(projects, safe=False)

@require_http_methods(["GET"])
def get_project(request, project_id):
    """Get a specific project by ID"""
    projects = load_projects_from_file()
    for project in projects:
        if project["id"] == project_id:
            return JsonResponse(project)
    return JsonResponse({"error": "Project not found"}, status=404)

@csrf_exempt
@require_http_methods(["POST"])
def create_project(request):
    """Create a new project"""
    try:
        data = json.loads(request.body)
        current_time = datetime.now().isoformat()
        project_id = str(uuid.uuid4())
        
        # Create project folder structure
        project_folder = get_project_folder(project_id)
        document_folder = get_project_document_folder(project_id)
        
        new_project = {
            "id": project_id,
            "title": data.get("title", ""),
            "description": data.get("description", ""),
            "created_at": current_time,
            "updated_at": current_time,
            "sources_count": 0
        }
        
        # Add to projects file
        projects = load_projects_from_file()
        projects.append(new_project)
        save_projects_to_file(projects)
        
        # Also save to database (future-proofing)
        Project.objects.create(
            id=uuid.UUID(project_id),
            title=data.get("title", ""),
            description=data.get("description", ""),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            sources_count=0
        )
        
        return JsonResponse(new_project)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

@csrf_exempt
@require_http_methods(["PATCH"])
def update_project(request, project_id):
    """Update a project"""
    try:
        data = json.loads(request.body)
        
        # Update title if provided
        if "title" not in data or not data["title"].strip():
            return JsonResponse({"error": "Title cannot be empty"}, status=400)
        
        new_title = data["title"].strip()
        
        # Update in file
        projects = load_projects_from_file()
        updated = False
        for project in projects:
            if project["id"] == project_id:
                project["title"] = new_title
                project["updated_at"] = datetime.now().isoformat()
                updated = True
                updated_project = project
                break
        
        if not updated:
            return JsonResponse({"error": "Project not found"}, status=404)
        
        save_projects_to_file(projects)
        
        # Also update in database (future-proofing)
        try:
            db_project = Project.objects.get(id=project_id)
            db_project.title = new_title
            db_project.save()
        except Project.DoesNotExist:
            pass  # Not critical for now
        
        return JsonResponse(updated_project)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_project(request, project_id):
    """Delete a project"""
    try:
        # Remove from file
        projects = load_projects_from_file()
        project_to_delete = None
        project_index = None
        
        for i, project in enumerate(projects):
            if project["id"] == project_id:
                project_to_delete = project
                project_index = i
                break
                
        if project_index is None:
            return JsonResponse({"error": "Project not found"}, status=404)
        
        deleted_project = projects.pop(project_index)
        save_projects_to_file(projects)
        
        # Remove project folder if it exists
        project_folder = get_project_folder(project_id)
        if os.path.exists(project_folder):
            import shutil
            shutil.rmtree(project_folder)
        
        # Delete from database (future-proofing)
        try:
            Project.objects.filter(id=project_id).delete()
        except Exception:
            pass  # Not critical for now
        
        # Remove document QA instance if exists
        from .document_qa import document_qa_instances
        if project_id in document_qa_instances:
            del document_qa_instances[project_id]
        
        return JsonResponse({"message": f"Project '{deleted_project['title']}' deleted successfully"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# API Views for Document QA
@csrf_exempt
@require_http_methods(["POST"])
def ask_question(request):
    """Process a question using DocumentQA"""
    try:
        data = json.loads(request.body)
        query = data.get("query", "").strip()
        project_id = data.get("project_id")
        
        if not query:
            return JsonResponse({"error": "Query cannot be empty"}, status=400)
        
        # Get the document QA system for the specific project
        qa_system = get_document_qa(project_id)
        
        answer, sources = qa_system.answer_question(query)
        return JsonResponse({"answer": answer, "sources": sources})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def summarize_document(request):
    """Generate a document summary"""
    try:
        data = json.loads(request.body)
        project_id = data.get("project_id")
        filename = data.get("filename")
        
        # Get the document QA system for the specific project
        qa_system = get_document_qa(project_id)
        
        # Generate the summary
        summary = qa_system.summarize_document(filename)
        
        # Check if the summary begins with "Summary of" or similar text
        lines = summary.split('\n')
        cleaned_summary = summary
        
        # Check for common header patterns in the first 2 lines and remove them if found
        if len(lines) > 1:
            # Check if first line contains the filename or "Summary of"
            if ("Summary of" in lines[0] and filename in lines[0]) or lines[0].strip() == '':
                cleaned_summary = '\n'.join(lines[1:]).strip()
        
        return JsonResponse({"summary": cleaned_summary, "filename": filename})
    except FileNotFoundError as e:
        return JsonResponse({"error": str(e)}, status=404)
    except Exception as e:
        return JsonResponse({"error": f"Error generating summary: {str(e)}"}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def reload_document_index(request, project_id):
    """Reload the document index for a project"""
    try:
        result = rebuild_index(project_id)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def check_file_exists(request, project_id):
    """Check if a file exists in a project"""
    try:
        data = json.loads(request.body)
        filename = data.get("filename")
        if not filename:
            return JsonResponse({"error": "filename field missing"}, status=400)
        
        document_folder = get_project_document_folder(project_id)
        file_path = os.path.join(document_folder, filename)
        
        return JsonResponse({"exists": os.path.exists(file_path)})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def upload_source(request, project_id):
    """Upload a file to a project"""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file provided"}, status=400)
        
        file = request.FILES['file']
        document_folder = get_project_document_folder(project_id)
        fs = FileSystemStorage(location=document_folder)
        filename = fs.save(file.name, file)
        
        print(f"File successfully saved to {os.path.join(document_folder, filename)}")
        
        # Update the source count for this project
        projects = load_projects_from_file()
        for project in projects:
            if project["id"] == project_id:
                sources_count = count_project_sources(project_id)
                project["sources_count"] = sources_count
                project["updated_at"] = datetime.now().isoformat()
                save_projects_to_file(projects)
                break
                
        # Also update in database (future-proofing)
        try:
            db_project = Project.objects.get(id=project_id)
            db_project.sources_count = count_project_sources(project_id)
            db_project.save()
        except Project.DoesNotExist:
            pass  # Not critical for now
        
        return JsonResponse({"message": f"File '{filename}' uploaded successfully."})
    except Exception as e:
        error_message = f"Error uploading file: {str(e)}"
        print(error_message)
        return JsonResponse({"error": error_message}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def rebuild_document_index(request, project_id):
    """Rebuild the document index for a project"""
    try:
        result = rebuild_index(project_id)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_http_methods(["GET"])
def list_sources(request, project_id):
    """List all sources in a project"""
    try:
        document_folder = get_project_document_folder(project_id)
        
        sources = []
        for filename in os.listdir(document_folder):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(document_folder, filename)
                file_size = os.path.getsize(file_path)
                sources.append({
                    "filename": filename,
                    "size": file_size,
                    "date_added": os.path.getctime(file_path)
                })
        
        return JsonResponse({"sources": sources})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_source(request, project_id, filename):
    """Delete a source from a project"""
    try:
        document_folder = get_project_document_folder(project_id)
        file_path = os.path.join(document_folder, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
            # Reinitialize document system for this project
            result = rebuild_index(project_id)
            
            # Update the source count for this project
            projects = load_projects_from_file()
            for project in projects:
                if project["id"] == project_id:
                    sources_count = count_project_sources(project_id)
                    project["sources_count"] = sources_count
                    project["updated_at"] = datetime.now().isoformat()
                    save_projects_to_file(projects)
                    break
                    
            # Also update in database (future-proofing)
            try:
                db_project = Project.objects.get(id=project_id)
                db_project.sources_count = count_project_sources(project_id)
                db_project.save()
            except Project.DoesNotExist:
                pass  # Not critical for now
            
            return JsonResponse({"message": f"File '{filename}' deleted and index updated."})
        else:
            return JsonResponse({"error": "File not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# New Model-specific Endpoints
@csrf_exempt
@require_http_methods(["POST"])
def model_inference(request):
    """Direct model inference endpoint without document context"""
    try:
        data = json.loads(request.body)
        
        # Required parameters
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return JsonResponse({"error": "Prompt cannot be empty"}, status=400)
        
        # Optional parameters
        context = data.get("context", "")
        
        # Generate response using the model - no longer passing model_name
        response = generate_response(prompt, context)
        
        return JsonResponse({
            "generated_text": response,
            "model_name": "gemma3"  # Just hardcode this for the response
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@require_http_methods(["GET"])
def model_info(request):
    """Get information about the available models and their status"""
    try:
        # No longer need to get model_name from request
        
        # Load the model to get information
        model_data = load_model()  # No longer passing the model_name parameter
        
        # Get model information
        model_info = {
            "name": "gemma3",
            "full_name": "google/gemma-3-4b-it",
            "status": "loaded",
            "type": "image-text-to-text",
            "device": model_data["device"],
        }
        
        return JsonResponse(model_info)
    except Exception as e:
        return JsonResponse({
            "name": "gemma3",
            "status": "error",
            "error": str(e)
        }, status=500)

@require_http_methods(["GET"])
def model_health(request):
    """Health check endpoint for the model service"""
    try:
        # Simple health check - try loading the default model
        model_data = load_model()  # No longer passing the model_name parameter
        return JsonResponse({
            "status": "healthy",
            "models_available": ["gemma3"],
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return JsonResponse({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, status=500)

# HTML Views
def index_view(request):
    """Serve the index.html page"""
    return render(request, 'index.html')

def project_view(request, project_id=None):
    """Serve the web.html page"""
    return render(request, 'web.html')
