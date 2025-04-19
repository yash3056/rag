import os
import json
import uuid
from datetime import datetime
from pathlib import Path

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import FileSystemStorage
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

from .models import Project, Source
from .document_qa import (
    DocumentQA, get_document_qa, get_project_folder, 
    get_project_document_folder, count_project_sources, rebuild_index
)

# Import model-specific functionality
from model import generate_response, load_model, process_query_with_context

# Authentication Views
def login_view(request):
    """Handle user login"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            return render(request, 'auth/login.html', {'error_message': 'Invalid username or password'})
    
    return render(request, 'auth/login.html')

def logout_view(request):
    """Handle user logout"""
    logout(request)
    return redirect('login')

def register_view(request):
    """Handle user registration"""
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email', '')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        
        # Validation
        if password1 != password2:
            return render(request, 'auth/register.html', 
                          {'error_message': 'Passwords do not match'})
        
        # Check if user already exists
        if User.objects.filter(username=username).exists():
            return render(request, 'auth/register.html', 
                          {'error_message': 'Username already exists'})
        
        # Create user
        user = User.objects.create_user(username=username, email=email, password=password1)
        login(request, user)
        return redirect('index')
    
    return render(request, 'auth/register.html')

# API Views for Projects
@login_required
@require_http_methods(["GET"])
def get_projects(request):
    """Get all projects for the current user"""
    # Use Django ORM to get only the current user's projects
    projects = [project.to_dict() for project in 
                Project.objects.filter(user=request.user).order_by('-updated_at')]
    return JsonResponse(projects, safe=False)

@login_required
@require_http_methods(["GET"])
def get_project(request, project_id):
    """Get a specific project by ID"""
    try:
        # Check that the project belongs to the current user
        project = Project.objects.get(id=project_id, user=request.user)
        return JsonResponse(project.to_dict())
    except Project.DoesNotExist:
        return JsonResponse({"error": "Project not found"}, status=404)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def create_project(request):
    """Create a new project"""
    try:
        data = json.loads(request.body)
        project_id = str(uuid.uuid4())
        
        # Create project folder structure
        project_folder = get_project_folder(project_id)
        document_folder = get_project_document_folder(project_id)
        
        try:
            # Create project in database with the current user
            project = Project.objects.create(
                id=uuid.UUID(project_id),
                user=request.user,
                title=data.get("title", ""),
                description=data.get("description", ""),
                sources_count=0
            )
            
            return JsonResponse(project.to_dict())
        except Exception as e:
            # Remove project folder if database creation fails
            if os.path.exists(project_folder):
                import shutil
                shutil.rmtree(project_folder)
            
            error_msg = f"Database error: {str(e)}"
            print(error_msg)
            return JsonResponse({"error": error_msg}, status=500)
    except Exception as e:
        error_msg = f"Failed to create project: {str(e)}"
        print(error_msg)
        return JsonResponse({"error": error_msg}, status=400)

@login_required
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
        
        try:
            # Ensure project belongs to current user
            project = Project.objects.get(id=project_id, user=request.user)
            project.title = new_title
            project.save()
            return JsonResponse(project.to_dict())
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

@login_required
@csrf_exempt
@require_http_methods(["DELETE"])
def delete_project(request, project_id):
    """Delete a project"""
    try:
        try:
            # Ensure project belongs to current user
            project = Project.objects.get(id=project_id, user=request.user)
            project_title = project.title
            
            # Remove project folder if it exists
            project_folder = get_project_folder(project_id)
            if os.path.exists(project_folder):
                import shutil
                shutil.rmtree(project_folder)
            
            # Delete from database
            project.delete()
            
            # Remove document QA instance if exists
            from .document_qa import document_qa_instances
            if project_id in document_qa_instances:
                del document_qa_instances[project_id]
            
            return JsonResponse({"message": f"Project '{project_title}' deleted successfully"})
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# API Views for Document QA
@login_required
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
        
        # Verify project belongs to user
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found or access denied"}, status=403)
        
        # Get the document QA system for the specific project
        qa_system = get_document_qa(project_id)
        
        answer, sources = qa_system.answer_question(query)
        return JsonResponse({"answer": answer, "sources": sources})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def summarize_document(request):
    """Generate a document summary"""
    try:
        data = json.loads(request.body)
        project_id = data.get("project_id")
        filename = data.get("filename")
        
        # Verify project belongs to user
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found or access denied"}, status=403)
        
        # Log the start of summarization
        print(f"Starting document summarization for {filename} in project {project_id}")
        
        # Get the document QA system for the specific project
        qa_system = get_document_qa(project_id)
        
        try:
            # Generate the summary with a time limit (to prevent server overload)
            # We'll use a try-except block to catch any internal errors during summarization
            summary = qa_system.summarize_document(filename)
            print(f"Successfully completed summarization for {filename} ({len(summary.split())} words)")
            
            # Check if the summary begins with "Summary of" or similar text
            lines = summary.split('\n')
            cleaned_summary = summary
            
            # Check for common header patterns in the first 2 lines and remove them if found
            if len(lines) > 1:
                # Check if first line contains the filename or "Summary of"
                if ("Summary of" in lines[0] and filename in lines[0]) or lines[0].strip() == '':
                    cleaned_summary = '\n'.join(lines[1:]).strip()
            
            # Return just the cleaned text, not as JSON
            return HttpResponse(cleaned_summary, content_type="text/plain")
        except Exception as e:
            # Log any internal errors during summarization
            print(f"Error during summarization process: {str(e)}")
            return JsonResponse({"error": f"Error during summarization: {str(e)}"}, status=500)
            
    except FileNotFoundError as e:
        print(f"File not found error in summarize_document: {str(e)}")
        return JsonResponse({"error": str(e)}, status=404)
    except Exception as e:
        print(f"Unexpected error in summarize_document: {str(e)}")
        return JsonResponse({"error": f"Error generating summary: {str(e)}"}, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def reload_document_index(request, project_id):
    """Reload the document index for a project"""
    try:
        # Verify project belongs to user
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found or access denied"}, status=403)
        
        result = rebuild_index(project_id)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def check_file_exists(request, project_id):
    """Check if a file exists in a project"""
    try:
        data = json.loads(request.body)
        filename = data.get("filename")
        if not filename:
            return JsonResponse({"error": "filename field missing"}, status=400)
        
        # Verify project belongs to user
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found or access denied"}, status=403)
        
        document_folder = get_project_document_folder(project_id)
        file_path = os.path.join(document_folder, filename)
        
        return JsonResponse({"exists": os.path.exists(file_path)})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def upload_source(request, project_id):
    """Upload a file to a project"""
    try:
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file provided"}, status=400)
        
        # Verify project belongs to user
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found or access denied"}, status=403)
        
        file = request.FILES['file']
        document_folder = get_project_document_folder(project_id)
        fs = FileSystemStorage(location=document_folder)
        filename = fs.save(file.name, file)
        
        print(f"File successfully saved to {os.path.join(document_folder, filename)}")
        
        try:
            # Update project in database
            project.sources_count = count_project_sources(project_id)
            project.save()
            
            # Add source to database
            file_path = os.path.join(document_folder, filename)
            file_size = os.path.getsize(file_path)
            
            # Check if source already exists
            source, created = Source.objects.get_or_create(
                project_id=project_id,
                filename=filename,
                defaults={
                    'size': file_size
                }
            )
            
            if not created:
                # Update existing source
                source.size = file_size
                source.save()
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found"}, status=404)
        
        return JsonResponse({"message": f"File '{filename}' uploaded successfully."})
    except Exception as e:
        error_message = f"Error uploading file: {str(e)}"
        print(error_message)
        return JsonResponse({"error": error_message}, status=500)

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def rebuild_document_index(request, project_id):
    """Rebuild the document index for a project"""
    try:
        # Verify project belongs to user
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found or access denied"}, status=403)
        
        result = rebuild_index(project_id)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@login_required
@require_http_methods(["GET"])
def list_sources(request, project_id):
    """List all sources in a project"""
    try:
        # Verify project belongs to user
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found or access denied"}, status=403)
        
        # First check if we have sources in the database
        db_sources = Source.objects.filter(project_id=project_id)
        
        if db_sources.exists():
            sources = [source.to_dict() for source in db_sources]
        else:
            # Fallback to file system if no sources in database
            document_folder = get_project_document_folder(project_id)
            
            sources = []
            for filename in os.listdir(document_folder):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(document_folder, filename)
                    file_size = os.path.getsize(file_path)
                    
                    # Create source in database
                    source = Source.objects.create(
                        project_id=project_id,
                        filename=filename,
                        size=file_size
                    )
                    
                    sources.append(source.to_dict())
        
        return JsonResponse({"sources": sources})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@login_required
@csrf_exempt
@require_http_methods(["DELETE"])
def delete_source(request, project_id, filename):
    """Delete a source from a project"""
    try:
        # Verify project belongs to user
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Project not found or access denied"}, status=403)
        
        document_folder = get_project_document_folder(project_id)
        file_path = os.path.join(document_folder, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
            # Reinitialize document system for this project
            result = rebuild_index(project_id)
            
            try:
                # Update project in database
                project.sources_count = count_project_sources(project_id)
                project.save()
                
                # Remove source from database
                Source.objects.filter(project_id=project_id, filename=filename).delete()
            except Project.DoesNotExist:
                pass
            
            return JsonResponse({"message": f"File '{filename}' deleted and index updated."})
        else:
            return JsonResponse({"error": "File not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# New Model-specific Endpoints
@login_required
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

@login_required
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

@login_required
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

# User Profile Endpoints
@login_required
@csrf_exempt
@require_http_methods(["PATCH"])
def update_user_profile(request):
    """Update user profile information"""
    try:
        data = json.loads(request.body)
        email = data.get('email', '').strip()
        
        # Update the user's email
        user = request.user
        user.email = email
        user.save()
        
        return JsonResponse({"success": True, "message": "Profile updated successfully"})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)

@login_required
@csrf_exempt
@require_http_methods(["PATCH"])
def update_user_password(request):
    """Update user password"""
    try:
        data = json.loads(request.body)
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        # Verify current password
        user = authenticate(username=request.user.username, password=current_password)
        if user is None:
            return JsonResponse({"success": False, "error": "Current password is incorrect"}, status=400)
        
        # Update password
        user.set_password(new_password)
        user.save()
        
        # Log the user back in with the new password
        login(request, user)
        
        return JsonResponse({"success": True, "message": "Password updated successfully"})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)

# HTML Views
@login_required
def index_view(request):
    """Serve the index.html page"""
    return render(request, 'index.html')

@login_required
def user_profile_view(request):
    """Serve the user profile page"""
    return render(request, 'account/profile.html')

@login_required
def settings_view(request):
    """Serve the settings page"""
    return render(request, 'account/settings.html')

@login_required
def project_view(request, project_id=None):
    """Serve the web.html page"""
    # Check if project exists and belongs to user
    if project_id:
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return redirect('index')
    
    return render(request, 'web.html')
