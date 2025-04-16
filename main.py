import os
import sys
from doctoembed import process_all, query_index, load_data, extract_text_from_pdf
from model import load_model, process_query_with_context, progressive_summarization

# NEW IMPORTS FOR FASTAPI
from fastapi import FastAPI, HTTPException, File, UploadFile, Query, Path
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from datetime import datetime
import json
import uuid
import shutil

# Add import for increasing file upload limits
from starlette.requests import Request

# Define Project model
class Project(BaseModel):
    id: str
    title: str
    description: Optional[str] = ""
    created_at: str
    updated_at: Optional[str] = None
    sources_count: int = 0

class ProjectCreate(BaseModel):
    title: str
    description: Optional[str] = ""

class QuestionRequest(BaseModel):
    query: str
    project_id: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

class SummaryRequest(BaseModel):
    project_id: str
    filename: str

# Project data storage (in-memory for simplicity, could be replaced with a database)
projects_data = []
PROJECTS_FILE = "projects.json"
PROJECTS_BASE_DIR = "projects"

# Create projects directory if it doesn't exist
os.makedirs(PROJECTS_BASE_DIR, exist_ok=True)

# Load existing projects from file if it exists
def load_projects():
    try:
        if os.path.exists(PROJECTS_FILE):
            with open(PROJECTS_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading projects: {e}")
        return []

# Save projects to file
def save_projects():
    with open(PROJECTS_FILE, 'w') as f:
        json.dump(projects_data, f, indent=2)

# Get project folder path
def get_project_folder(project_id):
    return os.path.join(PROJECTS_BASE_DIR, project_id)

# Get project document folder path
def get_project_document_folder(project_id):
    project_folder = get_project_folder(project_id)
    document_folder = os.path.join(project_folder, "document")
    os.makedirs(document_folder, exist_ok=True)
    return document_folder

# Count sources in a project
def count_project_sources(project_id):
    document_folder = get_project_document_folder(project_id)
    count = 0
    if os.path.exists(document_folder):
        for filename in os.listdir(document_folder):
            if filename.lower().endswith(".pdf"):
                count += 1
    return count

# Update source count for a project
def update_project_sources_count(project_id):
    for project in projects_data:
        if project["id"] == project_id:
            sources_count = count_project_sources(project_id)
            project["sources_count"] = sources_count
            project["updated_at"] = datetime.now().isoformat()
            save_projects()
            return sources_count
    return 0

# Initialize projects on startup
projects_data = load_projects()

# Create a dictionary to store DocumentQA instances for each project
document_qa_instances = {}

class DocumentQA:
    def __init__(self, pdf_folder="document", model_name="gemma3"):
        self.model_name = load_model(model_name)
        self.initialize_document_system(pdf_folder)
        
    def initialize_document_system(self, pdf_folder):
        """Initialize the document embedding system"""
        # Store the document folder path
        self.document_folder = pdf_folder
        
        # Create project-specific index and metadata files
        index_file = os.path.join(os.path.dirname(pdf_folder), "document_index.faiss")
        metadata_file = os.path.join(os.path.dirname(pdf_folder), "metadata.pkl")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            print(f"Loading existing document index from {index_file}...")
            self.index, self.metadata = load_data(index_file, metadata_file)
            
            # We need to reload the model and chunks
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Reconstruct chunks from documents
            from doctoembed import process_documents, create_chunks
            documents = process_documents(pdf_folder)
            self.all_chunks, _ = create_chunks(documents)
        else:
            print(f"Processing documents and creating index for {pdf_folder}...")
            self.all_chunks, self.metadata, self.index, self.model = process_all(
                pdf_folder, index_file, metadata_file
            )
            
        print(f"System initialized with {self.index.ntotal} document chunks for {pdf_folder}.")
    
    def answer_question(self, query, top_k=5):
        """Answer a question using document context and the model"""
        # Search relevant document chunks
        search_results = query_index(
            query, 
            self.index, 
            self.model, 
            self.metadata, 
            self.all_chunks, 
            top_k=top_k
        )
        
        # Process query with retrieved context using the LLM
        answer = process_query_with_context(query, search_results, self.model_name)
        return answer, search_results
    
    def display_sources(self, search_results):
        print("\nSources used:")
        for i, result in enumerate(search_results):
            relevance = 1.0/(result['distance'] + 0.01)
            filename = result['metadata']['filename']
            chunk_index = result['metadata']['chunk_index']
            
            print(f"\n{i+1}. {filename} (Chunk #{chunk_index}, Relevance: {relevance:.2f})")
            print("-" * 40)
            
            # Display the chunk text that was used (truncate if too long)
            chunk_text = result.get('text', 'No text available')
            max_display_length = 300
            if len(chunk_text) > max_display_length:
                displayed_text = chunk_text[:max_display_length] + "..."
            else:
                displayed_text = chunk_text
                
            print(displayed_text)
            print("-" * 40)
    
    def summarize_document(self, filename):
        """Summarize a specific document from this project's documents"""
        file_path = os.path.join(self.document_folder, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document {filename} not found")
        
        print(f"Starting summarization of {filename} from {file_path}")
        
        # Extract text from the PDF
        document_text = extract_text_from_pdf(file_path)
        
        # Check if we got any text
        if not document_text or not document_text.strip():
            error_msg = f"Could not extract text from {filename}. The PDF may be empty, password-protected, or contain only images."
            print(error_msg)
            return error_msg
            
        print(f"Extracted {len(document_text)} characters from {filename}")
        
        # Use the progressive summarization technique to handle long documents
        summary = progressive_summarization(document_text, self.model_name)
        return summary

# Function to get or create a DocumentQA instance for a project
def get_document_qa(project_id):
    if project_id not in document_qa_instances:
        document_folder = get_project_document_folder(project_id)
        document_qa_instances[project_id] = DocumentQA(pdf_folder=document_folder)
    return document_qa_instances[project_id]

# Define FastAPI application and endpoints
app = FastAPI()

# Enable CORS to allow requests from the webpage
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modify request body size limit for larger file uploads
@app.middleware("http")
async def increase_request_body_size(request: Request, call_next):
    # Increase request body size limit to 100MB (adjust as needed)
    setattr(request.state, "max_body_size", 100 * 1024 * 1024)
    response = await call_next(request)
    return response

# Project API endpoints
@app.get("/api/projects")
def get_projects():
    return projects_data

@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    for project in projects_data:
        if project["id"] == project_id:
            return project
    raise HTTPException(status_code=404, detail="Project not found")

@app.post("/api/projects")
def create_project(project: ProjectCreate):
    current_time = datetime.now().isoformat()
    project_id = str(uuid.uuid4())
    
    # Create project folder structure
    project_folder = get_project_folder(project_id)
    document_folder = get_project_document_folder(project_id)
    
    new_project = {
        "id": project_id,
        "title": project.title,
        "description": project.description,
        "created_at": current_time,
        "updated_at": current_time,
        "sources_count": 0
    }
    
    projects_data.append(new_project)
    save_projects()
    return new_project

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Get the document QA system for the specific project
    qa_system = get_document_qa(request.project_id)
    
    answer, sources = qa_system.answer_question(request.query)
    return {"answer": answer, "sources": sources}

@app.post("/reload/{project_id}")
def reload_index(project_id: str):
    document_folder = get_project_document_folder(project_id)
    
    # Remove existing instance if it exists
    if project_id in document_qa_instances:
        del document_qa_instances[project_id]
    
    # Create a new instance
    get_document_qa(project_id)
    
    return {"message": f"Document index for project {project_id} reloaded."}

@app.post("/check_file/{project_id}")
async def check_file(project_id: str, payload: dict):
    filename = payload.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="filename field missing")
    
    document_folder = get_project_document_folder(project_id)
    file_path = os.path.join(document_folder, filename)
    
    if os.path.exists(file_path):
        return {"exists": True}
    else:
        return {"exists": False}

@app.post("/upload_source/{project_id}")
async def upload_source(project_id: str, file: UploadFile = File(...)):
    try:
        document_folder = get_project_document_folder(project_id)
        file_location = os.path.join(document_folder, file.filename)
        
        # Process file in chunks to handle large files better
        with open(file_location, "wb") as buffer:
            # Process file in chunks of 2MB (increased from 1MB)
            chunk_size = 2 * 1024 * 1024  # 2MB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
        
        print(f"File successfully saved to {file_location}")
        
        # For multiple file uploads, defer rebuilding the index until all files are uploaded
        # Just note that a file changed
        if project_id not in document_qa_instances:
            # If no instance exists yet, create it
            get_document_qa(project_id)
        else:
            # Mark that we need to reload (but don't actually reload yet)
            print(f"Deferring index rebuild for project {project_id}")
        
        # Update the source count for this project
        update_project_sources_count(project_id)
        
        return {"message": f"File '{file.filename}' uploaded successfully."}
    except Exception as e:
        # Log the error details
        error_message = f"Error uploading file: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

# Add new endpoint to rebuild a project's index after multiple files uploaded
@app.post("/rebuild_index/{project_id}")
def rebuild_index(project_id: str):
    # Remove existing instance if it exists
    if project_id in document_qa_instances:
        del document_qa_instances[project_id]
    
    # Create a new instance - this rebuilds the index
    get_document_qa(project_id)
    
    return {"message": f"Document index for project {project_id} rebuilt."}

@app.get("/list_sources/{project_id}")
async def list_sources(project_id: str):
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
    
    return {"sources": sources}

@app.delete("/delete_source/{project_id}/{filename}")
async def delete_source(project_id: str, filename: str):
    try:
        document_folder = get_project_document_folder(project_id)
        file_path = os.path.join(document_folder, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
            # Reinitialize document system for this project
            if project_id in document_qa_instances:
                del document_qa_instances[project_id]
            get_document_qa(project_id)
            
            # Update the source count for this project
            update_project_sources_count(project_id)
            
            return {"message": f"File '{filename}' deleted and index updated."}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    # Find the project by ID
    project_index = None
    for i, project in enumerate(projects_data):
        if project["id"] == project_id:
            project_index = i
            break
            
    if project_index is None:
        raise HTTPException(status_code=404, detail="Project not found")
    
    try:
        # Remove the project from the list
        deleted_project = projects_data.pop(project_index)
        
        # Remove the project folder if it exists
        project_folder = get_project_folder(project_id)
        if os.path.exists(project_folder):
            shutil.rmtree(project_folder)
            
        # Remove any document QA instance
        if project_id in document_qa_instances:
            del document_qa_instances[project_id]
        
        # Save updated projects list
        save_projects()
        
        return {"message": f"Project '{deleted_project['title']}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting project: {str(e)}")

@app.patch("/api/projects/{project_id}")
def update_project(project_id: str, update_data: dict):
    # Validate the update data - only title can be updated for now
    if "title" not in update_data or not update_data["title"].strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    
    # Find and update the project
    for project in projects_data:
        if project["id"] == project_id:
            project["title"] = update_data["title"].strip()
            project["updated_at"] = datetime.now().isoformat()
            save_projects()
            return project
            
    raise HTTPException(status_code=404, detail="Project not found")

@app.post("/summarize")
def summarize_document(request: SummaryRequest):
    try:
        # Get the document QA system for the specific project
        qa_system = get_document_qa(request.project_id)
        
        # Generate the summary
        summary = qa_system.summarize_document(request.filename)
        
        return {"summary": summary, "filename": request.filename}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

def main():
    # Check if document folder exists, create if not
    if not os.path.exists("document"):
        os.makedirs("document")
        print("Created 'document' folder. Please add your PDF documents to this folder.")
        print("Then restart the application.")
        return
        
    # Create a local instance of DocumentQA for CLI use
    qa_system = DocumentQA(pdf_folder="document")
        
    print("\n===== Document Question Answering System =====")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'reload' to reload the document index.")
    print("Type 'summarize <filename>' to generate a document summary.")
    print("Type 'list' to show available documents.")
    
    while True:
        query = input("\nEnter your command or question: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if query.lower() == 'reload':
            qa_system.initialize_document_system("document")
            print("Document index reloaded.")
            continue
            
        if query.lower() == 'list':
            files = os.listdir("document")
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            if pdf_files:
                print("\nAvailable documents:")
                for i, file in enumerate(pdf_files, 1):
                    print(f"{i}. {file}")
            else:
                print("No PDF documents found in the document folder.")
            continue
        
        if query.lower().startswith('summarize '):
            filename = query[9:].strip()  # Get filename after 'summarize '
            
            # Check if file exists
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'  # Add .pdf extension if missing
                
            if not os.path.exists(os.path.join("document", filename)):
                print(f"Error: File '{filename}' not found in document folder.")
                continue
                
            print(f"\nGenerating summary for {filename}...")
            try:
                summary = qa_system.summarize_document(filename)
                print("\n===== Document Summary =====")
                print(summary)
                print("=" * 40)
            except Exception as e:
                print(f"Error generating summary: {str(e)}")
            continue
        
        if not query:
            continue
            
        print("Searching for information and generating answer...")
        answer, sources = qa_system.answer_question(query)
        
        print("\n===== Answer =====")
        print(answer)
        
        # Display sources
        qa_system.display_sources(sources)

if __name__ == "__main__":
    # To run the API server, pass 'serve' as an argument (e.g., python main.py serve)
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Update uvicorn settings to increase timeouts and limits
        uvicorn.run(
            "main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=True,
            timeout_keep_alive=120,  # Increase keep-alive timeout
            limit_concurrency=10    # Limit concurrent connections
        )
    else:
        main()