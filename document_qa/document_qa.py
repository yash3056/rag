import os
import json
import uuid
from pathlib import Path
from datetime import datetime
import shutil

from django.conf import settings
from django.core.files.storage import FileSystemStorage

from doctoembed import process_all, query_index, load_data, extract_text_from_pdf
from model import load_model, process_query_with_context, progressive_summarization

# Create a dictionary to store DocumentQA instances for each project
document_qa_instances = {}

class DocumentQA:
    def __init__(self, pdf_folder="document"):
        """Initialize the DocumentQA system with the specified PDF folder"""
        # No longer need to pass model_name
        self.model = load_model()
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
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Reconstruct chunks from documents
            from doctoembed import process_documents, create_chunks
            documents = process_documents(pdf_folder)
            self.all_chunks, _ = create_chunks(documents)
        else:
            print(f"Processing documents and creating index for {pdf_folder}...")
            self.all_chunks, self.metadata, self.index, self.embedding_model = process_all(
                pdf_folder, index_file, metadata_file
            )
            
        print(f"System initialized with {self.index.ntotal if hasattr(self.index, 'ntotal') else 0} document chunks for {pdf_folder}.")
    
    def answer_question(self, query, top_k=5, selected_sources=None):
        """Answer a question using document context and the model
        
        Args:
            query (str): The query to answer
            top_k (int): Number of top search results to use
            selected_sources (list): Optional list of filenames to filter results by
        """
        # Search relevant document chunks
        search_results = query_index(
            query, 
            self.index, 
            self.embedding_model, 
            self.metadata, 
            self.all_chunks, 
            top_k=top_k
        )
        
        # Filter results by selected sources if provided
        if selected_sources and len(selected_sources) > 0:
            filtered_results = [result for result in search_results 
                               if result['metadata']['filename'] in selected_sources]
            
            # If we have filtered results, use those, otherwise fall back to all results
            if filtered_results:
                search_results = filtered_results
        
        # Process query with retrieved context using the LLM
        answer = process_query_with_context(query, search_results)
        
        # Extract only the content after 'Task:' line if present
        if "Task:" in answer:
            task_parts = answer.split("Task:", 1)
            if len(task_parts) > 1:
                # Find the actual response after the task description
                # Look for an empty line after the task line which typically separates task from response
                task_part = task_parts[1].strip()
                parts = task_part.split('\n\n', 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                else:
                    answer = task_part
        
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
        # No longer passing model_name parameter
        summary = progressive_summarization(document_text)
        return summary


# Helper functions for project management
def get_project_folder(project_id):
    """Get the folder path for a project"""
    return os.path.join(settings.PROJECTS_BASE_DIR, str(project_id))

def get_project_document_folder(project_id):
    """Get the document folder path for a project"""
    project_folder = get_project_folder(project_id)
    document_folder = os.path.join(project_folder, "document")
    os.makedirs(document_folder, exist_ok=True)
    return document_folder

def get_document_qa(project_id):
    """Get or create a DocumentQA instance for a project"""
    if project_id not in document_qa_instances:
        document_folder = get_project_document_folder(project_id)
        document_qa_instances[project_id] = DocumentQA(pdf_folder=document_folder)
    return document_qa_instances[project_id]

def count_project_sources(project_id):
    """Count the number of PDF sources in a project"""
    document_folder = get_project_document_folder(project_id)
    count = 0
    if os.path.exists(document_folder):
        for filename in os.listdir(document_folder):
            if filename.lower().endswith(".pdf"):
                count += 1
    return count

def rebuild_index(project_id):
    """Rebuild the document index for a project"""
    # Remove existing instance if it exists
    if project_id in document_qa_instances:
        del document_qa_instances[project_id]
    
    # Create a new instance - this rebuilds the index
    get_document_qa(project_id)
    
    return {"message": f"Document index for project {project_id} rebuilt."}