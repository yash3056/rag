import os
import sys
from doctoembed import process_all, query_index, load_data
from model import load_model, process_query_with_context

# NEW IMPORTS FOR FASTAPI
from fastapi import FastAPI, HTTPException, File, UploadFile  # NEW import
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

class DocumentQA:
    def __init__(self, pdf_folder="document", model_name="gemma3"):
        self.model_name = load_model(model_name)
        self.initialize_document_system(pdf_folder)
        
    def initialize_document_system(self, pdf_folder):
        """Initialize the document embedding system"""
        if os.path.exists("document_index.faiss") and os.path.exists("metadata.pkl"):
            print("Loading existing document index...")
            self.index, self.metadata = load_data()
            
            # We need to reload the model and chunks
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Reconstruct chunks from documents
            from doctoembed import process_documents, create_chunks
            documents = process_documents(pdf_folder)
            self.all_chunks, _ = create_chunks(documents)
        else:
            print("Processing documents and creating index...")
            self.all_chunks, self.metadata, self.index, self.model = process_all(pdf_folder)
            
        print(f"System initialized with {self.index.ntotal} document chunks.")
    
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
        """Display the source documents and chunks used for the answer"""
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
        
# Global DocumentQA instance for FastAPI & CLI
qa_system = DocumentQA()

# NEW: Define FastAPI application and endpoints
app = FastAPI()

# NEW: Enable CORS to allow requests from the webpage
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    answer, sources = qa_system.answer_question(request.query)
    return {"answer": answer, "sources": sources}

@app.post("/reload")
def reload_index():
    qa_system.initialize_document_system("document")
    return {"message": "Document index reloaded."}

# NEW: Endpoint to add a new source and rebuild embeddings
@app.post("/add_source")
def add_source(payload: dict):
    filename = payload.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="filename field missing")
    # Assume the file has been added manually to the 'document' folder.
    qa_system.initialize_document_system("document")
    return {"message": f"Source '{filename}' added and document index updated."}

# New endpoint to check if file exists
@app.post("/check_file")
async def check_file(payload: dict):
    filename = payload.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="filename field missing")
    
    document_folder = "document"
    file_path = os.path.join(document_folder, filename)
    
    if os.path.exists(file_path):
        return {"exists": True}
    else:
        return {"exists": False}

# Improve the upload_source endpoint 
@app.post("/upload_source")
async def upload_source(file: UploadFile = File(...)):
    try:
        document_folder = "document"
        os.makedirs(document_folder, exist_ok=True)
        file_location = os.path.join(document_folder, file.filename)
        
        # Process file in chunks to handle large files better
        with open(file_location, "wb") as buffer:
            # Process file in chunks of 1MB
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
        
        print(f"File successfully saved to {file_location}")
        
        # Reinitialize document system
        qa_system.initialize_document_system(document_folder)
        return {"message": f"File '{file.filename}' uploaded and document index updated."}
    except Exception as e:
        # Log the error details
        error_message = f"Error uploading file: {str(e)}"
        print(error_message)
        # Consider saving the file anyway, even if indexing fails
        raise HTTPException(status_code=500, detail=error_message)

# NEW: Endpoint to list all PDF files in the document directory
@app.get("/list_sources")
async def list_sources():
    document_folder = "document"
    if not os.path.exists(document_folder):
        os.makedirs(document_folder, exist_ok=True)
        return {"sources": []}
    
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

@app.delete("/delete_source/{filename}")
async def delete_source(filename: str):
    try:
        document_folder = "document"
        file_path = os.path.join(document_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            # Reinitialize document system after deletion
            qa_system.initialize_document_system(document_folder)
            return {"message": f"File '{filename}' deleted and index updated."}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    # Check if document folder exists, create if not
    if not os.path.exists("document"):
        os.makedirs("document")
        print("Created 'document' folder. Please add your PDF documents to this folder.")
        print("Then restart the application.")
        return
        
    print("\n===== Document Question Answering System =====")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'reload' to reload the document index.")
    
    while True:
        query = input("\nEnter your question: ")
        query = query.strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if query.lower() == 'reload':
            qa_system.initialize_document_system("document")
            print("Document index reloaded.")
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
    # if len(sys.argv) > 1 and sys.argv[1] == "serve":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # else:
    #     main()
