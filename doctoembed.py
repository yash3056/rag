import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Step 1: Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file with enhanced error handling"""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            if len(reader.pages) == 0:
                print(f"Warning: PDF has no pages: {pdf_path}")
                return ""
                
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Log the first 100 chars for debugging
            preview = text[:100].replace('\n', ' ') if text else "No text extracted!"
            print(f"Extracted {len(text)} characters from {pdf_path}: {preview}...")
            
            if not text.strip():
                print(f"Warning: Extracted text is empty for {pdf_path}")
            
            return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap helps with context continuity
    return chunks

def process_documents(pdf_folder="document"):
    """Process PDF documents in the specified folder"""
    documents = []
    
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(full_path)
            documents.append({"filename": filename, "text": text})
    
    return documents

def create_chunks(documents):
    """Create text chunks from documents"""
    all_chunks = []
    metadata = []  # to store info about each chunk
    
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({"filename": doc["filename"], "chunk_index": i})
    
    return all_chunks, metadata

def create_embeddings(all_chunks, model_name='all-MiniLM-L6-v2'):
    """Create embeddings from text chunks"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    return np.array(embeddings).astype("float32"), model

def build_index(embeddings):
    """Build a FAISS index from embeddings"""
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

def save_data(index, metadata, index_path="document_index.faiss", metadata_path="metadata.pkl"):
    """Save the index and metadata to files"""
    # Make sure directory exists
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

def load_data(index_path="document_index.faiss", metadata_path="metadata.pkl"):
    """Load the index and metadata from files"""
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def query_index(query_text, index, model, metadata, all_chunks=None, top_k=5):
    """Query the index with the given text"""
    query_embedding = model.encode([query_text]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        # Handle invalid float values to prevent the "FloatObject invalid" error
        try:
            # Convert distance to a valid Python float and handle potential errors
            distance_value = float(dist)
            if not np.isfinite(distance_value):  # Check for NaN, inf, -inf
                print(f"Warning: Invalid distance value {dist} found, using 0.0 instead")
                distance_value = 0.0
        except (ValueError, TypeError) as e:
            print(f"Warning: Error converting distance {dist} to float: {e}, using 0.0 instead")
            distance_value = 0.0
            
        result = {
            "distance": distance_value,
            "metadata": metadata[idx]
        }
        if all_chunks:
            result["text"] = all_chunks[idx]
        results.append(result)
    
    return results

def process_all(pdf_folder="document", index_path="document_index.faiss", metadata_path="metadata.pkl"):
    """Process documents and create searchable index"""
    documents = process_documents(pdf_folder)
    all_chunks, metadata = create_chunks(documents)
    embeddings, model = create_embeddings(all_chunks)
    index = build_index(embeddings)
    save_data(index, metadata, index_path, metadata_path)
    return all_chunks, metadata, index, model

def main():
    """Main function when script is run directly"""
    all_chunks, metadata, index, model = process_all()
    print(f"Total vectors indexed: {index.ntotal}")
    
    # Example query
    query = "Explain the main concept of retrieval augmented generation."
    results = query_index(query, index, model, metadata, all_chunks)
    
    for result in results:
        print(f"Distance: {result['distance']:.4f} - Document: {result['metadata']}")
        print(f"Chunk: {result['text'][:200]}...\n")

if __name__ == "__main__":
    # Execute only when run as a script, not when imported
    main()
