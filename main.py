import os
import sys
from doctoembed import process_all, query_index, load_data
from model import load_model, process_query_with_context

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
        
def main():
    # Check if document folder exists, create if not
    if not os.path.exists("document"):
        os.makedirs("document")
        print("Created 'document' folder. Please add your PDF documents to this folder.")
        print("Then restart the application.")
        return
        
    # Initialize the QA system
    qa_system = DocumentQA()
    
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
            qa_system.initialize_document_system()
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
    main()
