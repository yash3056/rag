import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import math

def load_model():
    """Load the Gemma 3 model directly using Hugging Face Transformers"""
    model_id = "google/gemma-3-4b-it"
        
    print(f"Loading model {model_id}...")
    
    # Determine the appropriate device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load model and processor directly instead of using pipeline
    processor = AutoProcessor.from_pretrained(model_id)
    
    # When loading model, use device_map="auto" instead of the specific device
    # This lets HF Transformers handle device mapping automatically
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    return {"model": model, "processor": processor, "device": device}

# Lazy loading of model as a module-level variable
_model_cache = None

def generate_response(prompt, context=None):
    """Generate a response using the model with the given prompt and optional context"""
    global _model_cache
    
    # Lazy-load the model on first use
    if _model_cache is None:
        _model_cache = load_model()
    
    model = _model_cache["model"]
    processor = _model_cache["processor"]
    device = _model_cache["device"]
    
    # Create a system message that sets the context for the model
    system_message = "You are a helpful AI assistant focused on document analysis and summarization."
    
    # Prepare the prompt with context
    if context:
        full_prompt = f"{system_message}\n\nContent to process:\n\n{context}\n\nTask: {prompt}"
    else:
        full_prompt = f"{system_message}\n\n{prompt}"
    
    # Process input using the processor
    inputs = processor(text=full_prompt, return_tensors="pt")
    
    # Move input tensors to the appropriate device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Decode the output
    response_text = processor.decode(output[0], skip_special_tokens=True)
    
    # Clean up response
    response_text = response_text.strip()
    
    # Remove thinking tags if present
    while '<think>' in response_text and '</think>' in response_text:
        think_start = response_text.find('<think>')
        think_end = response_text.find('</think>') + len('</think>')
        response_text = response_text[:think_start] + response_text[think_end:]
    
    return response_text.strip()

def process_query_with_context(query, search_results):
    """Process a query using retrieved document chunks as context"""
    if not search_results or len(search_results) == 0:
        return "No relevant information found to answer your question."
    
    # Prepare context from search results
    context = ""
    for i, result in enumerate(search_results):
        source_info = f"Source {i+1}: {result['metadata']['filename']}"
        chunk_text = result.get('text', 'No text available')
        context += f"{source_info}\n{chunk_text}\n\n"
    
    # Generate response using the context
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided document sources. "
        "Include relevant information from the sources and cite them when appropriate."
    )
    
    # Add system prompt to the beginning of context
    context_with_prompt = f"{system_prompt}\n\n{context}"
    
    return generate_response(query, context_with_prompt)

def summarize_document(document_text):
    """Summarize a document using the model"""
    if not document_text or not document_text.strip():
        return "Error: No text content provided for summarization."
    
    # Create the prompt for summarization
    prompt = "Please create a detailed summary of the following document"
    
    # Get the full response from the model
    full_response = generate_response(prompt, document_text)
    
    # Extract only the summary part
    # Look for markers that indicate the start of the summary
    summary_markers = ["Summary:", "summary:", "SUMMARY:", "Here's a summary:", "Here is a summary:"]
    
    # Find the earliest marker that exists in the response
    summary_start = -1
    for marker in summary_markers:
        pos = full_response.find(marker)
        if pos >= 0:
            if summary_start == -1 or pos < summary_start:
                summary_start = pos
    
    # If we found a summary marker, extract the text after it
    if summary_start >= 0:
        # Find the marker that matched
        matched_marker = None
        for marker in summary_markers:
            if full_response[summary_start:summary_start+len(marker)] == marker:
                matched_marker = marker
                break
        
        if matched_marker:
            # Extract everything after the marker
            summary = full_response[summary_start + len(matched_marker):].strip()
            return summary
    
    # If no marker is found or extraction fails, return the original response
    # but truncate any obvious input document parts
    if "Task: Please create a detailed summary of the following document" in full_response:
        parts = full_response.split("Task: Please create a detailed summary of the following document")
        if len(parts) > 1:
            # Return everything after the task prompt
            return parts[-1].strip()
    
    return full_response

# Functions for advanced document summarization
def chunk_text_for_summary(text, chunk_size=10000, overlap=200):
    """
    Split text into overlapping chunks for summarization
    
    Args:
        text: The text to be chunked
        chunk_size: Maximum number of words per chunk (default: 10000)
        overlap: Number of words to overlap between chunks (default: 200)
        
    Returns:
        List of text chunks with specified overlap
    """
    if not text or not text.strip():
        print("Warning: Empty text passed to chunking function")
        return []
        
    # Split on whitespace to get words
    words = text.split()
    print(f"Chunking text with {len(words)} words")
    
    chunks = []
    start = 0
    
    while start < len(words):
        # Calculate end position with overlap
        end = min(start + chunk_size, len(words))
        
        # Create the chunk
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start += chunk_size - overlap
        
        # First chunk doesn't need overlap
        if start == chunk_size - overlap:
            # Adjust for the first chunk which doesn't need overlap
            start += overlap
    
    print(f"Created {len(chunks)} chunks from text")
    return chunks

def summarize_chunk(chunk):
    """Summarize a single chunk of text"""
    if not chunk or not chunk.strip():
        print("Warning: Empty chunk passed to summarization")
        return "No content to summarize."
    
    print("\n" + "="*80)
    print("CHUNK TEXT:")
    print("-"*40)
    print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    print("-"*40)
    
    prompt = "Please summarize this section of text concisely and accurately"
    summary = generate_response(prompt, chunk)
    
    print("\nCHUNK SUMMARY:")
    print("-"*40)
    print(summary)
    print("="*80 + "\n")
    
    return summary

def progressive_summarization(text, max_length=10000):
    """
    Summarize a document using overlapping chunks
    
    Args:
        text: The document text to summarize
        max_length: Maximum approximate length to process at once (default: 10000)
        
    Returns:
        Complete summary of the document
    """
    if not text or not text.strip():
        print("Error: Empty document text passed to progressive_summarization")
        return "Could not generate summary: No text content extracted from the document."
    
    print(f"Starting progressive summarization of text with length: {len(text)}")
    
    # If text is short enough, summarize directly
    if len(text.split()) <= max_length:
        print("Text is short enough for direct summarization")
        return summarize_document(text)
    
    # Use our improved chunking function with 10,000 words per chunk and 200 word overlap
    chunks = chunk_text_for_summary(text, chunk_size=max_length, overlap=200)
    print(f"Created {len(chunks)} chunks with {max_length} words per chunk and 200 word overlap")
    
    if not chunks:
        print("Error: No chunks created from text")
        return "Could not generate summary: Failed to process document text."
    
    # If we only have one chunk after all, summarize directly
    if len(chunks) == 1:
        print("Only one chunk created, summarizing directly")
        return summarize_document(chunks[0])
    
    # Process each chunk to get intermediate summaries
    intermediate_summaries = []
    print("\nStarting chunk-by-chunk summarization:")
    print("="*80)
    
    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i+1}/{len(chunks)}:")
        # Get summary for this chunk, but don't include the chunk text in the prompt
        chunk_summary = summarize_chunk(chunk)
        if chunk_summary and chunk_summary.strip():
            intermediate_summaries.append(chunk_summary)
            print(f"✓ Chunk {i+1} summary added ({len(chunk_summary.split())} words)")
        else:
            print(f"⚠ Warning: Empty summary for chunk {i+1}, skipping")
    
    if not intermediate_summaries:
        print("\n❌ Error: No valid intermediate summaries were generated")
        return "Could not generate summary: Failed to create section summaries."
    
    # Instead of recursively summarizing, link summaries into a coherent whole
    # Don't include the original chunks in the final prompt
    combined_summary = "\n\n---SECTION---\n\n".join(intermediate_summaries)
    print(f"\nCombined summary length: {len(combined_summary.split())} words")
    
    print("\nCreating final coherent summary...")
    # Create a new prompt that doesn't include chunk content
    prompt = "The following are summaries of different sections of a document. Create a comprehensive and coherent single summary that integrates them into a well-structured, flowing document. Don't summarize the summaries further, but link them together logically:"
    
    # Generate the final summary with only the section summaries, not the original chunks
    final_summary = generate_response(prompt, combined_summary)
    print(f"Final summary length: {len(final_summary.split())} words")
    return final_summary

def recursive_summarize_sections(sections, max_batch_size=5000):
    """
    Recursively summarize sections in batches to avoid memory issues
    
    Args:
        sections: List of text sections (summaries) to combine
        max_batch_size: Maximum approximate batch size in words
        
    Returns:
        Final coherent summary
    """
    print(f"\nRecursive summarization with {len(sections)} sections")
    
    # Base case: if we have a small enough batch, summarize directly
    total_words = sum(len(section.split()) for section in sections)
    print(f"Total word count in current batch: {total_words}")
    
    # If we're under the batch size limit, we can summarize directly
    if total_words < max_batch_size or len(sections) <= 2:
        # Combine sections with clear section markers
        combined_text = "\n\n--- Section Summary ---\n\n".join(sections)
        print(f"\nCreating summary from {len(sections)} sections ({total_words} words)")
        
        prompt = "Using the following section summaries, create a comprehensive and coherent summary of the entire document:"
        summary = generate_response(prompt, combined_text)
        print(f"Created summary with {len(summary.split())} words")
        return summary
    
    # For larger batches, process in smaller groups recursively
    print(f"Batch too large ({total_words} words), splitting into smaller batches")
    
    # Calculate batch size based on number of sections
    batch_size = max(2, len(sections) // 4)  # Divide into roughly 4 batches, minimum 2 sections per batch
    
    sub_summaries = []
    for i in range(0, len(sections), batch_size):
        batch = sections[i:i+batch_size]
        print(f"Processing sub-batch {i//batch_size + 1} with {len(batch)} sections")
        
        # Create mini-summary of this batch
        combined_batch = "\n\n--- Section Summary ---\n\n".join(batch)
        batch_word_count = len(combined_batch.split())
        
        prompt = "Summarize these connected document sections into a cohesive summary:"
        batch_summary = generate_response(prompt, combined_batch)
        
        print(f"Processed sub-batch {i//batch_size + 1}: {batch_word_count} words → {len(batch_summary.split())} words")
        sub_summaries.append(batch_summary)
    
    # Recursive call with the new sub-summaries
    print(f"Created {len(sub_summaries)} sub-summaries, recursively summarizing them")
    return recursive_summarize_sections(sub_summaries, max_batch_size)
