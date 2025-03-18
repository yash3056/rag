import ollama

def load_model(model_name="gemma3"):
    """Load the Gemma 3 model from Ollama"""
    # Ollama handles model loading on demand, so we just return the name
    return model_name

def generate_response(prompt, context=None, model_name="gemma3"):
    """Generate a response from the Gemma 3 model using the prompt and optional context"""
    # Prepare the message with context if available
    if context:
        full_prompt = f"Context information:\n{context}\n\nQuestion: {prompt}\n\nAnswer based on the context:"
    else:
        full_prompt = prompt
        
    # Generate response using Ollama
    stream_response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}],
        stream=False,  # Changed to False to get a direct response object instead of a stream
        options={
            'num_predict': 12800,
            'temperature': 0.8,
            'top_p': 0.9,
        },
    )
    
    # Get the response content
    response_text = stream_response['message']['content']
    
    # Remove any text within <think> </think> tags
    while '<think>' in response_text and '</think>' in response_text:
        think_start = response_text.find('<think>')
        think_end = response_text.find('</think>') + len('</think>')
        if think_start >= 0 and think_end > 0:
            response_text = response_text[:think_start] + response_text[think_end:]
    
    return response_text

def process_query_with_context(query, search_results, model_name="gemma3"):
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
    
    return generate_response(query, context_with_prompt, model_name)

def summarize_document(document_text, model_name="gemma3"):
    """Summarize a document using the Gemma 3 model"""
    prompt = "Please provide a comprehensive summary of the following document:"
    return generate_response(prompt, document_text, model_name)
