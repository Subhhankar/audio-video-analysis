# chat_functions.py
import json
import re
import math
import threading
from collections import Counter

# Global variables for storing analysis
analysis_data = {}
index_lock = threading.Lock()

def initialize_chat():
    """Initialize any necessary components for chat"""
    print("Chat system initialized successfully")

def tokenize_text(text):
    """Convert text to lowercase tokens with simple preprocessing"""
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split into tokens and remove empty strings
    return [token for token in text.split() if token]

def compute_tf(text):
    """Compute term frequencies for a text"""
    tokens = tokenize_text(text)
    counter = Counter(tokens)
    return {word: count / len(tokens) for word, count in counter.items()}

def compute_similarity(query_tf, doc_tf):
    """Compute cosine similarity between term frequency dictionaries"""
    # Find common words
    common_words = set(query_tf.keys()) & set(doc_tf.keys())
    
    # If no common words, return 0
    if not common_words:
        return 0
    
    # Compute dot product
    dot_product = sum(query_tf[word] * doc_tf[word] for word in common_words)
    
    # Compute magnitudes
    query_magnitude = math.sqrt(sum(value ** 2 for value in query_tf.values()))
    doc_magnitude = math.sqrt(sum(value ** 2 for value in doc_tf.values()))
    
    # Return cosine similarity
    if query_magnitude > 0 and doc_magnitude > 0:
        return dot_product / (query_magnitude * doc_magnitude)
    else:
        return 0

def create_analysis_vectors(analysis_result, analysis_id, profile_text=None):
    """Store analysis results for retrieval during chat"""
    global analysis_data
    
    # Extract all text data from the analysis
    chunks = []
    
    # Add profile text if available
    if profile_text and profile_text.strip():
        chunks.append({
            'text': f"User profile information: {profile_text}",
            'tf': None  # Will compute later
        })
    
    # Add basic analysis data
    chunks.append({
        'text': f"Primary emotion: {analysis_result.get('primary_emotion', 'unknown')}",
        'tf': None  # Will compute later
    })
    
    chunks.append({
        'text': f"Analysis summary: {analysis_result.get('analysis_summary', '')}",
        'tf': None
    })
    
    # Add emotion scores
    for emotion in analysis_result.get('emotions', []):
        chunks.append({
            'text': f"Emotion: {emotion.get('name')} - Score: {emotion.get('score')}/10. {emotion.get('justification', '')}",
            'tf': None
        })
    
    # Add behavioral indicators
    for indicator in analysis_result.get('behavioral_indicators', []):
        chunks.append({
            'text': f"Behavioral indicator: {indicator}",
            'tf': None
        })
    
    # Add timeline data
    for entry in analysis_result.get('timeline', []):
        chunks.append({
            'text': f"At time {entry.get('timestamp')} seconds: {entry.get('emotion')} - {entry.get('description')}. {entry.get('key_sentence', '')}",
            'tf': None
        })
    
    # Add modality analysis
    modality = analysis_result.get('modality_analysis', {})
    
    # Visual cues
    for cue in modality.get('visual', []):
        chunks.append({
            'text': f"Visual cue: {cue.get('cue')} - Interpretation: {cue.get('interpretation')}",
            'tf': None
        })
    
    # Audio cues
    for cue in modality.get('audio', []):
        chunks.append({
            'text': f"Audio cue: {cue.get('cue')} - Interpretation: {cue.get('interpretation')}",
            'tf': None
        })
    
    # Verbal content
    for content in modality.get('verbal', []):
        chunks.append({
            'text': f"Verbal content: '{content.get('content')}' - Emotional meaning: {content.get('emotional_meaning')}",
            'tf': None
        })
    
    # Add media type and task type information
    chunks.append({
        'text': f"Media type: {analysis_result.get('media_type', 'unknown')}",
        'tf': None
    })
    
    chunks.append({
        'text': f"Task type: {analysis_result.get('task_type', 'unknown')}",
        'tf': None
    })
    
    # Music recommendations
    for rec in analysis_result.get('music_recommendations', []):
        chunks.append({
            'text': f"Music recommendation: '{rec.get('title')}' by {rec.get('artist')}. {rec.get('explanation', '')}",
            'tf': None
        })
    
    # Pre-compute term frequencies for each chunk
    for chunk in chunks:
        chunk['tf'] = compute_tf(chunk['text'])
    
    with index_lock:
        # Store the data
        analysis_data = {
            'chunks': chunks,
            'analysis_id': analysis_id,
            'result': analysis_result
        }
    
    print(f"Created analysis data with {len(chunks)} context chunks")

def search_relevant_context(query, num_results=5):
    """Search for text chunks relevant to the user's query"""
    global analysis_data
    
    if not analysis_data:
        return []
    
    with index_lock:
        chunks = analysis_data.get('chunks', [])
        
        if not chunks:
            return []
        
        # Compute TF for query
        query_tf = compute_tf(query)
        
        # Compute similarity scores for each chunk
        scores = []
        for i, chunk in enumerate(chunks):
            similarity = compute_similarity(query_tf, chunk['tf'])
            scores.append((i, similarity))
        
        # Sort by similarity score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top matches
        top_indices = [idx for idx, _ in scores[:num_results]]
        relevant_chunks = [chunks[idx]['text'] for idx in top_indices]
        
        return relevant_chunks

def generate_chat_response(user_message, contexts, analysis_result, genai_client):
    """Generate a response using Google Gemini model"""
    try:
        # Prepare a prompt with context
        context_text = "\n".join(contexts)
        
        primary_emotion = analysis_result.get('primary_emotion', 'unknown')
        analysis_summary = analysis_result.get('analysis_summary', 'No summary available')
        
        prompt = f"""You are an empathetic AI psychologist having a conversation with someone about their emotional analysis results. 
        Maintain a calm, supportive, and professional tone. Respond in a conversational way.
        
        Here's what we know about their analysis:
        - Primary emotion: {primary_emotion}
        - Summary: {analysis_summary}
        
        Additional relevant context from the analysis:
        {context_text}
        
        User's message: {user_message}
        
        Provide a helpful, insightful response that directly addresses their question or comment. 
        Use your psychological expertise to offer meaningful insights about their emotional state based on the analysis data.
        Keep your response concise (3-5 sentences), personalized, and focused on their message.
        """
        
        # Generate response using Gemini
        model = "gemini-2.0-flash"
        response = genai_client.models.generate_content(
            model=model,
            contents=prompt
        )
        
        return response.text.strip()
    except Exception as e:
        print(f"Error generating chat response: {e}")
        return "I'm sorry, I'm having trouble analyzing the results right now. Could you try asking in a different way?"

def verify_analysis_session(analysis_id):
    """Verify that the requested analysis ID matches the current session"""
    with index_lock:
        if not analysis_data:
            return False
            
        if analysis_data.get('analysis_id') != analysis_id:
            return False
            
        return True