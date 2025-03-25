import streamlit as st
import tiktoken
import time
from groq import Groq
import threading
import queue
from dotenv import load_dotenv
import os
from spoke import get_wikipedia_info
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

load_dotenv()

st.set_page_config(layout="wide", page_title="Medical LLM Arena")

api_key = os.environ.get('groq_api_key')
groq_client = Groq(api_key=api_key)

# Define available models
MODELS = {
    "Llama3-8b": "llama3-8b-8192",
    "Llama3.3-70b": "llama-3.3-70b-versatile",
    "Mixtral": "mixtral-8x7b-32768",
    "DeepSeek-distill-32b":"deepseek-r1-distill-llama-70b",
    "Qwen2.5-32b":"qwen-2.5-32b"
}

# Initialize session state for chat history
if "chat_history_left" not in st.session_state:
    st.session_state.chat_history_left = []

if "chat_history_right" not in st.session_state:
    st.session_state.chat_history_right = []

if "streaming_complete" not in st.session_state:
    st.session_state.streaming_complete = {"left": False, "right": False}

if "medical_context" not in st.session_state:
    st.session_state.medical_context = {}

if "evaluations" not in st.session_state:
    st.session_state.evaluations = {}

# Custom CSS for better UI
st.markdown("""
<style>
    .user-message {
        background-color: #159ed1 ;
        color: white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .bot-message {
        background-color: #71797E;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .context-message {
        background-color: #F0F8FF;
        color: #333;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 3px solid #159ed1;
    }
    .stSelectbox {
        margin-bottom: 20px;
    }
    .title {
        text-align: center;
        margin-bottom: 20px;
    }
    .context-box {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        font-size: 14px;
    }
    .context-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .evaluation-box {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
        border-left: 3px solid #558b2f;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
    }
    .metric-label {
        font-weight: bold;
    }
    .metric-value {
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract medical entities using SpaCy
def extract_medical_entities(text):
    keywords = []
    with open('keywords.txt', 'r') as f:
        medical_keywords = [line.rstrip('\n') for line in f]
    
    tokens = text.replace('.', '').split(' ')
    for token in tokens:
        print(token.lower())
        if token.lower() in medical_keywords:
            keywords.append(token)
    return keywords 

# Function to fetch medical context information
def get_medical_context(entities):
    context = "### Context for Medical Terms\n"
    for term in entities:
        context += f"- {term}:{get_wikipedia_info(term)}" 
        context+="\n\n"
    return context

# Application title
st.markdown("<h1 class='title'>Medical LLM Arena</h1>", unsafe_allow_html=True)

# Create two columns for the models
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2 class='title'>No Context</h2>", unsafe_allow_html=True)
    model_left = st.selectbox("Select Model", list(MODELS.keys()), key="model_left")

with col2:
    st.markdown("<h2 class='title'>With Medical Context</h2>", unsafe_allow_html=True)
    model_right = st.selectbox("Select Model", list(MODELS.keys()), key="model_right")

# Function to evaluate LLM responses
def evaluate_responses(response_without_context, response_with_context, context):
    # Initialize evaluation metrics
    evaluation = {}
    
    # Calculate response length
    eval_without_context = {
        "length": len(response_without_context.split()),
        "context_relevance": 0
    }
    
    eval_with_context = {
        "length": len(response_with_context.split()),
        "context_relevance": 0
    }
    
    # Calculate context relevance if context exists
    if context:
        # Using TF-IDF and cosine similarity to measure relevance to context
        vectorizer = TfidfVectorizer().fit_transform([context, response_without_context, response_with_context])
        cosine_similarities = cosine_similarity(vectorizer)
        
        # Similarity between context and responses
        eval_without_context["context_relevance"] = round(cosine_similarities[0, 1] * 100, 2)
        eval_with_context["context_relevance"] = round(cosine_similarities[0, 2] * 100, 2)
    
    # Calculate response similarity
    if response_without_context and response_with_context:
        vectorizer = TfidfVectorizer().fit_transform([response_without_context, response_with_context])
        response_similarity = cosine_similarity(vectorizer)[0, 1]
        similarity_percentage = round(response_similarity * 100, 2)
        
        evaluation["response_similarity"] = similarity_percentage
    
    evaluation["without_context"] = eval_without_context
    evaluation["with_context"] = eval_with_context
    
    return evaluation

# Function to render evaluation metrics
def render_evaluation(evaluation, side):
    if side == "left":
        metrics = evaluation["without_context"]
        other_side = "with_context"
    else:
        metrics = evaluation["with_context"]
        other_side = "without_context"
    
    html = "<div class='evaluation-box'>"
    html += "<div class='context-title'>Response Evaluation</div>"
    
    html += "<div class='metric-container'>"
    html += "<span class='metric-label'>Response Length:</span>"
    html += f"<span class='metric-value'>{metrics['length']} words</span>"
    html += "</div>"
    
    if "context_relevance" in metrics:
        html += "<div class='metric-container'>"
        html += "<span class='metric-label'>Context Relevance:</span>"
        html += f"<span class='metric-value'>{metrics['context_relevance']}%</span>"
        html += "</div>"
    
    if "response_similarity" in evaluation:
        html += "<div class='metric-container'>"
        html += "<span class='metric-label'>Response Similarity:</span>"
        html += f"<span class='metric-value'>{evaluation['response_similarity']}%</span>"
        html += "</div>"
    
    html += "</div>"
    
    return html

# Display chat history for both models
def display_chat_history(col, side):
    with col:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        chat_history = st.session_state.chat_history_left if side == "left" else st.session_state.chat_history_right
        
        for i, message in enumerate(chat_history):
            message_id = f"{side}_{i}"
            
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
                
                # Display context after user message but only on the right side
                if side == "right" and message_id in st.session_state.medical_context:
                    st.markdown(f"<div class='context-message'><div class='context-title'>Medical Context</div>{st.session_state.medical_context[message_id]}</div>", unsafe_allow_html=True)
            
            elif message["role"] == "assistant":
                st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)
                
                # Display evaluation after assistant message if available
                if message_id in st.session_state.evaluations:
                    st.markdown(render_evaluation(st.session_state.evaluations[message_id], side), unsafe_allow_html=True)
        
        # Container for streaming content
        if side == "left":
            st.session_state.left_response_container = st.empty()
        else:
            st.session_state.right_response_container = st.empty()
        
        st.markdown("</div>", unsafe_allow_html=True)

# Display chat history for both sides
display_chat_history(col1, "left")
display_chat_history(col2, "right")

def truncate_text(text, max_tokens=5000):
    """Truncates the text to fit within the token limit."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)

# Function to stream responses from Groq
def get_llm_response(model_name, prompt, side, response_queue, with_context=False, context=""):
    try:
        response_stream = ""
        
        # Prepare the messages with or without context
        if with_context and context:
            truncated_context = truncate_text(context, max_tokens=5000)
            messages = [
                {"role": "system", "content": f"You are a medical assistant. Use the following medical context to inform your response, but don't explicitly mention you're using this context: {truncated_context}"},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        completion = groq_client.chat.completions.create(
            model=MODELS[model_name],
            messages=messages,
            stream=True,
            max_tokens=1024,
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response_stream += chunk.choices[0].delta.content
                response_queue.put((side, response_stream))
                time.sleep(0.05)  # Small delay to simulate realistic typing
        
        # Signal that streaming is complete for this side
        response_queue.put((f"{side}_complete", ""))
        
    except Exception as e:
        response_queue.put((side, f"Error: {str(e)}"))
        response_queue.put((f"{side}_complete", ""))

# Function to reset all chat history and context
def reset_chat():
    st.session_state.chat_history_left = []
    st.session_state.chat_history_right = []
    st.session_state.streaming_complete = {"left": False, "right": False}
    st.session_state.medical_context = {}
    st.session_state.evaluations = {}
    st.rerun()

# Input for user message
with st.container():
    user_input = st.text_area("Enter your medical question:", key="user_input", height=100)
    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.button("Submit", use_container_width=True)
    with col2:
        reset_button = st.button("Reset Chat", use_container_width=True)

# Handle reset button
if reset_button:
    reset_chat()

# Process user input and get model responses
if submit_button and user_input:
    # Reset streaming complete flags
    st.session_state.streaming_complete = {"left": False, "right": False}
    
    # Calculate message index
    message_index = len(st.session_state.chat_history_left)
    left_message_id = f"left_{message_index}"
    right_message_id = f"right_{message_index}"
    
    # Add user message to both chat histories
    st.session_state.chat_history_left.append({"role": "user", "content": user_input})
    st.session_state.chat_history_right.append({"role": "user", "content": user_input})
    
    # Extract medical entities and get context for right-side LLM
    medical_entities = extract_medical_entities(user_input)
    with st.spinner("Collecting context from graph this will take time"):
        medical_context = get_medical_context(medical_entities)
        st.session_state.medical_context[right_message_id] = medical_context
    
    # Create placeholders for model responses
    left_response = ""
    right_response = ""
    
    # Create a queue for thread communication
    response_queue = queue.Queue()
    
    # Start threads for both models - left without context, right with context
    left_thread = threading.Thread(target=get_llm_response, 
                                  args=(model_left, user_input, "left", response_queue, False, ""))
    
    right_thread = threading.Thread(target=get_llm_response, 
                                   args=(model_right, user_input, "right", response_queue, True, medical_context))
    
    left_thread.start()
    right_thread.start()
    

    # Continue streaming responses if not complete
    if not (st.session_state.streaming_complete.get("left", True) and 
            st.session_state.streaming_complete.get("right", True)):
        
        # Create placeholders for streaming text if they don't exist
        if not hasattr(st.session_state, "left_response_container"):
            st.session_state.left_response_container = st.empty()
        if not hasattr(st.session_state, "right_response_container"):
            st.session_state.right_response_container = st.empty()
        
        # Create a queue for thread communication if it doesn't exist
        if not hasattr(st.session_state, "response_queue"):
            st.session_state.response_queue = queue.Queue()
        
        left_placeholder = st.session_state.left_response_container.empty()
        right_placeholder = st.session_state.right_response_container.empty()

        # Continue until both models have finished streaming
        while not (st.session_state.streaming_complete.get("left", True) and 
                st.session_state.streaming_complete.get("right", True)):
            try:
                # Get the latest update from the queue with a timeout
#                side, content = st.session_state.response_queue
                side, content = response_queue.get(timeout=0.1)
                # Check if this is a completion signal
                if side == "left_complete":
                    st.session_state.streaming_complete["left"] = True
                    # Store final response
                    left_response = content
                    continue
                elif side == "right_complete":
                    st.session_state.streaming_complete["right"] = True
                    # Store final response
                    right_response = content
                    continue
                
                # Update the appropriate placeholder
                if side == "left":
                    st.session_state.streaming_responses["left"] = content
                    st.session_state.left_placeholder.markdown(f"<div class='bot-message'>{content}</div>", unsafe_allow_html=True)
                else:  # side == "right"
                    st.session_state.streaming_responses["right"] = content
                    right_placeholder.markdown(f"<div class='bot-message'>{content}</div>", unsafe_allow_html=True)
                    
            except queue.Empty:
                # No new updates in the queue, continue
                pass
            
            # Small delay to prevent UI lag
            time.sleep(0.1)
        
        # Make sure threads are finished
        if hasattr(st.session_state, 'left_thread') and hasattr(st.session_state, 'right_thread'):
            st.session_state.left_thread.join()
            st.session_state.right_thread.join()
        
        # Add final responses to chat history
        message_index = len(st.session_state.chat_history_left) - 1
        left_message_id = f"left_{message_index}"
        right_message_id = f"right_{message_index}"
        
        left_response = st.session_state.streaming_responses["left"]
        right_response = st.session_state.streaming_responses["right"]
        
        st.session_state.chat_history_left.append({"role": "assistant", "content": left_response})
        st.session_state.chat_history_right.append({"role": "assistant", "content": right_response})
        
        # Evaluate responses
        if left_response and right_response:
            context = st.session_state.medical_context.get(right_message_id, "")
            evaluation = evaluate_responses(left_response, right_response, context)
            st.session_state.evaluations[left_message_id] = evaluation
            st.session_state.evaluations[right_message_id] = evaluation
        
        # Force a rerun to update the UI with the complete chat history
        st.rerun()
