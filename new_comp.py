import streamlit as st
import tiktoken
import time
from groq import Groq
import threading
import queue
from dotenv import load_dotenv
import os
from spoke import get_wikipedia_info

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
    st.session_state.medical_context = ""

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
    
    # Display current medical context if available
    if st.session_state.medical_context:
        with st.expander("View Current Medical Context", expanded=False):
            st.markdown(st.session_state.medical_context)

# Display chat history for both models
with col1:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.chat_history_left:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Container for streaming content
    left_response_container = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.chat_history_right:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Container for streaming content
    right_response_container = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)


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

# Input for user message
with st.container():
    user_input = st.text_area("Enter your medical question:", key="user_input", height=100)
    submit_button = st.button("Submit", use_container_width=True)

# Process user input and get model responses
if submit_button and user_input:
    # Reset streaming complete flags
    st.session_state.streaming_complete = {"left": False, "right": False}
    
    # Add user message to both chat histories
    st.session_state.chat_history_left.append({"role": "user", "content": user_input})
    st.session_state.chat_history_right.append({"role": "user", "content": user_input})
    
    # Extract medical entities and get context for right-side LLM
    medical_entities = extract_medical_entities(user_input)
    with st.spinner("Collecting context from graph this will take time"):
        medical_context = get_medical_context(medical_entities)
        st.session_state.medical_context = medical_context
    
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
    
    # Create placeholders for streaming text
    left_placeholder = left_response_container.empty()
    right_placeholder = right_response_container.empty()
    
    # Continue until both models have finished streaming
    while not (st.session_state.streaming_complete["left"] and st.session_state.streaming_complete["right"]):
        try:
            # Get the latest update from the queue with a timeout
            side, content = response_queue.get(timeout=0.1)
            
            # Check if this is a completion signal
            if side == "left_complete":
                st.session_state.streaming_complete["left"] = True
                continue
            elif side == "right_complete":
                st.session_state.streaming_complete["right"] = True
                continue
            
            # Update the appropriate placeholder
            if side == "left":
                left_response = content
                left_placeholder.markdown(f"<div class='bot-message'>{left_response}</div>", unsafe_allow_html=True)
            else:  # side == "right"
                right_response = content
                right_placeholder.markdown(f"<div class='bot-message'>{right_response}</div>", unsafe_allow_html=True)
                
        except queue.Empty:
            # No new updates in the queue, continue
            pass
        
        # Small delay to prevent UI lag
        time.sleep(0.1)
    
    # Make sure threads are finished
    left_thread.join()
    right_thread.join()
    
    # Add final responses to chat history
    st.session_state.chat_history_left.append({"role": "assistant", "content": left_response})
    st.session_state.chat_history_right.append({"role": "assistant", "content": right_response})
    
    # Clear the input box
    # st.session_state.user_input = ""
    
    # Force a rerun to update the UI with the complete chat history
    st.rerun()
