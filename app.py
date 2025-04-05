import streamlit as st
import os
import json
import uuid
import datetime
from pathlib import Path
import base64
import re
import openai
from google import generativeai as genai


# Folders to store data
HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)
CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(exist_ok=True)
API_KEYS_FILE = CONFIG_DIR / "api_keys.json"

# Function to load and apply custom CSS
def apply_custom_css():
    # Base CSS to apply the Poppins font to all elements
    base_css = """
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    """
    

    css_file = Path("src/styles/chat.css")
    css_content = base_css
    
    try:
        if css_file.exists():
            with open(css_file, "r") as f:
                custom_css = f.read()

                if "@import url('https://fonts.googleapis.com/css2?family=Poppins" not in custom_css:
                    css_content = base_css + custom_css
                else:
                    css_content = custom_css
    except Exception as e:
        st.warning(f"Could not load custom CSS file: {e}. Using default styling.")
    

    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Apply custom CSS at startup
apply_custom_css()

# Function to load API keys
def load_api_keys():
    if API_KEYS_FILE.exists():
        try:
            with open(API_KEYS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"openai": "", "deepseek": "", "gemini": ""}
    return {"openai": "", "deepseek": "", "gemini": ""}

# Function to save API keys
def save_api_keys(api_keys):
    """Save API keys with secure file permissions"""

    CONFIG_DIR.mkdir(exist_ok=True)
    
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(api_keys, f)
    
    # Set file permissions to be readable/writable only by the owner
    try:
        import os
        import stat
        # 0o600 = read/write only for the owner
        os.chmod(API_KEYS_FILE, stat.S_IRUSR | stat.S_IWUSR)
    except Exception as e:
        st.warning(f"Could not set secure permissions on API keys file: {e}")

# Initialize session state for chat histories if not exists
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# Initialize separate API keys for each user session
if "user_api_keys" not in st.session_state:
    # Load API keys from disk when running locally
    if not is_streamlit_cloud() and API_KEYS_FILE.exists():
        st.session_state.user_api_keys = load_api_keys()
    else:
        st.session_state.user_api_keys = {"openai": "", "deepseek": "", "gemini": ""}

# Keep backward compatibility for pre-existing sessions
if "api_keys" not in st.session_state:
    if not is_streamlit_cloud() and API_KEYS_FILE.exists():
        st.session_state.api_keys = load_api_keys()
    else:
        st.session_state.api_keys = {"openai": "", "deepseek": "", "gemini": ""}

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o"
    

for chat_id, chat in st.session_state.get("chat_histories", {}).items():
    if chat.get("model") == "gemini":
        chat["model"] = "gemini-2.0-flash"

if "show_api_config" not in st.session_state:
    st.session_state.show_api_config = False

def save_chat_history(chat_id, history):
    """Save chat history to a JSON file"""
    try:
        file_path = HISTORY_DIR / f"{chat_id}.json"
        with open(file_path, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")

def load_chat_histories():
    """Load all saved chat histories"""
    histories = {}
    

    HISTORY_DIR.mkdir(exist_ok=True)
    
    # Validate chat_id to prevent directory traversal attacks
    def is_valid_chat_id(filename):
        return bool(re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.json$', filename))
    
    for file_path in HISTORY_DIR.glob("*.json"):

        if not is_valid_chat_id(file_path.name):
            continue
            
        chat_id = file_path.stem
        try:
            with open(file_path, 'r') as f:
                try:
                    chat_data = json.load(f)

                    if not all(key in chat_data for key in ["title", "messages", "model"]):
                        continue
                        

                    if "timestamp" not in chat_data:

                        chat_data["timestamp"] = datetime.datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    histories[chat_id] = chat_data
                except json.JSONDecodeError:
                    st.error(f"Error loading chat history: {file_path}")
        except Exception as e:
            st.error(f"Failed to read chat file {file_path}: {e}")
    return histories

def delete_chat_history(chat_id):
    """Delete a chat history file"""
    try:

        if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', chat_id):
            st.error("Invalid chat ID format.")
            return False
            
        file_path = HISTORY_DIR / f"{chat_id}.json"
        if file_path.exists():
            file_path.unlink()
            if chat_id in st.session_state.chat_histories:
                del st.session_state.chat_histories[chat_id]
            
            # If the deleted chat was the current one, reset current_chat_id
            if st.session_state.current_chat_id == chat_id:
                st.session_state.current_chat_id = None
                
            return True
        return False
    except Exception as e:
        st.error(f"Failed to delete chat history: {e}")
        return False

def create_new_chat():
    """Create a new chat with a unique ID"""
    chat_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now()
    st.session_state.chat_histories[chat_id] = {
        "title": f"New Chat",
        "model": st.session_state.selected_model,
        "messages": [],
        "timestamp": timestamp.isoformat()
    }
    save_chat_history(chat_id, st.session_state.chat_histories[chat_id])
    return chat_id

def update_api_key(provider, value):
    """Update API key in the user's session only"""
    if provider not in ["openai", "deepseek", "gemini"]:
        st.error(f"Invalid API provider: {provider}")
        return False
        
    if value and not value.strip():
        st.error("API key cannot be empty or just whitespace")
        return False
        
    if provider == "openai" and value and not value.startswith("sk-"):
        st.warning("OpenAI API keys typically start with 'sk-'. Please check your key.")
        
    if provider == "gemini" and value and len(value.strip()) < 10:
        st.warning("Google Gemini API key appears too short. Please check your key.")
        
    # Store API key in user's session
    st.session_state.user_api_keys[provider] = value.strip() if value else ""
    
    # Also save to disk if running locally (not on Streamlit Cloud)
    if not is_streamlit_cloud():
        try:
            # Also update the shared api_keys for backward compatibility
            st.session_state.api_keys[provider] = st.session_state.user_api_keys[provider]
            save_api_keys(st.session_state.api_keys)
        except Exception as e:
            st.error(f"Failed to save API key to disk: {e}")
    
    return True

# Function to detect if running on Streamlit Cloud
def is_streamlit_cloud():
    """Check if the app is running on Streamlit Cloud"""
    # Streamlit Cloud sets this environment variable
    return os.environ.get('STREAMLIT_SHARING') == 'true' or os.environ.get('IS_STREAMLIT_CLOUD') == 'true'

def toggle_api_config():
    """Toggle API configuration visibility"""
    st.session_state.show_api_config = not st.session_state.show_api_config

def generate_chat_title(messages, max_length=40):
    """Generate a title based on the first user message in the chat"""
    if not messages:
        return "New Empty Chat"
    

    first_user_message = None
    for msg in messages:
        if msg["role"] == "user":
            first_user_message = msg["content"]
            break
    
    if not first_user_message:
        return "New Chat"
    
    # Clean and truncate the message to create a title
    title = first_user_message.strip().split('\n')[0]  # Get first line
    
    # Remove special characters but preserve basic punctuation
    title = ''.join(c for c in title if c.isalnum() or c.isspace() or c in '.,!?-')
    

    if len(title) > max_length:

        cutoff = max_length
        while cutoff > 0 and not title[cutoff].isspace():
            cutoff -= 1
        if cutoff == 0:
            cutoff = max_length
        title = title[:cutoff] + '...'
    
    return title

def update_chat_title(chat_id):
    """Update the chat title based on its content"""
    if chat_id in st.session_state.chat_histories:
        chat = st.session_state.chat_histories[chat_id]
        if chat["messages"]:
            chat["title"] = generate_chat_title(chat["messages"])
            save_chat_history(chat_id, chat)

def get_openai_response(messages):
    """Get response from OpenAI API"""
    try:
        api_key = st.session_state.user_api_keys["openai"]
        if not api_key:
            return "Error: OpenAI API key is missing. Please add your API key in the settings."
            
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except openai.RateLimitError:
        return "Error: Rate limit exceeded for OpenAI API. Please try again later."
    except openai.AuthenticationError:
        return "Error: Invalid OpenAI API key. Please check your API key in settings and try again."
    except openai.APIConnectionError:
        return "Error: Could not connect to OpenAI API. Please check your internet connection."
    except Exception as e:
        st.error(f"Error from OpenAI API: {str(e)}")
        return f"Error: Could not get response from OpenAI API. {str(e)}"

def get_gemini_response(messages):
    """Get response from Google's Gemini API"""
    try:
        api_key = st.session_state.user_api_keys["gemini"]
        if not api_key:
            return "Error: Google Gemini API key is missing. Please add your API key in the settings."
            
        genai.configure(api_key=api_key)
        
        # Convert chat format to Gemini format - handle system message specially
        gemini_messages = []
        system_content = ""
        
        # Extract system message but don't try to use it directly as system_instruction
        for msg in messages:
            if msg["role"] == "system":

                system_content = msg["content"]
                continue
            role = "user" if msg["role"] == "user" else "model"
            
            # If this is a user message and we have a system message, prepend it
            if role == "user" and system_content and not any(m["role"] == "user" for m in gemini_messages):
                content = f"[System: {system_content}]\n\n{msg['content']}"
                gemini_messages.append({"role": role, "parts": [content]})
            else:
                gemini_messages.append({"role": role, "parts": [msg["content"]]})
        
        # Use the gemini-2.0-flash model
        model = genai.GenerativeModel('gemini-2.0-flash')
        

        if not gemini_messages:
            return "Error: No messages to process. Please send a message first."
            

        try:

            if len(gemini_messages) > 1:
                chat = model.start_chat(history=gemini_messages[:-1])
                response = chat.send_message(gemini_messages[-1]["parts"][0])
            else:

                response = model.generate_content(gemini_messages[0]["parts"][0])
                
            if not response.text:
                return "Sorry, Gemini didn't generate a response. Please try again."
                
            return response.text
        except Exception as inner_e:
            # Log the error for debugging
            st.error(f"Gemini chat error: {str(inner_e)}")
            
            # Fallback to direct generation if chat interface fails
            try:

                last_user_msg = ""
                for msg in reversed(messages):
                    if msg["role"] == "user":
                        last_user_msg = msg["content"]
                        break
                if last_user_msg:
                    response = model.generate_content(last_user_msg)
                    return response.text
                else:
                    return "Error: No user message found to generate a response."
            except Exception as fallback_e:
                st.error(f"Gemini fallback error: {str(fallback_e)}")
                return f"Error: Failed to get a response from Gemini. {str(fallback_e)}"
            
    except Exception as e:
        st.error(f"Error from Gemini API: {str(e)}")
        return f"Error: Could not get response from Gemini API. {str(e)}"

def get_deepseek_response(messages):
    """Get response from DeepSeek API (placeholder)"""
    try:
        api_key = st.session_state.user_api_keys["deepseek"]
        if not api_key:
            return "Error: DeepSeek API key is missing. Please add your API key in the settings."
            
        # DeepSeek implementation would go here
        # For this example, we'll just return a placeholder
        message = """
This is a placeholder for the DeepSeek API integration. 

To implement DeepSeek in a production environment:
1. Install the official DeepSeek SDK
2. Follow the DeepSeek documentation for API integration
3. Update this function with proper API calls

For now, this is just a demonstration of the interface.
"""
        return message
    except Exception as e:
        st.error(f"Error from DeepSeek API: {str(e)}")
        return f"Error: Could not get response from DeepSeek API. {str(e)}"

def get_model_response(messages, model):
    """Get response from the selected model"""
    if model == "gpt-4o":
        return get_openai_response(messages)
    elif model == "gemini" or model == "gemini-2.0-flash":
        return get_gemini_response(messages)
    elif model == "deepseek":
        return get_deepseek_response(messages)
    else:
        return "Error: Unknown model selected."

# Load existing chat histories
st.session_state.chat_histories.update(load_chat_histories())

# Sidebar
with st.sidebar:
    st.title("DayaChat")
    
    # New chat button
    if st.button("New Chat"):
        st.session_state.current_chat_id = create_new_chat()
        st.rerun()
    
    # Chat history - sort by timestamp (newest first)
    st.subheader("Chat History")
    if not st.session_state.chat_histories:
        st.info("No chat history yet. Start a new chat!")
    else:
        # Get all chats and sort by timestamp (newest first)
        sorted_chats = sorted(
            st.session_state.chat_histories.items(),
            key=lambda x: x[1].get("timestamp", ""),
            reverse=True  # Newest first
        )
        
        for chat_id, chat_data in sorted_chats:
            col1, col2 = st.columns([4, 1])
            with col1:
                # Truncate title if too long
                display_title = chat_data["title"]
                if len(display_title) > 25:
                    display_title = display_title[:25] + "..."
                
                if st.button(display_title, key=f"select_{chat_id}", use_container_width=True):
                    st.session_state.current_chat_id = chat_id

                    st.session_state.selected_model = chat_data["model"]
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{chat_id}", help="Delete this chat"):
                    delete_chat_history(chat_id)
                    st.rerun()

# Main chat interface
st.title("DayaChat")

# Model Selection and API Configuration button
if st.button("Choose Model", key="top_settings", use_container_width=False):
    toggle_api_config()

# Show API configuration only if the button is clicked
if st.session_state.show_api_config:
    st.subheader("Model & API Settings")
    
    # Model selection inside the API config
    selected_model = st.selectbox(
        "Select Model",
        ["gpt-4o", "gemini-2.0-flash", "deepseek"],
        index=["gpt-4o", "gemini-2.0-flash", "deepseek"].index(st.session_state.selected_model if st.session_state.selected_model != "gemini" else "gemini-2.0-flash"),
        key="main_model_selector"
    )
    
    # Update the selected model in session state
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        # If there's a current chat, update its model
        if st.session_state.current_chat_id:
            current_chat = st.session_state.chat_histories[st.session_state.current_chat_id]
            current_chat["model"] = selected_model
            save_chat_history(st.session_state.current_chat_id, current_chat)
    
    # Show only the API key input for the currently selected model
    if st.session_state.selected_model == "gpt-4o":
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.user_api_keys["openai"],
            type="password",
            key="main_openai_key_input"
        )
        if st.button("Save OpenAI API Key", key="save_openai_key"):
            if update_api_key("openai", openai_key):
                st.success("OpenAI API Key saved successfully!")
            
    elif st.session_state.selected_model == "gemini-2.0-flash":
        gemini_key = st.text_input(
            "Google Gemini API Key",
            value=st.session_state.user_api_keys["gemini"],
            type="password",
            key="main_gemini_key_input"
        )
        if st.button("Save Gemini API Key", key="save_gemini_key"):
            if update_api_key("gemini", gemini_key):
                st.success("Gemini API Key saved successfully!")
            
    elif st.session_state.selected_model == "deepseek":
        deepseek_key = st.text_input(
            "DeepSeek API Key",
            value=st.session_state.user_api_keys["deepseek"],
            type="password",
            key="main_deepseek_key_input"
        )
        if st.button("Save DeepSeek API Key", key="save_deepseek_key"):
            if update_api_key("deepseek", deepseek_key):
                st.success("DeepSeek API Key saved successfully!")

# Display current chat and handle new messages
if st.session_state.current_chat_id is None and len(st.session_state.chat_histories) > 0:
    # Set the first chat as current if available
    sorted_chats = sorted(
        st.session_state.chat_histories.items(),
        key=lambda x: x[1].get("timestamp", ""),
        reverse=True  # Newest first
    )
    
    if sorted_chats:
        st.session_state.current_chat_id = sorted_chats[0][0]
        # Update the selected model to match the chat's model
        current_chat = st.session_state.chat_histories[st.session_state.current_chat_id]
        st.session_state.selected_model = current_chat["model"]
        if st.session_state.selected_model == "gemini":
            st.session_state.selected_model = "gemini-2.0-flash"

if st.session_state.current_chat_id is None:
    # No chats available, create a new one
    st.session_state.current_chat_id = create_new_chat()
    st.rerun()

# Make sure the current chat ID is valid and exists
if st.session_state.current_chat_id not in st.session_state.chat_histories:
    st.session_state.current_chat_id = create_new_chat()
    st.rerun()

current_chat = st.session_state.chat_histories[st.session_state.current_chat_id]
current_model = current_chat["model"]
if current_model == "gemini":
    current_model = "gemini-2.0-flash"
    current_chat["model"] = "gemini-2.0-flash"

# Display chat messages
for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle new message
user_input = st.chat_input("Type your message here...")
if user_input:

    with st.chat_message("user"):
        st.write(user_input)
    
    current_chat["messages"].append({"role": "user", "content": user_input})
    

    if len(current_chat["messages"]) == 1:
        current_chat["title"] = generate_chat_title([{"role": "user", "content": user_input}])
    

    current_chat["model"] = st.session_state.selected_model
    current_model = current_chat["model"]
    

    current_chat["timestamp"] = datetime.datetime.now().isoformat()
    

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            messages_for_model = []
            
            
            model_identity = {
                "gpt-4o": "GPT-4o by OpenAI",
                "gemini-2.0-flash": "Gemini 2.0 Flash by Google",
                "deepseek": "DeepSeek model"
            }.get(current_model, current_model)
            
            model_id_message = {
                "role": "system", 
                "content": f"You are {model_identity}. When asked about your identity, which model you are, or who made you, always accurately identify yourself as {model_identity}."
            }
            messages_for_model.append(model_id_message)
            
            
            messages_for_model.extend([
                {"role": msg["role"], "content": msg["content"]} 
                for msg in current_chat["messages"]
            ])
            
            response = get_model_response(messages_for_model, current_model)
            st.write(response)
    
    
    current_chat["messages"].append({"role": "assistant", "content": response})
    
    # Save updated chat history
    save_chat_history(st.session_state.current_chat_id, current_chat) 
