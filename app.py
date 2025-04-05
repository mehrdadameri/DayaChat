import streamlit as st
import os
import json
import uuid
import datetime
import hashlib
from pathlib import Path
import base64
import re
import openai
from google import generativeai as genai

# Attempt to import the cookie manager
try:
    import extra_streamlit_components as stx
    COOKIE_MANAGER_AVAILABLE = True
except ImportError:
    COOKIE_MANAGER_AVAILABLE = False

# Folders to store data
HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)
CONFIG_DIR = Path("config")
CONFIG_DIR.mkdir(exist_ok=True)
API_KEYS_FILE = CONFIG_DIR / "api_keys.json"

# Function to detect if running on Streamlit Cloud
def is_streamlit_cloud():
    """Check if the app is running on Streamlit Cloud"""
    # Streamlit Cloud sets these environment variables
    return (os.environ.get('STREAMLIT_SHARING') == 'true' or 
            os.environ.get('IS_STREAMLIT_CLOUD') == 'true' or
            os.environ.get('STREAMLIT_RUN_PATH', '').startswith('/mount/src'))

# Get cookie manager instance
def get_cookie_manager():
    if COOKIE_MANAGER_AVAILABLE:
        return stx.CookieManager()
    return None

# Initialize cookie manager (singleton)
if "cookie_manager" not in st.session_state and COOKIE_MANAGER_AVAILABLE:
    st.session_state.cookie_manager = get_cookie_manager()

# Backup session ID method
def get_or_create_session_id():
    """Get or create a unique session ID for the current user"""
    if "session_id" not in st.session_state:
        # Create a unique ID that includes browser fingerprinting
        user_agent = os.environ.get('HTTP_USER_AGENT', '')
        remote_addr = os.environ.get('REMOTE_ADDR', '')
        fingerprint = hashlib.md5(f"{user_agent}{remote_addr}".encode()).hexdigest()
        st.session_state.session_id = f"{fingerprint}_{str(uuid.uuid4())}"
    return st.session_state.session_id

# Function to get a truly unique session ID using cookies
def get_secure_session_id():
    if is_streamlit_cloud():
        # Try using cookie manager for persistent session ID
        if COOKIE_MANAGER_AVAILABLE and "cookie_manager" in st.session_state:
            # Get existing cookie or set a new one
            cookie_name = "daya_chat_session_id"
            cookie_val = st.session_state.cookie_manager.get(cookie_name)
            
            if not cookie_val:
                # Cookie doesn't exist, set a new one
                new_id = str(uuid.uuid4())
                st.session_state.cookie_manager.set(
                    cookie_name, new_id,
                    expires_at=datetime.datetime.now() + datetime.timedelta(days=30)
                )
                return new_id
            return cookie_val
            
        # Fallback to session state
        return get_or_create_session_id()
    else:
        # For local deployment, use simple session state
        return "local_session"

# Initialize a unique session ID
session_id = get_secure_session_id()

# Create a session-specific namespace for API keys
def get_session_key(key):
    """Create a session-specific key name"""
    if is_streamlit_cloud():
        return f"{session_id}_{key}"
    return key

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
if get_session_key("chat_histories") not in st.session_state:
    st.session_state[get_session_key("chat_histories")] = {}

if get_session_key("current_chat_id") not in st.session_state:
    st.session_state[get_session_key("current_chat_id")] = None

# Initialize separate API keys for each user session
if get_session_key("user_api_keys") not in st.session_state:
    # Load API keys from disk when running locally
    if not is_streamlit_cloud() and API_KEYS_FILE.exists():
        st.session_state[get_session_key("user_api_keys")] = load_api_keys()
    else:
        st.session_state[get_session_key("user_api_keys")] = {"openai": "", "deepseek": "", "gemini": ""}

# Keep backward compatibility for pre-existing sessions
if "api_keys" not in st.session_state:
    if not is_streamlit_cloud() and API_KEYS_FILE.exists():
        st.session_state.api_keys = load_api_keys()
    else:
        st.session_state.api_keys = {"openai": "", "deepseek": "", "gemini": ""}

if get_session_key("selected_model") not in st.session_state:
    st.session_state[get_session_key("selected_model")] = "gpt-4o"
    
if get_session_key("show_api_config") not in st.session_state:
    st.session_state[get_session_key("show_api_config")] = False

# Shortcuts for session state access
def get_chat_histories():
    return st.session_state[get_session_key("chat_histories")]

def get_current_chat_id():
    return st.session_state[get_session_key("current_chat_id")]

def set_current_chat_id(chat_id):
    st.session_state[get_session_key("current_chat_id")] = chat_id

def get_user_api_keys():
    return st.session_state[get_session_key("user_api_keys")]

def get_selected_model():
    return st.session_state[get_session_key("selected_model")]

def set_selected_model(model):
    st.session_state[get_session_key("selected_model")] = model

def get_show_api_config():
    return st.session_state[get_session_key("show_api_config")]

def set_show_api_config(value):
    st.session_state[get_session_key("show_api_config")] = value

# Update chat models if needed
for chat_id, chat in get_chat_histories().items():
    if chat.get("model") == "gemini":
        chat["model"] = "gemini-2.0-flash"

def save_chat_history(chat_id, history):
    """Save chat history to a JSON file"""
    try:
        # For local deployment, save to disk
        if not is_streamlit_cloud():
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
            if chat_id in get_chat_histories():
                del get_chat_histories()[chat_id]
            
            # If the deleted chat was the current one, reset current_chat_id
            if get_current_chat_id() == chat_id:
                set_current_chat_id(None)
                
            return True
        return False
    except Exception as e:
        st.error(f"Failed to delete chat history: {e}")
        return False

def create_new_chat():
    """Create a new chat with a unique ID"""
    chat_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now()
    get_chat_histories()[chat_id] = {
        "title": f"New Chat",
        "model": get_selected_model(),
        "messages": [],
        "timestamp": timestamp.isoformat()
    }
    save_chat_history(chat_id, get_chat_histories()[chat_id])
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
    get_user_api_keys()[provider] = value.strip() if value else ""
    
    # Also save to disk if running locally (not on Streamlit Cloud)
    if not is_streamlit_cloud():
        try:
            # Also update the shared api_keys for backward compatibility
            st.session_state.api_keys[provider] = get_user_api_keys()[provider]
            save_api_keys(st.session_state.api_keys)
        except Exception as e:
            st.error(f"Failed to save API key to disk: {e}")
    
    return True

# Function to toggle API configuration
def toggle_api_config():
    """Toggle API configuration visibility"""
    set_show_api_config(not get_show_api_config())

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
    if chat_id in get_chat_histories():
        chat = get_chat_histories()[chat_id]
        if chat["messages"]:
            chat["title"] = generate_chat_title(chat["messages"])
            save_chat_history(chat_id, chat)

def get_openai_response(messages):
    """Get response from OpenAI API"""
    try:
        api_key = get_user_api_keys()["openai"]
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
        api_key = get_user_api_keys()["gemini"]
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
        api_key = get_user_api_keys()["deepseek"]
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

# Load existing chat histories for local deployment only
if not is_streamlit_cloud():
    get_chat_histories().update(load_chat_histories())

# Sidebar
with st.sidebar:
    st.title("DayaChat")
    
    # Session indicator
    if is_streamlit_cloud():
        st.caption(f"Session: {session_id[:8]}...")
    
    # New chat button
    if st.button("New Chat"):
        set_current_chat_id(create_new_chat())
        st.rerun()
    
    # Chat history - sort by timestamp (newest first)
    st.subheader("Chat History")
    if not get_chat_histories():
        st.info("No chat history yet. Start a new chat!")
    else:
        # Get all chats and sort by timestamp (newest first)
        sorted_chats = sorted(
            get_chat_histories().items(),
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
                    set_current_chat_id(chat_id)

                    set_selected_model(chat_data["model"])
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
if get_show_api_config():
    st.subheader("Model & API Settings")
    
    # Model selection inside the API config
    selected_model = st.selectbox(
        "Select Model",
        ["gpt-4o", "gemini-2.0-flash", "deepseek"],
        index=["gpt-4o", "gemini-2.0-flash", "deepseek"].index(get_selected_model() if get_selected_model() != "gemini" else "gemini-2.0-flash"),
        key="main_model_selector"
    )
    
    # Update the selected model in session state
    if selected_model != get_selected_model():
        set_selected_model(selected_model)
        # If there's a current chat, update its model
        if get_current_chat_id():
            current_chat = get_chat_histories()[get_current_chat_id()]
            current_chat["model"] = selected_model
            save_chat_history(get_current_chat_id(), current_chat)
    
    # Show only the API key input for the currently selected model
    if get_selected_model() == "gpt-4o":
        openai_key = st.text_input(
            "OpenAI API Key",
            value=get_user_api_keys()["openai"],
            type="password",
            key="main_openai_key_input"
        )
        if st.button("Save OpenAI API Key", key="save_openai_key"):
            if update_api_key("openai", openai_key):
                st.success("OpenAI API Key saved successfully!")
            
    elif get_selected_model() == "gemini-2.0-flash":
        gemini_key = st.text_input(
            "Google Gemini API Key",
            value=get_user_api_keys()["gemini"],
            type="password",
            key="main_gemini_key_input"
        )
        if st.button("Save Gemini API Key", key="save_gemini_key"):
            if update_api_key("gemini", gemini_key):
                st.success("Gemini API Key saved successfully!")
            
    elif get_selected_model() == "deepseek":
        deepseek_key = st.text_input(
            "DeepSeek API Key",
            value=get_user_api_keys()["deepseek"],
            type="password",
            key="main_deepseek_key_input"
        )
        if st.button("Save DeepSeek API Key", key="save_deepseek_key"):
            if update_api_key("deepseek", deepseek_key):
                st.success("DeepSeek API Key saved successfully!")

# Display current chat and handle new messages
if get_current_chat_id() is None and len(get_chat_histories()) > 0:
    # Set the first chat as current if available
    sorted_chats = sorted(
        get_chat_histories().items(),
        key=lambda x: x[1].get("timestamp", ""),
        reverse=True  # Newest first
    )
    
    if sorted_chats:
        set_current_chat_id(sorted_chats[0][0])
        # Update the selected model to match the chat's model
        current_chat = get_chat_histories()[sorted_chats[0][0]]
        set_selected_model(current_chat["model"])
        if current_chat["model"] == "gemini":
            set_selected_model("gemini-2.0-flash")

if get_current_chat_id() is None:
    # No chats available, create a new one
    set_current_chat_id(create_new_chat())
    st.rerun()

# Make sure the current chat ID is valid and exists
if get_current_chat_id() not in get_chat_histories():
    set_current_chat_id(create_new_chat())
    st.rerun()

current_chat = get_chat_histories()[get_current_chat_id()]
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
    

    current_chat["model"] = get_selected_model()
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
    save_chat_history(get_current_chat_id(), current_chat) 
