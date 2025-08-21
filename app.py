import streamlit as st
import os
from datetime import datetime
import nest_asyncio
from logic import load_vector_store, create_faculty_search_agent

nest_asyncio.apply()

st.set_page_config(
    page_title="FLAME Faculty Agent",
    layout="centered"
)

st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #121212;
    }
    .stMarkdown, .stChatMessage {
        color: #e0e0e0 !important;
    }
    .chat-bubble {
        border: 1px solid #444;
        border-radius: 10px;
        padding: 12px 15px;
        margin: 8px 0;
        background-color: #1e1e1e;
    }
    .chat-bubble.assistant {
        border-left: 4px solid #7c4dff;
    }
    .chat-message {
        display: flex;
        align-items: flex-start;
        gap: 8px;
        margin: 6px 0;
    }
    .chat-icon {
        font-size: 20px;
        margin-top: 2px;
    }
    .chat-user-text {
        font-weight: bold;
    }
    .info-card {
        background-color: #1e1e1e;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #e0e0e0;
    }
    .status-success {
        color: #4caf50;
        font-weight: bold;
    }
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

google_api_key = st.secrets.get("GOOGLE_API_KEY")

def log_question_to_file(question: str):
    """Log user questions to file for analytics"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {question}\n"
        with open("user_questions_log.txt", "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Logging error: {e}")

@st.cache_resource(show_spinner="Loading vector store database...")
def cached_load_vector_store(api_key: str):
    """Load the existing vector store from disk"""
    vector_store, summary_stats = load_vector_store(api_key)
    if isinstance(vector_store, str): 
        st.error(f"❌ {vector_store}")
        st.stop()
    return vector_store, summary_stats

@st.cache_resource(show_spinner="Creating faculty search agent...")
def cached_create_search_agent(_vector_store, api_key: str, _summary_stats):
    """Create the faculty search agent"""
    return create_faculty_search_agent(_vector_store, api_key, _summary_stats)

def display_database_info(summary_stats):
    """Display database information in the sidebar"""
    if summary_stats:
       
        if summary_stats.get('departments'):
            with st.expander("📚 Available Departments"):
                for dept in summary_stats['departments']:
                    st.write(f"• {dept}")

st.title("FLAME Faculty Agent")
st.markdown("*Your AI assistant for faculty research and expertise discovery*")

with st.sidebar:
    st.header("📖 How to Use")
    
    instructions = [
        "🔍 Search for faculty by name, department, or research area",
        "📄 Ask about specific research papers and publications", 
        "🎯 Find experts in particular research domains",
        "📧 Get contact information for collaboration",
        "🏛️ Explore research activities by department"
    ]
    
    for step in instructions:
        st.markdown(
            f"""
            <div style="background-color: #1e1e1e; color: #e0e0e0; 
                        padding: 10px; border-radius: 8px; 
                        margin-bottom: 8px; font-size: 13px;
                        border-left: 3px solid #7c4dff;">
                {step}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")

if not google_api_key:
    st.error("⚠️ GOOGLE_API_KEY not found in Streamlit secrets.")
    st.stop()

try:
    
    vector_store, summary_stats = cached_load_vector_store(google_api_key)
    
    search_agent = cached_create_search_agent(vector_store, google_api_key, summary_stats)
  
except Exception as e:
    st.markdown(
        f"""
        <div class="info-card">
            <span class="status-error">🔴 System Status: Error</span><br>
            <small>Error: {str(e)}</small>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.error(f"❌ Setup error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"<div class='chat-message'><div class='chat-icon'>👤</div><div class='chat-user-text'>{message['content']}</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-message'><div class='chat-icon'>🤖</div>"
            f"<div class='chat-bubble assistant'>{message['content']}</div></div>",
            unsafe_allow_html=True
        )

if prompt := st.chat_input("Ask about faculty, research, or publications..."):
    log_question_to_file(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f"<div class='chat-message'><div class='chat-icon'>👤</div><div class='chat-user-text'>{prompt}</div></div>",
        unsafe_allow_html=True
    )

    with st.spinner("Thinking..."):
        try:
            response = search_agent.search(prompt)
            answer = response.get("answer", "Sorry, I couldn't find an answer.")

        except Exception as e:
            answer = f"❌ Error processing your request: {str(e)}"
            print(f"Search error: {e}")  # For debugging

    st.markdown(
        f"<div class='chat-message'><div class='chat-icon'>🤖</div>"
        f"<div class='chat-bubble assistant'>{answer}</div></div>",
        unsafe_allow_html=True
    )
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
