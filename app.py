import streamlit as st
import os
from datetime import datetime
import nest_asyncio
from logic import load_and_create_vector_store, create_agent_chain
from pymongo import MongoClient
import urllib.parse

nest_asyncio.apply()

st.set_page_config(
    page_title="FLAME Faculty Agent",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS for styling
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
    .chat-message-user {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        margin-bottom: 10px;
    }
    .user-text-bubble {
        background-color: #7c4dff;
        color: white;
        padding: 10px 15px;
        border-radius: 20px 20px 5px 20px;
        font-weight: bold;
        max-width: 70%;
    }
    .chat-icon {
        font-size: 24px;
        margin: 0 10px;
    }
    </style>
""", unsafe_allow_html=True)

google_api_key = st.secrets.get("GOOGLE_API_KEY")
mongo_url = st.secrets.get("MONGO_URL")

def init_mongo_client(mongo_url):
    try:
        parsed_url = urllib.parse.urlparse(mongo_url)
        if not parsed_url.scheme or not parsed_url.hostname:
            st.error("Invalid MongoDB connection URL. Please check your `MONGO_URL` secret.")
            return None
        client = MongoClient(mongo_url)
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

def log_question_to_db(question: str):
    client = init_mongo_client(mongo_url)
    if client:
        try:
            db = client.get_database("flame_faculty_agent_db")
            logs_collection = db.get_collection("user_questions")
            log_entry = {
                "timestamp": datetime.now(),
                "question": question
            }
            logs_collection.insert_one(log_entry)
            print("Question logged to MongoDB successfully.")
        except Exception as e:
            st.error(f"Logging to MongoDB failed: {e}")
        finally:
            if client:
                client.close()

@st.cache_resource(show_spinner="Loading faculty data...")
def cached_load_vector_store(api_key: str):
    # These file paths are placeholders since we're loading a pre-existing store
    vector_store, error, df_faculty, df_papers, summary_stats = load_and_create_vector_store(
        "faculty_data.xlsx", 
        "research_papers.xlsx", 
        api_key
    )
    if error:
        st.error(error)
        st.stop()
    return vector_store, df_faculty, df_papers, summary_stats

@st.cache_resource(show_spinner="Creating agent...")
def cached_create_agent(_vector_store, api_key: str, _df_faculty, _df_papers, _summary_stats):
    return create_agent_chain(_vector_store, api_key, _df_faculty, _df_papers, _summary_stats)

st.title("FLAME University Faculty Agent")

with st.sidebar:
    st.header("📖 How to Use")
    st.info("""
    - Ask for faculty by department (e.g., `List all computer science faculty`).
    - Ask about a specific professor's research (e.g., `What is John Doe working on?`).
    - Inquire about research in a specific field (e.g., `Who is doing research in AI ethics?`).
    - Request the abstract of a paper (e.g., `What is the abstract for 'A Study of AI Ethics'?`).
    """)

if not google_api_key:
    st.error("⚠️ GOOGLE_API_KEY not found in Streamlit secrets.")
    st.stop()

try:
    vector_store, df_faculty, df_papers, summary_stats = cached_load_vector_store(google_api_key)
    agent_chain = cached_create_agent(vector_store, google_api_key, df_faculty, df_papers, summary_stats)
except Exception as e:
    st.error(f"❌ Setup error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask about faculty research or publications..."):
    log_question_to_db(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent_chain.invoke({"input": prompt})
                answer = response.get("answer", "Sorry, I couldn't find an answer.")
            except Exception as e:
                answer = f"❌ Error: {e}"
            
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})