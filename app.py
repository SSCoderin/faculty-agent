import streamlit as st
import os
from datetime import datetime
import nest_asyncio
from logic import load_and_create_vector_store, create_agent_chain

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
    </style>
""", unsafe_allow_html=True)

google_api_key = st.secrets.get("GOOGLE_API_KEY")

def log_question_to_file(question: str):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {question}\n"
        with open("user_questions_log.txt", "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Logging error: {e}")

@st.cache_resource(show_spinner="Loading faculty data...")
def cached_load_vector_store(api_key: str):
    vector_store, error, df_faculty, df_papers, summary_stats = load_and_create_vector_store(
        "faculty_data_with_interests.xlsx", 
        "latest_research_papers.xlsx", 
        api_key
    )
    if error:
        st.error(error)
        st.stop()
    return vector_store, df_faculty, df_papers, summary_stats

@st.cache_resource(show_spinner="Creating agent...")
def cached_create_agent(_vector_store, api_key: str, _df_faculty, _df_papers, _summary_stats):
    return create_agent_chain(_vector_store, api_key, _df_faculty, _df_papers, _summary_stats)

st.title("FLAME Faculty Agent")

with st.sidebar:
    st.header("📖 How to Use")

    instructions = [
        "Ask about faculty research, publications, or expertise",

    ]

    for step in instructions:
        st.markdown(
            f"""
            <div style="background-color: black; color: white; 
                        padding: 10px; border-radius: 10px; 
                        margin-bottom: 10px; font-size: 13px;">
                {step}
            </div>
            """,
            unsafe_allow_html=True
        )

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

if prompt := st.chat_input("Ask your question..."):
    log_question_to_file(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f"<div class='chat-message'><div class='chat-icon'>👤</div><div class='chat-user-text'>{prompt}</div></div>",
        unsafe_allow_html=True
    )

    with st.spinner("Thinking..."):
        try:
            response = agent_chain.invoke({"input": prompt})
            answer = response.get("answer", "Sorry, I couldn't find an answer.")
        except Exception as e:
            answer = f"❌ Error: {e}"

    st.markdown(
        f"<div class='chat-message'><div class='chat-icon'>🤖</div>"
        f"<div class='chat-bubble assistant'>{answer}</div></div>",
        unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "assistant", "content": answer})