

import streamlit as st
import os
from datetime import datetime

import nest_asyncio
nest_asyncio.apply()
from logic import load_and_create_vector_store, create_agent_chain


st.set_page_config(
    page_title="Faculty Research Agent",
    layout="centered"
)

google_api_key = st.secrets["GOOGLE_API_KEY"]   

def log_question_to_file(question):
    """Log user questions to a text file with timestamp."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {question}\n"
        
        # Append to questions log file
        with open("user_questions_log.txt", "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        # Don't interrupt the app if logging fails
        print(f"Logging error: {e}")

@st.cache_resource(show_spinner="Loading and processing faculty data...")
def cached_load_vector_store(api_key):
    """Caches the vector store to avoid reloading on every rerun."""
    vector_store, error, df_summary = load_and_create_vector_store("faculty_data_with_interests.xlsx", "latest_research_papers.xlsx", api_key)
    if error:
        st.error(error)
        st.stop()
    return vector_store, df_summary

@st.cache_resource(show_spinner="Creating AI agent...")
def cached_create_agent(_vector_store, api_key, _df_summary):
    """Caches the agent chain to avoid recreating on every rerun."""
    return create_agent_chain(_vector_store, api_key, _df_summary)

st.title("FLAME Faculty Agent")
st.write("Ask me anything about our faculty's research interests, expertise, and statistics!")



if not google_api_key:
    st.error("⚠️ Error: GOOGLE_API_KEY not found.")
    st.info("Please create a `.env` file and add your Google API Key to it.")
    st.stop()

try:
    vector_store, df_summary = cached_load_vector_store(google_api_key)
    agent_chain = cached_create_agent(vector_store, google_api_key, df_summary)
            
except Exception as e:
    st.error(f"❌ An error occurred during setup: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about faculty research, statistics, or specific expertise..."):
    # Log the question to file
    log_question_to_file(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching faculty database..."):
            try:
                response = agent_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                # If the response seems too short for statistical queries, enhance it
                if any(keyword in prompt.lower() for keyword in ['how many', 'total', 'count', 'statistics', 'distribution']):
                    if len(answer) < 100:  # If answer is too brief
                        # Add more context for statistical queries
                        if df_summary is not None:
                            stats_context = f"\n\n📈 **Additional Statistics:**\n"
                            stats_context += f"- **Total Faculty Members:** {len(df_summary)}\n"
                            stats_context += f"- **Total Publications:** {int(df_summary['paper_count'].sum())}\n"
                            stats_context += f"- **Active Departments:** {len(df_summary['Department'].unique())}\n"
                            
                            dept_dist = df_summary['Department'].value_counts().head(3)
                            stats_context += f"- **Top Departments:** {', '.join([f'{dept} ({count})' for dept, count in dept_dist.items()])}\n"
                            
                            answer += stats_context
                
            except Exception as e:
                answer = f"❌ An error occurred while processing your request: {e}"
                st.error("Please try rephrasing your question or contact support if the issue persists.")
            
            st.markdown(answer)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

