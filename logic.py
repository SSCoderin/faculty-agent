
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os
import json
from typing import List, Dict, Any

import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

def load_existing_vector_store(google_api_key: str, vector_store_dir: str = "./vector_store_db"):
    """Load existing vector store and processed data."""
    try:
        print("Loading existing vector store from disk...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        vector_store = Chroma(persist_directory=vector_store_dir, embedding_function=embeddings)
        
        df_faculty, df_papers, summary_stats = load_processed_data(vector_store_dir)
        
        if df_faculty is not None and df_papers is not None:
            print(f"Successfully loaded vector store with {len(df_faculty)} faculty and {len(df_papers)} papers")
            return vector_store, None, df_faculty, df_papers, summary_stats
        else:
            print("Could not load processed data, but vector store exists. Continuing with vector store only.")
            return vector_store, None, None, None, None
            
    except Exception as e:
        return None, f"Error loading vector store: {e}", None, None, None

def load_processed_data(vector_store_dir: str):
    """Load processed data from JSON files."""
    try:
        metadata_dir = os.path.join(vector_store_dir, "metadata")
        
        with open(os.path.join(metadata_dir, "faculty_data.json"), 'r', encoding='utf-8') as f:
            faculty_data = json.load(f)
        df_faculty = pd.DataFrame(faculty_data)
        
        with open(os.path.join(metadata_dir, "paper_data.json"), 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        df_papers = pd.DataFrame(paper_data)
        
        with open(os.path.join(metadata_dir, "summary_stats.json"), 'r', encoding='utf-8') as f:
            summary_stats = json.load(f)
        
        print(f"Successfully loaded processed data from {metadata_dir}")
        return df_faculty, df_papers, summary_stats
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None, None, None

def create_smart_agent_chain(vector_store, google_api_key: str, df_faculty=None, df_papers=None, summary_stats=None):
    """Creates an intelligent retrieval chain with structured responses."""
    
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=google_api_key, 
        temperature=0.1,
        max_output_tokens=2048
    )
    
    basic_listing_prompt = PromptTemplate(
        template="""You are a faculty information assistant. Provide a well-structured, minimal response using the context.

        RESPONSE FORMAT:
        - Use bullet points for lists of faculty.
        - For each faculty member, provide: Name, Position, Department, and Email.

        CONTEXT:
        {context}

        QUESTION: {input}

        STRUCTURED RESPONSE:""",
        input_variables=["context", "input"]
    )
    
    research_prompt = PromptTemplate(
        template="""You are a research information assistant. Provide a structured research overview using the context.

        RESPONSE FORMAT:
        📚 **Research Overview**
        👤 **Faculty:** [Name]
        🏢 **Department:** [Department]
        📧 **Email:** [Email]
        🔬 **Research Interests:** [Research areas]
        📄 **Publications:** [Key papers]

        CONTEXT:
        {context}

        QUESTION: {input}

        STRUCTURED RESEARCH RESPONSE:""",
        input_variables=["context", "input"]
    )
    
    abstract_prompt = PromptTemplate(
        template="""You are a paper abstract assistant. Provide the paper's abstract information from the context.

        RESPONSE FORMAT:
        📄 **Paper Title:** [Title]
        👤 **Author:** [Faculty Name]
        📝 **Abstract:**
        [Complete abstract text]

        CONTEXT:
        {context}

        QUESTION: {input}

        STRUCTURED ABSTRACT RESPONSE:""",
        input_variables=["context", "input"]
    )
    
    class IntelligentFacultyChain:
        def __init__(self, vector_store, llm, prompts):
            self.vector_store = vector_store
            self.llm = llm
            self.basic_prompt = prompts['basic']
            self.research_prompt = prompts['research']
            self.abstract_prompt = prompts['abstract']
        
        def _classify_query(self, query: str) -> str:
            """Classify query type to determine appropriate retrieval strategy."""
            query_lower = query.lower()
            
            abstract_keywords = ['abstract for', 'what is the abstract', 'paper abstract']
            if any(keyword in query_lower for keyword in abstract_keywords):
                return 'abstract'
            
            research_keywords = [
                'research', 'working on', 'work on', 'publications', 
                'papers by', 'expertise', 'specializes in', 'area of'
            ]
            if any(keyword in query_lower for keyword in research_keywords):
                return 'research'
            
            listing_keywords = ['list all', 'who are the faculty', 'faculty from', "who is"]
            if any(keyword in query_lower for keyword in listing_keywords):
                return 'basic'
                
            return 'basic'
        
        def _get_relevant_documents(self, query: str, query_type: str):
            """Get relevant documents based on query type."""
            try:
                if query_type == 'abstract':
                    retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5, "filter": {"document_type": "paper_abstract"}}
                    )
                elif query_type == 'basic':
                    retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 20, "filter": {"document_type": "faculty_basic"}}
                    )
                else:
                    retriever = self.vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.7, "filter": {"document_type": "faculty_research"}}
                    )
                
                return retriever.get_relevant_documents(query)
            except Exception as e:
                print(f"Error retrieving documents: {e}")
                return []
        
        def _format_context(self, docs: List[Document]):
            """Formats documents into a single string for the prompt."""
            return "\n---\n".join([doc.page_content for doc in docs])
        
        def invoke(self, query_dict: Dict[str, Any]):
            """Main invoke method with intelligent query handling."""
            try:
                query = query_dict["input"]
                query_type = self._classify_query(query)
                
                print(f"Query type classified as: {query_type}")
                
                docs = self._get_relevant_documents(query, query_type)
                
                if not docs:
                    return {
                        "input": query, "context": [],
                        "answer": "❌ No relevant information found. Please try rephrasing your question.",
                        "query_type": query_type
                    }
                
                context = self._format_context(docs)
                
                if query_type == 'basic':
                    prompt = self.basic_prompt
                elif query_type == 'abstract':
                    prompt = self.abstract_prompt
                else:
                    prompt = self.research_prompt
                
                formatted_prompt = prompt.format(context=context, input=query)
                
                try:
                    response = self.llm.invoke(formatted_prompt)
                except Exception as llm_error:
                    print(f"LLM error: {llm_error}")
                    response = "❌ I encountered an error processing your request."
                
                return {"input": query, "context": docs, "answer": response, "query_type": query_type}
            except Exception as e:
                print(f"Chain invoke error: {e}")
                return {
                    "input": query_dict.get("input", ""), "context": [],
                    "answer": "❌ A system error occurred. Please try again.",
                    "query_type": "error"
                }
    
    prompts = {'basic': basic_listing_prompt, 'research': research_prompt, 'abstract': abstract_prompt}
    return IntelligentFacultyChain(vector_store, llm, prompts)

def load_and_create_vector_store(faculty_data_path: str, faculty_paper_path: str, google_api_key: str):
    """Simplified function that only loads existing vector store."""
    return load_existing_vector_store(google_api_key)

create_agent_chain = create_smart_agent_chain