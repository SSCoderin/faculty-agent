import pandas as pd
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import re
import os
import shutil
import json
import pickle
from typing import List, Dict, Any

import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

def reset_vector_store(vector_store_dir: str = "./vector_store_db"):
    """
    Reset/delete the existing vector store to force recreation.
    Use this when there are compatibility issues with stored data.
    """
    try:
        if os.path.exists(vector_store_dir):
            shutil.rmtree(vector_store_dir)
            print(f"Successfully removed vector store directory: {vector_store_dir}")
            return True
    except Exception as e:
        print(f"Error removing vector store directory: {e}")
        return False
    return True

def load_processed_data(vector_store_dir: str):
    """
    Load processed data from JSON files instead of Excel files.
    """
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
        print(f"Faculty: {len(df_faculty)}, Papers: {len(df_papers)}")
        return df_faculty, df_papers, summary_stats
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None, None, None

def load_and_create_vector_store(google_api_key: str):
    """
    Load the existing vector store, independent of Excel files.
    """
    vector_store_dir = "./vector_store_db"
    
    if os.path.exists(vector_store_dir) and os.listdir(vector_store_dir):
        try:
            print("Loading existing vector store from disk...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            vector_store = Chroma(persist_directory=vector_store_dir, embedding_function=embeddings)
            
            df_faculty, df_papers, summary_stats = load_processed_data(vector_store_dir)
            
            if df_faculty is not None and df_papers is not None:
                print(f"Successfully loaded existing vector store with {len(df_faculty)} faculty and {len(df_papers)} papers")
                return vector_store, None, df_faculty, df_papers, summary_stats
            else:
                print("Could not load processed data, but vector store exists. Continuing with vector store only.")
                return vector_store, None, None, None, None
                
        except Exception as e:
            return None, f"Error loading existing vector store: {e}", None, None, None
    else:
        return None, "Vector store does not exist. Please create it first.", None, None, None

def create_agent_chain(vector_store, google_api_key: str, df_faculty=None, df_papers=None, summary_stats=None):
    """
    Creates an enhanced retrieval chain that works independently of Excel files.
    """
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=google_api_key, 
        temperature=0.1,
        max_output_tokens=5072  
    )

    summary_text = ""
    if summary_stats:
        summary_text = f"""
        COMPREHENSIVE FACULTY DATABASE STATISTICS:
        - Total Faculty Members: {summary_stats['total_faculty']}
        - Research Papers Available: {summary_stats['total_papers']}
        - Active Departments: {len(summary_stats['departments'])}
        - Departments: {', '.join(summary_stats['departments'])}
        - Research Areas: {len(summary_stats['research_interests'])}
        - Last Updated: {summary_stats['last_updated']}
        """
    elif df_faculty is not None:
        dept_counts = df_faculty['Department'].value_counts().to_dict()
        total_papers = len(df_papers) if df_papers is not None else 0
        
        summary_text = f"""
        COMPREHENSIVE FACULTY DATABASE STATISTICS:
        - Total Faculty Members: {len(df_faculty)}
        - Research Papers Available: {total_papers}
        - Active Departments: {len(df_faculty['Department'].unique())}
        - Faculty Distribution by Department: {dept_counts}
        - Research Interests: {len(df_faculty['Research Interest'].unique())} unique areas
        """

    prompt = PromptTemplate(
        template='''

        You are a comprehensive faculty information assistant with access to complete faculty profiles, research information, and publication data. 

        IMPORTANT INSTRUCTIONS:
        - ALWAYS provide COMPLETE, DETAILED answers using ALL relevant context information without truncation, summarization, or omissions (no 'and more').
        - For faculty queries: List all matching faculty with name, position, and email only; always start with the total number of faculty found; retrieve and include ALL faculty members.
        - For research paper queries: Include complete title, full abstract, author details, and department.
        - For statistical queries: Use the database overview below.
        - Always organize responses clearly with proper formatting; include contact email when relevant.
        - For publication searches: Provide complete paper titles and full abstracts; when listing multiple items, give comprehensive info for each.
        - Do NOT mention publication counts or numbers of papers.
        - For queries on faculty research/publications: Go through ALL retrieved research_paper documents and metadata. List each paper in a numbered list with:
        - Full Title
        - Complete Abstract (no truncation)
        - Author Department
        - Author Position
        - Author Research Interest
        - Contact Email
        Organize by faculty name if multiple; connect papers to faculty using 'faculty_name' in metadata, ignoring case. Analyze all metadata fields like 'paper_title', 'paper_abstract', 'faculty_name'.

        DATABASE OVERVIEW:
        {summary_stats}

        RETRIEVED CONTEXT (including content and metadata):
        {context}

        USER QUESTION: {input}

        COMPREHENSIVE ANSWER:
        ''',
            input_variables=["context", "input", "summary_stats"],
        )

    classification_prompt = PromptTemplate(
        template="""Classify the following query into one of two categories:

1. General Faculty Information - Queries about faculty members, departments, positions, contacts, bios, education, or general information that do not explicitly mention research.

2. Research Information - Queries that explicitly mention research, publications, papers, abstracts, or ask about faculty's research work, writings, or similar.

Only output the category number (1 or 2).

Query: {input}""",
        input_variables=["input"],
    )
    
    class EnhancedIndependentChain:
        def __init__(self, vector_store, llm, prompt, classification_prompt, summary_text):
            self.vector_store = vector_store
            self.llm = llm
            self.prompt = prompt
            self.classification_prompt = classification_prompt
            self.summary_text = summary_text
        
        def invoke(self, query_dict):
            try:
                formatted_class_prompt = self.classification_prompt.format(input=query_dict["input"])
                classification_response = self.llm.invoke(formatted_class_prompt).strip()
                
                if "2" in classification_response:
                    search_filter = None
                    k = 100
                else:
                    search_filter = {"document_type": {"$in": ["faculty_profile", "department_mapping", "research_profile"]}}
                    k = 100  # Increased to retrieve all
                
                retriever = self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,
                        "fetch_k": k * 2,
                        "lambda_mult": 0.3,
                        "filter": search_filter
                    }
                )
                
                docs = retriever.get_relevant_documents(query_dict["input"])
                organized_context = self._organize_context(docs)
                
                formatted_prompt = self.prompt.format(
                    context=organized_context,
                    input=query_dict["input"],
                    summary_stats=self.summary_text
                )
                
                try:
                    response = self.llm.invoke(formatted_prompt)
                except Exception as llm_error:
                    print(f"LLM error: {llm_error}")
                    response = "I apologize, but I encountered an error processing your request. Please try rephrasing your question."
                
                return {
                    "input": query_dict["input"],
                    "context": docs,
                    "answer": response
                }
            except Exception as e:
                print(f"Chain invoke error: {e}")
                return {
                    "input": query_dict["input"],
                    "context": [],
                    "answer": "I apologize, but I encountered an error processing your request. Please try again."
                }
        
        def _organize_context(self, docs):
            """Organize retrieved documents by type for better context structure, including metadata."""
            context_sections = {
                'faculty_profile': [],
                'research_profile': [],
                'research_paper': [],
                'department_mapping': []
            }
            
            for doc in docs:
                doc_type = doc.metadata.get('document_type', 'faculty_profile')
                if doc_type in context_sections:
                    section_content = f"Content: {doc.page_content}\nMetadata: {json.dumps(doc.metadata)}"
                    context_sections[doc_type].append(section_content)
            
            organized_parts = []
            
            if context_sections['faculty_profile']:
                organized_parts.append("FACULTY PROFILES:\n" + "\n---\n".join(context_sections['faculty_profile']))
            
            if context_sections['research_profile']:
                organized_parts.append("RESEARCH INFORMATION:\n" + "\n---\n".join(context_sections['research_profile']))
            
            if context_sections['research_paper']:
                organized_parts.append("RESEARCH PAPERS:\n" + "\n---\n".join(context_sections['research_paper']))
            
            if context_sections['department_mapping']:
                organized_parts.append("DEPARTMENT INFORMATION:\n" + "\n---\n".join(context_sections['department_mapping']))
            
            return "\n\n".join(organized_parts)
    
    return EnhancedIndependentChain(vector_store, llm, prompt, classification_prompt, summary_text)

def get_vector_store_info(vector_store_dir: str = "./vector_store_db"):
    """
    Get information about the existing vector store without needing Excel files.
    """
    try:
        if not os.path.exists(vector_store_dir):
            return None, "Vector store does not exist"
        
        metadata_dir = os.path.join(vector_store_dir, "metadata")
        if os.path.exists(os.path.join(metadata_dir, "summary_stats.json")):
            with open(os.path.join(metadata_dir, "summary_stats.json"), 'r') as f:
                summary_stats = json.load(f)
            return summary_stats, None
        else:
            return None, "Summary statistics not found"
            
    except Exception as e:
        return None, f"Error reading vector store info: {e}"

def test_retrieval(vector_store, query: str, k: int = 100):
    """Test function to see what documents are being retrieved for a query."""
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k*2, "lambda_mult": 0.3}
    )
    
    docs = retriever.get_relevant_documents(query)
    
    print(f"\nRetrieved {len(docs)} documents for query: '{query}'")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(f"Type: {doc.metadata.get('document_type', 'unknown')}")
        print(f"Faculty: {doc.metadata.get('faculty_name', 'N/A')}")
        print(f"Department: {doc.metadata.get('department', 'N/A')}")
        if doc.metadata.get('document_type') == 'research_paper':
            print(f"Paper: {doc.metadata.get('paper_title', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")
    
    return docs