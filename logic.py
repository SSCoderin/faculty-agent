import pandas as pd
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
import os
import json
import shutil

import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

def reset_vector_store(vector_store_dir: str = "./vector_store_db"):
    """
    Reset/delete the existing vector store if needed.
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

def load_vector_store(google_api_key: str, vector_store_dir: str = "./vector_store_db"):
    """
    Load existing vector store from disk.
    """
    if not os.path.exists(vector_store_dir) or not os.listdir(vector_store_dir):
        return None, "Vector store does not exist. Please create one first."
    
    try:
        print("Loading existing vector store from disk...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=google_api_key
        )
        vector_store = Chroma(
            persist_directory=vector_store_dir, 
            embedding_function=embeddings
        )
        
        # Load summary statistics if available
        summary_stats = load_summary_stats(vector_store_dir)
        
        print("Successfully loaded vector store")
        return vector_store, summary_stats
        
    except Exception as e:
        return None, f"Error loading vector store: {e}"

def load_summary_stats(vector_store_dir: str):
    """
    Load summary statistics from metadata.
    """
    try:
        metadata_dir = os.path.join(vector_store_dir, "metadata")
        stats_file = os.path.join(metadata_dir, "summary_stats.json")
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print("Summary statistics not found")
            return None
            
    except Exception as e:
        print(f"Error loading summary statistics: {e}")
        return None

def create_faculty_search_agent(vector_store, google_api_key: str, summary_stats=None):
    """
    Creates a comprehensive faculty search agent using the vector store.
    """
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=google_api_key, 
        temperature=0.1,
        max_output_tokens=5216
    )

    # Create summary text from statistics
    summary_text = ""
    if summary_stats:
        summary_text = f"""
        FACULTY DATABASE OVERVIEW:
        - Total Faculty Members: {summary_stats.get('total_faculty', 'N/A')}
        - Research Papers Available: {summary_stats.get('total_papers', 'N/A')}
        - Active Departments: {len(summary_stats.get('departments', []))}
        - Departments: {', '.join(summary_stats.get('departments', []))}
        - Research Areas: {len(summary_stats.get('research_interests', []))}
        - Last Updated: {summary_stats.get('last_updated', 'N/A')}
        """

    prompt = PromptTemplate(
        template='''
        You are a comprehensive faculty information assistant with access to complete faculty profiles, research information, and publication data.

        IMPORTANT INSTRUCTIONS:
        1. ALWAYS provide COMPLETE and DETAILED answers - never truncate information
        2. Use ALL relevant information from the context below
        3. For faculty queries, include: name, position, department, email, research interests, and publications
        4. For research paper queries, include: complete title, full abstract, author details, and department
        5. For statistical queries, use the database overview below
        6. Always organize responses clearly with proper formatting
        7. Include contact information (email) when available and relevant
        8. For publication searches, provide complete paper titles and full abstracts
        9. When listing multiple items, provide comprehensive information for each
        10. Do NOT mention publication counts or numbers of papers
        11. When asked about research, publications, or related topics, go through ALL retrieved research paper documents. List each paper comprehensively with:
           - Full Title
           - Complete Abstract (without truncation)
           - Author Department
           - Author Position
           - Author Research Interest
           - Contact Email
        12. Be thorough: Include every relevant detail from the context without summarization

        Think step by step:
        1. Analyze the user query carefully
        2. Review all retrieved context, including metadata
        3. If the query relates to research/publications, identify ALL research_paper documents
        4. Extract complete information from each document
        5. Structure the response comprehensively

        DATABASE OVERVIEW:
        {summary_stats}

        RETRIEVED CONTEXT:
        {context}

        USER QUESTION: {input}

        COMPREHENSIVE ANSWER:
        ''',
        input_variables=["context", "input", "summary_stats"],
    )

    # Enhanced retriever for comprehensive results
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 80,
            "fetch_k": 160,
            "lambda_mult": 0.3
        }
    )
    
    class FacultySearchAgent:
        def __init__(self, retriever, llm, prompt, summary_text):
            self.retriever = retriever
            self.llm = llm
            self.prompt = prompt
            self.summary_text = summary_text
        
        def search(self, query: str):
            """
            Main search function for faculty information.
            """
            try:
                # Retrieve relevant documents
                docs = self.retriever.get_relevant_documents(query)
                organized_context = self._organize_context(docs)
                
                # Format and send prompt
                formatted_prompt = self.prompt.format(
                    context=organized_context,
                    input=query,
                    summary_stats=self.summary_text
                )
                
                # Get response from LLM
                response = self.llm.invoke(formatted_prompt)
                
                return {
                    "query": query,
                    "answer": response,
                    "documents_found": len(docs)
                }
                
            except Exception as e:
                print(f"Search error: {e}")
                return {
                    "query": query,
                    "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                    "documents_found": 0
                }
        
        def _organize_context(self, docs):
            """Organize retrieved documents by type for better context structure."""
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
    
    return FacultySearchAgent(retriever, llm, prompt, summary_text)

def get_vector_store_info(vector_store_dir: str = "./vector_store_db"):
    """
    Get information about the existing vector store.
    """
    try:
        if not os.path.exists(vector_store_dir):
            return None, "Vector store does not exist"
        
        summary_stats = load_summary_stats(vector_store_dir)
        if summary_stats:
            return summary_stats, None
        else:
            return None, "Summary statistics not found"
            
    except Exception as e:
        return None, f"Error reading vector store info: {e}"

def test_search(vector_store, query: str, k: int = 25):
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
