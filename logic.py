# import pandas as pd
# from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.document_loaders import DataFrameLoader
# from langchain_community.vectorstores import Chroma
# from langchain.schema import Document
# import re
# import os
# import shutil
# import json
# import pickle
# from typing import List, Dict, Any

# # Override sqlite3 with pysqlite3 if available to resolve version compatibility
# import sys
# try:
#     __import__('pysqlite3')
#     sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# except ImportError:
#     pass

# def reset_vector_store(vector_store_dir: str = "./vector_store_db"):
#     """
#     Reset/delete the existing vector store to force recreation.
#     Use this when there are compatibility issues with stored data.
#     """
#     try:
#         if os.path.exists(vector_store_dir):
#             shutil.rmtree(vector_store_dir)
#             print(f"Successfully removed vector store directory: {vector_store_dir}")
#             return True
#     except Exception as e:
#         print(f"Error removing vector store directory: {e}")
#         return False
#     return True

# def normalize_faculty_name(name: str) -> str:
#     """
#     Normalize faculty names for better matching.
#     Handles variations like 'Prof. Name' vs 'Name', extra spaces, etc.
#     """
#     if pd.isna(name) or not name:
#         return ''
    
#     # Remove common prefixes and clean up
#     name = str(name).strip()
#     name = re.sub(r'^(Prof\.?\s*|Dr\.?\s*|Mr\.?\s*|Ms\.?\s*|Mrs\.?\s*)', '', name, flags=re.IGNORECASE)
#     name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with single space
#     return name.strip()

# def save_processed_data(faculty_data: pd.DataFrame, paper_data: pd.DataFrame, vector_store_dir: str):
#     """
#     Save processed data as JSON for future use without Excel files.
#     """
#     try:
#         # Create metadata directory
#         metadata_dir = os.path.join(vector_store_dir, "metadata")
#         os.makedirs(metadata_dir, exist_ok=True)
        
#         # Save faculty data
#         faculty_json = faculty_data.to_dict('records')
#         with open(os.path.join(metadata_dir, "faculty_data.json"), 'w', encoding='utf-8') as f:
#             json.dump(faculty_json, f, ensure_ascii=False, indent=2)
        
#         # Save paper data
#         paper_json = paper_data.to_dict('records')
#         with open(os.path.join(metadata_dir, "paper_data.json"), 'w', encoding='utf-8') as f:
#             json.dump(paper_json, f, ensure_ascii=False, indent=2)
        
#         # Save summary statistics (without paper count)
#         summary_stats = {
#             'total_faculty': len(faculty_data),
#             'total_papers': len(paper_data),
#             'departments': faculty_data['Department'].unique().tolist(),
#             'research_interests': faculty_data['Research Interest'].unique().tolist(),
#             'last_updated': pd.Timestamp.now().isoformat()
#         }
        
#         with open(os.path.join(metadata_dir, "summary_stats.json"), 'w', encoding='utf-8') as f:
#             json.dump(summary_stats, f, ensure_ascii=False, indent=2)
            
#         print(f"Successfully saved processed data to {metadata_dir}")
#         return True
#     except Exception as e:
#         print(f"Error saving processed data: {e}")
#         return False

# def load_processed_data(vector_store_dir: str):
#     """
#     Load processed data from JSON files instead of Excel files.
#     """
#     try:
#         metadata_dir = os.path.join(vector_store_dir, "metadata")
        
#         # Load faculty data
#         with open(os.path.join(metadata_dir, "faculty_data.json"), 'r', encoding='utf-8') as f:
#             faculty_data = json.load(f)
#         df_faculty = pd.DataFrame(faculty_data)
        
#         # Load paper data
#         with open(os.path.join(metadata_dir, "paper_data.json"), 'r', encoding='utf-8') as f:
#             paper_data = json.load(f)
#         df_papers = pd.DataFrame(paper_data)
        
#         # Load summary stats
#         with open(os.path.join(metadata_dir, "summary_stats.json"), 'r', encoding='utf-8') as f:
#             summary_stats = json.load(f)
        
#         print(f"Successfully loaded processed data from {metadata_dir}")
#         print(f"Faculty: {len(df_faculty)}, Papers: {len(df_papers)}")
#         return df_faculty, df_papers, summary_stats
#     except Exception as e:
#         print(f"Error loading processed data: {e}")
#         return None, None, None

# def create_comprehensive_documents(df_faculty: pd.DataFrame, df_papers: pd.DataFrame):
#     """
#     Create comprehensive documents for vector store including all data.
#     Each research paper gets its own dedicated document.
#     """
#     documents = []
    
#     # Create paper lookup dictionary for faster access
#     paper_lookup = {}
#     if not df_papers.empty:
#         for _, paper_row in df_papers.iterrows():
#             faculty_name_norm = normalize_faculty_name(paper_row['faculty_name'])
#             if faculty_name_norm not in paper_lookup:
#                 paper_lookup[faculty_name_norm] = []
#             paper_lookup[faculty_name_norm].append({
#                 'title': paper_row['paper_title'],
#                 'abstract': paper_row['paper_abstract']
#             })
    
#     # Process each faculty member
#     for _, faculty_row in df_faculty.iterrows():
#         faculty_name = faculty_row['Name']
#         faculty_name_norm = normalize_faculty_name(faculty_name)
#         faculty_papers = paper_lookup.get(faculty_name_norm, [])
        
#         # === Document 1: Complete Faculty Profile ===
#         profile_content = f"""Faculty Profile: {faculty_name}
# Position: {faculty_row.get('Position', 'Not specified')}
# Department: {faculty_row.get('Department', 'Not specified')}
# Email: {faculty_row.get('Email', 'Not available')}
# Research Interest: {faculty_row.get('Research Interest', 'Not specified')}"""
        
#         # Add additional fields if available
#         if 'Link to Detail' in faculty_row and pd.notna(faculty_row['Link to Detail']):
#             profile_content += f"\nProfile Link: {faculty_row['Link to Detail']}"
            
#         if 'Faculty Education' in faculty_row and pd.notna(faculty_row['Faculty Education']):
#             profile_content += f"\nEducation: {faculty_row['Faculty Education']}"
            
#         if 'Bio' in faculty_row and pd.notna(faculty_row['Bio']):
#             profile_content += f"\nBio: {faculty_row['Bio']}"
            
#         if 'Research' in faculty_row and pd.notna(faculty_row['Research']):
#             profile_content += f"\nResearch Details: {faculty_row['Research']}"
        
#         # Add paper titles overview (without count)
#         if faculty_papers:
#             paper_titles = [paper['title'] for paper in faculty_papers if pd.notna(paper['title'])]
#             if paper_titles:
#                 profile_content += f"\nPublications: {' | '.join(paper_titles[:5])}"  # Show first 5 titles
        
#         profile_doc = Document(
#             page_content=profile_content,
#             metadata={
#                 'faculty_name': faculty_name,
#                 'department': faculty_row.get('Department', 'Not specified'),
#                 'position': faculty_row.get('Position', 'Not specified'),
#                 'research_interest': faculty_row.get('Research Interest', 'Not specified'),
#                 'email': faculty_row.get('Email', 'Not available'),
#                 'document_type': 'faculty_profile',
#                 'education': faculty_row.get('Faculty Education', 'Not available'),
#                 'bio': faculty_row.get('Bio', 'Not available'),
#                 'profile_link': faculty_row.get('Link to Detail', 'Not available')
#             }
#         )
#         documents.append(profile_doc)
        
#         # === Document 2: Research Focus Document ===
#         research_content = f"""Research Profile: {faculty_name}
# Department: {faculty_row.get('Department', 'Not specified')}
# Primary Research Interest: {faculty_row.get('Research Interest', 'Not specified')}
# Academic Position: {faculty_row.get('Position', 'Not specified')}"""
        
#         if 'Research' in faculty_row and pd.notna(faculty_row['Research']):
#             research_content += f"\nDetailed Research Information: {faculty_row['Research']}"
            
#         if faculty_papers:
#             # Add research themes from paper titles (without count)
#             paper_titles = [paper['title'] for paper in faculty_papers if pd.notna(paper['title'])]
#             if paper_titles:
#                 research_content += f"\nResearch Areas from Publications: {' | '.join(paper_titles)}"
        
#         research_doc = Document(
#             page_content=research_content,
#             metadata={
#                 'faculty_name': faculty_name,
#                 'department': faculty_row.get('Department', 'Not specified'),
#                 'research_interest': faculty_row.get('Research Interest', 'Not specified'),
#                 'position': faculty_row.get('Position', 'Not specified'),
#                 'document_type': 'research_profile'
#             }
#         )
#         documents.append(research_doc)
        
#         # === Document 3: Department Mapping Document ===
#         dept_content = f"""Department Information
# Department: {faculty_row.get('Department', 'Not specified')}
# Faculty Member: {faculty_name}
# Position: {faculty_row.get('Position', 'Not specified')}
# Research Area: {faculty_row.get('Research Interest', 'Not specified')}
# Contact: {faculty_row.get('Email', 'Not available')}"""
        
#         dept_doc = Document(
#             page_content=dept_content,
#             metadata={
#                 'faculty_name': faculty_name,
#                 'department': faculty_row.get('Department', 'Not specified'),
#                 'research_interest': faculty_row.get('Research Interest', 'Not specified'),
#                 'position': faculty_row.get('Position', 'Not specified'),
#                 'document_type': 'department_mapping'
#             }
#         )
#         documents.append(dept_doc)
        
#         # === Dedicated Documents for Each Research Paper ===
#         for i, paper in enumerate(faculty_papers, 1):
#             if pd.notna(paper['title']) and str(paper['title']).strip():
#                 paper_title = str(paper['title']).strip()
#                 paper_abstract = str(paper['abstract']).strip() if pd.notna(paper['abstract']) else 'Abstract not available'
                
#                 # Create a dedicated document for each paper
#                 paper_content = f"""Research Paper: {paper_title}
# Author: {faculty_name}
# Author Department: {faculty_row.get('Department', 'Not specified')}
# Author Position: {faculty_row.get('Position', 'Not specified')}
# Author Research Interest: {faculty_row.get('Research Interest', 'Not specified')}
# Author Email: {faculty_row.get('Email', 'Not available')}

# Title: {paper_title}

# Abstract: {paper_abstract}"""
                
#                 paper_doc = Document(
#                     page_content=paper_content,
#                     metadata={
#                         'faculty_name': faculty_name,
#                         'department': faculty_row.get('Department', 'Not specified'),
#                         'research_interest': faculty_row.get('Research Interest', 'Not specified'),
#                         'position': faculty_row.get('Position', 'Not specified'),
#                         'paper_title': paper_title,
#                         'paper_abstract': paper_abstract,
#                         'document_type': 'research_paper',
#                         'paper_index': i,
#                         'author_email': faculty_row.get('Email', 'Not available')
#                     }
#                 )
#                 documents.append(paper_doc)
    
#     print(f"Created {len(documents)} documents for vector store")
#     return documents

# def load_and_create_vector_store(faculty_data_path: str, faculty_paper_path: str, google_api_key: str):
#     """
#     Enhanced function to load data, create vector store, and make it independent of Excel files.
#     """
#     vector_store_dir = "./vector_store_db"
    
#     # Check if vector store already exists with processed data
#     if os.path.exists(vector_store_dir) and os.listdir(vector_store_dir):
#         try:
#             print("Loading existing vector store from disk...")
#             embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
#             vector_store = Chroma(persist_directory=vector_store_dir, embedding_function=embeddings)
            
#             # Try to load processed data
#             df_faculty, df_papers, summary_stats = load_processed_data(vector_store_dir)
            
#             if df_faculty is not None and df_papers is not None:
#                 print(f"Successfully loaded existing vector store with {len(df_faculty)} faculty and {len(df_papers)} papers")
#                 return vector_store, None, df_faculty, df_papers, summary_stats
#             else:
#                 print("Could not load processed data, but vector store exists. Continuing with vector store only.")
#                 return vector_store, None, None, None, None
                
#         except Exception as e:
#             print(f"Error loading existing vector store: {e}. Creating new one...")
#             try:
#                 shutil.rmtree(vector_store_dir)
#                 print("Removed corrupted vector store directory")
#             except Exception as cleanup_error:
#                 print(f"Could not clean up vector store directory: {cleanup_error}")
    
#     # Load and process Excel files (only needed for first time setup)
#     print("Creating new vector store from Excel files...")
    
#     # Load faculty data
#     try:
#         df_faculty = pd.read_excel(faculty_data_path)
#         print(f"Loaded faculty data: {len(df_faculty)} records")
        
#         # Ensure required columns exist and clean data
#         required_columns = ['Name']
#         missing_columns = [col for col in required_columns if col not in df_faculty.columns]
#         if missing_columns:
#             return None, f"Error: Missing required columns in faculty data: {missing_columns}", None, None, None
            
#         # Clean and standardize faculty data
#         df_faculty['Name'] = df_faculty['Name'].fillna('Unknown').astype(str).str.strip()
#         df_faculty['Department'] = df_faculty['Department'].fillna('Unknown Department').astype(str).str.strip()
#         df_faculty['Position'] = df_faculty['Position'].fillna('Unknown Position').astype(str).str.strip()
#         df_faculty['Research Interest'] = df_faculty['Research Interest'].fillna('Not specified').astype(str).str.strip()
#         df_faculty['Email'] = df_faculty['Email'].fillna('Not available').astype(str).str.strip()
        
#         # Handle optional columns
#         optional_columns = ['Link to Detail', 'Faculty Education', 'Bio', 'Research']
#         for col in optional_columns:
#             if col not in df_faculty.columns:
#                 df_faculty[col] = 'Not available'
#             else:
#                 df_faculty[col] = df_faculty[col].fillna('Not available').astype(str).str.strip()
            
#     except FileNotFoundError:
#         return None, "Error: The faculty data file was not found.", None, None, None
#     except Exception as e:
#         return None, f"Error loading faculty data: {e}", None, None, None

#     # Load paper data
#     try:
#         df_papers = pd.read_excel(faculty_paper_path)
#         print(f"Loaded paper data: {len(df_papers)} records")
        
#         # Clean paper data
#         df_papers['faculty_name'] = df_papers['faculty_name'].fillna('Unknown').astype(str).str.strip()
#         df_papers['paper_title'] = df_papers['paper_title'].fillna('Untitled').astype(str).str.strip()
#         df_papers['paper_abstract'] = df_papers['paper_abstract'].fillna('Abstract not available').astype(str).str.strip()
        
#     except FileNotFoundError:
#         print("Warning: The paper data file was not found. Proceeding with faculty data only.")
#         df_papers = pd.DataFrame(columns=['faculty_name', 'paper_title', 'paper_abstract'])
#     except Exception as e:
#         print(f"Warning: Could not read paper data file. Error: {e}. Proceeding with faculty data only.")
#         df_papers = pd.DataFrame(columns=['faculty_name', 'paper_title', 'paper_abstract'])

#     # Create comprehensive documents
#     documents = create_comprehensive_documents(df_faculty, df_papers)
    
#     # Create vector store with all documents
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
#         vector_store = Chroma.from_documents(
#             documents, 
#             embeddings, 
#             persist_directory=vector_store_dir
#         )
        
#         # Persist the vector store
#         vector_store.persist()
#         print(f"Vector store created and saved to {vector_store_dir}")
        
#         # Save processed data for future independence from Excel files
#         save_processed_data(df_faculty, df_papers, vector_store_dir)
        
#         # Create summary statistics (without paper count display)
#         summary_stats = {
#             'total_faculty': len(df_faculty),
#             'total_papers': len(df_papers),
#             'departments': df_faculty['Department'].unique().tolist(),
#             'research_interests': df_faculty['Research Interest'].unique().tolist(),
#             'last_updated': pd.Timestamp.now().isoformat()
#         }
        
#         return vector_store, None, df_faculty, df_papers, summary_stats
        
#     except Exception as e:
#         return None, f"Error creating vector store: {e}", None, None, None

# def create_agent_chain(vector_store, google_api_key: str, df_faculty=None, df_papers=None, summary_stats=None):
#     """
#     Creates an enhanced retrieval chain that works independently of Excel files.
#     """
#     llm = GoogleGenerativeAI(
#         model="gemini-1.5-flash", 
#         google_api_key=google_api_key, 
#         temperature=0.1,
#         max_output_tokens=3072  # Increased for comprehensive responses
#     )

#     # Create comprehensive summary statistics (without paper count)
#     summary_text = ""
#     if summary_stats:
#         summary_text = f"""
#         COMPREHENSIVE FACULTY DATABASE STATISTICS:
#         - Total Faculty Members: {summary_stats['total_faculty']}
#         - Research Papers Available: {summary_stats['total_papers']}
#         - Active Departments: {len(summary_stats['departments'])}
#         - Departments: {', '.join(summary_stats['departments'])}
#         - Research Areas: {len(summary_stats['research_interests'])}
#         - Last Updated: {summary_stats['last_updated']}
#         """
#     elif df_faculty is not None:
#         # Fallback to dataframe statistics
#         dept_counts = df_faculty['Department'].value_counts().to_dict()
#         total_papers = len(df_papers) if df_papers is not None else 0
        
#         summary_text = f"""
#         COMPREHENSIVE FACULTY DATABASE STATISTICS:
#         - Total Faculty Members: {len(df_faculty)}
#         - Research Papers Available: {total_papers}
#         - Active Departments: {len(df_faculty['Department'].unique())}
#         - Faculty Distribution by Department: {dept_counts}
#         - Research Interests: {len(df_faculty['Research Interest'].unique())} unique areas
#         """

#     prompt = PromptTemplate(
#         template="""
#         You are a comprehensive faculty information assistant with access to complete faculty profiles, research information, and publication data. 

#         IMPORTANT INSTRUCTIONS:
#         1. ALWAYS provide COMPLETE and DETAILED answers - never truncate information
#         2. Use ALL relevant information from the context below
#         3. For faculty queries, include: name, position, department, email, research interests, and publications
#         4. For research paper queries, include: complete title, full abstract, author details, and department
#         5. For statistical queries, use the database overview below
#         6. Always organize responses clearly with proper formatting
#         7. Include contact information (email) when available and relevant
#         8. For publication searches, provide complete paper titles and full abstracts
#         9. When listing multiple items, provide comprehensive information for each
#         10. Do NOT mention publication counts or numbers of papers
#         11. When asked about a faculty member's research, publications, or related topics, go through ALL retrieved research paper documents. List each paper comprehensively in a numbered list with:
#            - Full Title
#            - Complete Abstract (without truncation)
#            - Author Department
#            - Author Position
#            - Author Research Interest
#            - Contact Email
#            Organize the list by faculty name if multiple faculties are involved. Do not summarize or omit any details from the papers.
#         12. Be thorough: Include every relevant detail from the context without summarization, truncation, or saying 'and more'. Ensure all paper titles and abstracts are fully covered.

#         Think step by step:
#         1. Analyze the user query carefully.
#         2. Review all retrieved context, including the metadata for each document (which connects paper titles and abstracts to faculty names).
#         3. If the query relates to research, publications, or similar topics, identify ALL research_paper documents and their metadata. Go through each one, extracting paper titles, abstracts, and connections to faculty names.
#         4. Compile a comprehensive list using the metadata and content, ensuring all details are included without omission.
#         5. Structure the response based on the query, using the database overview if needed.

#         DATABASE OVERVIEW:
#         {summary_stats}

#         RETRIEVED CONTEXT (including content and metadata):
#         {context}

#         USER QUESTION: {input}

#         COMPREHENSIVE ANSWER:
#         """,
#         input_variables=["context", "input", "summary_stats"],
#     )

#     # Enhanced retriever for comprehensive results
#     retriever = vector_store.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": 50,  # Increased for better coverage of all papers
#             "fetch_k": 100,
#             "lambda_mult": 0.5
#         }
#     )
    
#     class EnhancedIndependentChain:
#         def __init__(self, retriever, llm, prompt, summary_text):
#             self.retriever = retriever
#             self.llm = llm
#             self.prompt = prompt
#             self.summary_text = summary_text
        
#         def invoke(self, query_dict):
#             try:
#                 docs = self.retriever.get_relevant_documents(query_dict["input"])
#                 organized_context = self._organize_context(docs)
                
#                 formatted_prompt = self.prompt.format(
#                     context=organized_context,
#                     input=query_dict["input"],
#                     summary_stats=self.summary_text
#                 )
                
#                 try:
#                     response = self.llm.invoke(formatted_prompt)
#                 except Exception as llm_error:
#                     print(f"LLM error: {llm_error}")
#                     response = "I apologize, but I encountered an error processing your request. Please try rephrasing your question."
                
#                 return {
#                     "input": query_dict["input"],
#                     "context": docs,
#                     "answer": response
#                 }
#             except Exception as e:
#                 print(f"Chain invoke error: {e}")
#                 return {
#                     "input": query_dict["input"],
#                     "context": [],
#                     "answer": "I apologize, but I encountered an error processing your request. Please try again."
#                 }
        
#         def _organize_context(self, docs):
#             """Organize retrieved documents by type for better context structure, including metadata."""
#             context_sections = {
#                 'faculty_profile': [],
#                 'research_profile': [],
#                 'research_paper': [],
#                 'department_mapping': []
#             }
            
#             for doc in docs:
#                 doc_type = doc.metadata.get('document_type', 'faculty_profile')
#                 if doc_type in context_sections:
#                     section_content = f"Content: {doc.page_content}\nMetadata: {json.dumps(doc.metadata)}"
#                     context_sections[doc_type].append(section_content)
            
#             organized_parts = []
            
#             if context_sections['faculty_profile']:
#                 organized_parts.append("FACULTY PROFILES:\n" + "\n---\n".join(context_sections['faculty_profile']))
            
#             if context_sections['research_profile']:
#                 organized_parts.append("RESEARCH INFORMATION:\n" + "\n---\n".join(context_sections['research_profile']))
            
#             if context_sections['research_paper']:
#                 organized_parts.append("RESEARCH PAPERS:\n" + "\n---\n".join(context_sections['research_paper']))
            
#             if context_sections['department_mapping']:
#                 organized_parts.append("DEPARTMENT INFORMATION:\n" + "\n---\n".join(context_sections['department_mapping']))
            
#             return "\n\n".join(organized_parts)
    
#     return EnhancedIndependentChain(retriever, llm, prompt, summary_text)

# def get_vector_store_info(vector_store_dir: str = "./vector_store_db"):
#     """
#     Get information about the existing vector store without needing Excel files.
#     """
#     try:
#         if not os.path.exists(vector_store_dir):
#             return None, "Vector store does not exist"
        
#         # Try to load summary stats
#         metadata_dir = os.path.join(vector_store_dir, "metadata")
#         if os.path.exists(os.path.join(metadata_dir, "summary_stats.json")):
#             with open(os.path.join(metadata_dir, "summary_stats.json"), 'r') as f:
#                 summary_stats = json.load(f)
#             return summary_stats, None
#         else:
#             return None, "Summary statistics not found"
            
#     except Exception as e:
#         return None, f"Error reading vector store info: {e}"

# # Testing and debugging function
# def test_retrieval(vector_store, query: str, k: int = 50):
#     """Test function to see what documents are being retrieved for a query."""
#     retriever = vector_store.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": k, "fetch_k": k*2, "lambda_mult": 0.5}
#     )
    
#     docs = retriever.get_relevant_documents(query)
    
#     print(f"\nRetrieved {len(docs)} documents for query: '{query}'")
#     for i, doc in enumerate(docs, 1):
#         print(f"\nDocument {i}:")
#         print(f"Type: {doc.metadata.get('document_type', 'unknown')}")
#         print(f"Faculty: {doc.metadata.get('faculty_name', 'N/A')}")
#         print(f"Department: {doc.metadata.get('department', 'N/A')}")
#         if doc.metadata.get('document_type') == 'research_paper':
#             print(f"Paper: {doc.metadata.get('paper_title', 'N/A')}")
#         print(f"Content: {doc.page_content[:200]}...")
    
#     return docs



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

# Override sqlite3 with pysqlite3 if available to resolve version compatibility
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

def normalize_faculty_name(name: str) -> str:
    """
    Normalize faculty names for better matching.
    Handles variations like 'Prof. Name' vs 'Name', extra spaces, etc.
    """
    if pd.isna(name) or not name:
        return ''
    
    # Remove common prefixes and clean up
    name = str(name).strip().lower()
    name = re.sub(r'^(prof\.?\s*|dr\.?\s*|mr\.?\s*|ms\.?\s*|mrs\.?\s*)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with single space
    return name.strip()

def save_processed_data(faculty_data: pd.DataFrame, paper_data: pd.DataFrame, vector_store_dir: str):
    """
    Save processed data as JSON for future use without Excel files.
    """
    try:
        # Create metadata directory
        metadata_dir = os.path.join(vector_store_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Save faculty data
        faculty_json = faculty_data.to_dict('records')
        with open(os.path.join(metadata_dir, "faculty_data.json"), 'w', encoding='utf-8') as f:
            json.dump(faculty_json, f, ensure_ascii=False, indent=2)
        
        # Save paper data
        paper_json = paper_data.to_dict('records')
        with open(os.path.join(metadata_dir, "paper_data.json"), 'w', encoding='utf-8') as f:
            json.dump(paper_json, f, ensure_ascii=False, indent=2)
        
        # Save summary statistics (without paper count)
        summary_stats = {
            'total_faculty': len(faculty_data),
            'total_papers': len(paper_data),
            'departments': faculty_data['Department'].unique().tolist(),
            'research_interests': faculty_data['Research Interest'].unique().tolist(),
            'last_updated': pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(metadata_dir, "summary_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully saved processed data to {metadata_dir}")
        return True
    except Exception as e:
        print(f"Error saving processed data: {e}")
        return False

def load_processed_data(vector_store_dir: str):
    """
    Load processed data from JSON files instead of Excel files.
    """
    try:
        metadata_dir = os.path.join(vector_store_dir, "metadata")
        
        # Load faculty data
        with open(os.path.join(metadata_dir, "faculty_data.json"), 'r', encoding='utf-8') as f:
            faculty_data = json.load(f)
        df_faculty = pd.DataFrame(faculty_data)
        
        # Load paper data
        with open(os.path.join(metadata_dir, "paper_data.json"), 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        df_papers = pd.DataFrame(paper_data)
        
        # Load summary stats
        with open(os.path.join(metadata_dir, "summary_stats.json"), 'r', encoding='utf-8') as f:
            summary_stats = json.load(f)
        
        print(f"Successfully loaded processed data from {metadata_dir}")
        print(f"Faculty: {len(df_faculty)}, Papers: {len(df_papers)}")
        return df_faculty, df_papers, summary_stats
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None, None, None

def create_comprehensive_documents(df_faculty: pd.DataFrame, df_papers: pd.DataFrame):
    """
    Create comprehensive documents for vector store including all data.
    Each research paper gets its own dedicated document.
    """
    documents = []
    
    # Create paper lookup dictionary for faster access
    paper_lookup = {}
    if not df_papers.empty:
        for _, paper_row in df_papers.iterrows():
            faculty_name_norm = normalize_faculty_name(paper_row['faculty_name'])
            if faculty_name_norm not in paper_lookup:
                paper_lookup[faculty_name_norm] = []
            paper_lookup[faculty_name_norm].append({
                'title': paper_row['paper_title'],
                'abstract': paper_row['paper_abstract']
            })
    
    # Process each faculty member
    for _, faculty_row in df_faculty.iterrows():
        faculty_name = faculty_row['Name']
        faculty_name_norm = normalize_faculty_name(faculty_name)
        faculty_papers = paper_lookup.get(faculty_name_norm, [])
        
        # === Document 1: Complete Faculty Profile ===
        profile_content = f"""Faculty Profile: {faculty_name}
Position: {faculty_row.get('Position', 'Not specified')}
Department: {faculty_row.get('Department', 'Not specified')}
Email: {faculty_row.get('Email', 'Not available')}
Research Interest: {faculty_row.get('Research Interest', 'Not specified')}"""
        
        # Add additional fields if available
        if 'Link to Detail' in faculty_row and pd.notna(faculty_row['Link to Detail']):
            profile_content += f"\nProfile Link: {faculty_row['Link to Detail']}"
            
        if 'Faculty Education' in faculty_row and pd.notna(faculty_row['Faculty Education']):
            profile_content += f"\nEducation: {faculty_row['Faculty Education']}"
            
        if 'Bio' in faculty_row and pd.notna(faculty_row['Bio']):
            profile_content += f"\nBio: {faculty_row['Bio']}"
            
        if 'Research' in faculty_row and pd.notna(faculty_row['Research']):
            profile_content += f"\nResearch Details: {faculty_row['Research']}"
        
        # Add paper titles overview (without count)
        if faculty_papers:
            paper_titles = [paper['title'] for paper in faculty_papers if pd.notna(paper['title'])]
            if paper_titles:
                profile_content += f"\nPublications: {' | '.join(paper_titles[:5])}"  # Show first 5 titles
        
        profile_doc = Document(
            page_content=profile_content,
            metadata={
                'faculty_name': faculty_name,
                'department': faculty_row.get('Department', 'Not specified'),
                'position': faculty_row.get('Position', 'Not specified'),
                'research_interest': faculty_row.get('Research Interest', 'Not specified'),
                'email': faculty_row.get('Email', 'Not available'),
                'document_type': 'faculty_profile',
                'education': faculty_row.get('Faculty Education', 'Not available'),
                'bio': faculty_row.get('Bio', 'Not available'),
                'profile_link': faculty_row.get('Link to Detail', 'Not available')
            }
        )
        documents.append(profile_doc)
        
        # === Document 2: Research Focus Document ===
        research_content = f"""Research Profile: {faculty_name}
Department: {faculty_row.get('Department', 'Not specified')}
Primary Research Interest: {faculty_row.get('Research Interest', 'Not specified')}
Academic Position: {faculty_row.get('Position', 'Not specified')}"""
        
        if 'Research' in faculty_row and pd.notna(faculty_row['Research']):
            research_content += f"\nDetailed Research Information: {faculty_row['Research']}"
            
        if faculty_papers:
            # Add research themes from paper titles (without count)
            paper_titles = [paper['title'] for paper in faculty_papers if pd.notna(paper['title'])]
            if paper_titles:
                research_content += f"\nResearch Areas from Publications: {' | '.join(paper_titles)}"
        
        research_doc = Document(
            page_content=research_content,
            metadata={
                'faculty_name': faculty_name,
                'department': faculty_row.get('Department', 'Not specified'),
                'research_interest': faculty_row.get('Research Interest', 'Not specified'),
                'position': faculty_row.get('Position', 'Not specified'),
                'document_type': 'research_profile'
            }
        )
        documents.append(research_doc)
        
        # === Document 3: Department Mapping Document ===
        dept_content = f"""Department Information
Department: {faculty_row.get('Department', 'Not specified')}
Faculty Member: {faculty_name}
Position: {faculty_row.get('Position', 'Not specified')}
Research Area: {faculty_row.get('Research Interest', 'Not specified')}
Contact: {faculty_row.get('Email', 'Not available')}"""
        
        dept_doc = Document(
            page_content=dept_content,
            metadata={
                'faculty_name': faculty_name,
                'department': faculty_row.get('Department', 'Not specified'),
                'research_interest': faculty_row.get('Research Interest', 'Not specified'),
                'position': faculty_row.get('Position', 'Not specified'),
                'document_type': 'department_mapping'
            }
        )
        documents.append(dept_doc)
        
        # === Dedicated Documents for Each Research Paper ===
        for i, paper in enumerate(faculty_papers, 1):
            if pd.notna(paper['title']) and str(paper['title']).strip():
                paper_title = str(paper['title']).strip()
                paper_abstract = str(paper['abstract']).strip() if pd.notna(paper['abstract']) else 'Abstract not available'
                
                # Create a dedicated document for each paper
                paper_content = f"""Research Paper: {paper_title}
Author: {faculty_name}
Author Department: {faculty_row.get('Department', 'Not specified')}
Author Position: {faculty_row.get('Position', 'Not specified')}
Author Research Interest: {faculty_row.get('Research Interest', 'Not specified')}
Author Email: {faculty_row.get('Email', 'Not available')}

Title: {paper_title}

Abstract: {paper_abstract}"""
                
                paper_doc = Document(
                    page_content=paper_content,
                    metadata={
                        'faculty_name': faculty_name,
                        'department': faculty_row.get('Department', 'Not specified'),
                        'research_interest': faculty_row.get('Research Interest', 'Not specified'),
                        'position': faculty_row.get('Position', 'Not specified'),
                        'paper_title': paper_title,
                        'paper_abstract': paper_abstract,
                        'document_type': 'research_paper',
                        'paper_index': i,
                        'author_email': faculty_row.get('Email', 'Not available')
                    }
                )
                documents.append(paper_doc)
    
    print(f"Created {len(documents)} documents for vector store")
    return documents

def load_and_create_vector_store(faculty_data_path: str, faculty_paper_path: str, google_api_key: str):
    """
    Enhanced function to load data, create vector store, and make it independent of Excel files.
    """
    vector_store_dir = "./vector_store_db"
    
    # Check if vector store already exists with processed data
    if os.path.exists(vector_store_dir) and os.listdir(vector_store_dir):
        try:
            print("Loading existing vector store from disk...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            vector_store = Chroma(persist_directory=vector_store_dir, embedding_function=embeddings)
            
            # Try to load processed data
            df_faculty, df_papers, summary_stats = load_processed_data(vector_store_dir)
            
            if df_faculty is not None and df_papers is not None:
                print(f"Successfully loaded existing vector store with {len(df_faculty)} faculty and {len(df_papers)} papers")
                return vector_store, None, df_faculty, df_papers, summary_stats
            else:
                print("Could not load processed data, but vector store exists. Continuing with vector store only.")
                return vector_store, None, None, None, None
                
        except Exception as e:
            print(f"Error loading existing vector store: {e}. Creating new one...")
            try:
                shutil.rmtree(vector_store_dir)
                print("Removed corrupted vector store directory")
            except Exception as cleanup_error:
                print(f"Could not clean up vector store directory: {cleanup_error}")
    
    # Load and process Excel files (only needed for first time setup)
    print("Creating new vector store from Excel files...")
    
    # Load faculty data
    try:
        df_faculty = pd.read_excel(faculty_data_path)
        print(f"Loaded faculty data: {len(df_faculty)} records")
        
        # Ensure required columns exist and clean data
        required_columns = ['Name']
        missing_columns = [col for col in required_columns if col not in df_faculty.columns]
        if missing_columns:
            return None, f"Error: Missing required columns in faculty data: {missing_columns}", None, None, None
            
        # Clean and standardize faculty data
        df_faculty['Name'] = df_faculty['Name'].fillna('Unknown').astype(str).str.strip()
        df_faculty['Department'] = df_faculty['Department'].fillna('Unknown Department').astype(str).str.strip()
        df_faculty['Position'] = df_faculty['Position'].fillna('Unknown Position').astype(str).str.strip()
        df_faculty['Research Interest'] = df_faculty['Research Interest'].fillna('Not specified').astype(str).str.strip()
        df_faculty['Email'] = df_faculty['Email'].fillna('Not available').astype(str).str.strip()
        
        # Handle optional columns
        optional_columns = ['Link to Detail', 'Faculty Education', 'Bio', 'Research']
        for col in optional_columns:
            if col not in df_faculty.columns:
                df_faculty[col] = 'Not available'
            else:
                df_faculty[col] = df_faculty[col].fillna('Not available').astype(str).str.strip()
            
    except FileNotFoundError:
        return None, "Error: The faculty data file was not found.", None, None, None
    except Exception as e:
        return None, f"Error loading faculty data: {e}", None, None, None

    # Load paper data
    try:
        df_papers = pd.read_excel(faculty_paper_path)
        print(f"Loaded paper data: {len(df_papers)} records")
        
        # Clean paper data
        df_papers['faculty_name'] = df_papers['faculty_name'].fillna('Unknown').astype(str).str.strip()
        df_papers['paper_title'] = df_papers['paper_title'].fillna('Untitled').astype(str).str.strip()
        df_papers['paper_abstract'] = df_papers['paper_abstract'].fillna('Abstract not available').astype(str).str.strip()
        
    except FileNotFoundError:
        print("Warning: The paper data file was not found. Proceeding with faculty data only.")
        df_papers = pd.DataFrame(columns=['faculty_name', 'paper_title', 'paper_abstract'])
    except Exception as e:
        print(f"Warning: Could not read paper data file. Error: {e}. Proceeding with faculty data only.")
        df_papers = pd.DataFrame(columns=['faculty_name', 'paper_title', 'paper_abstract'])

    # Create comprehensive documents
    documents = create_comprehensive_documents(df_faculty, df_papers)
    
    # Create vector store with all documents
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        vector_store = Chroma.from_documents(
            documents, 
            embeddings, 
            persist_directory=vector_store_dir
        )
        
        # Persist the vector store
        vector_store.persist()
        print(f"Vector store created and saved to {vector_store_dir}")
        
        # Save processed data for future independence from Excel files
        save_processed_data(df_faculty, df_papers, vector_store_dir)
        
        # Create summary statistics (without paper count display)
        summary_stats = {
            'total_faculty': len(df_faculty),
            'total_papers': len(df_papers),
            'departments': df_faculty['Department'].unique().tolist(),
            'research_interests': df_faculty['Research Interest'].unique().tolist(),
            'last_updated': pd.Timestamp.now().isoformat()
        }
        
        return vector_store, None, df_faculty, df_papers, summary_stats
        
    except Exception as e:
        return None, f"Error creating vector store: {e}", None, None, None

def create_agent_chain(vector_store, google_api_key: str, df_faculty=None, df_papers=None, summary_stats=None):
    """
    Creates an enhanced retrieval chain that works independently of Excel files.
    """
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=google_api_key, 
        temperature=0.1,
        max_output_tokens=3072  # Increased for comprehensive responses
    )

    # Create comprehensive summary statistics (without paper count)
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
        # Fallback to dataframe statistics
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
        11. When asked about a faculty member's research, publications, or related topics, go through ALL retrieved research paper documents. List each paper comprehensively in a numbered list with:
           - Full Title
           - Complete Abstract (without truncation)
           - Author Department
           - Author Position
           - Author Research Interest
           - Contact Email
           Organize the list by faculty name if multiple faculties are involved. Do not summarize or omit any details from the papers.
        12. Be thorough: Include every relevant detail from the context without summarization, truncation, or saying 'and more'. Ensure all paper titles and abstracts are fully covered.
        13. For every user input about research, papers, or publications, meticulously go through all metadata and content of research_paper documents, connecting them to faculty names using 'faculty_name' in metadata.
        14. Take time to think and analyze all available data, including all metadata fields like 'paper_title', 'paper_abstract', 'faculty_name'.

        Think step by step:
        1. Analyze the user query carefully.
        2. Review all retrieved context, including the metadata for each document (which connects paper titles and abstracts to faculty names).
        3. If the query relates to research, publications, or similar topics, identify ALL research_paper documents and their metadata. Go through each one, extracting paper titles, abstracts, and connections to faculty names.
        4. Go through each document's metadata to ensure all papers are linked to the correct faculty, ignoring case differences.
        5. Compile a comprehensive list using the metadata and content, ensuring all details are included without omission.
        6. Structure the response based on the query, using the database overview if needed.

        DATABASE OVERVIEW:
        {summary_stats}

        RETRIEVED CONTEXT (including content and metadata):
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
            "k": 100,  # Increased for better coverage of all papers
            "fetch_k": 200,
            "lambda_mult": 0.3
        }
    )
    
    class EnhancedIndependentChain:
        def __init__(self, retriever, llm, prompt, summary_text):
            self.retriever = retriever
            self.llm = llm
            self.prompt = prompt
            self.summary_text = summary_text
        
        def invoke(self, query_dict):
            try:
                docs = self.retriever.get_relevant_documents(query_dict["input"])
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
    
    return EnhancedIndependentChain(retriever, llm, prompt, summary_text)

def get_vector_store_info(vector_store_dir: str = "./vector_store_db"):
    """
    Get information about the existing vector store without needing Excel files.
    """
    try:
        if not os.path.exists(vector_store_dir):
            return None, "Vector store does not exist"
        
        # Try to load summary stats
        metadata_dir = os.path.join(vector_store_dir, "metadata")
        if os.path.exists(os.path.join(metadata_dir, "summary_stats.json")):
            with open(os.path.join(metadata_dir, "summary_stats.json"), 'r') as f:
                summary_stats = json.load(f)
            return summary_stats, None
        else:
            return None, "Summary statistics not found"
            
    except Exception as e:
        return None, f"Error reading vector store info: {e}"

# Testing and debugging function
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