
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
from typing import List, Dict, Any

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
    name = str(name).strip()
    name = re.sub(r'^(Prof\.?\s*|Dr\.?\s*|Mr\.?\s*|Ms\.?\s*|Mrs\.?\s*)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+', ' ', name)  # Replace multiple spaces with single space
    return name.strip()

def load_and_create_vector_store(faculty_data_path: str, faculty_paper_path: str, google_api_key: str):
    """
    Loads faculty and paper data, combines it with enhanced metadata, and creates a Chroma vector store.
    Uses persistent storage and creates multiple document types for better retrieval.
    """
    
    # Define vector store directory
    vector_store_dir = "./vector_store_db"
    
    # Check if vector store already exists
    if os.path.exists(vector_store_dir) and os.listdir(vector_store_dir):
        try:
            print("Loading existing vector store from disk...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            vector_store = Chroma(persist_directory=vector_store_dir, embedding_function=embeddings)
            
            # Also load the faculty data for statistics and process it properly
            df_data = pd.read_excel(faculty_data_path)
            
            # Apply the same processing as we would for new data
            df_data['Department'] = df_data['Department'].fillna('Unknown Department').astype(str).str.strip()
            df_data['Research Interest'] = df_data['Research Interest'].fillna('Not specified').astype(str).str.strip()
            df_data['Position'] = df_data['Position'].fillna('Unknown Position').astype(str).str.strip()
            
            # Create Academic_Level column if it doesn't exist
            if 'Academic_Level' not in df_data.columns:
                df_data['Academic_Level'] = df_data['Position'].apply(
                    lambda x: 'Professor' if 'Professor' in str(x) else 
                             'Associate Professor' if 'Associate' in str(x) else 
                             'Assistant Professor' if 'Assistant' in str(x) else 
                             'Lecturer' if 'Lecturer' in str(x) else 'Other'
                )
            
            # Add paper_count column with default value if it doesn't exist
            if 'paper_count' not in df_data.columns:
                df_data['paper_count'] = 0
                
            print(f"Loaded existing vector store and processed faculty data: {len(df_data)} records")
            return vector_store, None, df_data
        except Exception as e:
            print(f"Error loading existing vector store: {e}. Creating new one...")
            # Clean up the corrupted vector store directory
            import shutil
            try:
                shutil.rmtree(vector_store_dir)
                print("Removed corrupted vector store directory")
            except Exception as cleanup_error:
                print(f"Could not clean up vector store directory: {cleanup_error}")
    
    # Load faculty data
    try:
        df_data = pd.read_excel(faculty_data_path)
        print(f"Loaded faculty data: {len(df_data)} records")
        
        # Check if required columns exist
        required_columns = ['Name', 'Position', 'Department', 'Research Interest']
        missing_columns = [col for col in required_columns if col not in df_data.columns]
        if missing_columns:
            return None, f"Error: Missing required columns in faculty data: {missing_columns}", None
            
    except FileNotFoundError:
        return None, "Error: The faculty data file was not found.", None
    except Exception as e:
        return None, f"Error loading faculty data: {e}", None

    # Load paper data and handle potential file errors
    try:
        df_paper = pd.read_excel(faculty_paper_path)
        print(f"Loaded paper data: {len(df_paper)} records")
    except FileNotFoundError:
        print("Warning: The paper data file was not found. Proceeding with faculty data only.")
        df_paper = pd.DataFrame(columns=['faculty_name', 'paper_title', 'paper_abstract'])
    except Exception as e:
        print(f"Warning: Could not read paper data file. Error: {e}. Proceeding with faculty data only.")
        df_paper = pd.DataFrame(columns=['faculty_name', 'paper_title', 'paper_abstract'])

    # === ENHANCED METADATA PROCESSING ===
    
    # Clean and standardize data
    df_data['Department'] = df_data['Department'].fillna('Unknown Department').astype(str).str.strip()
    df_data['Research Interest'] = df_data['Research Interest'].fillna('Not specified').astype(str).str.strip()
    df_data['Position'] = df_data['Position'].fillna('Unknown Position').astype(str).str.strip()
    
    # Extract Academic Level from Position
    df_data['Academic_Level'] = df_data['Position'].apply(
        lambda x: 'Professor' if 'Professor' in str(x) else 
                 'Associate Professor' if 'Associate' in str(x) else 
                 'Assistant Professor' if 'Assistant' in str(x) else 
                 'Lecturer' if 'Lecturer' in str(x) else 'Other'
    )
    
    # Normalize faculty names for better matching
    df_data['Name_Normalized'] = df_data['Name'].apply(normalize_faculty_name)
    
    # === ENHANCED PAPER PROCESSING ===
    
    if not df_paper.empty:
        # Normalize faculty names in paper data
        df_paper['faculty_name_normalized'] = df_paper['faculty_name'].apply(normalize_faculty_name)
        
        # Create a mapping from normalized names to original names
        name_mapping = dict(zip(df_data['Name_Normalized'], df_data['Name']))
        
        # Group papers by normalized faculty name
        df_papers_grouped = df_paper.groupby('faculty_name_normalized').agg({
            'paper_title': lambda x: list(x),
            'paper_abstract': lambda x: list(x)
        }).reset_index()
        
        # Create comprehensive paper information with better formatting
        def format_papers_info(row):
            titles = row['paper_title']
            abstracts = row['paper_abstract']
            papers_text = []
            
            for i, (title, abstract) in enumerate(zip(titles, abstracts), 1):
                # Clean and format each paper
                title_clean = str(title).strip() if pd.notna(title) else f"Paper {i}"
                abstract_clean = str(abstract).strip() if pd.notna(abstract) and str(abstract).strip() != 'nan' else 'Abstract not available'
                
                paper_text = f"Publication {i}: '{title_clean}' - {abstract_clean}"
                papers_text.append(paper_text)
            
            return " | ".join(papers_text)
        
        df_papers_grouped['papers_info'] = df_papers_grouped.apply(format_papers_info, axis=1)
        df_papers_grouped['paper_count'] = df_papers_grouped['paper_title'].apply(len)
        
        # Map back to original names
        df_papers_grouped['Name'] = df_papers_grouped['faculty_name_normalized'].map(name_mapping)
        df_papers_grouped = df_papers_grouped[['Name', 'papers_info', 'paper_count']].dropna(subset=['Name'])
        
        print(f"Successfully grouped papers for {len(df_papers_grouped)} faculty members")
    else:
        df_papers_grouped = pd.DataFrame(columns=['Name', 'papers_info', 'paper_count'])

    # Merge faculty data with paper data
    df_combined = pd.merge(df_data, df_papers_grouped, on='Name', how='left')

    # Fill NaN values for faculties with no papers
    df_combined = df_combined.fillna({
        'papers_info': 'No publications available', 
        'paper_count': 0
    })

    # === CREATE MULTIPLE DOCUMENT TYPES FOR BETTER RETRIEVAL ===
    
    documents = []
    
    for _, row in df_combined.iterrows():
        # Document 1: Basic faculty profile
        basic_content = f"""
        Faculty: {row['Name']}
        Position: {row['Position']} 
        Department: {row['Department']}
        Academic Level: {row['Academic_Level']}
        Research Interest: {row['Research Interest']}
        Email: {row.get('Email', 'Not available')}
        Total Publications: {int(row['paper_count'])}
        """
        
        # Add bio if available
        bio_text = ""
        if 'Bio' in row and pd.notna(row['Bio']) and str(row['Bio']).strip():
            bio_text = f"Bio: {row['Bio']}"
        elif 'Bio Information' in row and pd.notna(row['Bio Information']) and str(row['Bio Information']).strip():
            bio_text = f"Bio: {row['Bio Information']}"
            
        if bio_text:
            basic_content += f"\n{bio_text}"
            
        # Add research details if available
        if 'Research' in row and pd.notna(row['Research']) and str(row['Research']).strip():
            basic_content += f"\nResearch Details: {row['Research']}"
        
        # Create basic profile document
        basic_doc = Document(
            page_content=basic_content.strip(),
            metadata={
                'faculty_name': row['Name'],
                'department': row['Department'],
                'research_interest': row['Research Interest'],
                'position': row['Position'],
                'academic_level': row['Academic_Level'],
                'paper_count': int(row['paper_count']),
                'document_type': 'profile',
                'email': row.get('Email', 'Not available')
            }
        )
        documents.append(basic_doc)
        
        # Document 2: Research-focused document
        research_content = f"""
        {row['Name']} - Research Profile
        Department: {row['Department']}
        Research Interest: {row['Research Interest']}
        Academic Position: {row['Position']}
        Research Areas and Expertise: {row['Research Interest']}
        """
        
        if 'Research' in row and pd.notna(row['Research']) and str(row['Research']).strip():
            research_content += f"\nDetailed Research Information: {row['Research']}"
            
        research_doc = Document(
            page_content=research_content,
            metadata={
                'faculty_name': row['Name'],
                'department': row['Department'],
                'research_interest': row['Research Interest'],
                'position': row['Position'],
                'academic_level': row['Academic_Level'],
                'document_type': 'research',
                'paper_count': int(row['paper_count'])
            }
        )
        documents.append(research_doc)
        
        # Document 3: Publications document (if papers exist)
        if row['paper_count'] > 0 and row['papers_info'] != 'No publications available':
            pub_content = f"""
            Publications by {row['Name']} ({row['Department']})
            Research Area: {row['Research Interest']}
            Total Publications: {int(row['paper_count'])}
            
            Publications Details:
            {row['papers_info']}
            """
            
            pub_doc = Document(
                page_content=pub_content,
                metadata={
                    'faculty_name': row['Name'],
                    'department': row['Department'],
                    'research_interest': row['Research Interest'],
                    'position': row['Position'],
                    'document_type': 'publications',
                    'paper_count': int(row['paper_count'])
                }
            )
            documents.append(pub_doc)
        
        # Document 4: Department-research mapping document
        dept_research_content = f"""
        Department: {row['Department']}
        Faculty Member: {row['Name']}
        Position: {row['Position']}
        Research Focus: {row['Research Interest']}
        Publications: {int(row['paper_count'])} papers
        """
        
        dept_doc = Document(
            page_content=dept_research_content,
            metadata={
                'faculty_name': row['Name'],
                'department': row['Department'],
                'research_interest': row['Research Interest'],
                'position': row['Position'],
                'document_type': 'department_mapping',
                'paper_count': int(row['paper_count'])
            }
        )
        documents.append(dept_doc)

    print(f"Created {len(documents)} documents for vector store")
    
    # Create embeddings and vector store with persistence
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
        
        return vector_store, None, df_combined
    except Exception as e:
        return None, f"Error creating vector store: {e}", None

def create_agent_chain(vector_store, google_api_key: str, df_summary=None):
    """
    Creates an enhanced retrieval chain with improved prompt and higher k values for complete answers.
    """
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=google_api_key, 
        temperature=0.1,  # Lower temperature for more consistent responses
        max_output_tokens=2048  # Increased token limit for complete answers
    )

    # Create comprehensive summary statistics
    summary_stats = ""
    if df_summary is not None:
        total_faculty = len(df_summary)
        dept_counts = df_summary['Department'].value_counts().to_dict()
        level_counts = df_summary['Academic_Level'].value_counts().to_dict()
        total_papers = df_summary['paper_count'].sum()
        
        # Get research interest distribution
        research_interests = df_summary['Research Interest'].value_counts().head(10).to_dict()
        
        # Get faculty by department
        dept_faculty = {}
        for dept in df_summary['Department'].unique():
            dept_faculty[dept] = df_summary[df_summary['Department'] == dept]['Name'].tolist()
        
        summary_stats = f"""
        COMPREHENSIVE FACULTY DATABASE STATISTICS:
        - Total Faculty Members: {total_faculty}
        - Total Publications: {int(total_papers)}
        - Departments: {list(dept_counts.keys())}
        - Faculty Distribution by Department: {dept_counts}
        - Faculty by Academic Level: {level_counts}
        - Top Research Areas: {research_interests}
        - Faculty by Department: {dept_faculty}
        """

    prompt = PromptTemplate(
        template="""
        You are a comprehensive faculty information assistant. Your goal is to provide COMPLETE and DETAILED answers to all questions about faculty members, their research, departments, and publications.

        IMPORTANT INSTRUCTIONS:
        1. ALWAYS provide COMPLETE answers - never truncate or provide partial information
        2. Use ALL relevant information from the context and summary statistics
        3. For counting questions, use the summary statistics above for accuracy
        4. For specific faculty queries, use the detailed context below
        5. Always include faculty names with their departments and positions when mentioned
        6. Include contact information (email) when available and relevant
        7. For research queries, mention specific research interests and areas
        8. When listing papers, provide full titles and abstracts when available
        9. Organize your response clearly with proper formatting
        10. If asked about multiple faculty or departments, provide information for ALL of them

        DATABASE OVERVIEW:
        {summary_stats}

        CONTEXT INFORMATION:
        {context}

        USER QUESTION: {input}

        COMPREHENSIVE ANSWER (provide complete information, do not truncate):
        """,
        input_variables=["context", "input", "summary_stats"],
    )

    # Enhanced retriever with significantly higher k values for complete retrieval
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diverse results
        search_kwargs={
            "k": 20,  # Increased from 10 to 25 for more comprehensive results
            "fetch_k": 40,  # Increased from 25 to 50 for broader initial search
            "lambda_mult": 0.5  # Adjusted for better diversity vs relevance balance
        }
    )
    
    # Create enhanced chain with better document processing
    class EnhancedRetrievalChain:
        def __init__(self, retriever, llm, prompt, summary_stats):
            self.retriever = retriever
            self.llm = llm
            self.prompt = prompt
            self.summary_stats = summary_stats
        
        def invoke(self, query_dict):
            try:
                # Get retrieved documents with enhanced processing
                docs = self.retriever.get_relevant_documents(query_dict["input"])
                
                # Process and organize documents by type for better context
                organized_context = self._organize_context(docs)
                
                # Format the prompt with enhanced context
                formatted_prompt = self.prompt.format(
                    context=organized_context,
                    input=query_dict["input"],
                    summary_stats=self.summary_stats
                )
                
                # Get response from LLM with retry mechanism
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
            """Organize retrieved documents by type for better context structure."""
            context_sections = {
                'profile': [],
                'research': [],
                'publications': [],
                'department_mapping': []
            }
            
            for doc in docs:
                doc_type = doc.metadata.get('document_type', 'profile')
                if doc_type in context_sections:
                    context_sections[doc_type].append(doc.page_content)
            
            # Build organized context
            organized_parts = []
            
            if context_sections['profile']:
                organized_parts.append("FACULTY PROFILES:\n" + "\n---\n".join(context_sections['profile']))
            
            if context_sections['research']:
                organized_parts.append("RESEARCH INFORMATION:\n" + "\n---\n".join(context_sections['research']))
            
            if context_sections['publications']:
                organized_parts.append("PUBLICATIONS:\n" + "\n---\n".join(context_sections['publications']))
            
            if context_sections['department_mapping']:
                organized_parts.append("DEPARTMENT MAPPINGS:\n" + "\n---\n".join(context_sections['department_mapping']))
            
            return "\n\n".join(organized_parts)
    
    return EnhancedRetrievalChain(retriever, llm, prompt, summary_stats)

# Additional utility function for testing and debugging
def test_retrieval(vector_store, query: str, k: int = 25):
    """Test function to see what documents are being retrieved for a query."""
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k*2, "lambda_mult": 0.5}
    )
    
    docs = retriever.get_relevant_documents(query)
    
    print(f"\nRetrieved {len(docs)} documents for query: '{query}'")
    for i, doc in enumerate(docs, 1):
        print(f"\nDocument {i}:")
        print(f"Type: {doc.metadata.get('document_type', 'unknown')}")
        print(f"Faculty: {doc.metadata.get('faculty_name', 'N/A')}")
        print(f"Department: {doc.metadata.get('department', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")
    
    return docs