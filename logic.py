import pandas as pd
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import DataFrameLoader
from langchain_chroma import Chroma  # Updated import to fix deprecation warning
from langchain.schema import Document
import re
import os
import shutil
import json
import pickle
from typing import List, Dict, Any
import sqlite3

def check_sqlite_version():
    """Check and fix SQLite version compatibility for Streamlit deployment."""
    try:
        # Check current SQLite version
        version = sqlite3.sqlite_version
        print(f"Current SQLite version: {version}")
        
        # Parse version numbers
        version_parts = list(map(int, version.split('.')))
        required_parts = [3, 35, 0]
        
        # Check if version is sufficient
        for i, (current, required) in enumerate(zip(version_parts, required_parts)):
            if current > required:
                break
            elif current < required:
                print(f"SQLite version {version} is insufficient. Required: 3.35.0+")
                return False
        
        print(f"SQLite version {version} is compatible")
        return True
        
    except Exception as e:
        print(f"Error checking SQLite version: {e}")
        return False

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
        
        # Save summary statistics
        summary_stats = {
            'total_faculty': len(faculty_data),
            'total_papers': len(paper_data),
            'departments': faculty_data['Department'].unique().tolist() if 'Department' in faculty_data.columns else [],
            'research_interests': faculty_data['Research Interest'].unique().tolist() if 'Research Interest' in faculty_data.columns else [],
            'last_updated': pd.Timestamp.now().isoformat(),
            'faculty_paper_mapping': create_faculty_paper_mapping(faculty_data, paper_data)
        }
        
        with open(os.path.join(metadata_dir, "summary_stats.json"), 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)
            
        print(f"Successfully saved processed data to {metadata_dir}")
        return True
    except Exception as e:
        print(f"Error saving processed data: {e}")
        return False

def create_faculty_paper_mapping(faculty_data: pd.DataFrame, paper_data: pd.DataFrame) -> Dict:
    """Create a mapping of faculty to their papers for better tracking."""
    mapping = {}
    
    for _, faculty_row in faculty_data.iterrows():
        faculty_name = faculty_row['Name']
        faculty_name_norm = normalize_faculty_name(faculty_name)
        
        # Find papers for this faculty member
        faculty_papers = []
        for _, paper_row in paper_data.iterrows():
            paper_faculty_norm = normalize_faculty_name(paper_row['faculty_name'])
            if paper_faculty_norm == faculty_name_norm:
                faculty_papers.append({
                    'title': str(paper_row['paper_title']),
                    'abstract': str(paper_row['paper_abstract']),
                    'normalized_faculty_name': paper_faculty_norm
                })
        
        mapping[faculty_name] = {
            'normalized_name': faculty_name_norm,
            'papers': faculty_papers,
            'paper_count': len(faculty_papers),
            'department': faculty_row.get('Department', 'Unknown'),
            'research_interest': faculty_row.get('Research Interest', 'Not specified')
        }
    
    return mapping

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
        
        # Verify faculty-paper connections
        if 'faculty_paper_mapping' in summary_stats:
            mapping = summary_stats['faculty_paper_mapping']
            total_mapped_papers = sum(len(info['papers']) for info in mapping.values())
            print(f"Faculty-Paper mapping verified: {total_mapped_papers} papers mapped to faculty")
        
        return df_faculty, df_papers, summary_stats
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None, None, None

def create_comprehensive_documents(df_faculty: pd.DataFrame, df_papers: pd.DataFrame):
    """
    Create comprehensive documents for vector store with enhanced paper-faculty connection tracking.
    Each research paper gets its own dedicated document with complete faculty information.
    """
    documents = []
    
    # Create detailed paper lookup dictionary with better name matching
    paper_lookup = {}
    faculty_name_variants = {}  # Track all name variants for each faculty
    
    # First, create a mapping of all faculty name variants
    for _, faculty_row in df_faculty.iterrows():
        original_name = faculty_row['Name']
        normalized_name = normalize_faculty_name(original_name)
        
        if normalized_name not in faculty_name_variants:
            faculty_name_variants[normalized_name] = {
                'original_name': original_name,
                'faculty_data': faculty_row
            }
    
    # Process papers and match them to faculty with detailed logging
    if not df_papers.empty:
        print(f"Processing {len(df_papers)} papers for faculty matching...")
        matched_papers = 0
        unmatched_papers = 0
        
        for idx, paper_row in df_papers.iterrows():
            paper_faculty_name = str(paper_row['faculty_name']).strip()
            paper_faculty_norm = normalize_faculty_name(paper_faculty_name)
            
            # Try to find matching faculty
            matched_faculty = None
            if paper_faculty_norm in faculty_name_variants:
                matched_faculty = faculty_name_variants[paper_faculty_norm]
                matched_papers += 1
            else:
                # Try fuzzy matching for potential name variations
                for norm_name, faculty_info in faculty_name_variants.items():
                    if paper_faculty_norm.lower() in norm_name.lower() or norm_name.lower() in paper_faculty_norm.lower():
                        matched_faculty = faculty_info
                        matched_papers += 1
                        print(f"Fuzzy matched: '{paper_faculty_name}' -> '{faculty_info['original_name']}'")
                        break
                
                if not matched_faculty:
                    unmatched_papers += 1
                    print(f"Unmatched paper faculty: '{paper_faculty_name}' (normalized: '{paper_faculty_norm}')")
            
            # Store paper information
            if matched_faculty:
                faculty_key = matched_faculty['original_name']
                if faculty_key not in paper_lookup:
                    paper_lookup[faculty_key] = []
                
                paper_lookup[faculty_key].append({
                    'title': str(paper_row['paper_title']).strip() if pd.notna(paper_row['paper_title']) else 'Untitled',
                    'abstract': str(paper_row['paper_abstract']).strip() if pd.notna(paper_row['paper_abstract']) else 'Abstract not available',
                    'original_faculty_name': paper_faculty_name,
                    'normalized_faculty_name': paper_faculty_norm
                })
        
        print(f"Paper matching results: {matched_papers} matched, {unmatched_papers} unmatched")
    
    # Process each faculty member and create comprehensive documents
    for _, faculty_row in df_faculty.iterrows():
        faculty_name = faculty_row['Name']
        faculty_papers = paper_lookup.get(faculty_name, [])
        
        print(f"Processing faculty '{faculty_name}': {len(faculty_papers)} papers found")
        
        # === Document 1: Complete Faculty Profile ===
        profile_content = f"""Faculty Profile: {faculty_name}
Position: {faculty_row.get('Position', 'Not specified')}
Department: {faculty_row.get('Department', 'Not specified')}
Email: {faculty_row.get('Email', 'Not available')}
Research Interest: {faculty_row.get('Research Interest', 'Not specified')}"""
        
        # Add additional fields if available
        optional_fields = ['Link to Detail', 'Faculty Education', 'Bio', 'Research']
        for field in optional_fields:
            if field in faculty_row and pd.notna(faculty_row[field]) and str(faculty_row[field]).strip():
                profile_content += f"\n{field}: {faculty_row[field]}"
        
        # Add comprehensive paper information to profile
        if faculty_papers:
            profile_content += f"\n\nResearch Publications Summary:"
            profile_content += f"\nTotal Publications: {len(faculty_papers)}"
            
            # Add paper titles for searchability
            paper_titles = [paper['title'] for paper in faculty_papers if paper['title'] != 'Untitled']
            if paper_titles:
                profile_content += f"\nPublication Titles: {' | '.join(paper_titles)}"
        
        profile_doc = Document(
            page_content=profile_content,
            metadata={
                'faculty_name': faculty_name,
                'department': faculty_row.get('Department', 'Not specified'),
                'position': faculty_row.get('Position', 'Not specified'),
                'research_interest': faculty_row.get('Research Interest', 'Not specified'),
                'email': faculty_row.get('Email', 'Not available'),
                'document_type': 'faculty_profile',
                'paper_count': len(faculty_papers),
                'has_papers': len(faculty_papers) > 0,
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
        
        if 'Research' in faculty_row and pd.notna(faculty_row['Research']) and str(faculty_row['Research']).strip():
            research_content += f"\nDetailed Research Information: {faculty_row['Research']}"
        
        # Add research themes extracted from paper titles and abstracts
        if faculty_papers:
            research_content += f"\n\nResearch Areas from Publications:"
            for i, paper in enumerate(faculty_papers[:5], 1):  # Show first 5 for research profile
                research_content += f"\n{i}. {paper['title']}"
                if paper['abstract'] != 'Abstract not available':
                    # Add first 200 characters of abstract for research context
                    abstract_snippet = paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']
                    research_content += f"\n   Research Focus: {abstract_snippet}"
        
        research_doc = Document(
            page_content=research_content,
            metadata={
                'faculty_name': faculty_name,
                'department': faculty_row.get('Department', 'Not specified'),
                'research_interest': faculty_row.get('Research Interest', 'Not specified'),
                'position': faculty_row.get('Position', 'Not specified'),
                'document_type': 'research_profile',
                'paper_count': len(faculty_papers),
                'has_papers': len(faculty_papers) > 0
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
        
        if faculty_papers:
            dept_content += f"\nPublications: {len(faculty_papers)} research papers"
        
        dept_doc = Document(
            page_content=dept_content,
            metadata={
                'faculty_name': faculty_name,
                'department': faculty_row.get('Department', 'Not specified'),
                'research_interest': faculty_row.get('Research Interest', 'Not specified'),
                'position': faculty_row.get('Position', 'Not specified'),
                'document_type': 'department_mapping',
                'paper_count': len(faculty_papers),
                'has_papers': len(faculty_papers) > 0
            }
        )
        documents.append(dept_doc)
        
        # === Dedicated Documents for Each Research Paper ===
        for i, paper in enumerate(faculty_papers, 1):
            paper_title = paper['title']
            paper_abstract = paper['abstract']
            
            if paper_title and paper_title != 'Untitled':
                # Create a comprehensive document for each paper
                paper_content = f"""Research Paper Details:

Title: {paper_title}

Author: {faculty_name}
Author Department: {faculty_row.get('Department', 'Not specified')}
Author Position: {faculty_row.get('Position', 'Not specified')}
Author Research Interest: {faculty_row.get('Research Interest', 'Not specified')}
Author Email: {faculty_row.get('Email', 'Not available')}

Complete Abstract:
{paper_abstract}

Faculty Research Background: {faculty_row.get('Research Interest', 'Not specified')}"""
                
                # Add faculty bio if available for context
                if 'Bio' in faculty_row and pd.notna(faculty_row['Bio']) and str(faculty_row['Bio']).strip():
                    paper_content += f"\n\nAuthor Bio: {faculty_row['Bio']}"
                
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
                        'author_email': faculty_row.get('Email', 'Not available'),
                        'original_faculty_name_in_paper': paper.get('original_faculty_name', ''),
                        'normalized_faculty_name': paper.get('normalized_faculty_name', '')
                    }
                )
                documents.append(paper_doc)
    
    print(f"Created {len(documents)} documents for vector store")
    
    # Print summary of document types
    doc_type_counts = {}
    for doc in documents:
        doc_type = doc.metadata.get('document_type', 'unknown')
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
    
    print("Document type distribution:")
    for doc_type, count in doc_type_counts.items():
        print(f"  {doc_type}: {count}")
    
    return documents

def load_and_create_vector_store(faculty_data_path: str, faculty_paper_path: str, google_api_key: str):
    """
    Enhanced function to load data, create vector store with SQLite compatibility fixes.
    """
    vector_store_dir = "./vector_store_db"
    
    # Check SQLite compatibility
    if not check_sqlite_version():
        print("Warning: SQLite version might be incompatible. Attempting to continue...")
    
    # Check if vector store already exists with processed data
    if os.path.exists(vector_store_dir) and os.listdir(vector_store_dir):
        try:
            print("Loading existing vector store from disk...")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            
            # Try to load vector store with error handling for SQLite issues
            try:
                vector_store = Chroma(persist_directory=vector_store_dir, embedding_function=embeddings)
                print("Successfully loaded existing vector store")
            except Exception as vs_error:
                print(f"Error loading vector store (possibly SQLite version issue): {vs_error}")
                print("Attempting to recreate vector store...")
                shutil.rmtree(vector_store_dir)
                raise Exception("Recreating vector store due to compatibility issues")
            
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
                if os.path.exists(vector_store_dir):
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
        print(f"Faculty data columns: {list(df_faculty.columns)}")
        
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
        print(f"Paper data columns: {list(df_papers.columns)}")
        
        # Clean paper data
        df_papers['faculty_name'] = df_papers['faculty_name'].fillna('Unknown').astype(str).str.strip()
        df_papers['paper_title'] = df_papers['paper_title'].fillna('Untitled').astype(str).str.strip()
        df_papers['paper_abstract'] = df_papers['paper_abstract'].fillna('Abstract not available').astype(str).str.strip()
        
        # Log some sample data for verification
        print("\nSample paper data:")
        for i, row in df_papers.head(3).iterrows():
            print(f"  Faculty: {row['faculty_name']}")
            print(f"  Title: {row['paper_title'][:100]}...")
            print(f"  Abstract: {row['paper_abstract'][:150]}...")
            print()
        
    except FileNotFoundError:
        print("Warning: The paper data file was not found. Proceeding with faculty data only.")
        df_papers = pd.DataFrame(columns=['faculty_name', 'paper_title', 'paper_abstract'])
    except Exception as e:
        print(f"Warning: Could not read paper data file. Error: {e}. Proceeding with faculty data only.")
        df_papers = pd.DataFrame(columns=['faculty_name', 'paper_title', 'paper_abstract'])

    # Create comprehensive documents
    documents = create_comprehensive_documents(df_faculty, df_papers)
    
    # Create vector store with all documents and SQLite compatibility
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        
        # Create vector store with error handling for SQLite issues
        try:
            vector_store = Chroma.from_documents(
                documents, 
                embeddings, 
                persist_directory=vector_store_dir
            )
            print(f"Vector store created successfully")
        except Exception as vs_error:
            error_msg = str(vs_error).lower()
            if 'sqlite' in error_msg or 'unsupported version' in error_msg:
                return None, f"SQLite version error: {vs_error}. Please upgrade SQLite to version 3.35.0 or higher. For Streamlit Cloud, this may require updating the deployment environment.", None, None, None
            else:
                return None, f"Error creating vector store: {vs_error}", None, None, None
        
        # Persist the vector store
        try:
            vector_store.persist()
            print(f"Vector store persisted to {vector_store_dir}")
        except Exception as persist_error:
            print(f"Warning: Could not persist vector store: {persist_error}")
        
        # Save processed data for future independence from Excel files
        save_processed_data(df_faculty, df_papers, vector_store_dir)
        
        # Create comprehensive summary statistics
        summary_stats = {
            'total_faculty': len(df_faculty),
            'total_papers': len(df_papers),
            'departments': df_faculty['Department'].unique().tolist(),
            'research_interests': df_faculty['Research Interest'].unique().tolist(),
            'last_updated': pd.Timestamp.now().isoformat(),
            'faculty_paper_mapping': create_faculty_paper_mapping(df_faculty, df_papers)
        }
        
        return vector_store, None, df_faculty, df_papers, summary_stats
        
    except Exception as e:
        error_msg = str(e).lower()
        if 'sqlite' in error_msg or 'unsupported version' in error_msg:
            return None, f"SQLite compatibility error: {e}. This may be due to an outdated SQLite version in the deployment environment.", None, None, None
        else:
            return None, f"Error creating vector store: {e}", None, None, None

def create_agent_chain(vector_store, google_api_key: str, df_faculty=None, df_papers=None, summary_stats=None):
    """
    Creates an enhanced retrieval chain with improved paper abstract and title retrieval.
    """
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=google_api_key, 
        temperature=0.1,
        max_output_tokens=4096  # Increased for comprehensive responses with full abstracts
    )

    # Create comprehensive summary statistics
    summary_text = ""
    if summary_stats:
        faculty_paper_info = ""
        if 'faculty_paper_mapping' in summary_stats:
            mapping = summary_stats['faculty_paper_mapping']
            faculty_with_papers = sum(1 for info in mapping.values() if info['paper_count'] > 0)
            faculty_paper_info = f"""
        - Faculty with Publications: {faculty_with_papers}
        - Faculty without Publications: {summary_stats['total_faculty'] - faculty_with_papers}"""
        
        summary_text = f"""
        COMPREHENSIVE FACULTY DATABASE STATISTICS:
        - Total Faculty Members: {summary_stats['total_faculty']}
        - Total Research Papers: {summary_stats['total_papers']}
        - Active Departments: {len(summary_stats['departments'])}
        - Departments: {', '.join(summary_stats['departments'])}
        - Research Areas: {len(summary_stats['research_interests'])}
        - Last Updated: {summary_stats['last_updated'][:10]}{faculty_paper_info}
        """
    elif df_faculty is not None:
        # Fallback to dataframe statistics
        dept_counts = df_faculty['Department'].value_counts().to_dict()
        total_papers = len(df_papers) if df_papers is not None else 0
        
        summary_text = f"""
        COMPREHENSIVE FACULTY DATABASE STATISTICS:
        - Total Faculty Members: {len(df_faculty)}
        - Total Research Papers: {total_papers}
        - Active Departments: {len(df_faculty['Department'].unique())}
        - Faculty Distribution by Department: {dept_counts}
        - Research Interests: {len(df_faculty['Research Interest'].unique())} unique areas
        """

    prompt = PromptTemplate(
        template="""
        You are a comprehensive faculty information assistant with complete access to faculty profiles, research information, and publication data. 

        CRITICAL INSTRUCTIONS FOR RESEARCH PAPER QUERIES:
        1. When asked about research papers, publications, or any research-related topics, you MUST:
           - Go through ALL retrieved research paper documents thoroughly
           - Provide COMPLETE information for every relevant paper found
           - Include FULL paper titles (never truncate)
           - Include COMPLETE abstracts (never summarize or truncate)
           - Provide comprehensive author details for each paper

        2. For paper listings, use this EXACT format:
           **Paper [Number]: [Complete Paper Title]**
           - **Author:** [Faculty Name]
           - **Department:** [Department]
           - **Position:** [Academic Position]  
           - **Research Interest:** [Research Area]
           - **Email:** [Contact Email]
           - **Complete Abstract:** [Full abstract without any truncation or summarization]

        3. NEVER truncate, summarize, or omit any part of paper titles or abstracts
        4. NEVER say "and more" or indicate there are additional papers without listing them
        5. NEVER provide partial information - always include complete details
        6. When multiple papers are found, list ALL of them using the format above
        7. Organize papers by faculty name when multiple authors are involved

        GENERAL RESPONSE GUIDELINES:
        - Always provide COMPLETE and DETAILED answers
        - Use ALL relevant information from the context
        - For faculty queries, include: name, position, department, email, research interests
        - For statistical queries, use the database overview below
        - Always organize responses clearly with proper formatting
        - Include contact information when available and relevant
        - Be thorough and comprehensive in all responses

        DATABASE OVERVIEW:
        {summary_stats}

        RETRIEVED CONTEXT:
        {context}

        USER QUESTION: {input}

        COMPREHENSIVE ANSWER:
        """,
        input_variables=["context", "input", "summary_stats"],
    )

    # Enhanced retriever for comprehensive paper coverage
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 60,  # Increased for better coverage
            "fetch_k": 120,
            "lambda_mult": 0.5
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
                organized_context = self._organize_context_for_comprehensive_retrieval(docs, query_dict["input"])
                
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
        
        def _organize_context_for_comprehensive_retrieval(self, docs, query):
            """
            Organize retrieved documents with special focus on comprehensive paper retrieval.
            Ensures all paper abstracts and titles are properly included.
            """
            context_sections = {
                'faculty_profile': [],
                'research_profile': [],
                'research_paper': [],
                'department_mapping': []
            }
            
            # Track papers by faculty for comprehensive coverage
            papers_by_faculty = {}
            faculty_info = {}
            
            for doc in docs:
                doc_type = doc.metadata.get('document_type', 'faculty_profile')
                
                if doc_type in context_sections:
                    context_sections[doc_type].append(doc.page_content)
                
                # Special handling for research papers
                if doc_type == 'research_paper':
                    faculty_name = doc.metadata.get('faculty_name', 'Unknown')
                    if faculty_name not in papers_by_faculty:
                        papers_by_faculty[faculty_name] = []
                    
                    # Store complete paper information
                    paper_info = {
                        'title': doc.metadata.get('paper_title', 'Untitled'),
                        'abstract': doc.metadata.get('paper_abstract', 'Abstract not available'),
                        'content': doc.page_content,
                        'department': doc.metadata.get('department', 'Unknown'),
                        'position': doc.metadata.get('position', 'Unknown'),
                        'research_interest': doc.metadata.get('research_interest', 'Unknown'),
                        'email': doc.metadata.get('author_email', 'Not available')
                    }
                    papers_by_faculty[faculty_name].append(paper_info)
                
                # Collect faculty information
                if doc_type == 'faculty_profile':
                    faculty_name = doc.metadata.get('faculty_name', 'Unknown')
                    faculty_info[faculty_name] = {
                        'department': doc.metadata.get('department', 'Unknown'),
                        'position': doc.metadata.get('position', 'Unknown'),
                        'research_interest': doc.metadata.get('research_interest', 'Unknown'),
                        'email': doc.metadata.get('email', 'Not available'),
                        'paper_count': doc.metadata.get('paper_count', 0)
                    }
            
            organized_parts = []
            
            # Faculty profiles section
            if context_sections['faculty_profile']:
                organized_parts.append("FACULTY PROFILES:\n" + "\n---\n".join(context_sections['faculty_profile']))
            
            # Research profiles section
            if context_sections['research_profile']:
                organized_parts.append("RESEARCH INFORMATION:\n" + "\n---\n".join(context_sections['research_profile']))
            
            # Comprehensive research papers section - organized by faculty
            if papers_by_faculty:
                papers_section = "COMPLETE RESEARCH PAPERS DATABASE:\n"
                papers_section += f"Total Papers Found: {sum(len(papers) for papers in papers_by_faculty.values())}\n\n"
                
                for faculty_name, papers in papers_by_faculty.items():
                    papers_section += f"FACULTY: {faculty_name}\n"
                    if faculty_name in faculty_info:
                        info = faculty_info[faculty_name]
                        papers_section += f"Department: {info['department']}\n"
                        papers_section += f"Position: {info['position']}\n"
                        papers_section += f"Research Interest: {info['research_interest']}\n"
                        papers_section += f"Email: {info['email']}\n"
                    papers_section += f"Number of Papers: {len(papers)}\n\n"
                    
                    for i, paper in enumerate(papers, 1):
                        papers_section += f"PAPER {i}:\n"
                        papers_section += f"Title: {paper['title']}\n"
                        papers_section += f"Complete Abstract: {paper['abstract']}\n"
                        papers_section += f"Author Department: {paper['department']}\n"
                        papers_section += f"Author Position: {paper['position']}\n"
                        papers_section += f"Author Research Interest: {paper['research_interest']}\n"
                        papers_section += f"Author Email: {paper['email']}\n"
                        papers_section += "---\n"
                    
                    papers_section += "\n"
                
                organized_parts.append(papers_section)
            
            # Department information section
            if context_sections['department_mapping']:
                organized_parts.append("DEPARTMENT INFORMATION:\n" + "\n---\n".join(context_sections['department_mapping']))
            
            # Add metadata summary
            if papers_by_faculty:
                metadata_summary = f"\nRETRIEVAL METADATA:\n"
                metadata_summary += f"Total Documents Retrieved: {len(docs)}\n"
                metadata_summary += f"Research Papers Retrieved: {sum(len(papers) for papers in papers_by_faculty.values())}\n"
                metadata_summary += f"Faculty with Papers: {len(papers_by_faculty)}\n"
                metadata_summary += f"Document Types: {', '.join([t for t, docs in context_sections.items() if docs])}\n"
                organized_parts.append(metadata_summary)
            
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

def test_retrieval(vector_store, query: str, k: int = 50):
    """
    Test function to see what documents are being retrieved for a query.
    Enhanced to show paper-faculty connections.
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k*2, "lambda_mult": 0.5}
    )
    
    docs = retriever.get_relevant_documents(query)
    
    print(f"\nRetrieved {len(docs)} documents for query: '{query}'")
    
    # Organize by document type
    doc_types = {}
    papers_found = {}
    
    for i, doc in enumerate(docs, 1):
        doc_type = doc.metadata.get('document_type', 'unknown')
        if doc_type not in doc_types:
            doc_types[doc_type] = []
        doc_types[doc_type].append((i, doc))
        
        # Track papers
        if doc_type == 'research_paper':
            faculty = doc.metadata.get('faculty_name', 'Unknown')
            if faculty not in papers_found:
                papers_found[faculty] = []
            papers_found[faculty].append({
                'title': doc.metadata.get('paper_title', 'Untitled'),
                'abstract_length': len(doc.metadata.get('paper_abstract', ''))
            })
    
    print(f"\nDocument type distribution:")
    for doc_type, docs_list in doc_types.items():
        print(f"  {doc_type}: {len(docs_list)}")
    
    print(f"\nResearch papers found by faculty:")
    for faculty, papers in papers_found.items():
        print(f"  {faculty}: {len(papers)} papers")
        for paper in papers[:2]:  # Show first 2 papers per faculty
            print(f"    - {paper['title'][:80]}... (abstract: {paper['abstract_length']} chars)")
    
    # Show first few documents in detail
    print(f"\nFirst 5 documents in detail:")
    for i, doc in enumerate(docs[:5], 1):
        print(f"\nDocument {i}:")
        print(f"Type: {doc.metadata.get('document_type', 'unknown')}")
        print(f"Faculty: {doc.metadata.get('faculty_name', 'N/A')}")
        print(f"Department: {doc.metadata.get('department', 'N/A')}")
        if doc.metadata.get('document_type') == 'research_paper':
            print(f"Paper: {doc.metadata.get('paper_title', 'N/A')}")
            print(f"Abstract length: {len(doc.metadata.get('paper_abstract', ''))}")
        print(f"Content preview: {doc.page_content[:200]}...")
    
    return docs

def verify_faculty_paper_connections(df_faculty, df_papers):
    """
    Utility function to verify and debug faculty-paper connections.
    """
    print("VERIFYING FACULTY-PAPER CONNECTIONS:")
    print("=" * 50)
    
    # Check faculty names
    print(f"Total faculty: {len(df_faculty)}")
    print(f"Total papers: {len(df_papers)}")
    
    # Sample faculty names
    print("\nSample faculty names:")
    for name in df_faculty['Name'].head(5):
        print(f"  - '{name}' (normalized: '{normalize_faculty_name(name)}')")
    
    # Sample paper faculty names
    print("\nSample paper faculty names:")
    for name in df_papers['faculty_name'].head(5):
        print(f"  - '{name}' (normalized: '{normalize_faculty_name(name)}')")
    
    # Check for matches
    faculty_names_norm = set(normalize_faculty_name(name) for name in df_faculty['Name'])
    paper_names_norm = set(normalize_faculty_name(name) for name in df_papers['faculty_name'])
    
    matches = faculty_names_norm.intersection(paper_names_norm)
    faculty_only = faculty_names_norm - paper_names_norm
    papers_only = paper_names_norm - faculty_names_norm
    
    print(f"\nMatching names: {len(matches)}")
    print(f"Faculty without papers: {len(faculty_only)}")
    print(f"Papers without matching faculty: {len(papers_only)}")
    
    if papers_only:
        print("\nPapers without matching faculty (first 10):")
        for name in list(papers_only)[:10]:
            print(f"  - '{name}'")
    
    return {
        'total_faculty': len(df_faculty),
        'total_papers': len(df_papers),
        'matches': len(matches),
        'faculty_only': len(faculty_only),
        'papers_only': len(papers_only)
    }