import streamlit as st
import os
import shutil
import time
from typing import List, Dict, Optional
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize API keys from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["groq_api_key"] = st.secrets["GROQ_API_KEY"]
groq_api_key = os.environ["groq_api_key"]

# Constants
DATA_FOLDER = "./data"
ALLOWED_FILE_TYPES = ["pdf"]
MAX_FILE_SIZE_MB = 50  # Maximum allowed file size in MB

# Streamlit app configuration
st.set_page_config(page_title="Research Paper Analysis", layout="wide")
st.title("Research Paper Analysis using Retrieval Augmented Generation")
st.markdown("#### Leverage the power of language models to analyze research papers.")

def clean_previous_data() -> None:
    """Clean the data folder when the app reloads."""
    if os.path.exists(DATA_FOLDER):
        shutil.rmtree(DATA_FOLDER)
    os.makedirs(DATA_FOLDER)

def validate_uploaded_file(uploaded_file) -> bool:
    """Validate the uploaded file for type and size."""
    if uploaded_file.name.split('.')[-1].lower() not in ALLOWED_FILE_TYPES:
        return False
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:  # Convert MB to bytes
        return False
    return True

def save_uploaded_file(uploaded_file) -> str:
    """Save the uploaded file to the data folder and return the file path."""
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

def initialize_vector_store_from_upload(uploaded_files) -> FAISS:
    """
    Initialize vector embeddings and FAISS vector store from uploaded files.
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        FAISS vector store containing document embeddings
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = []

    for uploaded_file in uploaded_files:
        if not validate_uploaded_file(uploaded_file):
            st.warning(f"Skipping invalid file: {uploaded_file.name}")
            continue
            
        try:
            file_path = save_uploaded_file(uploaded_file)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue

    if not documents:
        raise ValueError("No valid documents were processed.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_documents = text_splitter.split_documents(documents)
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

def create_retrieval_chain_with_context(llm: ChatGroq, vectors: FAISS):
    """Create a retrieval chain using the LLM and vector store."""
    retriever = vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, document_chain)

def display_response(response: Dict) -> None:
    """Display the response to the user."""
    if not response:
        st.error("No response generated.")
        return
        
    st.text_area("Answer", value=response.get('answer', "No answer generated."), height=300)
    
    with st.expander("Relevant Documents:"):
        if response.get('context'):
            for i, doc in enumerate(response['context'], 1):
                st.subheader(f"Document {i}")
                st.write(doc.page_content)
                st.write("---")
        else:
            st.write("No relevant documents found.")

# Initialize LLM and prompt template
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

prompt_template = ChatPromptTemplate.from_template(
    """
    You are a research assistant analyzing scientific papers. Provide detailed, 
    accurate answers based on the provided context. If you're unsure, say so.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

# Clean data folder on reload
clean_previous_data()

# Sidebar for navigation and settings
with st.sidebar:
    st.header("Settings")
    st.markdown("""
        - **Upload Documents**: Add research PDFs to the database
        - **Ask Questions**: Query the uploaded documents
    """)
    
    if st.button("Clear Session"):
        st.session_state.clear()
        clean_previous_data()
        st.rerun()

# Upload Section
st.subheader("Upload Research PDFs")
uploaded_files = st.file_uploader(
    "Choose research documents (PDF only)",
    type=ALLOWED_FILE_TYPES,
    accept_multiple_files=True,
    help="Upload research papers in PDF format"
)

if uploaded_files:
    # Display uploaded files
    st.write("**Uploaded Files:**")
    for file in uploaded_files:
        st.write(f"- {file.name} ({file.size//1024} KB)")
    
    if st.button("Process Documents", key="init_embedding"):
        with st.spinner("Processing documents... This may take a few minutes."):
            try:
                st.session_state.vectors = initialize_vector_store_from_upload(uploaded_files)
                st.success("Documents processed successfully!")
            except Exception as e:
                st.error(f"Failed to process documents: {str(e)}")

# Question Input Section
st.subheader("Ask a Question")
question = st.text_input(
    "Enter your research question:",
    placeholder="E.g., What were the main findings of this study?",
    disabled=not st.session_state.get('vectors')
)

if question and "vectors" in st.session_state:
    with st.spinner("Searching for answers..."):
        try:
            if "retrieval_chain" not in st.session_state:
                st.session_state.retrieval_chain = create_retrieval_chain_with_context(
                    llm, 
                    st.session_state.vectors
                )
            
            start_time = time.time()
            response = st.session_state.retrieval_chain.invoke({'input': question})
            processing_time = time.time() - start_time
            
            st.success(f"Response generated in {processing_time:.2f} seconds")
            display_response(response)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    **Disclaimer**: This tool provides AI-generated responses based on uploaded documents. 
    Verify critical information with original sources.
""")