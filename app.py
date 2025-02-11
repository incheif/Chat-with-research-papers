import streamlit as st
import os
import shutil
import time
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

# Streamlit app title
st.title("Quantum Physics Research Analysis: Using Large Language Model")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", max_tokens=2048)

# Define prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    <context>
    {context}
    <context>

    {input}
    """
)

DATA_FOLDER = "./data"

def clean_previous_data():
    """Clean the `data` folder when the app reloads."""
    if os.path.exists(DATA_FOLDER):
        shutil.rmtree(DATA_FOLDER)
    os.makedirs(DATA_FOLDER)

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to the data folder and return the file path."""
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    return file_path

def initialize_vector_store_from_upload(uploaded_files):
    """Initialize vector embeddings and FAISS vector store from uploaded files."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = []

    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)  # Save the file and get the path
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(documents)  # Process all uploaded documents
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

def create_retrieval_chain_with_context(llm, vectors):
    """Create a retrieval chain using the LLM and vector store."""
    retriever = vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(retriever, document_chain)

def handle_user_question(question, retrieval_chain):
    """Handle user question and generate responses with context."""
    responses = {}

    # With context
    start = time.process_time()
    try:
        response_with_context = retrieval_chain.invoke({'input': question})
        responses['with_context'] = {
            "response_time": time.process_time() - start,
            "answer": response_with_context.get('answer', "No answer generated."),
            "context": response_with_context.get('context', [])
        }
    except Exception as e:
        st.error(f"Error during context-based response: {e}")
        responses['with_context'] = {"response_time": None, "answer": "Error generating response.", "context": []}

    return responses

# Clean data folder on reload
clean_previous_data()

# Streamlit UI
st.subheader("Upload PDFs")

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents for analysis:",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Initialize Document Embedding"):
        with st.spinner("Processing uploaded documents..."):
            st.session_state.vectors = initialize_vector_store_from_upload(uploaded_files)
        st.success("Vector store initialized successfully.")

question = st.text_input("Enter your question:")

if question and "vectors" in st.session_state:
    st.session_state.retrieval_chain = create_retrieval_chain_with_context(llm, st.session_state.vectors)
    responses = handle_user_question(question, st.session_state.retrieval_chain)

    if 'with_context' in responses:
        st.subheader("Response with Context:")
        st.text_area("Answer", responses['with_context']['answer'], height=600)
        with st.expander("Relevant Documents:"):
            if responses['with_context']['context']:
                for doc in responses['with_context']['context']:
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            else:
                st.write("No relevant documents found.")
