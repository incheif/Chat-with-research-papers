import streamlit as st
import os
import shutil
import time
import pdfplumber  # Add this import
import base64
import tempfile
import streamlit.components.v1 as components
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from streamlit_pdf_viewer import pdf_viewer  # Import the library

# Initialize API keys from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["groq_api_key"] = st.secrets["GROQ_API_KEY"]
groq_api_key = os.environ["groq_api_key"]

# Streamlit app title
st.set_page_config(page_title="Research Paper Analysis", layout="wide")
st.title("Research Paper Analysis using Retrieval Augmented Generation")
st.markdown("#### Leverage the power of language models to analyze research papers.")

try:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="Llama3-8b-8192",
        temperature=0.1,  # Added temperature parameter
        max_tokens=2048
    )
except Exception as e:
    st.error(f"Failed to initialize Groq client: {str(e)}")
    st.stop()  # 

    
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

def extract_highlighted_text(pdf_path, relevant_texts):
    """Extract and highlight relevant text from the PDF."""
    highlighted_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if any(text in page_text for text in relevant_texts):
                highlighted_pages.append((page_number + 1, page_text))
    return highlighted_pages

def display_pdf_with_highlights(pdf_path, relevant_texts):
    """Display the PDF with highlighted text."""
    highlighted_pages = extract_highlighted_text(pdf_path, relevant_texts)
    if highlighted_pages:
        st.markdown("### Highlighted Pages")
        for page_number, page_text in highlighted_pages:
            st.markdown(f"#### Page {page_number}")
            for text in relevant_texts:
                page_text = page_text.replace(text, f"**:blue[{text}]**")
            st.write(page_text)
    else:
        st.write("No relevant text found in the PDF.")

def embed_pdf(pdf_path):
    """Embed a PDF file in the Streamlit app."""
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe>'
    components.html(pdf_display, height=900)

def extract_highlighted_text_from_memory(file, relevant_texts):
    """Extract and highlight relevant text from an in-memory PDF file."""
    highlighted_pages = []
    with pdfplumber.open(file) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if any(text in page_text for text in relevant_texts):
                highlighted_pages.append((page_number + 1, page_text))
    return highlighted_pages

def display_pdf_with_highlights_from_memory(file, relevant_texts):
    """Display the PDF with highlighted text from an in-memory file."""
    highlighted_pages = extract_highlighted_text_from_memory(file, relevant_texts)
    if highlighted_pages:
        st.markdown("### Highlighted Pages")
        for page_number, page_text in highlighted_pages:
            st.markdown(f"#### Page {page_number}")
            for text in relevant_texts:
                page_text = page_text.replace(text, f"**:blue[{text}]**")
            st.write(page_text)
    else:
        st.write("No relevant text found in the PDF.")

def embed_pdf_from_memory(file):
    """Provide a download button for the PDF file in the Streamlit app."""
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    file.seek(0)  # Reset file pointer for reuse
    href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="document.pdf">Download PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

# Clean data folder on reload
clean_previous_data()

# Sidebar for navigation or settings
with st.sidebar:
    st.header("Assistant Panel")
    st.markdown("""
        - **Ask Questions** based on the uploaded research documents.
    """)

# Upload Section
st.subheader("Upload Research PDFs")

uploaded_files = st.file_uploader(
    "Choose one or more Research documents",
    type=["pdf"],
    accept_multiple_files=True
)

# Store uploaded files in a dictionary for later reference
uploaded_files_dict = {uploaded_file.name: uploaded_file for uploaded_file in uploaded_files}

# Display uploaded PDFs using streamlit_pdf_viewer
if uploaded_files:
    if st.button("Initialize Document Embedding", key="init_embedding"):
        with st.spinner("Processing uploaded documents..."):
            documents = []
            temp_file_paths = []  # Store paths of temporary files for cleanup later

            for uploaded_file in uploaded_files:
                # Save the uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                    temp_file_paths.append(temp_file_path)  # Keep track of the file path

                # Use PyPDFLoader with the temporary file path
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(documents)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)

        st.success("Vector store initialized successfully.")

        # Display uploaded PDFs using streamlit_pdf_viewer
        for uploaded_file in uploaded_files:
            st.markdown(f"### Uploaded File: {uploaded_file.name}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
                pdf_viewer(temp_file_path)  # Use pdf_viewer to display the PDF

        # Cleanup temporary files after processing
        for temp_file_path in temp_file_paths:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

# Question Input Section
question = st.text_input(
    "Enter your question :",
    help="You can ask a specific question related to the uploaded documents"
)

# Process the question if it's asked
if question and "vectors" in st.session_state:
    st.session_state.retrieval_chain = create_retrieval_chain_with_context(llm, st.session_state.vectors)
    responses = handle_user_question(question, st.session_state.retrieval_chain)

    if 'with_context' in responses:
        st.text_area("Answer", responses['with_context']['answer'], height=600)
        with st.expander("Relevant Documents:"):
            if responses['with_context']['context']:
                for doc in responses['with_context']['context']:
                    st.write(doc.page_content)
                    st.write("--------------------------------")
                    # Retrieve the correct file object using the file name
                    file_name = doc.metadata['source']
                    if file_name in uploaded_files_dict:
                        file_obj = uploaded_files_dict[file_name]
                        # Highlight relevant text in the PDF
                        display_pdf_with_highlights_from_memory(file_obj, [doc.page_content])
                    else:
                        st.write(f"File {file_name} not found.")
            else:
                st.write("No relevant documents found.")

# Footer or Additional Information Section
st.markdown("""
    **Disclaimer:**
    This tool is for research purposes and may not provide fully accurate or verified scientific conclusions.
""")
