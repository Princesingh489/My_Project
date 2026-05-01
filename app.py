import streamlit as st
import os
import sqlite3
import tempfile
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

st.set_page_config(page_title="RAG Knowledge Base", page_icon="📚", layout="wide")

# Set up API Key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    st.error("⚠️ API Key is missing! Please configure GEMINI_API_KEY in .streamlit/secrets.toml")
    st.stop()

# Set env var for Google Embeddings
os.environ["GOOGLE_API_KEY"] = api_key

# ----------------- DATABASE SETUP -----------------
DB_PATH = "knowledge.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_document(filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO documents (filename) VALUES (?)", (filename,))
    conn.commit()
    conn.close()

def save_query(question, answer):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO queries (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

def get_recent_queries(limit=5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT question, answer, created_at FROM queries ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

init_db()

# ----------------- RAG PIPELINE -----------------
@st.cache_resource
def get_vectorstore():
    # Return None if not initialized
    return None

def ocr_image_with_gemini(image_bytes):
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
    # Wrap image in a message for Gemini
    from langchain_core.messages import HumanMessage
    
    # LangChain Google GenAI supports passing image content directly
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Transcribe all the text you see in this image. Output only the text found."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"}}
        ]
    )
    response = llm.invoke([message])
    return response.content

def process_file_to_faiss(uploaded_file, faiss_path="faiss_index"):
    import base64
    from langchain_core.documents import Document
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    docs = []

    try:
        if file_extension == 'pdf':
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Load PDF
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            os.remove(tmp_file_path)
            
            # Check if text was extracted
            full_text = "".join([d.page_content for d in docs]).strip()
            if not full_text:
                st.info("No selectable text found in PDF. Attempting OCR with Gemini...")
                # For simplicity, we'll just treat the whole PDF as one "image" if it was small, 
                # but standard pypdf doesn't give images. 
                # Better: Tell user to upload images if PDF is scanned, or use a better loader.
                # However, Gemini can't take a PDF as an image directly in this SDK easily without conversion.
                st.error("This PDF seems to be a scanned image. Please upload the document as a JPG/PNG for OCR support.")
                return False
        
        elif file_extension in ['jpg', 'jpeg', 'png']:
            with st.spinner("Performing OCR with Gemini..."):
                image_base64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                transcribed_text = ocr_image_with_gemini(image_base64)
                docs = [Document(page_content=transcribed_text, metadata={"source": uploaded_file.name})]

        if not docs:
            st.error("Could not extract any content from the file.")
            return False

        # Split text (Optimized for CV headers)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        
        if os.path.exists(faiss_path):
            vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(splits)
        else:
            vectorstore = FAISS.from_documents(splits, embeddings)
            
        vectorstore.save_local(faiss_path)
        return True
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return False

def answer_question(question, faiss_path="faiss_index"):
    if not os.path.exists(faiss_path):
        return "Please upload a document first to build the knowledge base."
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    
    system_prompt = (
        "You are a robotic document analysis tool. "
        "Your task is to answer the QUESTION using ONLY the provided CONTEXT. "
        "Strict Rules:\n"
        "1. Use ONLY the provided CONTEXT. Do not use training data or outside knowledge.\n"
        "2. If the answer is not in the CONTEXT, say 'I am sorry, but that information is not in the document.'\n"
        "3. Do not offer clarifications about elections, politics, or other candidates.\n"
        "4. Focus on the candidate described in the document (the CV owner).\n"
        "\n"
        "CONTEXT:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 1. Retrieve the most relevant chunks (High-Resolution Search for speed)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(question)
    
    if not docs:
        return "⚠️ No relevant information was found in the knowledge base."

    # 2. Manual Grounding with Deep Context
    context_text = "\n\n".join([d.page_content for d in docs])
    
    prompt_text = f"""You are a professional Document Researcher. Use the CONTEXT below to answer the QUESTION.
Scanning Rule: Identify the person described in the document.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""

    try:
        response = llm.invoke(prompt_text)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Error during AI generation: {e}"

# ----------------- UI LAYOUT -----------------
st.title("📚 RAG-powered Knowledge Base")
st.write("*Upload company docs / PDFs. Ask questions and get cited answers.*")

# Sidebar for Uploads and History
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF or Image (JPG/PNG)", type=["pdf", "png", "jpg", "jpeg"])
    
    if st.button("Process & Index File", use_container_width=True) and uploaded_file:
        with st.spinner("Processing..."):
            success = process_file_to_faiss(uploaded_file)
            if success:
                save_document(uploaded_file.name)
                st.success(f"Successfully indexed: {uploaded_file.name}")
    
    st.divider()
    
    st.header("Recent Queries")
    history = get_recent_queries()
    if not history:
        st.info("No queries yet.")
    else:
        for row in history:
            with st.expander(f"Q: {row[0][:30]}..."):
                st.write(f"**Answer:** {row[1]}")
                st.caption(f"Time: {row[2]}")

# Main Chat Area
st.header("2. Ask Questions")

# Check if knowledge base exists
if not os.path.exists("faiss_index"):
    st.warning("👈 Please upload and process a document in the sidebar to start asking questions.")
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            answer = answer_question(prompt)
            st.markdown(answer)
            
            # Save query to database
            save_query(prompt, answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
