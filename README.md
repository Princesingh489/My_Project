# 📚 RAG-powered Knowledge Base (with OCR Support)

A powerful, AI-driven Knowledge Base application that allows users to upload PDF documents and images (JPG/PNG), index them using vector embeddings, and query them through a conversational chat interface. The system uses Retrieval-Augmented Generation (RAG) to ensure answers are strictly based on the provided documents.

## 🚀 Key Features
- **Multimodal Uploads**: Support for both PDF documents and Image files (JPG, JPEG, PNG).
- **Gemini OCR**: Automatic transcription of text from images and scanned PDFs using Google's Gemini models.
- **RAG Pipeline**: Semantic search using FAISS (Facebook AI Similarity Search) for high-performance context retrieval.
- **Strict Grounding**: Custom AI prompting to prevent hallucinations and ensure answers are document-based.
- **Query History**: Local SQLite database to track query history and document uploads.
- **Real-time Chat**: Interactive Streamlit-based chat interface.

## 🛠️ Technology Stack
- **Language**: Python 3.x
- **Web Framework**: [Streamlit](https://streamlit.io/)
- **AI Models**:
  - **LLM**: Google Gemini 3 Flash Preview (for reasoning and OCR)
  - **Embeddings**: Google Gemini Embedding 2 (for vectorization)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **Database**: SQLite3
- **OCR**: Integrated Gemini-based Multimodal OCR

## 📦 Core Libraries Used
- `streamlit`: UI and web server.
- `langchain`: Framework for building LLM-powered applications.
- `langchain-google-genai`: Integration with Google Gemini API.
- `faiss-cpu`: High-performance vector similarity search.
- `pypdf`: PDF text extraction.
- `sqlite3`: Metadata and history persistence.

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Project1
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**:
   Create a `.streamlit/secrets.toml` file or set an environment variable:
   ```toml
   # .streamlit/secrets.toml
   GEMINI_API_KEY = "YOUR_API_KEY_HERE"
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📖 How it Works
1. **Ingestion**: When a file is uploaded, the system extracts text (using `pypdf` for PDFs or `Gemini` for images).
2. **Chunking**: Text is split into small, manageable chunks (500 characters) to preserve context.
3. **Embedding**: Chunks are converted into vector representations using `gemini-embedding-2`.
4. **Indexing**: Vectors are stored in a local `FAISS` index for instant searching.
5. **Retrieval**: When a question is asked, the system finds the most relevant chunks (k=10).
6. **Generation**: The retrieved chunks are passed to `Gemini 3 Flash` with a strict prompt to generate a document-based answer.

---
*Created by Prince Kumar Singh*
