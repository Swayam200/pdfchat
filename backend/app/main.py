import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Read .env file
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://swayam200.me", "http://192.168.0.163:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHROMA_DIR = "./chroma_store"
UPLOAD_DIR = "./uploads"
EMBED_MODEL = "all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def ingest_pdf(filepath: str):
    """Run the full Phase 2 pipeline on a file path."""
    loader = PyPDFLoader(filepath)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    # Delete old chroma_store if it exists (we only keep latest PDF)
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    return len(chunks)

def query_rag(question: str) -> dict:
    """Retrieve relevant chunks + ask Gemini."""
    if not os.path.exists(CHROMA_DIR):
        raise HTTPException(
            status_code=400, 
            detail="No PDF ingested yet. Upload one first."
        )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    # Retrieve top 4 most similar chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    # Join chunks into one context string
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build the RAG prompt
    prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer isn't in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
    response = model.generate_content(prompt)

    # Return answer + sources
    sources = [
        {
            "text": doc.page_content[:200],  # preview
            "page": doc.metadata.get("page", "?"),
            "source": doc.metadata.get("source", "?")
        }
        for doc in docs
    ]

    return {"answer": response.text, "sources": sources}

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs allowed.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, file.filename)

    # Save the file
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    chunk_count = ingest_pdf(filepath)

    return {"message": f"Ingested {file.filename}", "chunks": chunk_count}


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    result = query_rag(request.question)
    return result

@app.post("/summarize")
def summarize(request: AskRequest):
    if not os.path.exists(CHROMA_DIR):
        raise HTTPException(status_code=400, detail="No PDF ingested yet.")

    # For summary we retrieve more chunks — 8 instead of 4
    # to get a broader picture of the whole document
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents(
        "introduction overview summary main topics"
    )
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Based on the document excerpts below, provide:
1. A summary in exactly 2-3 sentences describing what this document is about.
2. Exactly 5 suggested questions a reader might ask, each on a new line starting with "Q: "

Context:
{context}

Respond in this exact format:
SUMMARY: <your 2-3 sentence summary here>
QUESTIONS:
Q: <question 1>
Q: <question 2>
Q: <question 3>
Q: <question 4>
Q: <question 5>"""

    model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
    response = model.generate_content(prompt)
    return {"raw": response.text}