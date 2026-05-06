# ==========================================
# PDF Chat Backend (FastAPI + LangChain + Gemini)
# ==========================================
# This file handles the server logic for our PDF Chat application.
# It uses FastAPI to create web endpoints (like /upload and /ask).
# 
# Key Technologies:
# 1. FastAPI: A fast web framework for building APIs in Python.
# 2. LangChain: A framework to make working with LLMs (Large Language Models) easier.
# 3. ChromaDB: A vector database to store our PDF text chunks so we can search them.
# 4. Google Gemini: The AI model that reads the text and answers our questions.

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain imports for processing the PDF and creating the database
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Google Generative AI for Gemini
import google.generativeai as genai

# dotenv to load secret API keys from our .env file
from dotenv import load_dotenv

# ------------------------------------------
# Setup & Configuration
# ------------------------------------------

# Load environment variables (like our GEMINI_API_KEY)
load_dotenv()

# Configure the Google Gemini API with our secret key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the FastAPI app
app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
# This allows our Next.js frontend (running on a different port/IP) to talk to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://swayam200.me", "http://192.168.0.163:3000"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Constants for our directories and models
UPLOAD_DIR = "./uploads"       # Where we save the uploaded PDF files
EMBED_MODEL = "all-MiniLM-L6-v2" # The model used to convert text into numbers (vectors)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# We keep the vector database in memory so the user can upload multiple files in a single session
# without SQLite disk locking errors. It resets when the server restarts.
global_vectorstore = None

# ------------------------------------------
# Core Functions
# ------------------------------------------

def process_and_save_pdf(filepath: str):
    """
    Reads a PDF, splits it into small chunks, and saves those chunks into a database.
    This process is called 'ingestion'.
    """
    # 1. Load the PDF file
    loader = PyPDFLoader(filepath)
    pages = loader.load()

    # 2. Split the PDF into smaller pieces (chunks)
    # We do this because AI models can't read an entire book at once.
    # Chunk size is 500 characters, with 50 characters overlapping so we don't cut sentences in half.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    # 3. Clean up the old database if it exists
    # By simply reassigning the global variable, the old in-memory database is deleted
    # and replaced with the new one.
    global global_vectorstore

    # 4. Save the new chunks into ChromaDB (our in-memory vector database)
    global_vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Return how many pieces we split the document into
    return len(chunks)

def ask_pdf_question(question: str) -> dict:
    """
    Finds the most relevant text from the PDF and asks Gemini to answer the question using that text.
    This technique is called RAG (Retrieval-Augmented Generation).
    """
    global global_vectorstore
    
    # Check if a PDF has been uploaded first
    if global_vectorstore is None:
        raise HTTPException(
            status_code=400, 
            detail="No PDF ingested yet. Upload one first."
        )

    # 2. Search for the top 4 chunks of text that match the user's question
    retriever = global_vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)

    # 3. Combine those chunks into a single block of text (the context)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 4. Create the prompt for Gemini
    # We explicitly tell it to ONLY use the context we provide.
    prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer isn't in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    # 5. Send the prompt to the Gemini AI model
    model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
    response = model.generate_content(prompt)

    # 6. Format the sources so the frontend knows exactly where the answer came from
    sources = [
        {
            "text": doc.page_content[:200],  # A short preview of the text
            "page": doc.metadata.get("page", "?"), # The page number
            "source": doc.metadata.get("source", "?")
        }
        for doc in docs
    ]

    # Return the AI's answer along with the sources
    return {"answer": response.text, "sources": sources}

# ------------------------------------------
# API Endpoints (The routes the frontend calls)
# ------------------------------------------

# A simple endpoint to check if the server is running
@app.get("/health")
def health():
    return {"status": "ok, server is running!"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receives a PDF file from the frontend, saves it, and processes it.
    """
    # Make sure it's actually a PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs allowed.")

    # Create the uploads folder if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded file to our computer
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    # Process the PDF (split it and save to the database)
    chunk_count = process_and_save_pdf(filepath)

    return {"message": f"Successfully loaded {file.filename}", "chunks": chunk_count}


# We use a Pydantic model to define what the incoming data should look like
class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(request: AskRequest):
    """
    Receives a question from the frontend, gets the answer from the AI, and returns it.
    """
    # Make sure they actually typed a question
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    # Get the answer
    result = ask_pdf_question(request.question)
    return result


@app.post("/summarize")
def summarize(request: AskRequest):
    """
    Generates a quick summary of the PDF and suggests some questions the user can ask.
    """
    global global_vectorstore

    if global_vectorstore is None:
        raise HTTPException(status_code=400, detail="No PDF ingested yet.")

    # We retrieve 8 chunks instead of 4 to get a broader overview of the document
    retriever = global_vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents(
        "introduction overview summary main topics"
    )
    
    # Combine the text
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. Ask Gemini to provide a structured response
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
    
    # Return the raw text, the frontend will parse it
    return {"raw": response.text}