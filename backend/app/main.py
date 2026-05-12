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
import re
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple

# LangChain imports for processing the PDF and creating the database
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Google Generative AI for Gemini
import google.generativeai as genai

from tavily import TavilyClient

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://swayam200.me"],
    allow_origin_regex=r"(https://.*\.vercel\.app|http://(192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+):3000)",
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
global_current_page_count = 0

# The frontend extracts text with pdf.js and sends the raw text item order here.
# We keep a tiny page index so returned sources can point to exact text items in the viewer.
# This is the main design choice for highlighting: do not ask the frontend to
# guess where an answer came from. Store the page and pdf.js item range while
# indexing, return that same anchor with each source, and discard anything that
# does not belong to the currently loaded PDF.
#
# What to expect:
# - Source pages should always be within the current PDF's page count.
# - Highlighting should usually land on the supporting sentence, especially after
#   a fresh upload.
# - Answers can still be imperfect because RAG depends on retrieval quality and
#   the model's reading of the retrieved chunks.
global_page_index: Dict[int, Dict[str, Any]] = {}

CHUNK_SIZE = 900
CHUNK_OVERLAP = 160

STOP_WORDS = {
    "about", "after", "also", "and", "are", "because", "been", "but", "can",
    "could", "did", "does", "for", "from", "had", "has", "have", "how",
    "into", "its", "more", "not", "of", "off", "the", "their", "this",
    "that", "then", "there", "these", "they", "was", "were", "what", "when",
    "where", "which", "while", "who", "why", "with", "would", "you", "your",
}


class AskRequest(BaseModel):
    question: str
    web_search: bool = False  # Optional field with default value, backward compatible


class PdfTextItem(BaseModel):
    text: str = Field(default="", alias="str")


class PageText(BaseModel):
    page: int
    text: str = ""
    items: Optional[List[PdfTextItem]] = None


class IngestTextRequest(BaseModel):
    pages: List[PageText]
    filename: str
    total_pages: Optional[int] = None

# ------------------------------------------
# Core Functions
# ------------------------------------------

def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalized_with_map(text: str) -> Tuple[str, List[int]]:
    chars: List[str] = []
    index_map: List[int] = []
    previous_was_space = True

    for i, ch in enumerate(text):
        if ch.isspace():
            if chars and not previous_was_space:
                chars.append(" ")
                index_map.append(i)
            previous_was_space = True
        else:
            chars.append(ch.lower())
            index_map.append(i)
            previous_was_space = False

    if chars and chars[-1] == " ":
        chars.pop()
        index_map.pop()

    return "".join(chars), index_map


def _find_normalized_span(text: str, needle: str) -> Optional[Tuple[int, int]]:
    normalized_text, text_map = _normalized_with_map(text)
    normalized_needle, _ = _normalized_with_map(needle)
    if len(normalized_needle) < 12:
        return None

    match_start = normalized_text.find(normalized_needle)
    if match_start == -1:
        return None

    match_end = match_start + len(normalized_needle) - 1
    return text_map[match_start], text_map[match_end] + 1


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token not in STOP_WORDS
    }


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for match in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end - start >= 40:
            spans.append((start, end))
    return spans


def _word_window_spans(text: str, window_size: int = 34, step: int = 12) -> List[Tuple[int, int]]:
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    if len(words) <= window_size:
        return [(words[0].start(), words[-1].end())]

    spans: List[Tuple[int, int]] = []
    for start_i in range(0, len(words), step):
        end_i = min(start_i + window_size, len(words)) - 1
        spans.append((words[start_i].start(), words[end_i].end()))
        if end_i == len(words) - 1:
            break
    return spans


def _span_score(span_text: str, question_tokens: set[str], answer_tokens: set[str]) -> float:
    span_tokens = _tokens(span_text)
    if not span_tokens:
        return 0

    answer_overlap = len(span_tokens & answer_tokens)
    question_overlap = len(span_tokens & question_tokens)
    length_bonus = min(len(span_text) / 220, 1.0)
    length_penalty = max((len(span_text) - 520) / 520, 0)
    return (answer_overlap * 2.0) + question_overlap + length_bonus - length_penalty


def _best_highlight_span(
    doc: Document,
    question: str,
    answer: str,
    quoted_highlight: Optional[str],
) -> Tuple[str, Tuple[int, int]]:
    source_text = doc.page_content

    if quoted_highlight:
        exact_span = _find_normalized_span(source_text, quoted_highlight)
        if exact_span:
            return _clean_text(source_text[exact_span[0]:exact_span[1]]), exact_span

    candidates = _sentence_spans(source_text) + _word_window_spans(source_text)
    if not candidates:
        end = min(len(source_text), 260)
        return _clean_text(source_text[:end]), (0, end)

    question_tokens = _tokens(question)
    answer_tokens = _tokens(answer)
    best_start, best_end = max(
        candidates,
        key=lambda span: _span_score(source_text[span[0]:span[1]], question_tokens, answer_tokens),
    )

    return _clean_text(source_text[best_start:best_end]), (best_start, best_end)


def _item_range_for_doc_span(doc: Document, local_start: int, local_end: int) -> Tuple[Optional[int], Optional[int]]:
    page = doc.metadata.get("page")
    if not isinstance(page, int):
        return None, None

    page_index = global_page_index.get(page)
    chunk_start = doc.metadata.get("char_start")
    if not page_index or not isinstance(chunk_start, int):
        return None, None

    char_to_item: List[int] = page_index.get("char_to_item", [])
    if not char_to_item:
        return None, None

    global_start = max(0, chunk_start + local_start)
    global_end = min(len(char_to_item), chunk_start + local_end)
    if global_end <= global_start:
        return None, None

    item_indices = [i for i in char_to_item[global_start:global_end] if i >= 0]
    if not item_indices:
        return None, None

    return min(item_indices), max(item_indices)


def _valid_page_number(page: Any) -> bool:
    return (
        isinstance(page, int)
        and page >= 1
        and global_current_page_count > 0
        and page <= global_current_page_count
    )


def _extract_highlight_quotes(raw_response: str) -> Dict[int, str]:
    highlights_match = re.search(r"HIGHLIGHTS:\s*(.+)", raw_response, re.DOTALL)
    if not highlights_match:
        return {}

    highlights: Dict[int, str] = {}
    for line in highlights_match.group(1).splitlines():
        match = re.match(r"\s*(?:Source\s*)?(\d+)\s*:\s*(.+?)\s*$", line)
        if not match:
            continue
        source_index = int(match.group(1)) - 1
        quote = match.group(2).strip().strip('"').strip("'")
        if quote and quote.lower() != "none":
            highlights[source_index] = quote
    return highlights


def _answer_style_instruction(question: str) -> str:
    q = question.lower()
    wants_short = any(term in q for term in [
        "brief", "briefly", "concise", "short", "quick", "one sentence", "tl;dr",
    ])
    wants_detail = any(term in q for term in [
        "detail", "detailed", "deep", "explain", "elaborate", "analyze",
        "compare", "step by step", "why", "how does", "how do",
    ])

    if wants_short:
        return "The user is asking for a concise answer. Use 1-3 tight sentences."
    if wants_detail:
        return "The user is asking for explanation or analysis. Use short paragraphs or bullets with enough detail to be useful."
    if len(question.split()) <= 10 or q.strip().startswith(("what", "who", "when", "where", "which")):
        return "This looks like a direct question. Answer in 1-3 sentences unless the document requires a small list."
    return "Use a balanced answer: start with the direct answer, then add only the necessary supporting detail."


def _build_page_text_and_map(page_data: PageText) -> Tuple[str, List[int]]:
    if page_data.items:
        parts: List[str] = []
        char_to_item: List[int] = []
        previous_item_index = -1

        for item_index, item in enumerate(page_data.items):
            item_text = _clean_text(item.text)
            if not item_text:
                continue

            if parts:
                parts.append(" ")
                char_to_item.append(previous_item_index)

            parts.append(item_text)
            char_to_item.extend([item_index] * len(item_text))
            previous_item_index = item_index

        return "".join(parts), char_to_item

    fallback_text = _clean_text(page_data.text)
    return fallback_text, [0] * len(fallback_text)


def _choose_chunk_end(text: str, start: int, hard_end: int) -> int:
    if hard_end >= len(text):
        return len(text)

    search_from = start + int(CHUNK_SIZE * 0.55)
    best_end = -1
    for separator in ("\n\n", "\n", ". ", "? ", "! ", "; ", " "):
        pos = text.rfind(separator, search_from, hard_end)
        if pos > best_end:
            best_end = pos + len(separator)

    return best_end if best_end > start else hard_end


def _chunk_page_text(page: int, text: str, filename: str) -> List[Document]:
    chunks: List[Document] = []
    start = 0

    while start < len(text):
        hard_end = min(len(text), start + CHUNK_SIZE)
        end = _choose_chunk_end(text, start, hard_end)

        chunk_start, chunk_end = start, end
        while chunk_start < chunk_end and text[chunk_start].isspace():
            chunk_start += 1
        while chunk_end > chunk_start and text[chunk_end - 1].isspace():
            chunk_end -= 1

        if chunk_end > chunk_start:
            chunks.append(
                Document(
                    page_content=text[chunk_start:chunk_end],
                    metadata={
                        "page": page,
                        "source": filename,
                        "char_start": chunk_start,
                        "char_end": chunk_end,
                    },
                )
            )

        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)

    return chunks


def _search_web(query: str) -> List[Dict[str, str]]:
    """
    What Tavily does differently from a raw Google search API: it returns clean extracted content, 
    not just links — so the LLM can actually read it directly.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)
        return [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", "")
            }
            for result in response.get("results", [])
        ]
    except Exception as e:
        print(f"Tavily search failed: {e}")
        return []


def ask_pdf_question(question: str, web_search: bool = False) -> dict:
    """
    Finds the most relevant text from the PDF and asks Gemini to answer the question using that text.
    This technique is called RAG (Retrieval-Augmented Generation).

    The important safety rule here is that the model never invents source pages for us.
    We retrieve chunks first, number those chunks as Source 1/2/3, and only accept source
    numbers that point back to a real retrieved chunk from the currently ingested PDF.
    """
    global global_current_page_count, global_vectorstore
    
    if global_vectorstore is None:
        raise HTTPException(
            status_code=400, 
            detail="No PDF ingested yet. Upload one first."
        )

    # Retrieve the top chunks most similar to the question.
    # We also fetch their relevance scores so we can order them confidently.
    docs_with_scores = global_vectorstore.similarity_search_with_score(question, k=6)
    
    # Sort by score ascending (Chroma uses L2 distance: lower = more relevant)
    docs_with_scores.sort(key=lambda x: x[1])
    docs = [
        d
        for d, _ in docs_with_scores
        if _valid_page_number(d.metadata.get("page"))
    ]

    if not docs:
        return {
            "answer": "I cannot find this in the provided document.",
            "sources": [],
            "answer_snippet": "I cannot find this in the provided document.",
        }

    # Build a numbered context block so Gemini can tell us which source(s) it used.
    numbered_context = ""
    for i, doc in enumerate(docs):
        numbered_context += f"[Source {i+1}, Page {doc.metadata.get('page', '?')}]:\n{doc.page_content}\n\n"

    web_context = ""
    web_results = []
    if web_search:
        web_results = _search_web(question)
        if web_results:
            for i, result in enumerate(web_results):
                web_context += f"[Web {i+1}] \"{result['title']}\" ({result['url']}):\n{result['snippet']}\n\n"

    # Why we label PDF vs web context separately in the prompt:
    # so Gemini can reason about source priority (preferring PDF over Web)
    context_block = f"PDF CONTEXT:\n{numbered_context}"
    if web_context:
        context_block += f"\nWEB CONTEXT:\n{web_context}"

    # Ask Gemini to answer AND tell us which sources it actually relied on.
    # Critically: SOURCES_USED must only include sources whose text directly
    # contains the specific fact used in the answer — not just related context.
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.

Context:
{context_block}
Question: {question}

Length and detail guidance:
{_answer_style_instruction(question)}

Provide your response in this EXACT format (no deviations):
ANSWER: <your answer here>
SOURCES_USED: <comma-separated list of source numbers whose text DIRECTLY contains the specific fact(s) stated in your answer, e.g. "1,3">
HIGHLIGHTS:
1: <exact sentence or phrase copied from Source 1 that supports the answer>
3: <exact sentence or phrase copied from Source 3 that supports the answer>

Rules:
- If the answer is not in the context, write ANSWER: I cannot find this in the provided document. SOURCES_USED: none
- Do not infer that something is "used", "caused", or "proven" unless the context says that directly. A package name, command, or related phrase by itself is not enough evidence.
- SOURCES_USED must be ONLY sources that contain the specific sentence or data point you cited. Do NOT include sources that are merely related or provide background context.
- Each HIGHLIGHTS line must copy exact words from the matching source. Prefer one short supporting sentence or phrase, not a whole chunk."""

    if web_context:
        prompt += "\n- Prefer the PDF CONTEXT for your answers. Use the WEB CONTEXT only to supplement or if the PDF lacks the answer."

    model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Parse ANSWER and SOURCES_USED from the structured response
    answer_text = raw
    used_indices: set[int] = set()

    answer_match = re.search(r"ANSWER:\s*(.+?)(?=SOURCES_USED:|$)", raw, re.DOTALL)
    sources_match = re.search(r"SOURCES_USED:\s*(.+?)(?=HIGHLIGHTS:|$)", raw, re.DOTALL)
    highlight_quotes = _extract_highlight_quotes(raw)

    if answer_match:
        answer_text = answer_match.group(1).strip()
    if sources_match:
        raw_indices = sources_match.group(1).strip()
        if raw_indices.lower() != "none":
            for part in raw_indices.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1  # convert 1-indexed to 0-indexed
                    if 0 <= idx < len(docs):
                        used_indices.add(idx)

    # Do not invent citations if Gemini did not name a usable source. It is better
    # to show no badge than to make an unsupported answer look grounded.

    # Only return sources Gemini actually used, with exact highlight anchors.
    sources = []
    for i in sorted(used_indices):
        page = docs[i].metadata.get("page")
        if not _valid_page_number(page):
            continue

        highlight_text, local_span = _best_highlight_span(
            docs[i],
            question,
            answer_text,
            highlight_quotes.get(i),
        )
        item_start, item_end = _item_range_for_doc_span(docs[i], local_span[0], local_span[1])
        source = {
            "text": docs[i].page_content,
            "page": page,
            "source": docs[i].metadata.get("source", "?"),
            "highlight_text": highlight_text,
        }
        if item_start is not None and item_end is not None:
            source["item_start"] = item_start
            source["item_end"] = item_end
        sources.append(source)

    result = {
        "answer": answer_text,
        "sources": sources,
        "answer_snippet": sources[0]["highlight_text"] if sources else answer_text,
    }
    if web_search:
        result["web_sources"] = web_results

    return result

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
    Receives a PDF file from the frontend and saves it.
    Text extraction is now handled by the frontend using pdf.js,
    which ensures the same text extraction engine is used for both
    RAG indexing and PDF rendering/highlighting.
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

    return {"message": f"Successfully saved {file.filename}"}


@app.post("/ingest-text")
def ingest_text(request: IngestTextRequest):
    """
    Receives pre-extracted text from the frontend (extracted using pdf.js)
    and builds an item-aware vector store from it.

    Why we do it this way:
    - pdf.js is also what renders the visible PDF, so its text-item order is the
      closest thing we have to "what the user sees on screen".
    - We store page numbers and text-item ranges with every chunk. That gives the
      frontend a concrete highlight target instead of asking it to search vaguely
      around the right page.
    - Each upload gets a fresh Chroma collection. In dev, reusing the default
      collection can leak old chunks into a new PDF session, which is how impossible
      citations like page 13 can show up for a 6-page PDF.

    Expected accuracy:
    Answers are still only as good as retrieval plus Gemini's reading of the retrieved
    chunks, so users should verify important claims. Citations, however, should never
    point outside the currently loaded PDF's page count.
    """
    global global_current_page_count, global_page_index, global_vectorstore

    if not request.pages:
        raise HTTPException(status_code=400, detail="No pages provided.")

    global_vectorstore = None
    global_page_index = {}
    global_current_page_count = request.total_pages or len(request.pages)
    chunks: List[Document] = []
    seen_pages: set[int] = set()

    for page_data in request.pages:
        if page_data.page < 1 or page_data.page > global_current_page_count:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid page {page_data.page} for a {global_current_page_count}-page PDF.",
            )
        if page_data.page in seen_pages:
            raise HTTPException(status_code=400, detail=f"Duplicate page {page_data.page}.")
        seen_pages.add(page_data.page)

        page_text, char_to_item = _build_page_text_and_map(page_data)
        if not page_text.strip():
            continue

        global_page_index[page_data.page] = {
            "text": page_text,
            "char_to_item": char_to_item,
        }
        chunks.extend(_chunk_page_text(page_data.page, page_text, request.filename))

    if not chunks:
        raise HTTPException(status_code=400, detail="All pages were empty.")

    global_vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdfchat_{uuid.uuid4().hex}",
    )

    return {"message": f"Ingested {len(request.pages)} pages into {len(chunks)} chunks", "chunks": len(chunks)}


@app.post("/ask")
def ask(request: AskRequest):
    """
    Receives a question from the frontend, gets the answer from the AI, and returns it.
    """
    # Make sure they actually typed a question
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    # Get the answer
    result = ask_pdf_question(request.question, web_search=request.web_search)
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
    docs = [doc for doc in docs if _valid_page_number(doc.metadata.get("page"))]

    if not docs:
        raise HTTPException(status_code=400, detail="No valid PDF text found to summarize.")
    
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
