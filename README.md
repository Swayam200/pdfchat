# PDFChat

PDFChat is a full-stack web application that allows users to upload PDF documents and have interactive conversations with them. It leverages Retrieval-Augmented Generation (RAG) to provide accurate answers based directly on the contents of the uploaded PDFs.

## Tech Stack

- **Backend:** FastAPI (Python), Langchain, ChromaDB (Vector Store), Google Generative AI (Gemini)
- **Frontend:** Next.js (React, TypeScript), Tailwind CSS

## Features
- **Upload PDFs:** Ingests and processes PDF files, splitting them into manageable chunks.
- **Vector Search:** Embeds document chunks using HuggingFace models and stores them in ChromaDB.
- **Chat Interface:** Ask questions about the document and receive context-aware answers powered by Google Gemini.
- **Summarization:** Automatically generates a document summary and suggested questions.

## Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.9+)
- A Gemini API Key from Google AI Studio

### Starting the Backend

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment (if not already active):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your environment variables:
   Create a `.env` file in the `backend` directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
5. Run the FastAPI development server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --reload
   # The backend will be available at http://localhost:8000 and your network IP
   ```

### Starting the Frontend

1. Open a new terminal and navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the Next.js development server:
   ```bash
   npm run dev
   # The frontend will be available at http://localhost:3000
   ```

## Usage
1. Make sure both the frontend and backend servers are running.
2. Open your browser and navigate to `http://localhost:3000`.
3. Upload a PDF file using the interface.
4. Once processed, start asking questions about the document!
