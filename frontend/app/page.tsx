"use client";

// 1. React Imports
// useState manages our app's data (variables that change over time).
// useRef lets us target specific HTML elements (like scrolling to the bottom of the chat).
// useEffect lets us run code when something changes or when the component first loads.
import { useState, useRef, useEffect } from "react";

// 2. Next.js Imports
// dynamic allows us to load components only when they are needed on the browser,
// which is required for the PDF viewer because it uses browser-only features (like Canvas).
import dynamic from "next/dynamic";

// 3. Our Custom Components
import ResizablePanes from "./components/ResizablePanes";

// Dynamically import the PDF Viewer so it doesn't try to load on the server.
const PdfViewer = dynamic(() => import("./components/PdfViewer"), { ssr: false });

// ==========================================
// Types and Interfaces
// (These tell TypeScript what our data looks like)
// ==========================================

// Defines a single message in the chat
type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];               // Where the AI found the answer
  suggestedQuestions?: string[];    // Clickable questions for the user
};

// Defines a source citation
type Source = {
  text: string;
  page: number | string;
  source: string;
  highlight_text?: string;
  item_start?: number;
  item_end?: number;
};

type ExtractedPage = {
  page: number;
  text: string;
  items: { str: string }[];
};

// Defines the response we expect from the backend /upload endpoint
type UploadResponse = {
  message?: string;
  chunks?: number;
  detail?: string;
};

// Defines the response we expect from the backend /ask endpoint
type AskResponse = {
  answer?: string;
  sources?: Source[];
  detail?: string;
};

// The URL of our Python backend API
const CONFIGURED_API = process.env.NEXT_PUBLIC_API_URL;

function apiUrl() {
  if (typeof window === "undefined") return CONFIGURED_API ?? "http://localhost:8000";

  const configuredUrl = CONFIGURED_API ?? "http://localhost:8000";
  const pageHost = window.location.hostname;
  const configuredHost = new URL(configuredUrl).hostname;

  if (
    configuredHost === "localhost" &&
    pageHost !== "localhost" &&
    pageHost !== "127.0.0.1"
  ) {
    return `http://${pageHost}:8000`;
  }

  return configuredUrl;
}

// Approach in plain English:
// The browser already uses pdf.js to render the PDF, so we also use pdf.js to
// extract the text that gets indexed by the backend. That keeps answer sources
// and visual highlights speaking the same "PDF language". The backend sends back
// a page number plus text-item range; this file keeps those anchors with each
// source badge and refuses to show a source page outside the loaded PDF.
//
// Accuracy expectation:
// The page badge should be a real page in the current PDF, and highlighting should
// usually land on the supporting sentence. The answer itself can still be wrong if
// retrieval picked weak context or the model over-generalized, so important claims
// should still be verified against the highlighted source.

// ==========================================
// Main Application Component
// ==========================================
export default function Home() {

  // ----------------------------------------
  // State Variables (The memory of our app)
  // ----------------------------------------

  // Controls which screen is currently visible: the upload screen or the chat screen
  const [view, setView] = useState<"upload" | "chat">("upload");

  // Upload state
  const [file, setFile] = useState<File | null>(null);          // The actual PDF file selected
  const [uploading, setUploading] = useState(false);            // Is it currently uploading?
  const [uploadStatus, setUploadStatus] = useState("");         // Text to show while uploading
  const [docName, setDocName] = useState("");                   // The name of the PDF

  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);      // List of all chat messages
  const [question, setQuestion] = useState("");                 // The text currently typed in the input box
  const [asking, setAsking] = useState(false);                  // Is the AI currently thinking?

  // PDF Viewer state
  const [blobUrl, setBlobUrl] = useState<string | null>(null);  // A temporary URL pointing to the file in the browser's memory
  const [pdfPageCount, setPdfPageCount] = useState(0);
  const [highlightPage, setHighlightPage] = useState<number | null>(null); // Which page should the PDF viewer jump to?
  // The exact text chunk to highlight on the target page (amber Ctrl+F-style highlight)
  const [highlightText, setHighlightText] = useState<string | null>(null);
  const [highlightItemRange, setHighlightItemRange] = useState<{ start: number; end: number } | null>(null);
  // Increments each time a citation is clicked — forces re-trigger even if same page is clicked twice
  const [highlightKey, setHighlightKey] = useState(0);

  // ----------------------------------------
  // Effects (Code that runs automatically)
  // ----------------------------------------

  // When a new PDF is loaded, we create a temporary "Blob URL" for it.
  // When we're done (or when the component unmounts), we need to clean it up to prevent memory leaks.
  useEffect(() => {
    return () => {
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
      }
    };
  }, [blobUrl]);

  // Whenever the messages list updates, scroll smoothly to the very bottom of the chat.
  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ----------------------------------------
  // Actions (Functions that do things)
  // ----------------------------------------

  // This runs when the user clicks the "Upload & Start Chatting" button
  async function handleUpload() {
    if (!file) return; // Do nothing if no file is selected

    // Set UI to loading state
    setUploading(true);
    setUploadStatus("Uploading PDF...");

    try {
      // 1. Prepare the file to be sent to the backend (just saves the file)
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${apiUrl()}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = (await res.json()) as UploadResponse;

      if (!res.ok) {
        setUploadStatus(data.detail ?? "Upload failed.");
        return;
      }

      // 2. Extract text from the PDF using pdf.js — the SAME engine that renders it.
      //    This is the key architectural fix: previously PyPDFLoader on the backend
      //    extracted text in a different column order than pdf.js, so highlighting
      //    could never match. Now both sides use identical text.
      //
      //    WHY dynamic import: pdfjs-dist uses DOMMatrix which doesn't exist in Node.js.
      //    Next.js evaluates top-level imports during SSR, which would crash.
      //    Dynamic import() only runs in the browser where DOMMatrix exists.
      setUploadStatus("Extracting text from PDF...");

      const { pdfjs } = await import("react-pdf");
      // Unconditionally set workerSrc. react-pdf may set a default like 'pdf.worker.mjs' 
      // which fails to resolve in Next.js.
      pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

      const blob = URL.createObjectURL(file);
      const loadingTask = pdfjs.getDocument(blob);
      const pdfDoc = await loadingTask.promise;
      setPdfPageCount(pdfDoc.numPages);

      const pages: ExtractedPage[] = [];
      for (let i = 1; i <= pdfDoc.numPages; i++) {
        const page = await pdfDoc.getPage(i);
        const textContent = await page.getTextContent();
        const items = textContent.items.map((item) => ({
          str: "str" in item ? item.str ?? "" : "",
        }));
        const text = items.map((item) => item.str).join(" ");
        pages.push({ page: i, text, items });
      }

      // 3. Send pdf.js-extracted text to the backend for RAG indexing
      setUploadStatus("Building search index...");

      const ingestRes = await fetch(`${apiUrl()}/ingest-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pages, filename: file.name, total_pages: pdfDoc.numPages }),
      });
      const ingestData = await ingestRes.json();

      if (!ingestRes.ok) {
        setUploadStatus(ingestData.detail ?? "Ingestion failed.");
        return;
      }

      // 4. Upload successful! Switch to chat view.
      setDocName(file.name);
      setUploadStatus("");
      setHighlightPage(null);
      setHighlightText(null);
      setHighlightItemRange(null);
      setBlobUrl(blob);

      // Show a "Thinking..." message immediately
      setMessages([{ role: "assistant", content: "..." }]);
      setView("chat");

      // 5. Automatically ask the AI to summarize the document
      try {
        const sumRes = await fetch(`${apiUrl()}/summarize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: "summarize" }),
        });

        if (!sumRes.ok) throw new Error("Failed to summarize");

        const sumData = await sumRes.json();
        const rawText: string = (sumData.raw ?? "") as string;

        const summaryMatch = rawText.match(/SUMMARY:\s*(.+?)(?=QUESTIONS:|$)/s);
        const questionsMatch = rawText.match(/QUESTIONS:\s*([\s\S]+)/);

        const summary = summaryMatch?.[1]?.trim() ?? "Document loaded successfully.";

        const questions = questionsMatch?.[1]
          ?.split("\n")
          .filter((line: string) => line.trim().startsWith("Q:"))
          .map((line: string) => line.replace(/^Q:\s*/, "").trim())
          ?? [];

        setMessages([
          {
            role: "assistant",
            content: summary,
            sources: [],
            suggestedQuestions: questions,
          },
        ]);
      } catch {
        setMessages([
          { role: "assistant", content: `Ready! I've read **${file.name}** (${ingestData.chunks ?? "?"} chunks). Ask me anything about it.`, sources: [] },
        ]);
      }
    } catch (e) {
      setUploadStatus(e instanceof Error ? e.message : "Upload error.");
    } finally {
      setUploading(false);
    }
  }

  // This runs when the user clicks the "Send" button in the chat
  async function handleAsk() {
    if (!question.trim() || asking) return; // Do nothing if input is empty or AI is busy

    // 1. Add the user's question to the chat history
    const userMsg: Message = { role: "user", content: question };
    setMessages((prev) => [...prev, userMsg]);

    // Clear the input box and set to busy state
    setQuestion("");
    setAsking(true);

    // 2. Add a "Thinking..." placeholder message for the AI
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "..." },
    ]);

    try {
      // 3. Send the question to the backend API
      const res = await fetch(`${apiUrl()}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg.content }),
      });
      const data = (await res.json()) as AskResponse;

      // 4. Replace the "..." placeholder with the real answer from the AI
      setMessages((prev) => [
        ...prev.slice(0, -1),
        {
          role: "assistant",
          content: data.answer ?? data.detail ?? "No answer returned.",
          sources: data.sources,
        },
      ]);
    } catch {
      // If the backend fails, show an error message
      setMessages((prev) => [
        ...prev.slice(0, -1),
        { role: "assistant", content: "Something went wrong. Is the backend running?" },
      ]);
    } finally {
      setAsking(false); // Done thinking!
    }
  }

  // Helper function to allow sending messages by pressing the "Enter" key
  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault(); // Stop it from making a new line
      handleAsk();
    }
  }

  function sourcePage(sourceItem: Source) {
    const pageNum = typeof sourceItem.page === "number" ? sourceItem.page : parseInt(sourceItem.page, 10);
    if (Number.isNaN(pageNum)) return null;
    if (pdfPageCount > 0 && (pageNum < 1 || pageNum > pdfPageCount)) return null;
    return pageNum;
  }

  function showSourceInPdf(sourceItem: Source) {
    const pageNum = sourcePage(sourceItem);
    if (pageNum === null) return;

    const hasItemRange =
      typeof sourceItem.item_start === "number" &&
      typeof sourceItem.item_end === "number" &&
      sourceItem.item_end >= sourceItem.item_start;

    setHighlightPage(pageNum);
    setHighlightText(sourceItem.highlight_text ?? sourceItem.text);
    setHighlightItemRange(
      hasItemRange
        ? { start: sourceItem.item_start!, end: sourceItem.item_end! }
        : null
    );
    setHighlightKey((k) => k + 1);
  }

  // ==========================================
  // User Interface (What you actually see on screen)
  // ==========================================

  // ---- VIEW 1: UPLOAD SCREEN ----
  if (view === "upload") {
    return (
      <main className="flex min-h-screen items-center justify-center bg-black px-4 text-gray-100">
        {/* White container box */}
        <div className="w-full max-w-md rounded-2xl border border-white/5 bg-[#0a0a0a] p-8 shadow-md">
          <h1 className="text-2xl font-bold">ChatPDF</h1>
          <p className="mt-2 text-sm text-gray-400">
            Hi there! 👋 Upload any PDF document, and I&apos;ll help you understand it, summarize it, and answer any questions you have.
          </p>

          <div className="mt-6 space-y-4">

            {/* The file upload area (styled like a drag-and-drop box) */}
            <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-white/10 bg-[#111111] p-8 text-center hover:border-white/20 hover:bg-[#111111]/80 transition-all">
              <span className="text-3xl">📄</span>
              <span className="mt-2 text-sm font-medium text-gray-200">
                {file ? file.name : "Click to choose a PDF"}
              </span>
              <span className="text-xs text-gray-400 mt-1">
                {file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "PDF files only"}
              </span>
              <input
                type="file"
                accept="application/pdf"
                className="hidden" // We hide the default ugly file input and style the label instead
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </label>

            {/* The submit button */}
            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              className="w-full rounded-xl bg-emerald-600 py-3 text-sm font-medium text-white disabled:opacity-40 disabled:cursor-not-allowed hover:bg-emerald-500 transition-colors"
            >
              {uploading ? "Processing..." : "Start Chatting"}
            </button>

            {/* Error or loading status text */}
            {uploadStatus && (
              <p className="text-center text-sm text-gray-400">{uploadStatus}</p>
            )}
          </div>
        </div>
      </main>
    );
  }

  // ---- VIEW 2: CHAT INTERFACE (Right Panel) ----
  const chatContent = (
    <div className="flex h-full flex-col bg-[#0a0a0a] text-gray-100">

      {/* 1. Header (Top bar) */}
      <header className="flex items-center justify-between border-b border-white/5 bg-black px-6 py-3">
        <div>
          <h1 className="text-sm font-semibold text-gray-100">ChatPDF</h1>
          <p className="text-xs text-gray-400 truncate max-w-xs">{docName}</p>
        </div>
        {/* Reset button to go back to upload screen */}
        <button
          onClick={() => {
            setView("upload");
            setMessages([]);
            setFile(null);
            setDocName("");
            setPdfPageCount(0);
            setHighlightPage(null);
            setHighlightText(null);
            setHighlightItemRange(null);
            if (blobUrl) {
              URL.revokeObjectURL(blobUrl);
              setBlobUrl(null);
            }
          }}
          className="rounded-lg border border-white/10 bg-[#111111] px-3 py-1.5 text-xs text-gray-300 hover:bg-[#202020] transition-colors"
        >
          Upload new PDF
        </button>
      </header>

      {/* 2. Message History Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {messages.map((msg, i) => (
          // Align user messages to the right, AI messages to the left
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>

            {/* The message bubble */}
            <div className={`max-w-xl rounded-2xl px-5 py-4 text-sm ${msg.role === "user"
                ? "bg-[#111111] border border-white/5 text-gray-100 rounded-br-sm" // Darker bubble for user
                : "bg-transparent text-gray-100" // Transparent for AI (like ChatGPT)
              }`}>

              {/* Message text */}
              <p style={{ whiteSpace: "pre-wrap", lineHeight: "1.6" }}>
                {msg.content === "..." ? (
                  <span className="animate-pulse text-gray-400">Thinking...</span>
                ) : (
                  // Simple Markdown parsing: turn **text** into bold text
                  msg.content.split(/\*\*(.*?)\*\*/g).map((part, j) =>
                    j % 2 === 1 ? <strong key={j}>{part}</strong> : part
                  )
                )}
              </p>

              {/* Inline citation badges — deduplicated by page number. */}
              {msg.role === "assistant" && msg.content !== "..." && msg.sources && msg.sources.length > 0 && (() => {
                const seenPages = new Map<number, Source>();
                msg.sources!.forEach((s) => {
                  const p = sourcePage(s);
                  if (p === null) return;

                  const existing = seenPages.get(p);
                  const hasItemRange = typeof s.item_start === "number" && typeof s.item_end === "number";
                  const existingHasItemRange =
                    existing && typeof existing.item_start === "number" && typeof existing.item_end === "number";

                  if (!existing || (hasItemRange && !existingHasItemRange)) seenPages.set(p, s);
                });
                return (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {Array.from(seenPages.entries()).map(([pageNum, sourceItem]) => (
                      <button
                        key={pageNum}
                        onClick={() => showSourceInPdf(sourceItem)}
                        className="inline-flex items-center rounded bg-amber-50 border border-amber-200 px-1.5 py-0.5 text-[10px] font-semibold text-amber-700 hover:bg-amber-100 hover:border-amber-300 transition-colors cursor-pointer"
                        title={(sourceItem.highlight_text ?? sourceItem.text).slice(0, 100) + "..."}
                      >
                        p.{pageNum}
                      </button>
                    ))}
                  </div>
                );
              })()}

              {/* Clickable suggested questions (Only show on the very first AI message) */}
              {msg.role === "assistant" && msg.suggestedQuestions && msg.suggestedQuestions.length > 0 && i === 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {msg.suggestedQuestions.map((q, qi) => (
                    <button
                      key={qi}
                      onClick={() => {
                        setQuestion(q);
                        // Focus the input box slightly after setting the question
                        setTimeout(() => {
                          document.getElementById("chat-input")?.focus();
                        }, 50);
                      }}
                      className="rounded-full border border-white/10 bg-[#111111] px-3 py-1.5 text-xs text-gray-300 hover:bg-[#202020] transition-colors text-left"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}

              {/* Source citations (If the AI used specific parts of the PDF) */}
              {msg.sources && msg.sources.length > 0 && (() => {
                // Deduplicate sources by matching text & page to avoid showing identical chunks
                const uniqueSources: Source[] = [];
                const seenKeys = new Set<string>();
                msg.sources.forEach((s) => {
                  if (sourcePage(s) === null) return;
                  const key = `${s.page}-${s.text.trim()}`;
                  if (!seenKeys.has(key)) {
                    seenKeys.add(key);
                    uniqueSources.push(s);
                  }
                });

                if (uniqueSources.length === 0) return null;

                return (
                  <details className="mt-3 border-t border-white/5 pt-2">
                    <summary className="cursor-pointer text-xs text-gray-400 hover:text-gray-300">
                      {uniqueSources.length} source{uniqueSources.length > 1 ? "s" : ""}
                    </summary>
                    <div className="mt-2 space-y-2">
                      {uniqueSources.map((sourceItem, si) => (
                        <div
                          key={si}
                          onClick={() => showSourceInPdf(sourceItem)}
                          className="rounded-lg bg-[#111111] border border-white/5 px-3 py-2 text-xs text-gray-300 cursor-pointer hover:bg-[#202020] transition-all"
                        >
                          <span className="font-medium text-gray-100">Page {sourceItem.page}</span>
                          <p className="mt-0.5 line-clamp-2 text-gray-400">{sourceItem.highlight_text ?? sourceItem.text}</p>
                        </div>
                      ))}
                    </div>
                  </details>
                );
              })()}
            </div>
          </div>
        ))}
        {/* Invisible element at the bottom to scroll to */}
        <div ref={bottomRef} />
      </div>

      {/* 3. Input Area (Bottom bar) */}
      <div className="bg-[#0a0a0a] px-4 py-4 shrink-0 border-t border-white/5">
        <div className="mx-auto flex max-w-3xl gap-3">
          <textarea
            rows={1}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            id="chat-input"
            placeholder="Ask something about your PDF..."
            className="flex-1 resize-none rounded-xl border border-white/10 bg-[#111111] px-4 py-3 text-sm text-gray-100 focus:border-white/20 focus:outline-none placeholder-gray-500"
          />
          <button
            onClick={handleAsk}
            disabled={!question.trim() || asking}
            className="rounded-xl bg-emerald-600 px-6 py-3 text-sm font-medium text-white disabled:opacity-40 hover:bg-emerald-500 transition-colors"
          >
            Send
          </button>
        </div>
        <p className="mt-3 text-center text-xs text-gray-400">
          ChatPDF can make mistakes. Consider verifying important information.
        </p>
      </div>
    </div>
  );

  // Combine the PDF viewer (left side) and the Chat (right side) using our ResizablePanes component
  return (
    <ResizablePanes
      left={blobUrl ? <PdfViewer pdfUrl={blobUrl} highlightPage={highlightPage} highlightText={highlightText} highlightItemRange={highlightItemRange} highlightKey={highlightKey} /> : <div className="flex h-full items-center justify-center bg-black text-gray-600">No PDF</div>}
      right={chatContent}
    />
  );
}
