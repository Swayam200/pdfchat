'use client';
import { useState, useEffect, useRef } from "react";
import { Document, Page } from "react-pdf";
import type { PDFDocumentProxy } from "pdfjs-dist";
import { Virtuoso, VirtuosoHandle } from "react-virtuoso";
import "../lib/pdfWorker";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

interface PdfViewerProps {
  pdfUrl: string;
  highlightPage: number | null;
  highlightText: string | null;
  highlightItemRange: { start: number; end: number } | null;
  highlightKey: number;
}

interface RawTextItem {
  str?: string;
  transform?: number[];   // [a, b, c, d, tx, ty] — tx,ty = x,y in PDF user space
  width?: number;
  height?: number;
}

interface HighlightRect {
  x: number; y: number; w: number; h: number;
}

// Highlighting approach:
// The backend usually gives us exact pdf.js text-item indices for the source it used.
// That is much more reliable than searching for the generated answer text, because
// generated answers are paraphrases and PDFs can reorder text in multi-column layouts.
// When an older response has no item range, we fall back to a normalized text search.
//
// Accuracy expectation:
// Item-range highlights should land on the right source text as long as the PDF was
// uploaded with the current ingestion flow. The fallback search is best-effort: it
// should get the neighborhood right, but complex PDFs can still make it imperfect.

// ─────────────────────────────────────────────
// Step 1: find which PDF text items should be highlighted
// ─────────────────────────────────────────────

function normalizeForSearch(text: string) {
  return text.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function buildNormalizedPageText(items: RawTextItem[]) {
  let text = "";
  const charToItem: number[] = [];

  items.forEach((item, itemIndex) => {
    for (const char of (item.str ?? "").toLowerCase()) {
      if (/[a-z0-9]/.test(char)) {
        text += char;
        charToItem.push(itemIndex);
      }
    }
  });

  return { text, charToItem };
}

function candidateTexts(searchText: string) {
  const cleaned = searchText.replace(/\s+/g, " ").trim();
  const sentences = cleaned
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length >= 35);
  const words = cleaned.split(/\s+/).filter(Boolean);
  const windows: string[] = [];

  for (const size of [42, 28, 18]) {
    if (words.length < size) continue;
    const step = Math.max(8, Math.floor(size / 2));
    for (let start = 0; start < words.length; start += step) {
      windows.push(words.slice(start, Math.min(start + size, words.length)).join(" "));
      if (start + size >= words.length) break;
    }
  }

  return [cleaned, ...sentences, ...windows]
    .map((value) => ({ raw: value, normalized: normalizeForSearch(value) }))
    .filter((candidate) => candidate.normalized.length >= 18)
    .filter((candidate, index, arr) => (
      arr.findIndex((other) => other.normalized === candidate.normalized) === index
    ))
    .sort((a, b) => b.normalized.length - a.normalized.length);
}

function indicesFromItemRange(items: RawTextItem[], range: { start: number; end: number }) {
  const result = new Set<number>();
  const start = Math.max(0, Math.floor(range.start));
  const end = Math.min(items.length - 1, Math.floor(range.end));

  for (let i = start; i <= end; i++) {
    if (items[i]?.str?.trim()) result.add(i);
  }

  return result;
}

function indicesFromSearchText(items: RawTextItem[], searchText: string): Set<number> {
  const pageText = buildNormalizedPageText(items);
  if (pageText.text.length === 0) return new Set();

  let bestMatch: { start: number; end: number } | null = null;
  for (const candidate of candidateTexts(searchText)) {
    const matchStart = pageText.text.indexOf(candidate.normalized);
    if (matchStart === -1) continue;

    const matchEnd = matchStart + candidate.normalized.length - 1;
    if (!bestMatch || matchEnd - matchStart > bestMatch.end - bestMatch.start) {
      bestMatch = { start: matchStart, end: matchEnd };
    }
  }

  if (!bestMatch) return new Set();

  const result = new Set<number>();
  for (let i = bestMatch.start; i <= bestMatch.end; i++) {
    result.add(pageText.charToItem[i]);
  }

  return result;
}

// ─────────────────────────────────────────────
// Step 2: convert matched items → pixel rects
// ─────────────────────────────────────────────

async function computeHighlightRects(
  pdfDoc: PDFDocumentProxy,
  pageNumber: number,
  searchText: string,
  itemRange: { start: number; end: number } | null,
  scale: number
): Promise<HighlightRect[]> {
  const page = await pdfDoc.getPage(pageNumber);
  const viewport = page.getViewport({ scale });
  const textContent = await page.getTextContent();
  const items = textContent.items as RawTextItem[];

  const rangeIndices = itemRange ? indicesFromItemRange(items, itemRange) : new Set<number>();
  const indices = rangeIndices.size > 0 ? rangeIndices : indicesFromSearchText(items, searchText);
  const rects: HighlightRect[] = [];

  for (let i = 0; i < items.length; i++) {
    if (!indices.has(i)) continue;
    const item = items[i];
    if (!item.transform) continue;

    const itemH = (item.height ?? Math.abs(item.transform[3])) || 10;
    const itemW = item.width ?? Math.max((item.str ?? "").length * itemH * 0.45, 1);

    const [x1, y1, x2, y2] = viewport.convertToViewportRectangle([
      item.transform[4],
      item.transform[5],
      item.transform[4] + itemW,
      item.transform[5] + itemH,
    ]);

    rects.push({
      x: Math.min(x1, x2),
      y: Math.min(y1, y2) - 1,
      w: Math.abs(x2 - x1),
      h: Math.abs(y2 - y1) + 2,
    });
  }

  return rects;
}

// ─────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────

export default function PdfViewer({ pdfUrl, highlightPage, highlightText, highlightItemRange, highlightKey }: PdfViewerProps) {
  const [numPages, setNumPages] = useState(0);
  const [inputPage, setInputPage] = useState("1");
  const [scale, setScale] = useState(1.75);
  const [highlightRects, setHighlightRects] = useState<HighlightRect[]>([]);

  const pdfDocRef = useRef<PDFDocumentProxy | null>(null);
  const virtuosoRef = useRef<VirtuosoHandle>(null);

  useEffect(() => {
    if (!highlightPage || !highlightText || !pdfDocRef.current) {
      setHighlightRects([]);
      return;
    }
    computeHighlightRects(pdfDocRef.current, highlightPage, highlightText, highlightItemRange, scale)
      .then(setHighlightRects)
      .catch(() => setHighlightRects([]));
  }, [highlightPage, highlightText, highlightItemRange, highlightKey, scale]);

  useEffect(() => {
    if (!highlightPage || highlightPage < 1 || highlightPage > numPages) return;
    virtuosoRef.current?.scrollToIndex({ index: highlightPage - 1, align: "start", behavior: "auto" });
  }, [highlightPage, highlightKey, numPages]);

  function onDocumentLoadSuccess(pdf: PDFDocumentProxy) {
    setNumPages(pdf.numPages);
    pdfDocRef.current = pdf;
  }

  function updateVisiblePage(page: number) {
    setInputPage(page.toString());
  }

  function zoom(amount: number) {
    setScale(prev => Math.min(Math.max(prev + amount, 0.5), 2.0));
  }

  function handlePageSubmit(e: React.FormEvent) {
    e.preventDefault();
    let page = parseInt(inputPage, 10);
    if (isNaN(page) || page < 1) page = 1;
    if (page > numPages) page = numPages;
    setInputPage(page.toString());
    virtuosoRef.current?.scrollToIndex({ index: page - 1, align: "start", behavior: "auto" });
  }

  return (
    <div className="flex flex-col h-full bg-black overflow-hidden">

      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-3 bg-black border-b border-white/5 sticky top-0 shrink-0 z-10">
        <div className="flex items-center text-sm text-gray-400">
          {numPages > 0 ? (
            <form onSubmit={handlePageSubmit} className="flex items-center gap-2">
              <input
                type="number" value={inputPage}
                onChange={e => setInputPage(e.target.value)}
                onBlur={handlePageSubmit}
                className="w-12 bg-[#111] border border-white/10 rounded px-1 py-0.5 text-center text-gray-300 focus:outline-none focus:border-white/20"
                min={1} max={numPages}
              />
              <span>/ {numPages} pages</span>
            </form>
          ) : <span>Loading...</span>}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => zoom(-0.25)} disabled={scale <= 0.5}
            className="px-3 py-1 text-sm text-gray-400 bg-[#111] border border-white/10 rounded hover:bg-[#202020] disabled:opacity-40 transition-colors">−</button>
          <span className="text-sm text-gray-500 w-12 text-center font-medium">{Math.round(scale * 100)}%</span>
          <button onClick={() => zoom(0.25)} disabled={scale >= 2.0}
            className="px-3 py-1 text-sm text-gray-400 bg-[#111] border border-white/10 rounded hover:bg-[#202020] disabled:opacity-40 transition-colors">+</button>
        </div>
      </div>

      {/* PDF Document */}
      <div className="flex-1 overflow-hidden">
        <Document file={pdfUrl} onLoadSuccess={onDocumentLoadSuccess}
          loading={<div className="flex items-center justify-center h-64"><span className="animate-pulse text-gray-500">Loading PDF...</span></div>}
          className="h-full">
          {numPages > 0 && (
            <Virtuoso
              ref={virtuosoRef}
              totalCount={numPages}
              rangeChanged={({ startIndex }) => updateVisiblePage(startIndex + 1)}
              style={{ height: "100%" }}
              itemContent={(index) => {
                const pageNumber = index + 1;
                const isHighlighted = pageNumber === highlightPage;
                return (
                  <div className="flex justify-center py-6 px-2">
                    <div className="shadow-lg border border-gray-200 bg-white inline-block relative">
                      <Page
                        pageNumber={pageNumber}
                        scale={scale}
                        renderTextLayer={true}
                        renderAnnotationLayer={true}
                      />

                      {/* Pixel-perfect highlight overlay */}
                      {isHighlighted && highlightRects.map((rect, i) => (
                        <div
                          key={i}
                          style={{
                            position: "absolute",
                            left: rect.x,
                            top: rect.y,
                            width: rect.w,
                            height: rect.h,
                            backgroundColor: "rgba(253, 224, 71, 0.55)",
                            mixBlendMode: "multiply",
                            pointerEvents: "none",
                            borderRadius: 2,
                            zIndex: 10,
                          }}
                        />
                      ))}
                    </div>
                  </div>
                );
              }}
            />
          )}
        </Document>
      </div>
    </div>
  );
}
