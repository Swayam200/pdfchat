// frontend/app/lib/pdfWorker.ts
import { pdfjs } from 'react-pdf';

// We must set the workerSrc for pdf.js to function. 
// We use the unpkg CDN for convenience, matching the exact version of pdfjs installed by react-pdf.
// A Web Worker allows pdf.js to do the heavy lifting of parsing the PDF on a background thread
// so the UI doesn't freeze.
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;
