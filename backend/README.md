---
title: Pdfchat
emoji: 📄
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
---

FastAPI backend for pdfchat.

This Space runs only the Python API. The Next.js frontend should be deployed
separately, for example on Vercel, and configured with:

```txt
NEXT_PUBLIC_API_URL=https://swayam200-pdfchat.hf.space
```

The backend keeps uploaded PDF text and vector search state in memory. On free
Hugging Face CPU hardware, the Space can restart or sleep, so users may need to
upload the PDF again after a restart.
