# CNC Troubleshooter AI — Beginner RAG MVP

Beginner-friendly CNC alarm assistant using a machine manual as the main RAG source.

## Workflow

Machine manual PDF → PyMuPDF → chunks → Sentence Transformers → FAISS → Groq → short answer

## Language

The user can choose:
- English
- Roman Urdu

The answer is intentionally short:
1. What it means
2. What to do
3. Call maintenance
4. Do not do
5. If it still happens

The app does not continue a long troubleshooting conversation.

## Models

- Groq API
- OpenAI GPT OSS 120B: `openai/gpt-oss-120b`
- `all-MiniLM-L6-v2` for embeddings
- FAISS for vector retrieval

## GitHub files

```text
app.py
requirements.txt
README.md
```

## Streamlit Cloud

Upload the files to a GitHub repository. In Streamlit Community Cloud select the repository and `app.py`.

In **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Never commit the real API key to GitHub.

## Notes

- Manual processing is session-based in this MVP.
- Scanned PDFs may require OCR in a later version.
- For production, add persistent manual indexes, exact alarm-code retrieval, OCR, feedback storage, and multiple machine manuals.

## Safety

This is a support tool, not a replacement for qualified maintenance personnel. The AI must not instruct beginners to open live electrical cabinets, bypass safety circuits/interlocks, defeat emergency stops, or work on live equipment.
