# CNC Troubleshooter AI

Beginner-friendly CNC machine error troubleshooting assistant using:

- Streamlit
- Groq API
- FAISS
- Sentence Transformers
- PyMuPDF
- GitHub + Streamlit Community Cloud

## How it works

1. Upload the CNC machine manual PDF.
2. The app extracts the manual text.
3. The text is split into searchable sections.
4. Sentence Transformers creates embeddings.
5. FAISS retrieves the most relevant manual sections.
6. Groq explains the error in beginner-friendly language.
7. The app separates safe operator checks from technician-level work.

## GitHub files

The minimum deployment repository contains:

```text
cnc-troubleshooter/
├── app.py
├── requirements.txt
└── README.md
```

## Local setup

Create a virtual environment, then:

```bash
pip install -r requirements.txt
```

Set your Groq API key as an environment variable.

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
streamlit run app.py
```

## Streamlit Cloud setup

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Deploy the repository on Streamlit Community Cloud.
4. In the app's Settings / Secrets area, add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Do NOT put the API key in `app.py` or commit it to GitHub.

## Important limitation

The MVP processes the uploaded manual during the current app session. A production
version should pre-process manuals and store the FAISS index plus metadata so users
do not have to rebuild the index every session.

For scanned/image-only manuals, OCR should be added in a later version.

## Safety

This is an AI troubleshooting assistant, not a replacement for the machine
manufacturer's manual or a qualified maintenance technician.

The assistant should never be used to bypass safety interlocks, emergency stops,
electrical protection, or other machine safety systems.
