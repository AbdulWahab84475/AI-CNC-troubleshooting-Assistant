import os, re, hashlib
from typing import List, Dict
import faiss
import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

APP_TITLE = "CNC Troubleshooter AI"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title=APP_TITLE, page_icon="⚙️", layout="centered")

GENERAL_KNOWLEDGE = [
    {"text": "A servo alarm means the servo system detected a problem. The exact cause depends on the alarm number and machine manual."},
    {"text": "An overtravel or stroke-end condition means an axis reached a machine travel limit. The control may prevent movement for safety."},
    {"text": "A spindle alarm means the spindle system detected a problem. The exact cause should be checked against the machine manual and alarm code."},
    {"text": "An emergency-stop condition means the machine entered a safety stop. Never bypass or defeat a safety circuit."},
    {"text": "A tool-change alarm means the automatic tool-change sequence did not complete normally. The exact cause depends on the alarm and manual."},
]

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def general_index():
    model = load_embedding_model()
    texts = [x["text"] for x in GENERAL_KNOWLEDGE]
    v = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    idx = faiss.IndexFlatIP(v.shape[1])
    idx.add(v)
    return idx, texts

def groq_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")

def groq_client():
    key = groq_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured. Add it in Streamlit Secrets.")
    return Groq(api_key=key)

def extract_pages(uploaded):
    doc = fitz.open(stream=uploaded.getvalue(), filetype="pdf")
    out = []
    for n, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if text:
            out.append({"page": n, "text": text})
    doc.close()
    return out

def clean(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_pages(pages, size=1800, overlap=250):
    chunks = []
    for p in pages:
        text = clean(p["text"])
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            part = text[start:end].strip()
            if part:
                chunks.append({"page": p["page"], "text": part})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks

def build_index(chunks):
    model = load_embedding_model()
    texts = [x["text"] for x in chunks]
    v = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    idx = faiss.IndexFlatIP(v.shape[1])
    idx.add(v)
    return idx

def retrieve(query, chunks, idx, k=5):
    if not chunks or idx is None:
        return []
    model = load_embedding_model()
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    scores, ids = idx.search(q, min(k, len(chunks)))
    out = []
    for score, i in zip(scores[0], ids[0]):
        if i >= 0:
            r = dict(chunks[i]); r["score"] = float(score); out.append(r)
    return out

def retrieve_general(query, k=2):
    idx, texts = general_index()
    model = load_embedding_model()
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    scores, ids = idx.search(q, min(k, len(texts)))
    return [{"text": texts[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]

def fingerprint(uploaded):
    return hashlib.sha256(uploaded.getvalue()).hexdigest()

def make_prompt(language, error, when, behavior, observation, manual, general):
    manual_text = "\n\n".join(f"[Manual page {x['page']}]\n{x['text']}" for x in manual) or "No manual section retrieved."
    general_text = "\n\n".join(x["text"] for x in general)
    lang = (
        "Reply in simple English. Use short sentences and common words."
        if language == "English"
        else
        "Reply in simple Roman Urdu. Use easy everyday Roman Urdu, not Urdu script. Keep alarm names/codes in English."
    )
    return f"""
You are a professional CNC troubleshooting assistant for a BEGINNER operator.

{lang}

The uploaded machine manual is the PRIMARY source.
Never invent an exact alarm meaning.
If the manual does not support an exact diagnosis, say so.
Keep the answer SHORT.
Do not ask for more questions, screenshots, or details.
After the final guidance, stop. If the problem remains, tell the user to contact the relevant maintenance department or a qualified CNC technician.
Never tell a beginner to open an electrical cabinet, bypass an interlock, bypass a safety circuit, defeat an emergency stop, or work on live electrical equipment.

ERROR:
{error}

WHEN:
{when}

MACHINE BEHAVIOR:
{behavior}

OBSERVATION:
{observation or "None"}

MANUAL EVIDENCE:
{manual_text}

GENERAL FALLBACK:
{general_text}

Return ONLY:

### What it means
1-2 short sentences. Explain technical words simply.

### What to do
Only 1-3 safe checks a beginner can do without opening the machine or bypassing safety.

### Call maintenance
One short sentence saying when to contact maintenance.

### Do not do
One short sentence with the unsafe action to avoid.

### If it still happens
One short sentence:
"Stop here and contact the relevant maintenance department or a qualified CNC technician."
""".strip()

def ask_groq(prompt):
    response = groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Be concise, beginner-friendly, manual-grounded, and safety-first."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=900,
    )
    return response.choices[0].message.content.strip()

st.title("⚙️ CNC Troubleshooter AI")
st.caption("Don't understand the CNC alarm? Tell me exactly what the machine says.")

language = st.selectbox("Explanation language", ["English", "Roman Urdu"])
st.divider()

with st.expander("📘 Machine Manual", expanded=True):
    st.write("Upload the machine manual. The AI will use it as the main source.")
    uploaded = st.file_uploader("Upload PDF manual", type=["pdf"])
    if uploaded:
        fp = fingerprint(uploaded)
        if st.session_state.get("manual_fp") != fp:
            with st.spinner("Reading the manual..."):
                pages = extract_pages(uploaded)
                chunks = chunk_pages(pages)
                if chunks:
                    st.session_state["manual_fp"] = fp
                    st.session_state["manual_chunks"] = chunks
                    st.session_state["manual_index"] = build_index(chunks)
        if st.session_state.get("manual_chunks"):
            st.success(f"Manual ready: {len(st.session_state['manual_chunks'])} sections.")
        else:
            st.warning("No readable text was found. A scanned PDF may need OCR.")

st.divider()
st.subheader("1️⃣ Tell me the error")
error = st.text_area(
    "What exactly is shown on the CNC screen?",
    placeholder="Example: Servo alarm 18 - Main side encoder initial communication error",
    height=100,
)

st.subheader("2️⃣ What happened?")
when = st.selectbox("When did it happen?", [
    "I just turned the machine on", "During axis movement", "During machining",
    "During tool change", "During spindle operation",
    "During zero return / reference return", "Not sure"
])
behavior = st.selectbox("What did the machine do?", [
    "Machine stopped", "Axis stopped", "Spindle stopped",
    "Tool change stopped", "Machine will not move", "Not sure"
])
observation = st.text_input("Anything else you noticed? (optional)", placeholder="Example: X axis will not move")

if st.button("🔍 Explain the Error", type="primary", use_container_width=True):
    if not error.strip():
        st.error("Please enter the error shown on the machine.")
        st.stop()

    chunks = st.session_state.get("manual_chunks", [])
    idx = st.session_state.get("manual_index")
    query = f"Alarm/error: {error}\nWhen: {when}\nBehavior: {behavior}\nObservation: {observation}"
    manual = retrieve(query, chunks, idx, 5)
    general = retrieve_general(query, 2)

    try:
        with st.spinner("Checking the manual..."):
            answer = ask_groq(make_prompt(language, error.strip(), when, behavior, observation.strip(), manual, general))
        st.divider()
        st.subheader("3️⃣ Troubleshooting guide")
        st.markdown(answer)

        if manual:
            with st.expander("📖 Manual sections used"):
                for item in manual:
                    st.markdown(f"**Page {item['page']}** — relevance {item['score']:.2f}")
                    st.write(item["text"][:1200])
                    st.divider()
        else:
            st.info("No manual section was retrieved. This answer used general CNC knowledge only.")
    except Exception as exc:
        st.error(f"Could not generate the answer: {exc}")

st.divider()
st.caption("Safety note: Electrical, hydraulic, pneumatic, servo, and safety-system work should be handled by qualified maintenance personnel.")
