import os
import re
import hashlib
import tempfile
from pathlib import Path

import faiss
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="CNC Troubleshooter AI",
    page_icon="⚙️",
    layout="wide",
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


# -----------------------------
# General CNC fallback knowledge
# -----------------------------
GENERAL_KNOWLEDGE = [
    """CNC servo alarms can be related to excessive motor load, mechanical
    obstruction, insufficient lubrication, ball screw or guideway resistance,
    encoder feedback, damaged cables, servo amplifier faults, or excessive
    acceleration/deceleration. First identify the affected axis and whether
    the alarm occurs during cutting, rapid movement, jog, or startup.""",

    """CNC overtravel alarms can be caused by commanded movement beyond a
    travel limit, incorrect work offsets, tool offsets, machine coordinates,
    limit switch activation, or reference position problems. Do not bypass
    physical limits. Follow the machine manufacturer's procedure for safely
    moving away from a limit.""",

    """CNC spindle overload can be caused by excessive depth of cut, feed,
    unsuitable spindle speed, worn tooling, poor workholding, chip evacuation
    problems, or spindle mechanical/drive problems. Determine whether the
    overload happens only during cutting or also without cutting load.""",

    """ATC tool-change failures can involve spindle orientation, tool
    clamp/unclamp confirmation, magazine position, tool-change arm position,
    sensors, pneumatic/hydraulic actuators, or PLC sequence conditions.
    Follow the machine manufacturer's procedure before manually moving ATC
    components.""",

    """Low hydraulic pressure can be caused by low oil level, pump problems,
    filter restriction, leakage, incorrect adjustment, worn components,
    temperature/viscosity effects, or a faulty pressure switch. Do not adjust
    pressure beyond the manufacturer's specified value.""",

    """Low pneumatic pressure can prevent tool changes, chuck operation,
    doors, clamps, or other machine functions. Check the machine air-pressure
    indication, plant air supply, regulator/filter condition, and obvious leaks
    according to the manufacturer's procedure.""",

    """Coolant failure can be caused by low coolant level, blocked filters or
    nozzles, pump failure, electrical problems, or coolant delivery blockage.
    Check level, pump operation, filter/nozzle condition, and flow. Avoid
    running a pump dry when the pump is not designed for it.""",

    """Lubrication alarms can indicate low oil, lubrication pump failure,
    blocked lines, pressure problems, leakage, or a faulty sensor/switch.
    Do not ignore lubrication alarms because insufficient lubrication can
    damage guideways, ball screws, and other moving components.""",

    """Poor surface finish can be related to tool wear, chatter, excessive
    feed, unsuitable spindle speed, tool runout, poor rigidity, coolant
    problems, backlash, or mechanical condition. Diagnose systematically
    rather than changing many parameters at once.""",

    """Chatter is unwanted machining vibration. Common causes include
    insufficient rigidity, excessive tool overhang, unsuitable spindle speed,
    excessive engagement, worn tooling, poor workholding, or resonance.
    Improve rigidity and change cutting conditions systematically.""",
]


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    api_key = api_key or os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None


def extract_pdf_pages(uploaded_file):
    """Extract text page-by-page so manual page references can be shown."""
    data = uploaded_file.getvalue()
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_number, "text": text})
    doc.close()
    return pages


def chunk_pages(pages, chunk_size=1800, overlap=250):
    """Create overlapping chunks while preserving page numbers."""
    chunks = []
    for item in pages:
        text = re.sub(r"\s+", " ", item["text"]).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) >= 80:
                chunks.append({
                    "text": chunk,
                    "page": item["page"],
                    "source": "Uploaded machine manual",
                })
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


@st.cache_resource(show_spinner=False)
def build_index(texts):
    model = load_embedding_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(query, chunks, index, top_k=5):
    model = load_embedding_model()
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(vector, k)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunks):
            result = dict(chunks[idx])
            result["score"] = float(score)
            results.append(result)

    return results


def manual_fingerprint(uploaded_file):
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def build_general_index():
    return build_index(GENERAL_KNOWLEDGE)


def retrieve_general(query, top_k=3):
    index = build_general_index()
    chunks = [
        {"text": x, "page": None, "source": "General CNC knowledge"}
        for x in GENERAL_KNOWLEDGE
    ]
    return retrieve(query, chunks, index, top_k=top_k)


def make_prompt(user_error, simple_answers, manual_results, general_results):
    manual_context = "\n\n".join(
        [
            f"[MANUAL SOURCE {i+1} | Page {r['page']} | similarity {r['score']:.3f}]\n{r['text']}"
            for i, r in enumerate(manual_results)
        ]
    ) or "No relevant manual passage was retrieved."

    general_context = "\n\n".join(
        [
            f"[GENERAL CNC SOURCE {i+1}]\n{r['text']}"
            for i, r in enumerate(general_results)
        ]
    ) or "No general fallback knowledge was retrieved."

    return f"""
You are CNC Troubleshooter AI, an assistant for a BEGINNER CNC MACHINE OPERATOR.

The operator may know almost nothing about CNC troubleshooting. They may only
copy the alarm/error text from the machine screen. Your job is to explain the
problem in simple language and guide them safely.

IMPORTANT:
- The uploaded machine manual is the PRIMARY technical source.
- Prefer the manual over general knowledge.
- Never invent an exact alarm-code meaning.
- Alarm numbers can differ between machine manufacturers/controllers.
- If the manual does not clearly identify the alarm, say that it could not be
  confirmed from the supplied manual.
- Do not claim certainty.
- Do not tell the user to bypass interlocks or safety devices.
- Do not instruct a beginner to open an electrical cabinet, work on live
  electrical systems, change protected parameters, alter PLC logic, or adjust
  hydraulic/pneumatic settings unless the manual explicitly gives a safe
  operator procedure.
- Separate simple operator checks from technician-only checks.
- Explain technical terms in plain language.
- Do not overwhelm the user with unnecessary technical details.
- Ask for only a few useful follow-up details.

USER'S ERROR:
{user_error}

SIMPLE USER ANSWERS:
{simple_answers}

RETRIEVED MACHINE MANUAL:
{manual_context}

GENERAL CNC FALLBACK KNOWLEDGE:
{general_context}

Return the answer in this format:

## ⚠️ What does this error mean?
Explain it in beginner-friendly language. If it cannot be confirmed from the
manual, say so clearly.

## 🔎 What could be causing it?
Give 3 to 5 likely causes, only when supported by the manual or reasonable
general CNC knowledge. Keep each explanation simple.

## 🛠️ What should I do now?
Give a numbered sequence. Start with safe, simple checks.

## 🟢 Checks I can do safely
List simple observations/checks suitable for a beginner/operator.

## 🟡 When should I call maintenance?
Explain when a technician is needed.

## 🔴 Do not do this yourself
Mention only relevant unsafe actions.

## 📖 Manual reference
Give the relevant manual page numbers when available. If no exact page can
be confirmed, say "No exact manual page identified."

## ❓ If the problem is still there
Ask up to 3 simple follow-up questions that would materially improve diagnosis.

Keep the response practical and concise.
"""


def ask_groq(prompt):
    client = get_groq_client()
    if not client:
        return None

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=2200,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful senior CNC troubleshooting assistant. "
                    "The user is a beginner. Be practical, conservative and safety-focused."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# =============================
# UI
# =============================
st.title("⚙️ CNC Troubleshooter AI")
st.write("Don't understand the CNC alarm? **Tell me exactly what the machine says.**")

with st.sidebar:
    st.header("📚 Machine Manual")

    manual = st.file_uploader(
        "Upload the machine manual (PDF)",
        type=["pdf"],
        help="The manual is used as the primary RAG knowledge source.",
    )

    if manual:
        fingerprint = manual_fingerprint(manual)

        if st.session_state.get("manual_fingerprint") != fingerprint:
            with st.spinner("Reading your machine manual..."):
                pages = extract_pdf_pages(manual)
                chunks = chunk_pages(pages)

                if chunks:
                    index = build_index([c["text"] for c in chunks])
                    st.session_state["manual_fingerprint"] = fingerprint
                    st.session_state["manual_chunks"] = chunks
                    st.session_state["manual_index"] = index
                    st.session_state["manual_name"] = manual.name
                    st.success(
                        f"Manual ready: {len(pages)} pages, {len(chunks)} searchable sections."
                    )
                else:
                    st.session_state.pop("manual_fingerprint", None)
                    st.error(
                        "No readable text was found. This PDF may be scanned/image-only."
                    )
        else:
            st.success(
                f"Manual ready: {st.session_state.get('manual_name', 'PDF')}"
            )

    st.divider()
    st.caption("AI uses the uploaded manual as the primary source.")
    st.caption("Never bypass machine safety systems.")


st.header("1️⃣ Tell me the error")

error_text = st.text_area(
    "What exactly is shown on the CNC screen?",
    placeholder="Example: 414 SERVO ALARM",
    height=100,
)

st.header("2️⃣ Help me understand what happened")

col1, col2 = st.columns(2)

with col1:
    when = st.radio(
        "When did it happen?",
        [
            "When starting the machine",
            "While moving",
            "During machining",
            "During tool change",
            "During spindle operation",
            "I'm not sure",
        ],
    )

with col2:
    machine_behavior = st.radio(
        "What happened to the machine?",
        [
            "Machine stopped",
            "An axis stopped",
            "Spindle stopped",
            "Tool change stopped",
            "Other",
            "I'm not sure",
        ],
    )

extra = st.text_input(
    "Anything else you noticed? (optional)",
    placeholder="Example: X axis was moving when the alarm appeared.",
)

if st.button("🔍 Explain My Error", type="primary", use_container_width=True):
    if not error_text.strip():
        st.warning("Please enter the error exactly as it appears on the machine.")
        st.stop()

    manual_chunks = st.session_state.get("manual_chunks", [])
    manual_index = st.session_state.get("manual_index")

    query = f"""
    CNC alarm/error: {error_text}
    Timing: {when}
    Machine behavior: {machine_behavior}
    Additional observation: {extra}
    """

    with st.spinner("Checking the machine manual..."):
        if manual_chunks and manual_index is not None:
            manual_results = retrieve(
                query, manual_chunks, manual_index, top_k=5
            )
        else:
            manual_results = []

        general_results = retrieve_general(query, top_k=3)

    simple_answers = f"""
    When: {when}
    Machine behavior: {machine_behavior}
    Extra observation: {extra or "None"}
    """

    prompt = make_prompt(
        error_text,
        simple_answers,
        manual_results,
        general_results,
    )

    with st.spinner("Preparing a simple troubleshooting explanation..."):
        answer = ask_groq(prompt)

    if not answer:
        st.error(
            "Groq API key is not configured. Add GROQ_API_KEY to Streamlit Secrets."
        )
        st.stop()

    st.header("3️⃣ Your troubleshooting guide")
    st.markdown(answer)

    if manual_results:
        with st.expander("📖 See the manual sections used by AI"):
            for r in manual_results:
                st.markdown(
                    f"**Page {r['page']} — similarity {r['score']:.3f}**"
                )
                st.write(r["text"][:1200] + ("..." if len(r["text"]) > 1200 else ""))
                st.divider()
    else:
        st.info(
            "No machine manual was uploaded. The answer used general CNC knowledge only. "
            "For machine-specific alarm meanings, upload the machine manual."
        )

    st.divider()
    st.subheader("👍 Was this helpful?")
    feedback = st.radio(
        "Your feedback",
        ["Yes", "Partly", "No"],
        horizontal=True,
        key="feedback_radio",
    )
    feedback_note = st.text_input(
        "Optional: what was the actual cause?",
        placeholder="Example: Technician found a damaged encoder cable.",
        key="feedback_note",
    )

    if st.button("Save feedback"):
        st.success(
            "Feedback captured for this session. A future version can store "
            "these cases in a troubleshooting history."
        )

st.divider()
st.caption(
    "CNC Troubleshooter AI • FAISS RAG • Groq • Streamlit"
)
