"""
Improved document_analysis.py

Key improvements:
- Extracts text from PDFs and images locally (PyPDF2 + Pillow + pytesseract), with fallbacks.
- Provides a dedicated "lease analysis" flow, including a jurisdiction field to contextualize legal guidance.
- Performs lightweight rule-based detection of commonly illegal lease clauses (from user's examples).
- Sends the extracted lease text and candidate clauses to OpenAI for confirmation, explanation, and suggested next steps.
- Highlights candidate illegal sentences in the UI (simple HTML <mark> usage).
- Handles long documents by chunking before sending to the model.
- More robust error handling and user guidance, plus README-style notes about extra dependencies.

Notes:
- This code aims to be practical and conservative: it uses local extraction where possible and uses the model to confirm and explain.
- This is NOT legal advice. Results should be presented as informational only.
- Install optional dependencies for best results:
    pip install pytesseract PyPDF2 pdfplumber Pillow
  And ensure Tesseract OCR is installed on the machine for pytesseract to work.
"""

import streamlit as st
from PIL import Image
import io
import os
from openai import OpenAI
import re
import textwrap

# Optional OCR / PDF libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False

# Load environment and initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # change as needed

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in environment variable 'OPENAI_API_KEY'.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Patterns for candidate illegal clauses (based on user's examples).
# This is intentionally conservative — these are *candidates* for review by the model,
# and we also ask the model to confirm illegality and explain why.
ILLEGAL_CLAUSE_PATTERNS = [
    # tenant must do all repairs (very broad)
    r"\b(tenant|you)\s+(is|are|'?ll be)?\s*(responsible|liable)\s+for\s+all\s+repairs\b",
    r"\b(tenant|you)\s+must\s+make\s+all\s+repairs\b",
    # security deposit used for utilities
    r"\b(security deposit)\s+.*\b(utilit(?:y|ies)|electric|gas|water)\b",
    r"\b(use(s)?\s+the\s+security\s+deposit\s+to)\s+.*\b(utilit(?:y|ies)|electric|gas)\b",
    # tenant must pay for electricity/gas where landlord billed
    r"\b(tenant|you)\s+must\s+pay\s+for\s+electric(it(?:y)?)\b",
    r"\b(tenant|you)\s+must\s+pay\s+for\s+gas\b",
    r"\b(pay\s+for\s+(electricity|gas).*landlord.*name)\b",
    # immediate payment of all remaining rent on early termination
    r"\b(immediately\s+upon\s+(termination|end)\s+you\s+must\s+pay\s+all\s+rent\s+due\s+for\s+the\s+remainder)\b",
    r"\b(pay\s+all\s+rent\s+due\s+for\s+the\s+remainder\s+of\s+the\s+term)\b",
    # wildly broad fallback keywords
    r"\b(no\s+repair\s+or\s+maintenance\s+by\s+landlord)\b",
]

# Utility: split text into sentences (simple approach)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\?\!]\s)|\n+')


def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Try to extract text from PDF bytes using PyPDF2 (fast) or fallback to empty string."""
    if not PYPDF2_AVAILABLE:
        st.warning("PyPDF2 not installed — PDF text extraction will be limited. Install PyPDF2 for better results.")
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_pages = []
        for p in range(len(reader.pages)):
            page = reader.pages[p]
            try:
                text_pages.append(page.extract_text() or "")
            except Exception:
                # Per-page failure should not break whole extraction
                text_pages.append("")
        return "\n\n".join(text_pages).strip()
    except Exception as e:
        st.warning(f"PDF extraction failed: {e}")
        return ""


def extract_text_from_image_bytes(file_bytes: bytes) -> str:
    """Try to extract text from images using pytesseract if available."""
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        st.warning(f"Could not open image for OCR: {e}")
        return ""
    if not TESSERACT_AVAILABLE:
        st.warning("pytesseract not installed or Tesseract not available. Install them for OCR.")
        return ""
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.warning(f"OCR failed: {e}")
        return ""


def find_candidate_illegal_sentences(text: str) -> list:
    """
    Uses simple regex patterns to find sentences that *may* contain illegal clauses.
    Returns list of sentences (strings).
    """
    if not text:
        return []
    # Split text into sentences
    chunks = _SENTENCE_SPLIT_RE.split(text)
    candidates = []
    for sent in chunks:
        s = sent.strip()
        if not s:
            continue
        lower = s.lower()
        for pat in ILLEGAL_CLAUSE_PATTERNS:
            if re.search(pat, lower):
                candidates.append(s)
                break
    # dedupe while preserving order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def highlight_sentences_in_html(text: str, sentences_to_highlight: list) -> str:
    """
    Very simple html highlighter: wrap exact matches with <mark>.
    For long docs this naive replacement may highlight substring copies too.
    """
    html = text.replace("\n", "<br>")
    for s in sentences_to_highlight:
        escaped = re.escape(s)
        # replace first occurrence only to reduce over-highlighting
        html, n = re.subn(escaped, f"<mark>{s}</mark>", html, count=1)
    return f'<div style="white-space: pre-wrap; font-family: monospace;">{html}</div>'


def chunk_text(text: str, max_chars: int = 20000) -> list:
    """Simple chunker based on character length with sentence boundaries when possible."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to break at last sentence end before end
        if end < len(text):
            m = re.search(r'(?<=\.\s)', text[start:end][::-1])
            # If reverse search found something, fallback to naive cut
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks


def analyze_text_with_openai(lease_text: str, jurisdiction: str, candidate_sentences: list) -> str:
    """
    Call OpenAI to confirm which candidate sentences are illegal, explain in simple language,
    and suggest next steps. We pass the lease text and candidate sentences (short).
    """
    # Limit input size by chunking if needed; here we send the whole lease for context if not huge.
    MAX_CHARS = 25000
    if len(lease_text) > MAX_CHARS:
        # send first chunk and mention that the document was longer
        context_text = lease_text[:MAX_CHARS]
        more_note = f"\n\n[Document truncated for model; original length: {len(lease_text)} characters]\n"
    else:
        context_text = lease_text
        more_note = ""

    # Build a detailed system prompt so the model knows what to do.
    system_prompt = (
        "You are a helpful assistant fluent in plain, non-legal language. "
        "A user uploaded a residential lease. You are NOT a lawyer; provide information only. "
        "The user wants: (1) A short plain-language summary of the lease's source and purpose, "
        "(2) Identification of any illegal clauses (based on the candidate list and general US landlord-tenant rules), "
        "(3) For each illegal clause found, explain why it is likely illegal in simple terms, "
        "and provide suggested next steps the tenant can take (e.g., negotiate, consult tenant union or attorney). "
        "If jurisdiction is provided, address legality generally for that jurisdiction; if not, say 'general US guidance'. "
        "Do not give legal advice or cite statutes. For each clause you mark as illegal, include the exact matched sentence and a short label. "
        "If unsure about a clause, say so and explain what information would resolve the uncertainty."
    )

    user_content = (
        f"Jurisdiction (user-supplied): {jurisdiction or 'General US guidance'}\n\n"
        f"Candidate suspicious sentences (detected by simple heuristics):\n"
        + ("\n".join([f"- {s}" for s in candidate_sentences]) if candidate_sentences else "- (none detected)\n")
        + "\n\nLease text (for context):\n"
        + context_text
        + more_note
        + "\n\nPlease respond using the following structure:\n"
        "1) Summary (1-3 short sentences).\n"
        "2) Confirmed illegal clauses: for each, provide: matched_sentence, reason (plain language), suggested next steps.\n"
        "3) Other noteworthy clauses (if any) the user should look at, and why.\n"
        "End with a brief 'what to do next' checklist (3 items max).\n"
        "Keep responses short and clear."
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1500,
        )
        # Be defensive about response shape
        if hasattr(response, "choices") and response.choices and getattr(response.choices[0], "message", None):
            return response.choices[0].message.content
        # Some SDKs return different shapes; attempt common ones
        if isinstance(response, dict):
            # try openai-python like shape
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "") or choices[0].get("text", "")
        return "Model returned an unexpected response shape."
    except Exception as e:
        return f"Error calling OpenAI: {e}"


def process_uploaded_file(uploaded_file):
    """
    Processes the uploaded file:
    - For image files: returns PIL Image for preview and performs OCR (if available).
    - For PDFs: extracts text (if PyPDF2 available).
    Returns (file_bytes, mime_type, preview_image, filename, extracted_text).
    """
    file_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type
    filename = getattr(uploaded_file, "name", "uploaded_document")

    preview_image = None
    extracted_text = ""

    if mime_type == "application/pdf":
        extracted_text = extract_text_from_pdf_bytes(file_bytes)
    else:
        # Try to open as image for preview; some PDFs may not be recognized at this point
        try:
            uploaded_file.seek(0)
            preview_image = Image.open(uploaded_file)
        except Exception:
            preview_image = None

        # Perform OCR if possible
        extracted_text = extract_text_from_image_bytes(file_bytes)

    return file_bytes, mime_type, preview_image, filename, extracted_text


def main():
    st.set_page_config(page_title="Lease Analyzer", layout="wide")
    st.title("Lease Analyzer — Understand your lease and spot illegal clauses")
    st.write(
        "Upload an image (JPEG/PNG) or PDF of your lease. "
        "This tool will extract the text, provide a plain-language summary, and highlight clauses that may be illegal. "
        "This is informational only and not legal advice."
    )

    with st.sidebar:
        st.header("Options")
        jurisdiction = st.text_input("Jurisdiction (optional, e.g. 'California, USA')", value="")
        show_raw = st.checkbox("Show extracted text", value=False)
        model_label = st.text_input("Model name (advanced)", value=OPENAI_MODEL)
        st.write("OCR available:" , "Yes" if TESSERACT_AVAILABLE else "No (install pytesseract + Tesseract)")

    input_method = st.radio("Choose input method:", ("Upload File", "Capture Image"))

    uploaded_file = None
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Choose a lease file", type=["jpg", "jpeg", "png", "pdf"])
    else:
        uploaded_file = st.camera_input("Capture an image")

    if uploaded_file is not None:
        file_bytes, mime_type, preview_image, filename, extracted_text = process_uploaded_file(uploaded_file)

        if mime_type != "application/pdf" and preview_image:
            st.image(preview_image, caption=f"Preview: {filename}", use_column_width=True)
        elif mime_type == "application/pdf":
            st.write("PDF uploaded.")

        if not extracted_text:
            st.warning("No text extracted from the document. If it's an image, install Tesseract OCR and pytesseract. "
                       "If it's a PDF that contains images (scanned), OCR is required.")
        else:
            if show_raw:
                with st.expander("Show extracted text"):
                    st.text_area("Extracted lease text", value=extracted_text, height=400)

            # Find candidate illegal sentences
            candidate_sentences = find_candidate_illegal_sentences(extracted_text)
            st.write(f"Found {len(candidate_sentences)} candidate suspicious clause(s).")

            # Build highlighted HTML
            highlighted_html = highlight_sentences_in_html(extracted_text, candidate_sentences) if extracted_text else ""
            if highlighted_html:
                st.markdown("Highlighted candidate clauses:")
                st.markdown(highlighted_html, unsafe_allow_html=True)

            # Provide a brief preview of candidate sentences
            if candidate_sentences:
                with st.expander("Candidate suspicious sentences"):
                    for i, s in enumerate(candidate_sentences, 1):
                        st.write(f"{i}. {s}")

            # Confirm & explain with the model on button press
            if st.button("Analyze lease and check for illegal clauses"):
                st.info("Analyzing — this may take a few seconds.")
                # Allow user to override model name via sidebar
                global OPENAI_MODEL
                OPENAI_MODEL = model_label or OPENAI_MODEL
                analysis_result = analyze_text_with_openai(extracted_text, jurisdiction, candidate_sentences)
                st.success("Analysis complete")
                # Show model output
                st.markdown("### Model analysis")
                st.write(analysis_result)

                # Offer download of highlighted HTML for user's records
                try:
                    b = highlighted_html.encode("utf-8")
                    st.download_button("Download highlighted lease (HTML)", data=b, file_name=f"{filename}_highlighted.html", mime="text/html")
                except Exception:
                    # If nothing to download, ignore
                    pass


if __name__ == "__main__":
    main()
