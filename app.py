from model_utils import gemini_image_prompt
import os
import io
import base64
from pathlib import Path
import streamlit as st
from PIL import Image
import fitz

from model_utils import (
    create_vectordb,
    load_vector_db,
    save_vectordb,
    retrieve_doc,
)
import numpy as np
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)


def extract_images_base64_only(pdf_path):
    image_data_store = {}
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_id = f"page_{i}_img_{img_index}"
                image_data_store[image_id] = img_base64
            except Exception:
                continue
    return image_data_store


def decode_base64_image(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def build_page_to_image_map(image_data_store):
    page_to_image_ids = {}
    for image_id in image_data_store.keys():
        try:
            # image_id format: "page_{i}_img_{j}"
            page_idx = int(image_id.split("_")[1])
        except Exception:
            continue
        page_to_image_ids.setdefault(page_idx, []).append(image_id)
    return page_to_image_ids


def build_context_text(docs):
    parts = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        doc_type = meta.get("type")
        page_num = meta.get("page")
        if doc_type == "text" and d.page_content:
            content = d.page_content.strip()
            if len(content) > 1200:
                content = content[:1200] + "..."
            parts.append(f"[page {page_num}] {content}")
        elif doc_type == "image":
            image_id = meta.get("image_id")
            parts.append(f"[page {page_num}] [Image: {image_id}]")
    return "\n\n".join(parts)


def render_pdf(pdf_file_path):
    try:
        with open(pdf_file_path, "rb") as f:
            pdf_bytes = f.read()
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_display = f"""
            <iframe
                src="data:application/pdf;base64,{b64_pdf}#view=FitH"
                width="100%"
                height="700px"
                type="application/pdf">
            </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.download_button("Download PDF", data=pdf_bytes, file_name=Path(
            pdf_file_path).name, mime="application/pdf")
    except Exception as e:
        st.warning(f"Unable to display PDF: {e}")


st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal RAG with Gemini")


if "llm" not in st.session_state:
    api_key = st.secrets["GOOGLE_API_KEY"]
    st.session_state.llm = GoogleGenerativeAI(
        model="gemini-2.5-flash", api_key=api_key) if api_key else None


script_dir = os.path.dirname(os.path.abspath(__file__))
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "vector_db_path" not in st.session_state:
    st.session_state.vector_db_path = None

with st.sidebar:

    uploaded_file = st.file_uploader(
        "Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"], key="file_upload_sidebar", label_visibility="visible")
    if uploaded_file:
        # Clear chat automatically
        st.session_state["messages"] = []
        if uploaded_file.type == "application/pdf":
            temp_pdf_path = os.path.join(
                script_dir, "pdfs", uploaded_file.name)
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.read())
            st.session_state.pdf_path = temp_pdf_path
            st.session_state.vector_db_path = os.path.join(
                script_dir, "outputs", "vectorstore_" + Path(temp_pdf_path).stem)
            st.success(f"PDF '{uploaded_file.name}' loaded!", icon="📄")
            # Build index only when PDF is uploaded
            all_docs, all_embeddings, image_data_store_full = process_pdf(
                temp_pdf_path)
            embeddings_array = np.array(all_embeddings)
            docs_with_embeddings = list(zip(all_docs, embeddings_array))
            vector_db = create_vectordb(docs_with_embeddings)
            save_vectordb(vector_db, st.session_state.vector_db_path)
            st.session_state.image_data_store = extract_images_base64_only(
                temp_pdf_path)
            st.session_state.page_to_image_ids = build_page_to_image_map(
                st.session_state.image_data_store)
        else:
            # Image uploaded, store in session for one-time use, do not trigger model
            st.session_state["uploaded_image_bytes"] = uploaded_file.read()
            st.success(
                f"Image '{uploaded_file.name}' uploaded! Now type your question and press Send.", icon="🖼️")
    if st.button("Remove PDF", key="remove_pdf_btn_sidebar"):
        st.session_state.pdf_path = None
        st.session_state.vector_db_path = None
        st.session_state.image_data_store = {}
        st.session_state.page_to_image_ids = {}
        st.session_state.messages = []
        st.info("PDF removed. You can upload a new PDF or use image-only mode.")
    show_pdf = st.checkbox("View PDF", value=False)


# Only load vector_db if a PDF is uploaded
vector_db = None
if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
    vector_db = load_vector_db(st.session_state.vector_db_path)


if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Chat display: bottom-to-top (latest at bottom, like ChatGPT) ---
chat_placeholder = st.container()
with chat_placeholder:
    st.markdown('<div style="min-height:40px;"></div>',
                unsafe_allow_html=True)  # Top spacing
    st.markdown('<div style="margin-bottom:80px;"></div>',
                unsafe_allow_html=True)  # Bottom padding for chat container
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# --- Chat input bar fixed at the bottom with proper spacing ---
st.markdown("""
    <style>
    /* Main container modifications */
    div.block-container {
        padding-bottom: 100px !important;
    }

    /* Ensure chat messages don't go behind input */
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
        padding-bottom: 100px;
    }

    /* Chat input bar styling */
    .element-container:has([data-testid="stTextInput"]) {
        position: fixed !important;
        bottom: 0px !important;
        left: 15.625rem !important; /* Match Streamlit's sidebar width */
        right: 0 !important;
        width: auto !important; /* Use right:0 instead of width:100% */
        background: white !important;
        border-top: 1px solid #e0e0e0;
        z-index: 999;
        padding: 1rem 1.5rem !important;
        margin: 0 !important;
        box-shadow: 0 -4px 10px rgba(0,0,0,0.1);
    }
    
    /* Adjust for collapsed sidebar */
    [data-testid="collapsedControl"] ~ div .element-container:has([data-testid="stTextInput"]) {
        left: 3.375rem !important;
    }

    /* Ensure chat container scrolls properly */
    section[data-testid="stSidebar"] {
        z-index: 1000;
    }

    /* Hide duplicate elements */
    .element-container:has([data-testid="stTextInput"]) + .element-container:has([data-testid="stTextInput"]) {
        display: none !important;
    }

    /* Ensure proper spacing at bottom of chat */
    [data-testid="ChatMessageInputs"] {
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([6, 1])  # Create columns with ratio 6:1
with col1:
    user_input = st.text_input(
        "Type your message...",
        key="chat_input_box",
        label_visibility="collapsed"
    )

# --- Trigger send ---
should_respond = user_input and user_input.strip() != ""

if "main_chat_input_value" not in st.session_state:
    st.session_state["main_chat_input_value"] = ""


def clear_input():
    st.session_state["main_chat_input_value"] = ""


if should_respond:
    user_msg = user_input
    st.session_state["main_chat_input_value"] = ""  # Clear input after send
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    answer_text = None
    with st.spinner("Generating answer..."):
        uploaded_image_bytes = st.session_state.get(
            "uploaded_image_bytes", None)
        if uploaded_image_bytes:
            try:
                img = Image.open(io.BytesIO(uploaded_image_bytes))
                max_width = 120
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                    img = img.resize(new_size)
                st.image(img, caption="Uploaded Image",
                         use_container_width=False)

                if st.session_state.llm is None:
                    answer_text = "LLM is not configured. Please set GOOGLE_API_KEY."
                else:
                    with io.BytesIO(uploaded_image_bytes) as img_file:
                        answer_text = gemini_image_prompt(
                            img_file, user_input or "Describe this image")
            except Exception as e:
                answer_text = f"Image Vision API error: {e}"

            st.session_state["uploaded_image_bytes"] = None

        elif st.session_state.get('pdf_path') and os.path.exists(st.session_state.get('pdf_path', '')) and vector_db is not None:
            try:
                retrieved_docs = retrieve_doc(
                    user_input or "image", vector_db)
                context_text = build_context_text(retrieved_docs)
                if st.session_state.llm is None:
                    answer_text = "LLM is not configured. Please set GOOGLE_API_KEY in your environment."
                else:
                    prompt = f"""You are a helpful assistant. Use the context below to answer the question as best as possible. If the answer is not in the context, you can use your own knowledge, but prefer the context when possible.\n\n<context>\n{context_text}\n</context>\n\nQuestion: {user_input}"""
                    resp = st.session_state.llm.invoke(prompt)
                    answer_text = getattr(resp, "content", str(resp))
            except Exception as e:
                answer_text = f"LLM error: {e}"
        else:
            try:
                if st.session_state.llm is None:
                    answer_text = "LLM is not configured. Please set GOOGLE_API_KEY in your environment."
                else:
                    resp = st.session_state.llm.invoke(user_input)
                    answer_text = getattr(resp, "content", str(resp))
            except Exception as e:
                answer_text = f"LLM error: {e}"

    if answer_text is not None:
        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text})
        with st.chat_message("assistant"):
            st.markdown(answer_text)
