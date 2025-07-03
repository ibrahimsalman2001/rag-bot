import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()  # Load secrets from .env or Streamlit Cloud

# ----------- Function to Generate Text Using DeepSeek via OpenRouter ----------- #
def generate_text_from_deepseek(query, retrieved_docs, temperature=0.7):
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        return "❌ OPENROUTER_API_KEY not set."

    context = "\n".join(retrieved_docs)
    prompt = f"Query: {query}\nContext:\n{context}\nAnswer:"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "RAG-Bot",
    }

    data = {
        "model": "deepseek/deepseek-chat",
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a professional Pakistani tax compliance assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        if response.status_code != 200:
            return f"❌ OpenRouter Error {response.status_code}: {response.text}"
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Exception: {e}"

# ----------- Function to Extract Text from PDF Files ----------- #
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# ----------- Function to Split Text into Chunks ----------- #
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=100)
    return text_splitter.split_text(text)

# ----------- Function to Load All .txt Files from Outputs/ ----------- #
def load_text_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return ""

# ----------- Streamlit UI Setup ----------- #
st.set_page_config(page_title="RAG Bot", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .header {
        font-size: 36px;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        font-size: 18px;
        color: #616161;
        text-align: center;
        margin-bottom: 40px;
    }
    .file-uploader {
        border: 2px solid #1e88e5;
        padding: 20px;
        border-radius: 10px;
        background-color: #f1f1f1;
    }
    .file-uploader:hover {
        border-color: #0d47a1;
    }
    .response-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="header">RAG Bot: Ask Questions from PDF Documents</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload your PDFs and ask any question related to the documents. The bot will provide you answers based on the content.</div>', unsafe_allow_html=True)

# File uploader for PDFs
pdf_docs = st.file_uploader("Upload Your PDF Files", accept_multiple_files=True, type="pdf", key="pdf_uploader")

# Upload section UI
if pdf_docs:
    with st.spinner("Processing PDFs..."):
        extracted_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(extracted_text)

        output_dir = 'outputs/'
        os.makedirs(output_dir, exist_ok=True)
        extracted_file_path = os.path.join(output_dir, "extracted_text.txt")
        with open(extracted_file_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

        st.success("PDFs processed successfully!")

# Fetch documents from outputs folder
output_dir = 'outputs/'
retrieved_docs = []
for filename in os.listdir(output_dir):
    if filename.endswith("extracted_text.txt"):
        file_path = os.path.join(output_dir, filename)
        retrieved_docs.append(load_text_from_file(file_path))

# User question input
user_query = st.text_input("Ask a question related to the uploaded documents:")

if user_query and retrieved_docs:
    with st.spinner("Generating response..."):
        response = generate_text_from_deepseek(user_query, retrieved_docs)
        st.markdown('<div class="response-box">', unsafe_allow_html=True)
        st.write("### Answer:")
        st.write(response)
        st.markdown('</div>', unsafe_allow_html=True)

if not pdf_docs:
    st.warning("Please upload one or more PDF files to get started.")
