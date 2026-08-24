import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def clean_pdf_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"\.{4,}", "...", text)
    return text.strip()


def load_and_process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    for page in pages:
        page.page_content = clean_pdf_text(page.page_content)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    return chunks


def add_documents_to_index(chunks):
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")


def search_financial_docs(query: str, k: int = 5):
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        docs = vectorstore.similarity_search(query, k=k)
        return "\n\n---\n\n".join([doc.page_content for doc in docs])
    else:
        return "Sistemde kayıtlı bir finansal rapor veya doküman bulunamadı."