import os
import shutil
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from services.forecaster import predict_stock_price
from services.rag_service import search_financial_docs, load_and_process_pdf, add_documents_to_index
from services.agent_service import ask_financial_agent


class PredictRequest(BaseModel):
    symbol: str
    days: int


class DocSearchRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"
    lang: str = "tr" 


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Financial AI Agent API"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/predict")
async def predict(request: PredictRequest):
    result = predict_stock_price(request.symbol, request.days)
    return result


@app.post("/api/v1/search-docs")
async def search(request: DocSearchRequest):
    return search_financial_docs(request.query)


@app.post("/api/v1/chat")
async def chat(request: ChatRequest):
    answer = ask_financial_agent(request.message, request.session_id, request.lang)
    return {"response": answer}


@app.post("/api/v1/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"error": "Lütfen sadece PDF formatında dosya yükleyin."}

    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = load_and_process_pdf(file_path)
    add_documents_to_index(chunks)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_added": len(chunks),
        "message": "Doküman başarıyla işlendi ve FAISS vektör veritabanına eklendi."
    }