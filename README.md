# 📈 Financial AI Agent & Stock Forecasting Dashboard

An end-to-end financial intelligence platform combining **Deep Learning (GRU)** time-series forecasting, a **Dynamic Financial RAG** engine for PDF balance sheet analysis, and an **Autonomous Financial AI Agent** with multi-turn conversation memory.

---

## 🏛 Architecture

[Live Stock Data (yfinance)] ──> [Feature Eng. (17 Indicators)] ──> [GRU Model] ──┐
                                                                                  ├──> [FastAPI Backend] <──> [Streamlit UI]
[Financial Report (PDF)]    ──> [Multilingual Embeddings]      ──> [FAISS RAG]  ──┘          │
                                                                                              └──> [AI Agent (Groq / Qwen 27B)]

---

## 🚀 Key Features

1. **Deep Learning Stock Price Forecaster (GRU):**
   - Trained on historical price sequences using a 60-day sliding window.
   - 17 Engineered Financial Features: OHLCV, moving average ratios (MA_20, MA_60, MA_ratio), momentum indicators (RSI_14, MACD, MACD_signal, MACD_hist), 20-day volatility, and volume indicators.
   - Logarithmic transformations (np.log, np.log1p) with dual-scaler normalization (scaler_x, scaler_y).

2. **Dynamic Financial RAG Engine:**
   - Ingests and processes financial reports/balance sheets (PDF) on-the-fly.
   - Text cleaning via regex to normalize broken tables and whitespace.
   - Text chunking via RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=150).
   - Semantic vector search using sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 and FAISS.

3. **Hybrid Financial AI Agent:**
   - Powered by Groq (qwen/qwen3.6-27b) for ultra-fast, high-quality reasoning.
   - Dynamic Tool Calling: Decides whether to query the deep learning model, search vector documents, or synthesize both.
   - Multi-Turn Session Memory: Tracks context across sequential prompts using InMemoryChatMessageHistory.

4. **Interactive Dashboard:**
   - Built with Streamlit and Plotly.
   - Displays dynamic stock price history charts with forecasted trajectory.
   - Sidebar PDF drag-and-drop document ingestion and indexing.
   - Modern conversational chat UI.

---

## 🛠 Tech Stack

- **Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend:** Streamlit, Plotly
- **Machine Learning / Deep Learning:** TensorFlow/Keras, Scikit-Learn, NumPy, Pandas, yfinance
- **AI & RAG:** LangChain, LangChain-Groq, LangChain-HuggingFace, FAISS, PyPDF, Sentence-Transformers

---

## 📂 Project Structure

financial-ai-agent/
├── app.py                     # Streamlit frontend dashboard
├── main.py                    # FastAPI backend server
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (API Keys)
├── .gitignore                 # Git ignore rules
├── services/
│   ├── __init__.py
│   ├── forecaster.py          # GRU feature engineering & live inference
│   ├── rag_service.py         # PDF cleaning, chunking & FAISS indexer
│   └── agent_service.py       # LangChain agent, tool calling & memory
├── models/
│   ├── gru_model.keras        # Trained GRU model weights
│   ├── gru_scaler_x.pkl       # Input feature scaler
│   └── gru_scaler_y.pkl       # Target scaler
├── data/                      # Temporary storage for uploaded PDFs
└── faiss_index/               # Local FAISS vector index files

---

## 💻 Installation & Setup

1. Clone the repository and set up virtual environment:
   git clone https://github.com/OmerOrucc/financial-ai-agent.git
   cd financial-ai-agent
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt

2. Configure environment variables in .env file:
   GROQ_API_KEY=gsk_your_groq_api_key_here

3. Run the backend server (Terminal 1):
   uvicorn main:app --reload

4. Run the frontend dashboard (Terminal 2):
   streamlit run app.py

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Service health status check |
| POST | /api/v1/predict | Runs GRU model on live stock ticker |
| POST | /api/v1/upload-pdf | Ingests, splits, and indexes financial PDF |
| POST | /api/v1/search-docs | Performs semantic search in FAISS |
| POST | /api/v1/chat | AI Agent multi-tool chat with memory |