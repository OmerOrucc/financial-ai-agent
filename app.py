import streamlit as st
import requests
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import uuid

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Financial AI Agent & Dashboard",
    page_icon="📈",
    layout="wide"
)

# Çok Dilli UI Sözlüğü
TRANSLATIONS = {
    "TR": {
        "title": "📈 Finansal AI Analist ve Hisse Tahmin Sistemi",
        "caption": "Derin Öğrenme (GRU) Fiyat Tahmini, Dinamik RAG Bilanço Analizi ve Hibrit Finans Asistanı",
        "sidebar_doc_title": "📄 Finansal Rapor & Bilanço Yükle",
        "sidebar_doc_desc": "Şirket bilançolarını (PDF) yükleyerek vektör veritabanına aktarın.",
        "pdf_select": "PDF Dosyası Seçin",
        "index_btn": "Dokümanı İndeksle",
        "indexing_spinner": "Rapor okunuyor, parçalanıyor ve FAISS'e kaydediliyor...",
        "index_success": "✅ Başarılı! {count} metin parçası indekslendi.",
        "reset_chat": "🗑️ Sohbeti Sıfırla",
        "sys_status": "Sistem Durumu:",
        "backend_online": "FastAPI Backend: Çevrimiçi",
        "backend_offline": "FastAPI Backend: Çevrimdışı (Uvicorn'u başlatın)",
        "pred_header": "🎯 Derin Öğrenme (GRU) Fiyat Tahmini",
        "ticker_label": "Hisse Kodu (Örn: AAPL, THYAO.IS, ASELS.IS):",
        "predict_btn": "Tahmin Et",
        "predicting_spinner": "{sym} için canlı veriler çekilip model çalıştırılıyor...",
        "last_close": "Son Kapanış",
        "model_pred": "Model Tahmini (Ertesi Gün)",
        "trend_label": "Öngörülen Trend",
        "chart_title": "{sym} Fiyat Geçmişi ve GRU Tahmin Eğrisi",
        "chart_actual": "Gerçek Kapanış",
        "chart_pred": "Model Tahmini",
        "chat_header": "💬 Finansal AI Asistanı",
        "chat_caption": "Yüklenen bilanço raporlarını ve hisse beklentilerini doğal dille sorgulayın.",
        "chat_initial": "Merhaba! Ben Finansal Danışman asistanınızım. Yüklediğiniz raporların kâr/zarar durumunu analiz edebilir veya hisse tahminleri üretebilirim.",
        "chat_placeholder": "Bir soru sorun (Örn: Tüpraş'ın net kârı nedir? veya AAPL tahmini ne?)",
        "chat_spinner": "Finansal veriler ve raporlar taranıyor..."
    },
    "EN": {
        "title": "📈 Financial AI Analyst & Stock Forecaster",
        "caption": "Deep Learning (GRU) Forecasting, Dynamic RAG Balance Sheet Analysis & Hybrid AI Assistant",
        "sidebar_doc_title": "📄 Upload Financial Report & Balance Sheet",
        "sidebar_doc_desc": "Upload company reports (PDF) to index into the vector database.",
        "pdf_select": "Choose a PDF File",
        "index_btn": "Index Document",
        "indexing_spinner": "Reading report, chunking, and embedding into FAISS...",
        "index_success": "✅ Success! {count} text chunks indexed.",
        "reset_chat": "🗑️ Reset Chat",
        "sys_status": "System Status:",
        "backend_online": "FastAPI Backend: Online",
        "backend_offline": "FastAPI Backend: Offline (Start Uvicorn)",
        "pred_header": "🎯 Deep Learning (GRU) Price Forecaster",
        "ticker_label": "Stock Symbol (e.g. AAPL, MSFT, THYAO.IS):",
        "predict_btn": "Forecast",
        "predicting_spinner": "Fetching live data for {sym} and running model...",
        "last_close": "Last Close",
        "model_pred": "Model Forecast (Next Day)",
        "trend_label": "Predicted Trend",
        "chart_title": "{sym} Price History & GRU Forecast Trajectory",
        "chart_actual": "Actual Close",
        "chart_pred": "Model Forecast",
        "chat_header": "💬 Financial AI Assistant",
        "chat_caption": "Query uploaded financial statements and market outlook in natural language.",
        "chat_initial": "Hello! I am your Financial Analyst assistant. I can evaluate uploaded balance sheets or generate stock price predictions.",
        "chat_placeholder": "Ask a question (e.g., What is the net profit of the company? or Forecast AAPL)",
        "chat_spinner": "Scanning financial documents and vector database..."
    }
}

# ==================== SIDEBAR ====================
with st.sidebar:
    lang_code = st.radio("🌐 Dil / Language", ["TR", "EN"], horizontal=True)
    t = TRANSLATIONS[lang_code]
    api_lang = "tr" if lang_code == "TR" else "en"

    st.markdown("---")
    st.header(t["sidebar_doc_title"])
    st.write(t["sidebar_doc_desc"])

    uploaded_file = st.file_uploader(t["pdf_select"], type=["pdf"])

    if uploaded_file is not None:
        if st.button(t["index_btn"], use_container_width=True):
            with st.spinner(t["indexing_spinner"]):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{API_BASE_URL}/api/v1/upload-pdf", files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(t["index_success"].format(count=data.get('chunks_added', 0)))
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.markdown("---")
    if st.button(t["reset_chat"], use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": t["chat_initial"]}
        ]
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown(f"**{t['sys_status']}**")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            st.success(t["backend_online"])
    except:
        st.error(t["backend_offline"])

# Oturum kimliği oluşturma
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Mesaj geçmişi başlatma veya dil değiştiğinde ilk mesajı senkronize etme
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": t["chat_initial"]}
    ]
elif len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "assistant":
    # Henüz kullanıcı soru sormadıysa, dil seçimine göre ilk karşılama mesajını güncelle
    st.session_state.messages[0]["content"] = t["chat_initial"]

st.title(t["title"])
st.caption(t["caption"])

# ==================== ANA EKRAN İKİ SÜTUN ====================
col1, col2 = st.columns([1, 1])

# ----------------- 1. SÜTUN: FİYAT TAHMİNİ & GRAFİK -----------------
with col1:
    st.subheader(t["pred_header"])

    col_sym, col_btn = st.columns([3, 1])
    with col_sym:
        symbol_input = st.text_input(t["ticker_label"], value="AAPL").strip().upper()
    with col_btn:
        st.write("")
        st.write("")
        predict_btn = st.button(t["predict_btn"], use_container_width=True)

    if predict_btn:
        with st.spinner(t["predicting_spinner"].format(sym=symbol_input)):
            try:
                res = requests.post(
                    f"{API_BASE_URL}/api/v1/predict",
                    json={"symbol": symbol_input, "days": 1}
                )
                if res.status_code == 200:
                    pred_data = res.json()
                    last_price = pred_data["last_close_price"]
                    next_price = pred_data["predictions"][0]
                    trend = pred_data["trend"]

                    diff = round(next_price - last_price, 2)
                    diff_pct = round((diff / last_price) * 100, 2)

                    m1, m2, m3 = st.columns(3)
                    m1.metric(t["last_close"], f"{last_price} $ / TL")
                    m2.metric(t["model_pred"], f"{next_price} $ / TL", delta=f"{diff} ({diff_pct}%)")
                    m3.metric(t["trend_label"], trend)

                    history = yf.Ticker(symbol_input).history(period="3mo").reset_index()
                    if not history.empty:
                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=history["Date"],
                            y=history["Close"],
                            mode="lines",
                            name=t["chart_actual"],
                            line=dict(color="#1f77b4", width=2)
                        ))

                        last_date = history["Date"].iloc[-1]
                        next_date = last_date + pd.Timedelta(days=1)

                        fig.add_trace(go.Scatter(
                            x=[last_date, next_date],
                            y=[last_price, next_price],
                            mode="lines+markers",
                            name=t["chart_pred"],
                            line=dict(color="#2ca02c" if diff >= 0 else "#d62728", width=3, dash="dot"),
                            marker=dict(size=8)
                        ))

                        fig.update_layout(
                            title=t["chart_title"].format(sym=symbol_input),
                            xaxis_title="Date / Tarih",
                            yaxis_title="Price / Fiyat",
                            template="plotly_white",
                            height=400,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Error: {e}")

# ----------------- 2. SÜTUN: AI SOHBET EKRANI -----------------
with col2:
    st.subheader(t["chat_header"])
    st.caption(t["chat_caption"])

    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_query := st.chat_input(t["chat_placeholder"]):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner(t["chat_spinner"]):
                    try:
                        res = requests.post(
                            f"{API_BASE_URL}/api/v1/chat",
                            json={
                                "message": user_query,
                                "session_id": st.session_state.session_id,
                                "lang": api_lang
                            }
                        )
                        if res.status_code == 200:
                            ans = res.json().get("response", "No response received.")
                            st.markdown(ans)
                            st.session_state.messages.append({"role": "assistant", "content": ans})
                        else:
                            st.error(f"API Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")