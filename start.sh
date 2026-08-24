#!/bin/bash

# 1. FastAPI arka planda başlatılır
uvicorn main:app --host 127.0.0.1 --port 8000 &

# 2. Streamlit HF Spaces portunda (7860) ön planda başlatılır
streamlit run app.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false