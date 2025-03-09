# Procfile
web: gunicorn app.main:app --bind 0.0.0.0:$PORT
streamlit: streamlit run frontend/streamlit_main.py --server.port 8501 --server.enableCORS false
