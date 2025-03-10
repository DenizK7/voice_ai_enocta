# app/main.py
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='https://front-production-5a59.up.railway.app/', port=8080)
