@echo off
echo Starting Backend Server...
start cmd /k "cd backend && ..\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000"

echo Starting Frontend Server...
start cmd /k "cd frontend && ..\venv\Scripts\python.exe -m http.server 3000"

echo Both servers are starting...
echo Access Backend at http://localhost:8000/docs
echo Access Frontend at http://localhost:3000
