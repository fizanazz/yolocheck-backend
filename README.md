# YOLOCheck Backend

FastAPI backend for YOLOCheck — AI-powered mole screening using YOLOv11.

## Stack
- **Framework**: FastAPI + Uvicorn
- **Database**: Supabase (PostgreSQL)
- **Storage**: Supabase Storage
- **AI Model**: YOLOv11 (Ultralytics)
- **AI Chatbot**: Google Gemini

## Project Structure
```
yolocheck-backend/
├── app/
│   ├── main.py                    ← FastAPI app + middleware + routers
│   ├── api/routes/
│   │   ├── health.py              ← GET /health
│   │   ├── scan.py                ← POST /scan, GET /scan/{id}
│   │   ├── user.py                ← GET /user/{id}/history
│   │   └── chat.py                ← POST /chat
│   ├── core/
│   │   ├── config.py              ← Settings from .env
│   │   ├── exceptions.py          ← Domain exceptions + handlers
│   │   └── logging.py             ← Structured logging
│   ├── db/
│   │   └── supabase_client.py     ← Supabase client singleton
│   ├── schemas/
│   │   ├── scan.py                ← Scan request/response models
│   │   └── chat.py                ← Chat request/response models
│   ├── services/
│   │   ├── yolo_service.py        ← YOLOv11 inference
│   │   ├── scan_service.py        ← Scan orchestration + DB persistence
│   │   ├── chat_service.py        ← Gemini AI chatbot
│   │   ├── risk_engine.py         ← ABCD scoring algorithm
│   │   └── storage_service.py     ← Supabase Storage uploads
│   └── utils/
│       └── image_utils.py         ← Image decode, crop, draw
├── ml/
│   └── best.pt                    ← Your trained YOLOv11 weights (you provide)
├── scripts/
│   └── schema.sql                 ← Run once in Supabase SQL Editor
├── tests/
│   ├── test_api.py
│   └── test_risk_engine.py
├── .env.example                   ← Copy to .env and fill in
└── requirements.txt
```

## Quick Setup

### 1. Install Python 3.11+
Download from https://python.org/downloads — tick "Add to PATH"

### 2. Create virtual environment
```cmd
python -m venv venv
venv\Scripts\activate        (Windows)
source venv/bin/activate     (Mac/Linux)
pip install -r requirements.txt
```

### 3. Set up Supabase
1. Create a free project at https://supabase.com
2. Go to SQL Editor → paste and run scripts/schema.sql
3. Go to Storage → create a bucket called `scan-images` (set to Public)
4. Copy your project URL and API keys

### 4. Set up Google Gemini
Get a free API key from https://aistudio.google.com/app/apikey

### 5. Create .env file
```cmd
copy .env.example .env
```
Then edit `.env` and fill in all values.

### 6. Add your trained model
```cmd
copy path\to\your\best.pt ml\best.pt
```
If you skip this, the backend runs in stub mode (fake detections for testing).

### 7. Run the server
```cmd
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs to see all API endpoints.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /scan | Upload image + run YOLOv11 |
| GET | /scan/{id} | Get scan by ID |
| GET | /user/{id}/history | Get all scans for a user |
| POST | /chat | Ask the AI Health Assistant |

## Connecting to Frontend

Your Vite React frontend runs on http://localhost:5173 by default.
Make sure your .env has:
```
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

From your frontend, the base URL for all API calls is:
```
http://localhost:8000
```

## Running Tests
```cmd
pytest
```
