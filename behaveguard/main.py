import sys
from pathlib import Path
# Add project root to python path to avoid ModuleNotFoundError
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, BackgroundTasks, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import time
from typing import List, Dict, Any, Optional

from behaveguard.storage import init_db, save_session, get_enrolled_profiles, DATA_DIR, MODELS_DIR
from behaveguard.pipeline import train_user_models, score_session, identify_user, get_training_status

app = FastAPI(title="BehaveGuard Backend API", version="1.0.0")

# Enable CORS for Next.js and Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

@app.on_event("startup")
def startup_event():
    """Run database initialization on startup."""
    init_db()

@app.get("/api/health")
def health_check():
    """Return API health and system statistics."""
    # Count profiles
    profiles = get_enrolled_profiles()
    
    # Count total sessions
    sessions_count = 0
    sessions_file = DATA_DIR / "sessions.csv"
    if sessions_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(sessions_file)
            sessions_count = len(df)
        except Exception:
            pass

    return {
        "status": "healthy",
        "uptime_seconds": int(time.time() - START_TIME),
        "profiles_count": len(profiles),
        "total_sessions_recorded": sessions_count
    }

@app.get("/api/profiles")
def get_profiles():
    """List all enrolled user profiles."""
    profiles = get_enrolled_profiles()
    out = []
    for p in profiles:
        status = get_training_status(p)
        out.append({
            "subject_id": p,
            "is_trained": status["status"] == "completed",
            "training_status": status
        })
    return out

@app.post("/api/submit")
def submit_session(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    """
    Ingest a new session from the client.
    Appends to CSV database, triggers auto-training if it's the first session.
    """
    subject_id = payload.get("subject_id")
    if not subject_id:
        raise HTTPException(status_code=400, detail="Missing 'subject_id' in payload.")

    try:
        # Save session to CSV database
        save_session(payload)
        
        # Check if models are already trained
        status = get_training_status(subject_id)
        if status["status"] not in ["completed", "training"]:
            # Auto-trigger asynchronous training
            background_tasks.add_task(train_user_models, subject_id)
            auto_trained = True
        else:
            auto_trained = False
            
        return {
            "status": "success",
            "message": "Session data saved successfully.",
            "subject_id": subject_id,
            "auto_training_triggered": auto_trained
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving session: {str(e)}")

@app.post("/api/train")
def trigger_training(payload: Dict[str, Any] = Body(...), background_tasks: BackgroundTasks = None):
    """
    Explicitly trigger model training (SVM, LSTM, TCN) for a subject.
    Runs asynchronously in the background.
    """
    subject_id = payload.get("subject_id")
    if not subject_id:
        raise HTTPException(status_code=400, detail="Missing 'subject_id' in request.")

    # Check status
    status = get_training_status(subject_id)
    if status["status"] == "training":
        return {
            "status": "running",
            "message": "Training is already in progress.",
            "subject_id": subject_id
        }

    # Queue training
    background_tasks.add_task(train_user_models, subject_id)
    return {
        "status": "queued",
        "message": "Model training queued in the background.",
        "subject_id": subject_id
    }

@app.get("/api/profiles/{subject_id}/status")
def check_status(subject_id: str):
    """Check the model training status of a profile."""
    status = get_training_status(subject_id)
    return status

@app.post("/api/score")
def score_session_endpoint(payload: Dict[str, Any]):
    """
    Score a live session against a selected user profile.
    Payload: { "subject_id": "...", "session": { ... } }
    """
    subject_id = payload.get("subject_id")
    session_data = payload.get("session")
    
    if not subject_id or not session_data:
        raise HTTPException(status_code=400, detail="Missing 'subject_id' or 'session' data.")

    # Save session data for diagnostics
    try:
        save_session(session_data)
    except Exception as e:
        print(f"[ERROR] Failed to save score session: {e}")

    res = score_session(subject_id, session_data)
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
        
    return res

@app.post("/api/identify")
def identify_user_endpoint(payload: Dict[str, Any]):
    """
    Identify who is typing in a session from a list of candidate profiles.
    Payload: { "candidate_ids": ["...", "..."], "session": { ... } }
    """
    candidate_ids = payload.get("candidate_ids")
    session_data = payload.get("session")
    
    if not candidate_ids or not session_data:
        raise HTTPException(status_code=400, detail="Missing 'candidate_ids' or 'session' data.")

    # Save session data for diagnostics
    try:
        save_session(session_data)
    except Exception as e:
        print(f"[ERROR] Failed to save identify session: {e}")

    res = identify_user(candidate_ids, session_data)
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
        
    return res
