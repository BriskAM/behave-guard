import os
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from behaveguard.storage import MODELS_DIR, load_keyboard_events, load_mouse_data
from behaveguard.features import (
    extract_keystroke_sequences, extract_keystroke_aggregates,
    extract_mouse_kinematic_windows, extract_mouse_task_baselines
)
from behaveguard.models.svm import BehaveGuardSVM
from behaveguard.models.lstm import BehaveGuardLSTM
from behaveguard.models.tcn import BehaveGuardTCN

# Global in-memory training status tracker
TRAINING_STATUS: Dict[str, Dict[str, Any]] = {}

def get_training_status(subject_id: str) -> Dict[str, Any]:
    """Returns the current training status for a subject."""
    # Check in-memory first
    if subject_id in TRAINING_STATUS:
        return TRAINING_STATUS[subject_id]
    
    # Check if a model directory exists and has a status file
    status_file = MODELS_DIR / subject_id / "status.json"
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
            
    # Check if files exist to determine if already trained
    svm_path = MODELS_DIR / subject_id / "svm.pkl"
    lstm_path = MODELS_DIR / subject_id / "lstm.pkl"
    tcn_path = MODELS_DIR / subject_id / "tcn.pkl"
    
    if svm_path.exists() and lstm_path.exists() and tcn_path.exists():
        return {"status": "completed", "progress": 1.0, "message": "All models trained."}
        
    return {"status": "idle", "progress": 0.0, "message": "No training started yet."}

def save_status(subject_id: str, status: str, progress: float, message: str, error: Optional[str] = None):
    """Saves status to memory and to a local status.json file."""
    data = {
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": time.time()
    }
    if error:
        data["error"] = error
        
    TRAINING_STATUS[subject_id] = data
    
    subject_dir = MODELS_DIR / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    with open(subject_dir / "status.json", 'w') as f:
        json.dump(data, f)

def train_user_models(subject_id: str):
    """
    Synchronous training routine. Designed to be run in a BackgroundTask.
    """
    save_status(subject_id, "training", 0.1, "Loading keyboard events from database...")
    try:
        # 1. Load data
        events = load_keyboard_events(subject_id)
        if len(events) < 100:
            raise ValueError(f"Insufficient keystrokes for training ({len(events)}/100 required). Please type more.")

        # 2. Extract features
        save_status(subject_id, "training", 0.2, "Extracting sequence and aggregate features...")
        sequences = extract_keystroke_sequences(events, seq_len=50, stride=10) # dense stride for more training samples
        aggregates = extract_keystroke_aggregates(events, win_size=50, stride=10)

        if len(sequences) < 10 or len(aggregates) < 10:
            raise ValueError(f"Extracted only {len(sequences)} windows from {len(events)} keys. Need at least 10 windows. Please type more.")

        # Split data: 80% training, 20% calibration
        split_idx_seq = int(len(sequences) * 0.8)
        split_idx_agg = int(len(aggregates) * 0.8)

        train_seq = sequences[:split_idx_seq]
        cal_seq = sequences[split_idx_seq:]

        train_agg = aggregates[:split_idx_agg]
        cal_agg = aggregates[split_idx_agg:]

        subject_dir = MODELS_DIR / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        # 3. Train SVM
        save_status(subject_id, "training", 0.3, "Training One-Class SVM baseline...")
        svm_model = BehaveGuardSVM()
        svm_model.fit(train_agg)
        
        # Calibrate SVM on validation
        if cal_agg:
            scores = [svm_model.score_window(w)["raw_decision"] for w in cal_agg]
            svm_model.t_anomaly = float(np.percentile(scores, 95))
        svm_model.save(subject_dir / "svm.pkl")

        # 4. Train LSTM
        save_status(subject_id, "training", 0.5, "Training PyTorch LSTM Autoencoder (this may take a minute)...")
        lstm_model = BehaveGuardLSTM(epochs=60, batch_size=16)
        lstm_model.fit(train_seq)
        
        # Calibrate LSTM on validation
        if cal_seq:
            scores = [lstm_model.score_window(w)["raw_decision"] for w in cal_seq]
            lstm_model.t_anomaly = float(np.percentile(scores, 95))
        lstm_model.save(subject_dir / "lstm.pkl")

        # 5. Train TCN
        save_status(subject_id, "training", 0.7, "Training PyTorch TCN Autoencoder...")
        tcn_model = BehaveGuardTCN(epochs=60, batch_size=16)
        tcn_model.fit(train_seq)
        
        # Calibrate TCN on validation
        if cal_seq:
            scores = [tcn_model.score_window(w)["raw_decision"] for w in cal_seq]
            tcn_model.t_anomaly = float(np.percentile(scores, 95))
        tcn_model.save(subject_dir / "tcn.pkl")

        # 6. Train Mouse Dynamics Kinematic SVM & Task Baselines
        save_status(subject_id, "training", 0.85, "Extracting and training mouse dynamics models...")
        mouse_data = load_mouse_data(subject_id)
        passive = mouse_data.get("passive", [])
        dot_trials = mouse_data.get("dot_trials", [])
        drag_trials = mouse_data.get("drag_trials", [])

        # Train mouse SVM
        mouse_wins = extract_mouse_kinematic_windows(passive, win_size=30, stride=5)
        if len(mouse_wins) >= 10:
            split_idx = int(len(mouse_wins) * 0.8)
            train_m = mouse_wins[:split_idx]
            cal_m = mouse_wins[split_idx:]
            
            svm_mouse = BehaveGuardSVM()
            svm_mouse.fit(train_m)
            if cal_m:
                scores = [svm_mouse.score_window(w)["raw_decision"] for w in cal_m]
                svm_mouse.t_anomaly = float(np.percentile(scores, 95))
            svm_mouse.save(subject_dir / "svm_mouse.pkl")
            
        # Save mouse task baselines
        mouse_baselines = extract_mouse_task_baselines(dot_trials, drag_trials)
        with open(subject_dir / "mouse_baselines.json", 'w') as f:
            json.dump(mouse_baselines, f)

        save_status(subject_id, "completed", 1.0, "All models (SVM, LSTM, TCN, Mouse SVM) successfully trained.")
        print(f"Models successfully trained for '{subject_id}'!")
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        print(f"Error training models for {subject_id}: {error_msg}\n{trace}")
        save_status(subject_id, "failed", 1.0, f"Training failed: {error_msg}", error=trace)

# ------------------------------------------------------------------ #
# Scoring and Verification
# ------------------------------------------------------------------ #

def score_session(subject_id: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score an incoming session against a target subject's profile.
    Returns detail scores for SVM, LSTM, TCN, and a Fused score.
    """
    subject_dir = MODELS_DIR / subject_id
    svm_path = subject_dir / "svm.pkl"
    lstm_path = subject_dir / "lstm.pkl"
    tcn_path = subject_dir / "tcn.pkl"
    svm_mouse_path = subject_dir / "svm_mouse.pkl"
    baselines_path = subject_dir / "mouse_baselines.json"

    if not (svm_path.exists() and lstm_path.exists() and tcn_path.exists()):
        return {
            "error": f"Models are not fully trained/available for profile '{subject_id}'.",
            "is_trained": False
        }

    # Load models
    svm_model = BehaveGuardSVM()
    svm_model.load(svm_path)

    lstm_model = BehaveGuardLSTM()
    lstm_model.load(lstm_path)

    tcn_model = BehaveGuardTCN()
    tcn_model.load(tcn_path)

    svm_mouse_model = BehaveGuardSVM()
    if svm_mouse_path.exists():
        svm_mouse_model.load(svm_mouse_path)

    mouse_baselines = {}
    if baselines_path.exists():
        try:
            with open(baselines_path, 'r') as f:
                mouse_baselines = json.load(f)
        except Exception:
            pass

    # Extract features from session keyboard events
    events = session_data.get("keyboard", {}).get("events", [])
    if len(events) < 5:
        return {
            "verdict": "unknown",
            "message": "Not enough keystroke data to score (minimum 5 required).",
            "anomaly_rate": 0.0,
            "anomaly_score": 0.0
        }

    sequences = extract_keystroke_sequences(events, seq_len=50, stride=25)
    aggregates = extract_keystroke_aggregates(events, win_size=50, stride=25)

    # If the session is short and doesn't yield a full window, fallback to sizing down the window size
    if not sequences or not aggregates:
        sequences = extract_keystroke_sequences(events, seq_len=min(len(events)-1, 50), stride=25)
        aggregates = extract_keystroke_aggregates(events, win_size=min(len(events), 50), stride=25)

    if not sequences or not aggregates:
        # Still empty, fallback to simple event-level scoring or default values
        return {
            "verdict": "uncertain",
            "message": "Could not extract model-ready windows. Session too short.",
            "anomaly_rate": 0.5,
            "anomaly_score": 0.5
        }

    # Ensure all sequence windows are padded to seq_len=50 for PyTorch models (LSTM/TCN)
    padded_seqs = []
    for seq in sequences:
        if len(seq) < 50:
            pad_len = 50 - len(seq)
            padded = np.pad(seq, ((0, pad_len), (0, 0)), 'constant')
            padded_seqs.append(padded)
        else:
            padded_seqs.append(seq)
    sequences = padded_seqs

    # Score each window
    svm_wins = [svm_model.score_window(w) for w in aggregates]
    lstm_wins = [lstm_model.score_window(w) for w in sequences]
    tcn_wins = [tcn_model.score_window(w) for w in sequences]

    # Session aggregation
    svm_session = svm_model.score_session(svm_wins)
    lstm_session = lstm_model.score_session(lstm_wins)
    tcn_session = tcn_model.score_session(tcn_wins)

    kb_score = 0.70 * svm_session["mean_score"] + 0.15 * lstm_session["mean_score"] + 0.15 * tcn_session["mean_score"]
    kb_anomaly_rate = 0.70 * svm_session["anomaly_rate"] + 0.15 * lstm_session["anomaly_rate"] + 0.15 * tcn_session["anomaly_rate"]

    # Score Mouse Dynamics
    m_score = 0.0
    m_rate = 0.0
    m_verdict = "legitimate"
    m_trained = svm_mouse_path.exists()
    
    passive_pts = session_data.get("mouse", {}).get("passive_points", [])
    dot_trials = session_data.get("mouse", {}).get("dot_trials", [])
    drag_trials = session_data.get("mouse", {}).get("drag_trials", [])

    has_passive_data = False
    if passive_pts and svm_mouse_model.is_trained:
        passive_wins = extract_mouse_kinematic_windows(passive_pts, win_size=30, stride=15)
        if len(passive_wins) >= 30:
            scores = [svm_mouse_model.score_window(w) for w in passive_wins]
            m_session = svm_mouse_model.score_session(scores)
            m_score = m_session["mean_score"]
            m_rate = m_session["anomaly_rate"]
            m_verdict = m_session["session_verdict"]
            has_passive_data = True

    z_scores = []
    if dot_trials and "dot_travel_mean" in mouse_baselines:
        for t in dot_trials:
            travel = t.get("travel_time_ms")
            error = t.get("error_px")
            if travel is not None:
                z_scores.append(abs((travel - mouse_baselines["dot_travel_mean"]) / mouse_baselines["dot_travel_std"]))
            if error is not None:
                z_scores.append(abs((error - mouse_baselines["dot_error_mean"]) / mouse_baselines["dot_error_std"]))
                
    if drag_trials and "drag_duration_mean" in mouse_baselines:
        for t in drag_trials:
            dur = t.get("duration_ms")
            success = t.get("success")
            if dur is not None:
                z_scores.append(abs((dur - mouse_baselines["drag_duration_mean"]) / mouse_baselines["drag_duration_std"]))
            if success is not None and not success:
                z_scores.append(3.0)

    task_anomaly = 0.0
    if z_scores:
        mean_z = float(np.mean(z_scores))
        task_anomaly = float(np.clip(mean_z / 3.0, 0.0, 1.0))

    has_mouse_data = m_trained and (has_passive_data or len(dot_trials) > 0 or len(drag_trials) > 0)

    if has_mouse_data:
        fused_mouse_score = 0.5 * m_score + 0.5 * task_anomaly
        fused_mouse_rate = 0.5 * m_rate + 0.5 * task_anomaly
        
        fused_score = 0.5 * kb_score + 0.5 * fused_mouse_score
        fused_anomaly_rate = 0.5 * kb_anomaly_rate + 0.5 * fused_mouse_rate
    else:
        fused_mouse_score = 0.0
        fused_mouse_rate = 0.0
        fused_score = kb_score
        fused_anomaly_rate = kb_anomaly_rate

    if fused_anomaly_rate < 0.25:
        verdict = "legitimate"
    elif fused_anomaly_rate > 0.60:
        verdict = "impostor"
    else:
        verdict = "uncertain"

    # Action recommendation
    if fused_score < 0.35:
        action = "none"
    elif fused_score < 0.60:
        action = "soft_challenge"
    elif fused_score < 0.80:
        action = "hard_challenge"
    else:
        action = "lockout"

    return {
        "is_trained": True,
        "verdict": verdict,
        "anomaly_score": fused_score,
        "anomaly_rate": fused_anomaly_rate,
        "recommended_action": action,
        "keyboard_score": kb_score,
        "mouse_score": fused_mouse_score if has_mouse_data else None,
        "models": {
            "svm": {
                "verdict": svm_session["session_verdict"],
                "anomaly_rate": svm_session["anomaly_rate"],
                "anomaly_score": svm_session["mean_score"]
            },
            "lstm": {
                "verdict": lstm_session["session_verdict"],
                "anomaly_rate": lstm_session["anomaly_rate"],
                "anomaly_score": lstm_session["mean_score"]
            },
            "tcn": {
                "verdict": tcn_session["session_verdict"],
                "anomaly_rate": tcn_session["anomaly_rate"],
                "anomaly_score": tcn_session["mean_score"]
            },
            "mouse_svm": {
                "verdict": m_verdict,
                "anomaly_rate": m_rate,
                "anomaly_score": m_score,
                "task_anomaly": task_anomaly
            } if has_mouse_data else None
        }
    }

# ------------------------------------------------------------------ #
# Identity Identification (Multi-profile match)
# ------------------------------------------------------------------ #

def identify_user(candidate_ids: List[str], session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the session data against a list of candidate profiles and classify which profile 
    most likely matches the typist, including relative confidence scores.
    """
    scores = []
    valid_candidates = []

    for cid in candidate_ids:
        # Skip if candidate models are not trained
        subject_dir = MODELS_DIR / cid
        if not (subject_dir / "svm.pkl").exists():
            continue
            
        res = score_session(cid, session_data)
        if "error" in res:
            continue
            
        print(f"[DEBUG IDENTIFY] Candidate: {cid}")
        print(f"  Verdict: {res['verdict']}")
        print(f"  Anomaly Score: {res['anomaly_score']:.4f}")
        print(f"  Keyboard Score: {res['keyboard_score']:.4f}")
        print(f"  Mouse Score: {res.get('mouse_score')}")
        print(f"  Models Breakdown:")
        for m_name, m_res in res['models'].items():
            if m_res:
                print(f"    {m_name}: score={m_res['anomaly_score']:.4f}, verdict={m_res['verdict']}")
            
        # Match rate / similarity is the inverse of anomaly score
        match_score = max(0.0, 1.0 - res["anomaly_score"])
        scores.append(match_score)
        valid_candidates.append({
            "subject_id": cid,
            "match_score": match_score,
            "anomaly_score": res["anomaly_score"],
            "verdict": res["verdict"],
            "keyboard_score": res.get("keyboard_score"),
            "mouse_score": res.get("mouse_score")
        })

    if not valid_candidates:
        return {
            "error": "No valid trained candidate profiles selected.",
            "candidates": []
        }

    # Use Softmax normalization (with a temperature scaling factor of 6.0)
    # to polarize confidence scores and clearly separate legitimate typists from impostors.
    k = 6.0
    exp_scores = [np.exp(k * s) for s in scores]
    sum_exp = sum(exp_scores)
    
    if sum_exp > 0:
        for c, exp_s in zip(valid_candidates, exp_scores):
            c["confidence"] = float(exp_s / sum_exp)
    else:
        val = 1.0 / len(valid_candidates)
        for c in valid_candidates:
            c["confidence"] = val

    # Sort by confidence descending
    valid_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    best_candidate = valid_candidates[0]
    identified_subject = best_candidate["subject_id"] if best_candidate["confidence"] > 0.4 else "unknown"

    # Write to a persistent log file
    log_dir = Path("/Users/akshitmehta/Development/behave-guard/behaveguard/data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "identification.log"
    
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "keys_typed": len(session_data.get("keyboard", {}).get("events", [])),
        "mouse_passive_points": len(session_data.get("mouse", {}).get("passive_points", [])),
        "has_active_mouse_tasks": (len(session_data.get("mouse", {}).get("dot_trials", [])) > 0 or 
                                    len(session_data.get("mouse", {}).get("drag_trials", [])) > 0),
        "candidates": candidate_ids,
        "identified_subject_id": identified_subject,
        "confidence": best_candidate["confidence"],
        "candidates_breakdown": [
            {
                "subject_id": c["subject_id"],
                "confidence": c["confidence"],
                "verdict": c["verdict"],
                "keyboard_score": c["keyboard_score"],
                "mouse_score": c["mouse_score"]
            }
            for c in valid_candidates
        ]
    }
    
    try:
        with open(log_file, "a") as lf:
            lf.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error writing to identification log: {e}")

    return {
        "identified_subject_id": identified_subject,
        "confidence": best_candidate["confidence"],
        "candidates": valid_candidates
    }
