import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"

SESSIONS_CSV = DATA_DIR / "sessions.csv"
KEY_EVENTS_CSV = DATA_DIR / "key_events.csv"
MOUSE_PASSIVE_CSV = DATA_DIR / "mouse_passive.csv"
DOT_TRIALS_CSV = DATA_DIR / "dot_trials.csv"
DRAG_TRIALS_CSV = DATA_DIR / "drag_trials.csv"

def init_db(xlsx_path: str = "Behaveguard-client.xlsx"):
    """
    Convert the client generated Excel file into CSV files on first startup.
    Creates the behaveguard/data/ directory.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(xlsx_path):
        # Create empty CSVs if Excel file doesn't exist
        _init_empty_csvs()
        return

    # Check if CSVs already exist. If yes, skip conversion.
    if SESSIONS_CSV.exists() and KEY_EVENTS_CSV.exists():
        return

    print(f"Initializing database from Excel file: {xlsx_path}")
    try:
        xl = pd.ExcelFile(xlsx_path)
        
        # 1. Sessions
        if 'Sessions' in xl.sheet_names:
            # We noticed pandas parses this without headers or uses first row as header.
            # Let's load it with header=None, and specify columns.
            df_sess = xl.parse('Sessions', header=None)
            df_sess.columns = [
                'subject_id', 'collected_at', 'keyboard_events_count', 
                'mouse_passive_count', 'dot_trials_count', 'drag_trials_count', 'duration_ms'
            ]
            df_sess.to_csv(SESSIONS_CSV, index=False)
        else:
            _create_empty_sessions()

        # 2. KeyEvents
        if 'KeyEvents' in xl.sheet_names:
            df_keys = xl.parse('KeyEvents')
            df_keys.to_csv(KEY_EVENTS_CSV, index=False)
        else:
            _create_empty_key_events()

        # 3. MousePassive
        if 'MousePassive' in xl.sheet_names:
            df_mouse = xl.parse('MousePassive')
            df_mouse.to_csv(MOUSE_PASSIVE_CSV, index=False)
        else:
            _create_empty_mouse_passive()

        # 4. DotTrials
        if 'DotTrials' in xl.sheet_names:
            df_dots = xl.parse('DotTrials')
            df_dots.to_csv(DOT_TRIALS_CSV, index=False)
        else:
            _create_empty_dot_trials()

        # 5. DragTrials
        if 'DragTrials' in xl.sheet_names:
            df_drags = xl.parse('DragTrials')
            df_drags.to_csv(DRAG_TRIALS_CSV, index=False)
        else:
            _create_empty_drag_trials()
            
        print("Successfully migrated Excel data to CSVs!")
    except Exception as e:
        print(f"Error migrating Excel to CSVs: {e}. Fallback to empty CSVs.")
        _init_empty_csvs()

def _init_empty_csvs():
    _create_empty_sessions()
    _create_empty_key_events()
    _create_empty_mouse_passive()
    _create_empty_dot_trials()
    _create_empty_drag_trials()

def _create_empty_sessions():
    df = pd.DataFrame(columns=[
        'subject_id', 'collected_at', 'keyboard_events_count', 
        'mouse_passive_count', 'dot_trials_count', 'drag_trials_count', 'duration_ms'
    ])
    df.to_csv(SESSIONS_CSV, index=False)

def _create_empty_key_events():
    df = pd.DataFrame(columns=[
        'subject_id', 'collected_at', 'segment', 'key_id', 
        'key_category', 'press_ts', 'release_ts', 'dwell_ms'
    ])
    df.to_csv(KEY_EVENTS_CSV, index=False)

def _create_empty_mouse_passive():
    df = pd.DataFrame(columns=['subject_id', 'collected_at', 'x', 'y', 'ts', 'dx', 'dy', 'pressure'])
    df.to_csv(MOUSE_PASSIVE_CSV, index=False)

def _create_empty_dot_trials():
    df = pd.DataFrame(columns=[
        'subject_id', 'collected_at', 'trial_index', 'target_x', 'target_y', 
        'click_x', 'click_y', 'appeared_at', 'clicked_at', 'travel_time_ms', 'error_px'
    ])
    df.to_csv(DOT_TRIALS_CSV, index=False)

def _create_empty_drag_trials():
    df = pd.DataFrame(columns=[
        'subject_id', 'collected_at', 'trial_index', 'start_x', 'start_y', 
        'end_x', 'end_y', 'zone_x', 'zone_y', 'duration_ms', 'success'
    ])
    df.to_csv(DRAG_TRIALS_CSV, index=False)

def save_session(data: Dict[str, Any]):
    """
    Appends a new SessionData payload (conforming to client JSON) into the CSV database.
    Also saves a backup JSON file in data/backup_sessions/.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    subject_id = data.get("subject_id", "unknown")
    collected_at = data.get("collected_at", "")
    duration_ms = data.get("duration_ms", 0.0)

    kb = data.get("keyboard", {})
    kb_events = kb.get("events", [])
    
    mouse = data.get("mouse", {})
    mouse_passive = mouse.get("passive_points", [])
    dot_trials = mouse.get("dot_trials", [])
    drag_trials = mouse.get("drag_trials", [])

    # 1. Append to sessions.csv
    sess_row = pd.DataFrame([{
        'subject_id': subject_id,
        'collected_at': collected_at,
        'keyboard_events_count': len(kb_events),
        'mouse_passive_count': len(mouse_passive),
        'dot_trials_count': len(dot_trials),
        'drag_trials_count': len(drag_trials),
        'duration_ms': duration_ms
    }])
    sess_row.to_csv(SESSIONS_CSV, mode='a', header=not SESSIONS_CSV.exists(), index=False)

    # 2. Append to key_events.csv
    if kb_events:
        key_rows = []
        for e in kb_events:
            p_ts = e.get("press_ts", 0.0)
            r_ts = e.get("release_ts")
            dwell = (r_ts - p_ts) if r_ts is not None else 0.0
            key_rows.append({
                'subject_id': subject_id,
                'collected_at': collected_at,
                'segment': e.get("segment", "free"),
                'key_id': e.get("key_id", ""),
                'key_category': e.get("key_category", "alphanum"),
                'press_ts': p_ts,
                'release_ts': r_ts,
                'dwell_ms': dwell
            })
        pd.DataFrame(key_rows).to_csv(KEY_EVENTS_CSV, mode='a', header=not KEY_EVENTS_CSV.exists(), index=False)

    # 3. Append to mouse_passive.csv
    if mouse_passive:
        mp_rows = []
        for p in mouse_passive:
            mp_rows.append({
                'subject_id': subject_id,
                'collected_at': collected_at,
                'x': p.get("x", 0.0),
                'y': p.get("y", 0.0),
                'ts': p.get("ts", 0.0),
                'dx': p.get("dx", 0.0),
                'dy': p.get("dy", 0.0),
                'pressure': p.get("pressure", 0.5)
            })
        pd.DataFrame(mp_rows).to_csv(MOUSE_PASSIVE_CSV, mode='a', header=not MOUSE_PASSIVE_CSV.exists(), index=False)

    # 4. Append to dot_trials.csv
    if dot_trials:
        dot_rows = []
        for idx, t in enumerate(dot_trials):
            dot_rows.append({
                'subject_id': subject_id,
                'collected_at': collected_at,
                'trial_index': idx + 1,
                'target_x': t.get("target_x", 0.0),
                'target_y': t.get("target_y", 0.0),
                'click_x': t.get("click_x", 0.0),
                'click_y': t.get("click_y", 0.0),
                'appeared_at': t.get("appeared_at", 0.0),
                'clicked_at': t.get("clicked_at", 0.0),
                'travel_time_ms': t.get("travel_time_ms", 0.0),
                'error_px': t.get("error_px", 0.0)
            })
        pd.DataFrame(dot_rows).to_csv(DOT_TRIALS_CSV, mode='a', header=not DOT_TRIALS_CSV.exists(), index=False)

    # 5. Append to drag_trials.csv
    if drag_trials:
        drag_rows = []
        for idx, t in enumerate(drag_trials):
            drag_rows.append({
                'subject_id': subject_id,
                'collected_at': collected_at,
                'trial_index': idx + 1,
                'start_x': t.get("start_x", 0.0),
                'start_y': t.get("start_y", 0.0),
                'end_x': t.get("end_x", 0.0),
                'end_y': t.get("end_y", 0.0),
                'zone_x': t.get("zone_x", 0.0),
                'zone_y': t.get("zone_y", 0.0),
                'duration_ms': t.get("duration_ms", 0.0),
                'success': t.get("success", False)
            })
        pd.DataFrame(drag_rows).to_csv(DRAG_TRIALS_CSV, mode='a', header=not DRAG_TRIALS_CSV.exists(), index=False)

    # Backup raw JSON
    backup_dir = DATA_DIR / "backup_sessions"
    backup_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{subject_id}_{collected_at.replace(':', '-')}.json"
    with open(backup_dir / filename, 'w') as f:
        json.dump(data, f)

def get_enrolled_profiles() -> List[str]:
    """Returns list of unique subject IDs with recorded data."""
    if not SESSIONS_CSV.exists():
        return []
    df = pd.read_csv(SESSIONS_CSV)
    return sorted(df['subject_id'].unique().tolist())

def load_keyboard_events(subject_id: str) -> List[Dict[str, Any]]:
    """Loads all keyboard events for a user."""
    if not KEY_EVENTS_CSV.exists():
        return []
    df = pd.read_csv(KEY_EVENTS_CSV)
    user_df = df[df['subject_id'] == subject_id]
    return user_df.to_dict(orient='records')

def load_mouse_data(subject_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Loads all mouse datasets for a user."""
    out = {"passive": [], "dot_trials": [], "drag_trials": []}
    if MOUSE_PASSIVE_CSV.exists():
        df = pd.read_csv(MOUSE_PASSIVE_CSV)
        out["passive"] = df[df['subject_id'] == subject_id].to_dict(orient='records')
    if DOT_TRIALS_CSV.exists():
        df = pd.read_csv(DOT_TRIALS_CSV)
        out["dot_trials"] = df[df['subject_id'] == subject_id].to_dict(orient='records')
    if DRAG_TRIALS_CSV.exists():
        df = pd.read_csv(DRAG_TRIALS_CSV)
        out["drag_trials"] = df[df['subject_id'] == subject_id].to_dict(orient='records')
    return out
