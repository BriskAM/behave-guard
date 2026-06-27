import numpy as np
import time
import math
from typing import List, Dict, Any, Optional
from scipy.stats import skew, kurtosis

# Constants
WINDOW_SIZE = 50
KEY_CAT_INDEX = {
    "alphanum": 0,
    "symbol": 1,
    "special": 2,
    "space": 3
}

COMMON_DIGRAPHS = {
    ('t', 'h'), ('h', 'e'), ('i', 'n'), ('e', 'r'), ('a', 'n'),
    ('r', 'e'), ('o', 'n'), ('e', 'n'), ('a', 't'), ('e', 'd'),
    ('h', 'a'), ('t', 'o'), ('o', 'r'), ('i', 't'), ('e', 's'),
    ('s', 't'), ('i', 's'), ('n', 'd'), ('a', 's'), ('a', 'r'),
    ('o', 'u'), ('t', 'e'), ('n', 't'), ('n', 'g'), ('t', 'i'),
}

DIGRAPH_FREQUENCY = {d: 1.0 for d in COMMON_DIGRAPHS}

def encode_time(ts_ms: float) -> tuple[float, float]:
    """Encode timestamp (in ms) as cyclical time-of-day features. (Disabled to avoid scaling explosion)."""
    return 0.0, 0.0

def key_cat_onehot(category: str) -> list[float]:
    vec = [0.0] * 4
    idx = KEY_CAT_INDEX.get(category, 2)  # default to special if not found
    vec[idx] = 1.0
    return vec

# ------------------------------------------------------------------ #
# Keystroke Sequence Feature Extraction (LSTM/TCN input)
# ------------------------------------------------------------------ #

def clean_key_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean raw key events by filtering out dwell outliers and IQR-based session anomalies."""
    valid_events = []
    for e in events:
        press = e.get("press_ts")
        release = e.get("release_ts")
        if press is not None and release is not None:
            dwell = release - press
            if 5.0 < dwell < 400.0:
                valid_events.append(e)
                
    if len(valid_events) < 4:
        return valid_events
        
    dwells = [e["release_ts"] - e["press_ts"] for e in valid_events]
    q25, q75 = np.percentile(dwells, 25), np.percentile(dwells, 75)
    iqr = q75 - q25
    upper_bound = q75 + 2.5 * iqr
    
    cleaned_events = [e for e in valid_events if (e["release_ts"] - e["press_ts"]) <= upper_bound]
    return cleaned_events

def extract_keystroke_sequences(events: List[Dict[str, Any]], seq_len: int = 50, stride: int = 25) -> List[np.ndarray]:
    """
    Extract overlapping sequences of shape (seq_len, 10) from raw key events.
    Each event represents: [dwell_ms, flight_ms, digraph_ms, cat_alphanum, cat_symbol, cat_special, cat_space, t_sin, t_cos, freq_weight]
    """
    events = clean_key_events(events)
    if len(events) < 2:
        return []

    # 1. Parse consecutive events into pairs
    pairs = []
    for i in range(len(events) - 1):
        a, b = events[i], events[i+1]
        
        # Calculate timings
        a_press = a["press_ts"]
        a_release = a["release_ts"]
        b_press = b["press_ts"]
        
        if a_release is None or b_press is None or a_press is None:
            continue

        raw_dwell = a_release - a_press
        raw_flight = b_press - a_release
        if not (-1000 < raw_flight < 5000):
            continue

        dwell_ms = np.clip(raw_dwell, 5.0, 400.0)
        flight_ms = np.clip(raw_flight, -150.0, 600.0)
        digraph_ms = np.clip(b_press - a_press, 5.0, 1000.0)

        cat_a = a["key_category"]
        cat_b = b["key_category"]
        pair_key = (a.get("key_id", ""), b.get("key_id", ""))
        weight = DIGRAPH_FREQUENCY.get(pair_key, 0.3)
        
        t_sin, t_cos = encode_time(a_release)
        onehot = key_cat_onehot(cat_a)
        
        vec = np.array([
            dwell_ms,
            flight_ms,
            digraph_ms,
            *onehot,
            t_sin,
            t_cos,
            weight
        ], dtype=np.float32)
        pairs.append(vec)

    # 2. Slide window over pairs
    sequences = []
    i = 0
    while i + seq_len <= len(pairs):
        window = pairs[i : i + seq_len]
        sequences.append(np.stack(window))
        i += stride

    return sequences

# ------------------------------------------------------------------ #
# Keystroke Aggregate Feature Extraction (OC-SVM input)
# ------------------------------------------------------------------ #

def _safe_stats(arr: np.ndarray) -> list[float]:
    if len(arr) < 2:
        return [float(arr.mean()) if len(arr) else 0.0, 0.0]
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
    ]

def extract_keystroke_aggregates(events: List[Dict[str, Any]], win_size: int = 50, stride: int = 25) -> List[np.ndarray]:
    """
    Extract overlapping aggregate feature windows of shape (23,) for OC-SVM.
    """
    events = clean_key_events(events)
    if len(events) < 5:
        return []

    # Parse key timings
    dwells, flights, digraphs = [], [], []
    cat_dwells = {'alphanum': [], 'symbol': [], 'special': []} # group special/space together for SVM
    ikis = []
    pairs = []

    for i in range(len(events)):
        evt = events[i]
        press = evt["press_ts"]
        release = evt["release_ts"]
        if press is None or release is None:
            continue
            
        raw_dwell = release - press
        dwell = np.clip(raw_dwell, 5.0, 400.0)

        cat = evt["key_category"]
        cat_key = cat if cat in ['alphanum', 'symbol'] else 'special'
        
        # Add to window state helper list
        pairs.append({
            'dwell': dwell,
            'cat': cat_key,
            'press': press,
            'release': release,
            'key_id': evt.get("key_id", "")
        })

    # Slide window
    aggregates = []
    i = 0
    while i + win_size <= len(pairs):
        chunk = pairs[i : i + win_size]
        
        dwells_c, flights_c, digraphs_c = [], [], []
        cat_dwells_c = {'alphanum': [], 'symbol': [], 'special': []}
        ikis_c = []
        
        prev = None
        for item in chunk:
            dwell_norm = item['dwell']
            dwells_c.append(dwell_norm)
            cat_dwells_c[item['cat']].append(dwell_norm)
            
            if prev is not None:
                flight = np.clip(item['press'] - prev['release'], -150.0, 600.0)
                dgraph = np.clip(item['press'] - prev['press'], 5.0, 1000.0)
                iki = np.clip(item['press'] - prev['press'], 5.0, 1000.0)
                
                flights_c.append(flight)
                digraphs_c.append(dgraph)
                ikis_c.append(iki)
                
            prev = item
            
        # User-mean normalization of timing features
        mean_d = float(np.mean(dwells_c)) if dwells_c else 80.0
        mean_d = max(mean_d, 50.0)
        
        mean_g = float(np.mean(digraphs_c)) if digraphs_c else 200.0
        mean_g = max(mean_g, 100.0)
        
        # Divide by mean to achieve scale invariance
        dwells_c_norm = [d / mean_d for d in dwells_c]
        flights_c_norm = [f / mean_d for f in flights_c]
        digraphs_c_norm = [d / mean_g for d in digraphs_c]
        
        cat_dwells_c_norm = {}
        for cat in cat_dwells_c:
            cat_dwells_c_norm[cat] = [d / mean_d for d in cat_dwells_c[cat]] if cat_dwells_c[cat] else []
            
        ikis_c_norm = [ik / mean_g for ik in ikis_c] if ikis_c else []
        
        feature_groups = [
            np.array(dwells_c_norm),
            np.array(flights_c_norm) if flights_c_norm else np.array([0.0]),
            np.array(digraphs_c_norm) if digraphs_c_norm else np.array([0.0]),
            np.array(cat_dwells_c_norm['alphanum']) if cat_dwells_c_norm['alphanum'] else np.array([0.0]),
            np.array(cat_dwells_c_norm['symbol']) if cat_dwells_c_norm['symbol'] else np.array([0.0]),
            np.array(cat_dwells_c_norm['special']) if cat_dwells_c_norm['special'] else np.array([0.0]),
            np.array(ikis_c_norm) if ikis_c_norm else np.array([0.0]),
        ]
        
        feats = []
        for arr in feature_groups:
            feats.extend(_safe_stats(arr))  # 7 * 2 = 14
            
        fd_ratio = (np.array(flights_c) / (np.array(dwells_c[:len(flights_c)]) + 1e-6)
                    if flights_c else np.array([1.0]))
        feats.extend(_safe_stats(fd_ratio))  # + 2 = 16
        
        # Time encoding
        t_sin, t_cos = encode_time(chunk[0]['press'])
        
        # Ratios
        n = len(chunk)
        alphanum_ratio = sum(1 for x in chunk if x['cat'] == 'alphanum') / n
        symbol_ratio = sum(1 for x in chunk if x['cat'] == 'symbol') / n
        special_ratio = 1.0 - alphanum_ratio - symbol_ratio
        
        # WPM
        duration_min = ((chunk[-1]['press'] - chunk[0]['press']) / 60000.0) or 1e-6
        wpm = (n / 5.0) / duration_min
        
        feats.extend([t_sin, t_cos, alphanum_ratio, symbol_ratio, special_ratio, wpm / 100.0]) # + 6 = 22
        
        # Digraph coverage
        seen_pairs = set()
        for k in range(1, len(chunk)):
            seen_pairs.add((chunk[k-1]['key_id'], chunk[k]['key_id']))
        coverage = len(seen_pairs & COMMON_DIGRAPHS) / len(COMMON_DIGRAPHS)
        feats.append(coverage)  # + 1 = 23
        
        aggregates.append(np.array(feats, dtype=np.float32))
        i += stride

    return aggregates

# ------------------------------------------------------------------ #
# Mouse Dynamics Feature Extraction (aggregate stats)
# ------------------------------------------------------------------ #

def extract_mouse_aggregates(passive_points: List[Dict[str, Any]], 
                             dot_trials: List[Dict[str, Any]], 
                             drag_trials: List[Dict[str, Any]]) -> np.ndarray:
    """
    Extract a unified 12-dimensional mouse feature vector summarizing a session:
    [avg_passive_speed, std_passive_speed, avg_passive_accel, std_passive_accel,
     avg_passive_curvature, std_passive_curvature, avg_dot_travel_time, std_dot_travel_time,
     avg_dot_error, std_dot_error, drag_success_rate, avg_drag_duration]
    """
    # 1. Passive mouse kinematics
    speeds = []
    accels = []
    curvatures = []
    
    prev_speed = 0.0
    for i in range(1, len(passive_points) - 1):
        p1, p2, p3 = passive_points[i-1], passive_points[i], passive_points[i+1]
        
        dt = (p2["ts"] - p1["ts"]) / 1000.0  # seconds
        if dt <= 0.001:
            continue
            
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        dist = math.hypot(dx, dy)
        speed = dist / dt  # px/s
        speeds.append(speed)
        
        accel = (speed - prev_speed) / dt
        accels.append(accel)
        prev_speed = speed
        
        # Curvature
        v1 = (p2["x"] - p1["x"], p2["y"] - p1["y"])
        v2 = (p3["x"] - p2["x"], p3["y"] - p2["y"])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        norm = math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1])
        if norm > 1e-6:
            curvatures.append(cross / norm)

    avg_p_speed = np.mean(speeds) if speeds else 0.0
    std_p_speed = np.std(speeds) if speeds else 0.0
    avg_p_accel = np.mean(accels) if accels else 0.0
    std_p_accel = np.std(accels) if accels else 0.0
    avg_p_curv = np.mean(curvatures) if curvatures else 0.0
    std_p_curv = np.std(curvatures) if curvatures else 0.0

    # 2. Dot trials click metrics
    dot_travel_times = [t["travel_time_ms"] for t in dot_trials if "travel_time_ms" in t]
    dot_errors = [t["error_px"] for t in dot_trials if "error_px" in t]
    
    avg_dot_travel = np.mean(dot_travel_times) if dot_travel_times else 0.0
    std_dot_travel = np.std(dot_travel_times) if dot_travel_times else 0.0
    avg_dot_err = np.mean(dot_errors) if dot_errors else 0.0
    std_dot_err = np.std(dot_errors) if dot_errors else 0.0

    # 3. Drag trials metrics
    successes = [1.0 if t["success"] else 0.0 for t in drag_trials if "success" in t]
    drag_durations = [t["duration_ms"] for t in drag_trials if "duration_ms" in t]
    
    drag_success = np.mean(successes) if successes else 0.0
    avg_drag_dur = np.mean(drag_durations) if drag_durations else 0.0

    return np.array([
        avg_p_speed, std_p_speed,
        avg_p_accel, std_p_accel,
        avg_p_curv, std_p_curv,
        avg_dot_travel, std_dot_travel,
        avg_dot_err, std_dot_err,
        drag_success, avg_drag_dur
    ], dtype=np.float32)

def extract_mouse_kinematic_windows(passive_points: List[Dict[str, Any]], win_size: int = 30, stride: int = 15, avg_drag_duration: float = 1200.0) -> List[np.ndarray]:
    """Segment passive points and compute kinematic statistics per window (shape 7)."""
    if len(passive_points) < 5:
        return []
        
    events_data = []
    prev_speed = 0.0
    for i in range(1, len(passive_points) - 1):
        p1, p2, p3 = passive_points[i-1], passive_points[i], passive_points[i+1]
        dt = (p2["ts"] - p1["ts"]) / 1000.0
        if dt <= 0.001:
            continue
        dx = p2["x"] - p1["x"]
        dy = p2["y"] - p1["y"]
        dist = math.hypot(dx, dy)
        speed = float(np.clip(dist / dt, 0.0, 4000.0))
        accel = float(np.clip((speed - prev_speed) / dt, -60000.0, 60000.0))
        prev_speed = speed
        
        v1 = (p2["x"] - p1["x"], p2["y"] - p1["y"])
        v2 = (p3["x"] - p2["x"], p3["y"] - p2["y"])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        norm = math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1])
        curv = cross / norm if norm > 1e-6 else 0.0
        
        events_data.append([speed, accel, curv])

    windows = []
    i = 0
    while i + win_size <= len(events_data):
        chunk = events_data[i : i + win_size]
        chunk_arr = np.array(chunk)
        mean_vals = np.mean(chunk_arr, axis=0)
        std_vals = np.std(chunk_arr, axis=0)
        
        vec = np.array([
            mean_vals[0], std_vals[0],  # speed
            mean_vals[1], std_vals[1],  # accel
            mean_vals[2], std_vals[2],  # curvature
            avg_drag_duration
        ], dtype=np.float32)
        windows.append(vec)
        i += stride
        
    return windows

def _robust_mean_std(values: List[float], fallback_mean: float, fallback_std: float) -> tuple[float, float]:
    if not values:
        return fallback_mean, fallback_std
    arr = np.array(values)
    if len(arr) < 3:
        return float(np.mean(arr)), float(np.std(arr)) + 1e-8
    q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    filtered = arr[(arr >= lower) & (arr <= upper)]
    if len(filtered) < 3:
        filtered = arr
    return float(np.mean(filtered)), float(np.std(filtered)) + 1e-8

def extract_mouse_task_baselines(dot_trials: List[Dict[str, Any]], drag_trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute baseline statistics for dot clicking and drag task metrics."""
    dot_travels = [t["travel_time_ms"] for t in dot_trials if "travel_time_ms" in t]
    dot_errors = [t["error_px"] for t in dot_trials if "error_px" in t]
    drag_durs = [t["duration_ms"] for t in drag_trials if "duration_ms" in t]
    
    dt_mean, dt_std = _robust_mean_std(dot_travels, 1500.0, 500.0)
    de_mean, de_std = _robust_mean_std(dot_errors, 10.0, 5.0)
    dd_mean, dd_std = _robust_mean_std(drag_durs, 1200.0, 400.0)
    
    drag_success = [1.0 if t["success"] else 0.0 for t in drag_trials if "success" in t]
    ds_mean = float(np.mean(drag_success)) if drag_success else 1.0
    
    return {
        "dot_travel_mean": dt_mean,
        "dot_travel_std": dt_std,
        "dot_error_mean": de_mean,
        "dot_error_std": de_std,
        "drag_duration_mean": dd_mean,
        "drag_duration_std": dd_std,
        "drag_success_mean": ds_mean
    }

class Normalizer:
    """Normalizes features by subtracting mean and dividing by std."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("Normalizer not fitted.")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
