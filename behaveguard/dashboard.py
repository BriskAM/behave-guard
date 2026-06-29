import sys
from pathlib import Path
from datetime import datetime

# Add project root to python path to avoid ModuleNotFoundError
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

from behaveguard.storage import (
    DATA_DIR, get_enrolled_profiles, load_keyboard_events, load_mouse_data
)
from behaveguard.features import (
    extract_keystroke_aggregates, extract_keystroke_sequences,
    extract_trial_path_kinematics
)
from behaveguard.pipeline import get_training_status

def is_valid_num(val):
    try:
        if val is None:
            return False
        f = float(val)
        return not np.isnan(f) and not np.isinf(f)
    except (ValueError, TypeError):
        return False

# Page config
st.set_page_config(
    page_title="BehaveGuard Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@400;700&display=swap');
    
    .reportview-container {
        background-color: #0b0f19;
    }
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(135deg, #00E5FF 0%, #7D2AE8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'Roboto Mono', monospace;
        color: #8A99AD;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #121824;
        border: 1px solid #1f2a3f;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .metric-value {
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        font-size: 2.2rem;
        color: #00E5FF;
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #8A99AD;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🛡️ BehaveGuard Operations Center</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Continuous Behavioral Biometrics Analytics & Threat Detection Dashboard</p>", unsafe_allow_html=True)

# Check database
if not (DATA_DIR / "sessions.csv").exists():
    st.warning("⚠️ No database CSVs found yet. Run the FastAPI server first to initialize the database from the Excel file.")
    st.stop()

# Sidebar controls
st.sidebar.title("🛡️ Profile Selector")
profiles = get_enrolled_profiles()
if not profiles:
    st.sidebar.warning("No profiles found in the database. Enroll some profiles using the client.")
    st.stop()

selected_profile = st.sidebar.selectbox("Inspect Active User Profile", profiles)

# Tab structure
tab_summary, tab_keys, tab_mouse, tab_compare, tab_models, tab_clusters = st.tabs([
    "📊 User Session Summary", 
    "⌨️ Keystroke Rhythmics", 
    "🖱️ Active/Passive Kinematics", 
    "👥 Profile Comparison Space",
    "🤖 ML Models & Drift Monitor",
    "🌌 Biometric Cluster Space"
])

# Load selected profile data
events = load_keyboard_events(selected_profile)
mouse = load_mouse_data(selected_profile)
sessions_df = pd.read_csv(DATA_DIR / "sessions.csv")
profile_sessions = sessions_df[sessions_df["subject_id"] == selected_profile]

# Load recent backup raw JSON files for path coordinates mapping
backup_dir = DATA_DIR / "backup_sessions"
backup_files = []
if backup_dir.exists():
    backup_files = sorted(list(backup_dir.glob(f"{selected_profile}_*.json")), key=lambda x: x.stat().st_mtime, reverse=True)

# ------------------------------------------------------------------ #
# TAB 1: SUMMARY
# ------------------------------------------------------------------ #
with tab_summary:
    st.header(f"Profile Overview: {selected_profile}")
    
    # Overview metrics calculation
    if len(events) > 0:
        times = [e["press_ts"] for e in events if e.get("press_ts") is not None and not np.isnan(e.get("press_ts"))]
        span_min = (max(times) - min(times)) / 60000.0 if len(times) > 1 else 1e-6
        avg_wpm = (len(events) / 5.0) / max(span_min, 0.1)
        dwells = [e["dwell_ms"] for e in events if is_valid_num(e.get("dwell_ms")) and 0 < e.get("dwell_ms") < 1000]
        avg_dwell = np.mean(dwells) if dwells else 0.0
        
        flights = []
        for i in range(len(events)-1):
            a, b = events[i], events[i+1]
            p_a = a.get("press_ts")
            r_a = a.get("release_ts")
            p_b = b.get("press_ts")
            if p_a is None or np.isnan(p_a) or r_a is None or np.isnan(r_a) or p_b is None or np.isnan(p_b):
                continue
            fl = p_b - r_a
            if 0 < fl < 2000:
                flights.append(fl)
        avg_flight = np.mean(flights) if flights else 0.0
    else:
        avg_wpm, avg_dwell, avg_flight = 0.0, 0.0, 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="card"><div class="metric-label">Total Sessions</div><div class="metric-value">{len(profile_sessions)}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><div class="metric-label">Typing Speed</div><div class="metric-value">{avg_wpm:.1f} <span style="font-size: 1rem; color: #8A99AD;">WPM</span></div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><div class="metric-label">Avg Hold (Dwell)</div><div class="metric-value">{avg_dwell:.1f} <span style="font-size: 1rem; color: #8A99AD;">ms</span></div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="card"><div class="metric-label">Avg Interval (Flight)</div><div class="metric-value">{avg_flight:.1f} <span style="font-size: 1rem; color: #8A99AD;">ms</span></div></div>', unsafe_allow_html=True)

    # Sessions log
    st.subheader("Historical Sessions Log")
    st.dataframe(profile_sessions, use_container_width=True)

# ------------------------------------------------------------------ #
# TAB 2: KEYSTROKE RHYTHMICS
# ------------------------------------------------------------------ #
with tab_keys:
    st.header("Keyboard Rhythm & Timing Signature")
    
    if len(events) < 5:
        st.info("Insufficient keystroke events to render distributions.")
    else:
        df_keys = pd.DataFrame(events)
        
        # 1. Dwell and Flight distributions
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_keys[df_keys["dwell_ms"] < 600], x="dwell_ms", nbins=50, 
                               title="Key Dwell Time Distribution (Hold Time)", 
                               labels={"dwell_ms": "Dwell Time (ms)"},
                               color_discrete_sequence=["#00E5FF"], template="plotly_dark")
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            df_flights = pd.DataFrame({"flight_ms": flights}) if flights else pd.DataFrame(columns=["flight_ms"])
            if not df_flights.empty:
                fig = px.histogram(df_flights[df_flights["flight_ms"] < 1000], x="flight_ms", nbins=50, 
                                   title="Key Flight Time Distribution (Gap Time)", 
                                   labels={"flight_ms": "Flight Time (ms)"},
                                   color_discrete_sequence=["#7D2AE8"], template="plotly_dark")
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid flight times collected yet.")

        # 2. Transition Heatmap
        st.subheader("Key Transition Bigram Heatmap (Top Transitions)")
        bigrams = []
        for i in range(len(events)-1):
            a, b = events[i], events[i+1]
            p_a = a.get("press_ts")
            r_a = a.get("release_ts")
            p_b = b.get("press_ts")
            if p_a is None or np.isnan(p_a) or r_a is None or np.isnan(r_a) or p_b is None or np.isnan(p_b):
                continue
            fl = p_b - r_a
            if 0 < fl < 1000:
                bigrams.append({
                    "First": a["key_id"].upper(),
                    "Second": b["key_id"].upper(),
                    "Flight": fl
                })
        if bigrams:
            df_bi = pd.DataFrame(bigrams)
            top_keys = df_bi["First"].value_counts().head(10).index.tolist()
            df_pivot = df_bi[(df_bi["First"].isin(top_keys)) & (df_bi["Second"].isin(top_keys))]
            if not df_pivot.empty:
                pivot_tbl = df_pivot.pivot_table(index="First", columns="Second", values="Flight", aggfunc="mean").fillna(0)
                fig_heat = px.imshow(pivot_tbl, text_auto=".0f", color_continuous_scale="Purples",
                                     title="Transition Flight Gaps (ms) between Common Keys",
                                     labels=dict(color="Flight Time (ms)"), template="plotly_dark")
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Not enough key transition statistics to render transitions matrix.")
        else:
            st.info("No key transitions recorded.")

# ------------------------------------------------------------------ #
# TAB 3: ACTIVE/PASSIVE KINEMATICS
# ------------------------------------------------------------------ #
with tab_mouse:
    st.header("Active Target Trails & Pointer Pressure Dynamics")
    
    passive_pts = mouse.get("passive", [])
    dot_trials = mouse.get("dot_trials", [])
    drag_trials = mouse.get("drag_trials", [])
    
    # Attempt to load recent trial paths from raw backup JSONs
    recent_dot_paths = []
    recent_drag_paths = []
    latest_pressure_log = []
    
    if backup_files:
        try:
            with open(backup_files[0], "r") as f:
                raw_session = json.load(f)
            js_mouse = raw_session.get("mouse", {})
            recent_dot_paths = [t for t in js_mouse.get("dot_trials", []) if "path" in t and len(t["path"]) > 1]
            recent_drag_paths = [t for t in js_mouse.get("drag_trials", []) if "path" in t and len(t["path"]) > 1]
            
            # Extract pressure data from passive move points
            passive_events = js_mouse.get("passive_points", [])
            latest_pressure_log = [p.get("pressure", 0.5) for p in passive_events if "pressure" in p]
        except Exception as e:
            pass

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.subheader("🎯 Active Kinematics (Dot Clicking Task)")
        if recent_dot_paths:
            # Dropdown to select which dot trial path to visualize
            trial_sel = st.selectbox("Select Dot Trial Target Path to Plot", 
                                     options=range(len(recent_dot_paths)),
                                     format_func=lambda i: f"Trial {i+1} (Err: {recent_dot_paths[i].get('error_px', 0.0):.1f}px, Time: {recent_dot_paths[i].get('travel_time_ms', 0):.0f}ms)")
            
            t_data = recent_dot_paths[trial_sel]
            path_coords = t_data["path"]
            df_path = pd.DataFrame(path_coords)
            
            # Compute kinematics stats
            k_stats = extract_trial_path_kinematics(path_coords)
            
            # Plot path
            fig_path = go.Figure()
            # Draw straight line target vector
            fig_path.add_trace(go.Scatter(x=[path_coords[0]["x"], t_data["target_x"]],
                                          y=[path_coords[0]["y"], t_data["target_y"]],
                                          mode="lines", name="Target Straight Path",
                                          line=dict(color="#1f2a3f", dash="dash")))
            # Draw actual movement trajectory
            fig_path.add_trace(go.Scatter(x=df_path["x"], y=df_path["y"],
                                          mode="lines+markers", name="Actual Trajectory",
                                          line=dict(color="#00E5FF", width=3),
                                          marker=dict(size=6, color="#7D2AE8")))
            # Draw target circle
            fig_path.add_trace(go.Scatter(x=[t_data["target_x"]], y=[t_data["target_y"]],
                                          mode="markers", name="Target Endpoint",
                                          marker=dict(size=14, color="#00C853", symbol="circle-open", line=dict(width=3))))
            
            fig_path.update_yaxes(autorange="reversed")
            fig_path.update_layout(title="Active target movement deviation path", template="plotly_dark",
                                   xaxis_title="Screen X position (px)", yaxis_title="Screen Y position (px)")
            st.plotly_chart(fig_path, use_container_width=True)
            
            # Print calculated kinematics
            st.markdown(f"""
            **Calculated Kinematic Attributes for this Trial:**
            *   **Curvature Ratio**: `{k_stats['curvature']:.2f}` (1.0 = perfect straight path)
            *   **Peak Velocity**: `{k_stats['peak_velocity'] * 1000.0:.1f} px/sec`
            *   **Direction Reversals**: `{k_stats['direction_reversals']}`
            *   **Submovements Count**: `{k_stats['submovements']}`
            *   **Tremor index**: `{k_stats['tremor']:.3f}`
            """)
        elif dot_trials:
            df_dots = pd.DataFrame(dot_trials)
            fig = px.scatter(df_dots, x="travel_time_ms", y="error_px", 
                             title="Target Click Time vs Click Error",
                             labels={"travel_time_ms": "Travel Time (ms)", "error_px": "Click Error (pixels)"},
                             color_discrete_sequence=["#00E5FF"], template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dot target click trials recorded.")

    with col_k2:
        st.subheader("🎯 Active Kinematics (Drag and Drop Task)")
        if recent_drag_paths:
            # Dropdown to select which drag trial path to visualize
            drag_sel = st.selectbox("Select Drag Trial Target Path to Plot", 
                                     options=range(len(recent_drag_paths)),
                                     format_func=lambda i: f"Trial {i+1} (Success: {recent_drag_paths[i].get('success')}, Time: {recent_drag_paths[i].get('duration_ms', 0):.0f}ms)")
            
            t_data = recent_drag_paths[drag_sel]
            path_coords = t_data["path"]
            df_path = pd.DataFrame(path_coords)
            
            # Compute kinematics stats
            k_stats = extract_trial_path_kinematics(path_coords)
            
            # Plot path
            fig_path = go.Figure()
            # Draw actual movement trajectory
            fig_path.add_trace(go.Scatter(x=df_path["x"], y=df_path["y"],
                                          mode="lines+markers", name="Actual Trajectory",
                                          line=dict(color="#FF007F", width=3),
                                          marker=dict(size=6, color="#7D2AE8")))
            
            fig_path.update_yaxes(autorange="reversed")
            fig_path.update_layout(title="Active drag trajectory path", template="plotly_dark",
                                   xaxis_title="Screen X position (px)", yaxis_title="Screen Y position (px)")
            st.plotly_chart(fig_path, use_container_width=True)
            
            # Print calculated kinematics
            st.markdown(f"""
            **Calculated Kinematic Attributes for this Trial:**
            *   **Curvature Ratio**: `{k_stats['curvature']:.2f}` (1.0 = perfect straight path)
            *   **Peak Velocity**: `{k_stats['peak_velocity'] * 1000.0:.1f} px/sec`
            *   **Direction Reversals**: `{k_stats['direction_reversals']}`
            *   **Submovements Count**: `{k_stats['submovements']}`
            *   **Tremor index**: `{k_stats['tremor']:.3f}`
            """)
        elif drag_trials:
            df_drags = pd.DataFrame(drag_trials)
            fig = px.box(df_drags, y="duration_ms", points="all",
                         title="Drag Durations Boxplot",
                         labels={"duration_ms": "Duration (ms)"},
                         color_discrete_sequence=["#FF007F"], template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No drag and drop trials recorded.")

    # New Row: Passive Mouse Kinematics & Pressure logs
    st.write("---")
    st.subheader("🖱️ Passive Kinematics & Pointer Pressure Tracking")
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        if latest_pressure_log:
            fig_press = px.line(y=latest_pressure_log, title="Pointer Pressure Log during Passive Moves",
                                labels={"index": "Event sequence index", "y": "Pressure value (0.0 to 1.0)"},
                                color_discrete_sequence=["#00C853"], template="plotly_dark")
            st.plotly_chart(fig_press, use_container_width=True)
        else:
            st.info("No pointer pressure data log captured in the latest session. Default values (0.5) will be plotted.")
            
    with col_p2:
        if passive_pts:
            df_passive = pd.DataFrame(passive_pts).sort_values("ts").reset_index(drop=True)
            df_passive["dt"] = df_passive["ts"].diff() / 1000.0
            df_passive["dx"] = df_passive["x"].diff()
            df_passive["dy"] = df_passive["y"].diff()
            df_passive["dist"] = np.hypot(df_passive["dx"], df_passive["dy"])
            df_passive["speed"] = df_passive["dist"] / (df_passive["dt"] + 1e-6)
            
            df_clean = df_passive[(df_passive["dt"] > 0.001) & (df_passive["speed"] < 4000)].copy()
            
            fig_speed = px.histogram(df_clean, x="speed", nbins=50,
                                     title="Passive Movement Speed Distribution",
                                     labels={"speed": "Speed (pixels / second)"},
                                     color_discrete_sequence=["#00E5FF"], template="plotly_dark")
            fig_speed.update_layout(bargap=0.1)
            st.plotly_chart(fig_speed, use_container_width=True)
        else:
            st.info("No passive mouse trajectory points recorded yet.")

# ------------------------------------------------------------------ #
# TAB 4: PROFILE COMPARISON
# ------------------------------------------------------------------ #
with tab_compare:
    st.header("👥 Profile Rhythm Signatures Comparison")
    st.markdown("""
    Compare your typing rhythm speed, holds, and intervals side-by-side with other candidates.
    """)
    
    compare_selection = st.multiselect("Select Candidates to Compare", profiles, default=profiles[:min(3, len(profiles))])
    
    comp_data = []
    for p in compare_selection:
        p_events = load_keyboard_events(p)
        if p_events:
            p_times = [e["press_ts"] for e in p_events if e.get("press_ts") is not None and not np.isnan(e.get("press_ts"))]
            span_min = (max(p_times) - min(p_times)) / 60000.0 if len(p_times) > 1 else 1e-6
            p_wpm = (len(p_events) / 5.0) / max(span_min, 0.1)
            
            p_dwells = [e["dwell_ms"] for e in p_events if is_valid_num(e.get("dwell_ms")) and 0 < e.get("dwell_ms") < 1000]
            p_flights = []
            for i in range(len(p_events)-1):
                a, b = p_events[i], p_events[i+1]
                p_a = a.get("press_ts")
                r_a = a.get("release_ts")
                p_b = b.get("press_ts")
                if p_a is None or np.isnan(p_a) or r_a is None or np.isnan(r_a) or p_b is None or np.isnan(p_b):
                    continue
                fl = p_b - r_a
                if 0 < fl < 2000:
                    p_flights.append(fl)
            
            comp_data.append({
                "Subject ID": p,
                "WPM": float(p_wpm),
                "Avg Dwell (Hold)": float(np.mean(p_dwells)) if p_dwells else 0.0,
                "Avg Flight (Gap)": float(np.mean(p_flights)) if p_flights else 0.0,
                "Dwell Std": float(np.std(p_dwells)) if p_dwells else 0.0,
                "Flight Std": float(np.std(p_flights)) if p_flights else 0.0
            })
            
    if comp_data:
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, use_container_width=True)
        
        # Radar Chart comparison
        fig_radar = go.Figure()
        categories = ["WPM", "Avg Dwell (Hold)", "Avg Flight (Gap)", "Dwell Std", "Flight Std"]
        
        for idx, row in df_comp.iterrows():
            values = [row[cat] for cat in categories]
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=row["Subject ID"]
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(df_comp["Avg Flight (Gap)"].max() * 1.2, 300.0)])),
            showlegend=True, title="Profile Rhythm Polar Signatures (ms)", template="plotly_dark"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No comparative metrics available yet.")

# ------------------------------------------------------------------ #
# TAB 5: MODELS & DRIFT MONITOR
# ------------------------------------------------------------------ #
with tab_models:
    st.header("ML Verification Models & Longitudinal Drift")
    
    # Model config JSON status
    status = get_training_status(selected_profile)
    
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.subheader("Model Status Metadata")
        st.json(status)
        
    with col_m2:
        st.subheader("Profile Drift Over Stored Sessions")
        # Let's extract the actual keyboard statistics from the database sessions to draw an actual drift chart
        profile_sess_sorted = profile_sessions.sort_values("collected_at").reset_index(drop=True)
        if len(profile_sess_sorted) >= 2:
            # We compute a real drift proxy: standard deviation deviation from first session
            # using keyboard holds count and session duration
            means = []
            for idx, row in profile_sess_sorted.iterrows():
                # Estimate a rhythm rate metric
                rate = float(row.get("keyboard_events_count", 100)) / (float(row.get("duration_ms", 60000.0)) / 1000.0)
                means.append(rate)
            
            # Map distance from centroid baseline (first session value)
            dist_baseline = np.abs(np.array(means) - means[0])
            
            fig_drift = go.Figure()
            fig_drift.add_trace(go.Scatter(x=profile_sess_sorted["collected_at"], y=dist_baseline,
                                           mode="lines+markers", name="Keyboard Style Distance",
                                           line=dict(color="#00C853", width=3)))
            # Add alerts threshold
            fig_drift.add_trace(go.Scatter(x=profile_sess_sorted["collected_at"], y=[0.8]*len(means),
                                           mode="lines", name="Style Drift Warn Limit",
                                           line=dict(color="#FFD600", dash="dash")))
            fig_drift.add_trace(go.Scatter(x=profile_sess_sorted["collected_at"], y=[1.8]*len(means),
                                           mode="lines", name="Staleness / Retrain Limit",
                                           line=dict(color="#D50000", dash="dash")))
            
            fig_drift.update_layout(title="Rhythm rate deviation from baseline over time",
                                    xaxis_title="Session Timestamp", yaxis_title="Style Deviation score",
                                    template="plotly_dark")
            st.plotly_chart(fig_drift, use_container_width=True)
        else:
            # Draw standard simulator if only one session exists
            t = np.arange(1, 30)
            drift = 0.5 + 0.1 * np.sin(t / 2.0) + 0.01 * t
            fig_drift = go.Figure()
            fig_drift.add_trace(go.Scatter(x=t, y=drift, mode='lines+markers', name='Distance from Centroid', line=dict(color='#00C853')))
            fig_drift.add_trace(go.Scatter(x=t, y=[1.2]*29, mode='lines', name='Warn Limit', line=dict(color='#FFD600', dash='dash')))
            fig_drift.add_trace(go.Scatter(x=t, y=[1.8]*29, mode='lines', name='Retrain Limit', line=dict(color='#D50000', dash='dash')))
            fig_drift.update_layout(title="Simulated style deviation over 30 days (profile needs at least 2 sessions to compute actual drift)",
                                    xaxis_title="Session Index", yaxis_title="Style Deviation score", template="plotly_dark")
            st.plotly_chart(fig_drift, use_container_width=True)

    # Keyboard SVM feature consistency analysis
    st.write("---")
    st.subheader("🔍 Keyboard SVM Feature Consistency Analysis")
    st.markdown("""
    The One-Class SVM baseline models your typing consistency across 23 different timing and pattern features.
    Features with **lower standard deviation scales** indicate where your typing rhythm is the most consistent and unique!
    """)
    
    from behaveguard.models.svm import BehaveGuardSVM
    svm_path = Path("/Users/akshitmehta/Development/behave-guard/behaveguard/models") / selected_profile / "svm.pkl"
    if svm_path.exists():
        try:
            svm = BehaveGuardSVM()
            svm.load(svm_path)
            
            features_23 = [
                "Dwell Mean", "Dwell Std", "Flight Mean", "Flight Std", 
                "Digraph Mean", "Digraph Std", "Alphanum Dwell Mean", "Alphanum Dwell Std", 
                "Symbol Dwell Mean", "Symbol Dwell Std", "Special Dwell Mean", "Special Dwell Std", 
                "IKI Mean", "IKI Std", "FD Ratio Mean", "FD Ratio Std", 
                "Cyclical Time Sin", "Cyclical Time Cos", "Alphanum Ratio", "Symbol Ratio", 
                "Special Ratio", "WPM Metric", "Digraph Coverage"
            ]
            
            if hasattr(svm.scaler, 'scale_') and len(svm.scaler.scale_) == 23:
                df_features = pd.DataFrame({
                    "Feature": features_23,
                    "Consistency Scale (Std Dev)": svm.scaler.scale_,
                    "Baseline Mean": svm.scaler.mean_
                }).sort_values("Consistency Scale (Std Dev)").reset_index(drop=True)
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    fig_scales = px.bar(df_features, x="Consistency Scale (Std Dev)", y="Feature", orientation="h",
                                         title="Rhythm Consistency (Lower = More Consistent / Stable)",
                                         color="Consistency Scale (Std Dev)", color_continuous_scale="Viridis_r",
                                         template="plotly_dark")
                    fig_scales.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_scales, use_container_width=True)
                    
                with col_f2:
                    st.markdown("**Top 5 Most Consistent Typing Traits:**")
                    for i in range(min(5, len(df_features))):
                        row = df_features.iloc[i]
                        st.markdown(f"{i+1}. **{row['Feature']}** (Scale Variance: `{row['Consistency Scale (Std Dev)']:.2f}`)")
                    
                    st.markdown("""
                    > [!NOTE]
                    > During model calibration, a scale floor of `0.20` is enforced on keyboard aggregates to prevent overfitting.
                    > Traits resting exactly at `0.20` represent highly regular keystroke patterns.
                    """)
            else:
                st.info("The loaded SVM profile does not contain 23 aggregates features.")
        except Exception as e:
            st.error(f"Failed to render SVM feature dynamics: {str(e)}")
    else:
        st.info("No trained keyboard SVM model baseline found for this profile yet.")

# ------------------------------------------------------------------ #
# TAB 6: BIOMETRIC CLUSTER SPACE
# ------------------------------------------------------------------ #
with tab_clusters:
    st.header("Behavioral Biometric Projection")
    st.markdown("""
    PCA projection of keyboard aggregates across all enrolled candidates.
    Distinct grouping confirms that typing rhythms represent unique behavioral profiles.
    """)

    all_features = []
    labels = []
    
    for p in get_enrolled_profiles():
        p_events = load_keyboard_events(p)
        if len(p_events) >= 50:
            p_aggs = extract_keystroke_aggregates(p_events, win_size=50, stride=5)
            all_features.extend(p_aggs)
            labels.extend([p] * len(p_aggs))
            
    if len(all_features) < 3:
        st.warning("Not enough aggregate windows across enrolled profiles to run Principal Component Analysis (PCA). Keep typing in the client!")
    else:
        X = np.stack(all_features)
        scaler_all = StandardScaler()
        X_scaled = scaler_all.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        df_pca = pd.DataFrame(X_pca, columns=["Principal Component 1", "Principal Component 2"])
        df_pca["Subject ID"] = labels
        
        fig = px.scatter(df_pca, x="Principal Component 1", y="Principal Component 2", 
                         color="Subject ID", 
                         title="PCA rhythm space grouping with SVM Decision Boundary",
                         labels={"Principal Component 1": "PC1 (Rhythm Speed)", "Principal Component 2": "PC2 (Digraph Variance)"},
                         color_discrete_sequence=px.colors.qualitative.Bold, template="plotly_dark")
        
        # Plot 2D One-Class SVM decision boundary trained on the selected user's PCA coordinates
        # to represent their genuine zone in the 2D projection plane.
        try:
            X_pca_selected = X_pca[np.array(labels) == selected_profile]
            if len(X_pca_selected) >= 5:
                # Meshgrid bounding space
                x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
                y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                grid_pca = np.c_[xx.ravel(), yy.ravel()]
                
                # Fit a 2D OneClassSVM to align with the visual 2D PCA representation
                from sklearn.svm import OneClassSVM
                svm_2d = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
                svm_2d.fit(X_pca_selected)
                
                z = svm_2d.decision_function(grid_pca)
                z = z.reshape(xx.shape)
                
                # Overlay boundary contour line at value = 0.0
                fig.add_trace(go.Contour(
                    x=np.linspace(x_min, x_max, 100),
                    y=np.linspace(y_min, y_max, 100),
                    z=z,
                    showscale=False,
                    contours=dict(start=0.0, end=0.0, size=1),
                    contours_coloring='lines',
                    line=dict(color="#00E5FF", width=3, dash="dash"),
                    name=f"{selected_profile} SVM Boundary (Genuine Zone)"
                ))
            else:
                st.sidebar.warning(f"Not enough aggregate windows for {selected_profile} to compute 2D boundary.")
        except Exception as ex:
            st.sidebar.error(f"SVM Boundary Overlay Error: {str(ex)}")
                
        for trace in fig.data:
            if hasattr(trace, 'marker') and trace.type == 'scatter':
                trace.marker.size = 10
                trace.marker.opacity = 0.9
                trace.marker.line = dict(width=1.0, color='white')
                
        st.plotly_chart(fig, use_container_width=True)
