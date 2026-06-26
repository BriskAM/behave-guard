import sys
from pathlib import Path
# Add project root to python path to avoid ModuleNotFoundError
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from behaveguard.storage import (
    DATA_DIR, get_enrolled_profiles, load_keyboard_events, load_mouse_data
)
from behaveguard.features import extract_keystroke_aggregates, extract_keystroke_sequences
from behaveguard.pipeline import get_training_status

# Page config
st.set_page_config(
    page_title="BehaveGuard Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #FFB300 0%, #00E5FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-family: 'Roboto Mono', monospace;
        color: #757575;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>BehaveGuard</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Continuous Behavioral Authentication Engine & Fingerprint Analytics</p>", unsafe_allow_html=True)

# Check database
if not (DATA_DIR / "sessions.csv").exists():
    st.warning("⚠️ No database CSVs found yet. Run the FastAPI server first to initialize the database from the Excel file.")
    st.stop()

# Sidebar controls
st.sidebar.title("🛡️ Profile Inspector")
profiles = get_enrolled_profiles()
if not profiles:
    st.sidebar.warning("No profiles found in the database. Enroll some profiles using the client.")
    st.stop()

selected_profile = st.sidebar.selectbox("Select Profile to Inspect", profiles)

# Tab structure
tab_summary, tab_keys, tab_mouse, tab_models, tab_clusters = st.tabs([
    "📊 User Session Summary", 
    "⌨️ Keystroke Dynamics", 
    "🖱️ Mouse Dynamics", 
    "🤖 ML Models & Drift",
    "🌌 Biometric Cluster Space"
])

# Load selected profile data
events = load_keyboard_events(selected_profile)
mouse = load_mouse_data(selected_profile)
sessions_df = pd.read_csv(DATA_DIR / "sessions.csv")
profile_sessions = sessions_df[sessions_df["subject_id"] == selected_profile]

# ------------------------------------------------------------------ #
# TAB 1: SUMMARY
# ------------------------------------------------------------------ #
with tab_summary:
    st.header(f"Profile: {selected_profile}")
    
    # Overview metrics
    if len(events) > 0:
        # Calculate WPM
        times = [e["press_ts"] for e in events]
        span_min = (max(times) - min(times)) / 60000.0 if len(times) > 1 else 1e-6
        avg_wpm = (len(events) / 5.0) / max(span_min, 0.1)
        
        dwells = [e["dwell_ms"] for e in events]
        avg_dwell = np.mean(dwells)
        
        # Calculate average flight time
        flights = []
        for i in range(len(events)-1):
            fl = events[i+1]["press_ts"] - events[i]["release_ts"]
            if 0 < fl < 2000:
                flights.append(fl)
        avg_flight = np.mean(flights) if flights else 0.0
    else:
        avg_wpm, avg_dwell, avg_flight = 0.0, 0.0, 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions", len(profile_sessions))
    with col2:
        st.metric("Typing Speed (WPM)", f"{avg_wpm:.1f}")
    with col3:
        st.metric("Avg Key Dwell", f"{avg_dwell:.1f} ms")
    with col4:
        st.metric("Avg Key Flight", f"{avg_flight:.1f} ms")

    # Sessions log
    st.subheader("Recorded Sessions Log")
    st.dataframe(profile_sessions, use_container_width=True)

# ------------------------------------------------------------------ #
# TAB 2: KEYSTROKE DYNAMICS
# ------------------------------------------------------------------ #
with tab_keys:
    st.header("Keyboard Rhythm Analytics")
    
    if len(events) < 5:
        st.info("Insufficient keystroke events to render distributions.")
    else:
        df_keys = pd.DataFrame(events)
        
        # 1. Dwell and Flight distributions
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_keys[df_keys["dwell_ms"] < 1000], x="dwell_ms", nbins=50, 
                               title="Dwell Time Distribution (Key Hold Time)", 
                               labels={"dwell_ms": "Dwell Time (ms)"},
                               color_discrete_sequence=["#FFB300"])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            df_flights = pd.DataFrame({"flight_ms": flights}) if flights else pd.DataFrame(columns=["flight_ms"])
            if not df_flights.empty:
                fig = px.histogram(df_flights[df_flights["flight_ms"] < 1500], x="flight_ms", nbins=50, 
                                   title="Flight Time Distribution (Key-to-Key Gap)", 
                                   labels={"flight_ms": "Flight Time (ms)"},
                                   color_discrete_sequence=["#FF8F00"])
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid flight times collected yet.")

        # 2. Category distributions
        st.subheader("Key Category Ratios")
        cat_counts = df_keys["key_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.pie(cat_counts, names="Category", values="Count", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# TAB 3: MOUSE DYNAMICS
# ------------------------------------------------------------------ #
with tab_mouse:
    st.header("Mouse Trajectory & Click Analytics")
    
    passive_pts = mouse.get("passive", [])
    dot_trials = mouse.get("dot_trials", [])
    drag_trials = mouse.get("drag_trials", [])

    if not passive_pts and not dot_trials:
        st.info("No mouse dynamics data collected for this profile yet.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy: Travel Time vs Click Error")
            if dot_trials:
                df_dots = pd.DataFrame(dot_trials)
                fig = px.scatter(df_dots, x="travel_time_ms", y="error_px", 
                                 title="Target Clicks Accuracy",
                                 labels={"travel_time_ms": "Travel Time (ms)", "error_px": "Click Error (pixels)"},
                                 color_discrete_sequence=["#00E5FF"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No target-clicking dot task records.")
                
        with col2:
            st.subheader("Drag Task Metrics")
            if drag_trials:
                df_drags = pd.DataFrame(drag_trials)
                success_rate = df_drags["success"].mean() * 100.0
                avg_dur = df_drags["duration_ms"].mean()
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Drag Success Rate", f"{success_rate:.1f}%")
                with col_m2:
                    st.metric("Avg Drag Duration", f"{avg_dur:.1f} ms")
                
                fig = px.box(df_drags, y="duration_ms", points="all",
                             title="Drag Durations Boxplot",
                             labels={"duration_ms": "Duration (ms)"},
                             color_discrete_sequence=["#00B8D4"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No drag-and-drop task records.")

        # New Row: Passive Mouse Kinematics
        st.write("---")
        st.subheader("Passive Kinematics & Cursor Trajectory")
        if passive_pts:
            df_passive = pd.DataFrame(passive_pts).sort_values("ts").reset_index(drop=True)
            df_passive["dt"] = df_passive["ts"].diff() / 1000.0
            df_passive["dx"] = df_passive["x"].diff()
            df_passive["dy"] = df_passive["y"].diff()
            df_passive["dist"] = np.hypot(df_passive["dx"], df_passive["dy"])
            df_passive["speed"] = df_passive["dist"] / (df_passive["dt"] + 1e-6)
            
            # Clean outliers
            df_clean = df_passive[(df_passive["dt"] > 0.001) & (df_passive["speed"] < 5000)].copy()
            
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                fig_speed = px.histogram(df_clean, x="speed", nbins=50,
                                         title="Passive Movement Speed Distribution",
                                         labels={"speed": "Speed (pixels / second)"},
                                         color_discrete_sequence=["#00E5FF"])
                fig_speed.update_layout(bargap=0.1)
                st.plotly_chart(fig_speed, use_container_width=True)
                
            with col_p2:
                # Plot a subset of points (e.g. up to 400 points) to show a trajectory path
                df_traj = df_clean.head(400)
                fig_traj = px.line(df_traj, x="x", y="y", markers=True,
                                   title="Recent Mouse Cursor Path (First 400 points)",
                                   labels={"x": "X Position (px)", "y": "Y Position (px)"},
                                   color_discrete_sequence=["#D50000"])
                fig_traj.update_yaxes(autorange="reversed") # Match screen coordinate system
                st.plotly_chart(fig_traj, use_container_width=True)
        else:
            st.info("No passive mouse trajectory points recorded yet.")

# ------------------------------------------------------------------ #
# TAB 4: MODELS & DRIFT
# ------------------------------------------------------------------ #
with tab_models:
    st.header("Machine Learning Verification Status")
    
    status = get_training_status(selected_profile)
    st.subheader("Model Status Overview")
    
    # Progress visualization
    st.json(status)

    # Simulated longitudinal drift warning
    st.subheader("Continuous Authentication Drift Detector")
    st.markdown("""
    Continuous learning requires monitoring shifts in user habits over time (e.g. fatigue, coffee intake, keyboard changes).
    Below is the simulated profile drift monitor compared to the enrollment baseline:
    """)
    
    # Build simulated drift graph
    t = np.arange(1, 31)
    drift = 0.5 + 0.1 * np.sin(t / 2.0) + 0.01 * t # baseline + noise + linear drift
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=drift, mode='lines+markers', name='Distance from Profile Centroid', line=dict(color='#00C853')))
    # Threshold lines
    fig.add_trace(go.Scatter(x=t, y=[1.2]*30, mode='lines', name='Suspicious Drift Threshold (T_drift)', line=dict(color='#FFD600', dash='dash')))
    fig.add_trace(go.Scatter(x=t, y=[1.8]*30, mode='lines', name='Poisoning / Impostor Threshold (T_anomaly)', line=dict(color='#D50000', dash='dash')))
    
    fig.update_layout(
        title="Session Mahalanobis Distance Over 30 Days",
        xaxis_title="Session Index",
        yaxis_title="Mahalanobis Distance",
        legend_title="Threshold Status"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# TAB 5: BIOMETRIC CLUSTER SPACE
# ------------------------------------------------------------------ #
with tab_clusters:
    st.header("Biometric Fingerprint Space Visualization")
    st.markdown("""
    This PCA projection shows how keyboard rhythm aggregate features cluster users.
    Each dot represents a 50-keystroke aggregate window. Distinct clusters indicate unique timing signatures.
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
        
        # PCA projection to 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        df_pca = pd.DataFrame(X_pca, columns=["Principal Component 1", "Principal Component 2"])
        df_pca["Subject ID"] = labels
        
        fig = px.scatter(df_pca, x="Principal Component 1", y="Principal Component 2", 
                         color="Subject ID", 
                         title="PCA Behavioral Biometrics Projection (Keyboard Dynamics)",
                         labels={"Principal Component 1": "PC1 (Rhythm Speed / Variance)", "Principal Component 2": "PC2 (Digraph Patterns)"},
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)
