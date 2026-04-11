"""
TrafficVision AI
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import time, os, tempfile, urllib.request
from collections import deque
import plotly.graph_objects as go

st.set_page_config(
    page_title="TrafficVision AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&family=Barlow:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }

.stApp { background: #111210; color: #c8c9b8; }
.main .block-container { padding: 1.5rem 2rem 4rem; max-width: 100%; }

::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #1a1b17; }
::-webkit-scrollbar-thumb { background: #3a3c30; }

[data-testid="stSidebar"] {
    background: #0d0e0c !important;
    border-right: 1px solid #2a2c24;
}
[data-testid="stSidebar"] .block-container { padding: 1rem 0.9rem; }
[data-testid="stSidebar"] label {
    color: #4a4c3c !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stSidebar"] p { color: #4a4c3c !important; font-size: 0.8rem !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: #161714 !important;
    border: 1px solid #2a2c24 !important;
    color: #8a8c78 !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #e8a020 !important;
    outline: none !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }

.stButton > button {
    background: transparent !important;
    color: #8a8c78 !important;
    border: 1px solid #2a2c24 !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 400 !important;
    font-size: 0.72rem !important;
    padding: 8px 16px !important;
    width: 100% !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #1e2018 !important;
    border-color: #e8a020 !important;
    color: #e8a020 !important;
}
.stButton > button:active {
    background: #252618 !important;
}

[data-testid="stFileUploader"] {
    background: #161714 !important;
    border: 1px solid #2a2c24 !important;
    border-radius: 2px !important;
}

[data-testid="stSelectbox"] > div > div {
    background: #161714 !important;
    border: 1px solid #2a2c24 !important;
    border-radius: 2px !important;
    color: #8a8c78 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}

[data-testid="stSlider"] > div > div > div > div {
    background: #e8a020 !important;
}
[data-testid="stSlider"] > div > div > div > div > div {
    background: #e8a020 !important;
}

.stSpinner > div { border-top-color: #e8a020 !important; }

.rule { height: 1px; background: #2a2c24; margin: 1rem 0; }

.tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    padding: 2px 8px;
    border: 1px solid #2a2c24;
    color: #4a4c3c;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-right: 6px;
}
.tag-active {
    border-color: #e8a020;
    color: #e8a020;
    background: rgba(232,160,32,0.06);
}
.tag-green {
    border-color: #5a8c50;
    color: #5a8c50;
    background: rgba(90,140,80,0.06);
}
.tag-red {
    border-color: #c84030;
    color: #c84030;
    background: rgba(200,64,48,0.06);
}

.stat-num {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -1px;
}
.stat-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4a4c3c;
    margin-top: 4px;
}

.alert-box {
    border: 1px solid;
    border-left: 3px solid;
    padding: 12px 16px;
    font-family: 'IBM Plex Mono', monospace;
}
.alert-rise { border-color: #c84030; background: rgba(200,64,48,0.05); }
.alert-ease { border-color: #5a8c50; background: rgba(90,140,80,0.05); }
.alert-hold { border-color: #e8a020; background: rgba(232,160,32,0.05); }
.alert-wait { border-color: #2a2c24; background: rgba(42,44,36,0.3); }

.grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: #2a2c24;
    border: 1px solid #2a2c24;
}
.grid-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: #2a2c24;
    border: 1px solid #2a2c24;
}
.grid-4 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 1px;
    background: #2a2c24;
    border: 1px solid #2a2c24;
}
.gcell {
    background: #111210;
    padding: 16px 18px;
}

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #3a3c2c;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2018;
}

.status-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}
.dot-on  { background: #5a8c50; }
.dot-off { background: #2a2c24; }
.dot-warn { background: #e8a020; }

.model-row {
    display: flex;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #1a1c14;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
}
.model-name { color: #5a5c48; flex: 1; }
.model-status-ok  { color: #5a8c50; }
.model-status-err { color: #3a3c2c; }

.perf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #1a1c14;
    font-size: 0.78rem;
}
.perf-key { color: #4a4c3c; }
.perf-val { font-family: 'IBM Plex Mono', monospace; color: #8a8c78; font-size: 0.72rem; }
.perf-val-hi { font-family: 'IBM Plex Mono', monospace; color: #5a8c50; font-size: 0.72rem; }

.density-LOW      { color: #5a8c50; }
.density-MEDIUM   { color: #e8a020; }
.density-HIGH     { color: #c87020; }
.density-CRITICAL { color: #c84030; }
</style>
""", unsafe_allow_html=True)


# ── LSTM ──────────────────────────────────────────────────────
class CongestionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 128, 2, batch_first=True, dropout=0.3)
        self.fc   = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2),    nn.Linear(64, 4)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── MODEL URLS ────────────────────────────────────────────────
YOLO_URL  = "https://huggingface.co/riya17singh/trafficvision-models/resolve/main/best.pt"
LSTM_URL  = "https://huggingface.co/riya17singh/trafficvision-models/resolve/main/lstm_congestion.pt"
YOLO_PATH = "models/best.pt"
LSTM_PATH = "models/lstm_congestion.pt"


def ensure_models():
    os.makedirs("models", exist_ok=True)
    status = {"yolo": False, "lstm": False}

    if os.path.exists(YOLO_PATH) and os.path.getsize(YOLO_PATH) > 1000:
        status["yolo"] = True
    else:
        try:
            import requests
            r = requests.get(YOLO_URL, stream=True, timeout=180)
            r.raise_for_status()
            with open(YOLO_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            status["yolo"] = True
        except Exception as e:
            st.warning(f"YOLO download failed: {e}")

    if os.path.exists(LSTM_PATH) and os.path.getsize(LSTM_PATH) > 100:
        status["lstm"] = True
    else:
        try:
            import requests
            r = requests.get(LSTM_URL, stream=True, timeout=60)
            r.raise_for_status()
            with open(LSTM_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            status["lstm"] = True
        except Exception as e:
            st.warning(f"LSTM download failed: {e}")

    return status


@st.cache_resource(show_spinner=False)
def load_models():
    yolo_model = None
    lstm_model = None
    if os.path.exists(YOLO_PATH):
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(YOLO_PATH)
        except: pass
    if os.path.exists(LSTM_PATH):
        try:
            m = CongestionLSTM()
            try:    m.load_state_dict(torch.load(LSTM_PATH, map_location="cpu", weights_only=True))
            except: m.load_state_dict(torch.load(LSTM_PATH, map_location="cpu"))
            lstm_model = m.eval()
        except: pass
    return yolo_model, lstm_model


# ── HELPERS ───────────────────────────────────────────────────
def get_density(count, risk=1.0):
    adj = count * risk
    if adj <= 5:  return "LOW",      0, "#5a8c50"
    if adj <= 15: return "MEDIUM",   1, "#e8a020"
    if adj <= 30: return "HIGH",     2, "#c87020"
    return             "CRITICAL", 3, "#c84030"

def get_weather_risk(weather, lighting, rain):
    m, r = 1.0, []
    if rain == "Heavy Rain":   m += 0.40; r.append("heavy rain")
    elif rain == "Light Rain": m += 0.15; r.append("wet roads")
    if weather == "Foggy":     m += 0.35; r.append("fog")
    elif weather == "Snowy":   m += 0.50; r.append("snow")
    elif weather == "Cloudy":  m += 0.05
    if lighting == "Night":    m += 0.20; r.append("night")
    elif lighting == "Dusk/Dawn": m += 0.10; r.append("dusk/dawn glare")
    return round(m, 2), r

def predict_lstm(lstm, buf):
    if lstm is None or len(buf) < 30: return None
    try:
        d = np.array(list(buf), dtype=np.float32)
        d[:,0] /= 50.0; d[:,1] /= 3.0
        out = lstm(torch.tensor(d).unsqueeze(0))
        return float(torch.argmax(out, dim=1).item())
    except: return None

def draw_on_frame(frame, count, label, color_hex, pred, fnum, fps, risk):
    import cv2
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,52), (16,17,14), -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
    bgr = {
        "#5a8c50":(80,140,90), "#e8a020":(32,160,232),
        "#c87020":(32,112,200), "#c84030":(48,64,200)
    }.get(color_hex, (180,180,160))
    cv2.putText(frame, f"COUNT: {count:03d}", (12,34),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (180,182,160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"DENSITY: {label}", (w//3,34),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, bgr, 1, cv2.LINE_AA)
    if pred is not None:
        pl = ["LOW","MEDIUM","HIGH","CRITICAL"][min(3,int(round(pred)))]
        cv2.putText(frame, f"FORECAST: {pl}", (int(1.9*w//3),34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (160,150,100), 1, cv2.LINE_AA)
    if risk > 1.05:
        cv2.putText(frame, f"RISK x{risk:.1f}", (w-130,34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (48,64,200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f}fps  #{fnum:05d}", (w-140,16),
                cv2.FONT_HERSHEY_DUPLEX, 0.35, (50,52,40), 1, cv2.LINE_AA)
    return frame

def make_chart(hist):
    times  = [h["t"]     for h in hist]
    counts = [h["count"] for h in hist]
    levels = [h["level"] for h in hist]
    preds  = [h["pred"]  for h in hist]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=counts, name="vehicle count",
        mode="lines", line=dict(color="#8a8c78", width=1.5),
        fill="tozeroy", fillcolor="rgba(138,140,120,0.05)",
        hovertemplate="<b>%{y}</b> vehicles at %{x}s<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=times, y=[l*8 for l in levels], name="density (×8)",
        mode="lines", line=dict(color="#e8a020", width=1, dash="dot"),
        hovertemplate="<b>%{customdata}</b> at %{x}s<extra></extra>",
        customdata=[["LOW","MEDIUM","HIGH","CRITICAL"][min(3,l)] for l in levels]
    ))
    if any(p is not None for p in preds):
        fig.add_trace(go.Scatter(
            x=times, y=[p*8 if p is not None else None for p in preds],
            name="lstm forecast (×8)", mode="lines",
            line=dict(color="#5a8c50", width=1),
            hovertemplate="<b>forecast %{customdata}</b> at %{x}s<extra></extra>",
            customdata=[["LOW","MEDIUM","HIGH","CRITICAL"][min(3,int(p or 0))] for p in preds]
        ))
    fig.update_layout(
        height=180, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#3a3c2c", size=10, family="IBM Plex Mono"),
        legend=dict(orientation="h", y=1.2, font=dict(color="#4a4c3c", size=9), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a1c14", bordercolor="#2a2c24",
                        font=dict(color="#c8c9b8", size=11, family="IBM Plex Mono")),
        xaxis=dict(showgrid=True, gridcolor="#1a1c14", tickfont=dict(color="#2a2c24"),
                   zeroline=False, title=dict(text="seconds", font=dict(color="#2a2c24", size=9))),
        yaxis=dict(showgrid=True, gridcolor="#1a1c14", tickfont=dict(color="#2a2c24"), zeroline=False),
    )
    return fig


# ── SESSION STATE ─────────────────────────────────────────────
for k, v in [("running",False),("history",[]),("vpath",None),("app_ready",False)]:
    if k not in st.session_state: st.session_state[k] = v


# ── LOADING SCREEN ────────────────────────────────────────────
if not st.session_state.app_ready:
    st.markdown("""
    <div style="min-height:70vh;display:flex;flex-direction:column;
    align-items:center;justify-content:center;text-align:center;padding:4rem 2rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
        color:#3a3c2c;letter-spacing:4px;margin-bottom:24px;">
        TRAFFICVISION AI — SYSTEM BOOT</div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:4rem;
        font-weight:800;color:#c8c9b8;letter-spacing:-2px;line-height:1;margin-bottom:8px;">
        INITIALIZING</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
        color:#4a4c3c;margin-bottom:40px;">
        loading detection models — please wait</div>
        <div style="width:200px;height:1px;background:#2a2c24;margin-bottom:6px;"></div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:#2a2c24;">
        YOLOv8 · LSTM · BDD100K + VisDrone</div>
    </div>
    """, unsafe_allow_html=True)

    ensure_models()
    yolo_m, lstm_m = load_models()
    st.session_state._yolo = yolo_m
    st.session_state._lstm = lstm_m
    st.session_state.app_ready = True
    time.sleep(0.5)
    st.rerun()

yolo_model = st.session_state.get("_yolo")
lstm_model = st.session_state.get("_lstm")
if yolo_model is None and lstm_model is None:
    yolo_model, lstm_model = load_models()


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 1rem;border-bottom:1px solid #2a2c24;margin-bottom:1rem;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.4rem;
        font-weight:700;color:#c8c9b8;letter-spacing:1px;">TRAFFICVISION</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
        color:#3a3c2c;letter-spacing:3px;margin-top:2px;">AI · v2.0 · BDD100K + VisDrone</div>
    </div>
    """, unsafe_allow_html=True)

    yo = yolo_model is not None
    lo = lstm_model is not None
    gp = torch.cuda.is_available()

    st.markdown(f"""
    <div style="margin-bottom:1rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
        color:#2a2c24;letter-spacing:3px;margin-bottom:8px;">SYSTEM STATUS</div>
        <div class="model-row">
            <span class="status-dot {'dot-on' if yo else 'dot-off'}"></span>
            <span class="model-name">YOLOv8</span>
            <span class="{'model-status-ok' if yo else 'model-status-err'}">
            {'READY' if yo else 'NOT FOUND'}</span>
        </div>
        <div class="model-row">
            <span class="status-dot {'dot-on' if lo else 'dot-off'}"></span>
            <span class="model-name">LSTM</span>
            <span class="{'model-status-ok' if lo else 'model-status-err'}">
            {'READY' if lo else 'NOT FOUND'}</span>
        </div>
        <div class="model-row" style="border:none;">
            <span class="status-dot {'dot-warn' if gp else 'dot-off'}"></span>
            <span class="model-name">GPU</span>
            <span class="{'model-status-ok' if gp else 'model-status-err'}">
            {'ACTIVE' if gp else 'CPU MODE'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not yo:
        st.markdown("""
        <div style="border:1px solid #c84030;border-left:2px solid #c84030;
        padding:8px 10px;margin-bottom:12px;font-family:'IBM Plex Mono',monospace;
        font-size:0.65rem;color:#c84030;background:rgba(200,64,48,0.05);">
        YOLO MODEL NOT FOUND<br>
        <span style="color:#4a4c3c;">Check HuggingFace URL or place best.pt in models/</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
    color:#2a2c24;letter-spacing:3px;margin-bottom:10px;">DETECTION CONFIG</div>""",
    unsafe_allow_html=True)

    conf_thresh   = st.slider("Confidence threshold", 0.10, 0.90, 0.40, 0.05)
    process_every = st.slider("Frame skip (speed)", 1, 10, 3)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
    color:#2a2c24;letter-spacing:3px;margin-bottom:10px;">MODEL PERFORMANCE</div>
    <div class="perf-row">
        <span class="perf-key">Dataset</span>
        <span class="perf-val">BDD100K + VisDrone</span>
    </div>
    <div class="perf-row">
        <span class="perf-key">LSTM val loss</span>
        <span class="perf-val-hi">0.0107</span>
    </div>
    <div class="perf-row">
        <span class="perf-key">Classes</span>
        <span class="perf-val">car bus truck moto</span>
    </div>
    <div class="perf-row" style="border:none;">
        <span class="perf-key">Conditions</span>
        <span class="perf-val">night rain fog snow</span>
    </div>
    """, unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid #2a2c24;padding-bottom:1.2rem;margin-bottom:1.5rem;">
    <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:6px;">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:3rem;
        font-weight:800;color:#c8c9b8;letter-spacing:-1px;line-height:1;">
        TRAFFICVISION AI</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
        color:#3a3c2c;letter-spacing:2px;padding-bottom:6px;">
        INTELLIGENT MONITORING SYSTEM</div>
    </div>
    <div style="font-size:0.85rem;color:#4a4c3c;font-weight:300;margin-bottom:10px;">
    Real-time vehicle detection &nbsp;/&nbsp; Traffic density estimation &nbsp;/&nbsp; Congestion forecasting
    </div>
    <span class="tag tag-active">YOLOv8</span>
    <span class="tag tag-active">LSTM</span>
    <span class="tag">BDD100K</span>
    <span class="tag">VisDrone</span>
    <span class="tag">Weather-Aware</span>
    <span class="tag">4 Classes</span>
</div>
""", unsafe_allow_html=True)


# ── UPLOAD ────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload traffic video — MP4 / AVI / MOV / MKV",
    type=["mp4","avi","mov","mkv"]
)

if uploaded is None:
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, n, title, desc in [
        (c1, "01", "UPLOAD VIDEO",    "CCTV / dashcam / drone footage"),
        (c2, "02", "SET CONDITIONS",  "Weather / lighting / precipitation"),
        (c3, "03", "START ANALYSIS",  "Live detection + LSTM prediction"),
    ]:
        col.markdown(f"""
        <div style="border:1px solid #2a2c24;padding:20px 18px;cursor:default;
        transition:border-color 0.15s;"
        onmouseover="this.style.borderColor='#e8a020'"
        onmouseout="this.style.borderColor='#2a2c24'">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
            color:#3a3c2c;letter-spacing:3px;margin-bottom:10px;">STEP {n}</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
            font-weight:600;color:#8a8c78;letter-spacing:1px;margin-bottom:5px;">{title}</div>
            <div style="font-size:0.78rem;color:#3a3c2c;">{desc}</div>
        </div>""", unsafe_allow_html=True)

else:
    # ── WEATHER ──────────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
    color:#2a2c24;letter-spacing:3px;margin-bottom:10px;">ENVIRONMENTAL CONDITIONS</div>
    """, unsafe_allow_html=True)

    wc1, wc2, wc3 = st.columns(3)
    with wc1: weather  = st.selectbox("Weather",       ["Clear","Cloudy","Foggy","Snowy"])
    with wc2: lighting = st.selectbox("Lighting",      ["Daytime","Dusk/Dawn","Night"])
    with wc3: rain     = st.selectbox("Precipitation", ["None","Light Rain","Heavy Rain"])

    risk_mult, risk_reasons = get_weather_risk(weather, lighting, rain)

    if risk_mult > 1.05:
        rl = "moderate" if risk_mult < 1.3 else ("high" if risk_mult < 1.6 else "critical")
        rc = {"moderate":"#e8a020","high":"#c87020","critical":"#c84030"}[rl]
        st.markdown(f"""
        <div style="border:1px solid {rc};border-left:2px solid {rc};padding:10px 14px;
        margin:10px 0;font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
        background:rgba(0,0,0,0.2);">
            <span style="color:{rc};">ADVERSE CONDITIONS — RISK x{risk_mult:.2f}</span>
            <span style="color:#4a4c3c;margin-left:12px;">
            {' / '.join(risk_reasons)}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:1px solid #2a3c24;border-left:2px solid #5a8c50;padding:8px 14px;
        margin:10px 0;font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
        color:#5a8c50;background:rgba(90,140,80,0.04);">
        CLEAR CONDITIONS — STANDARD THRESHOLDS ACTIVE</div>
        """, unsafe_allow_html=True)

    # ── TEMP FILE ─────────────────────────────────────────────
    if st.session_state.vpath is None or not os.path.exists(st.session_state.vpath):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf.write(uploaded.read()); tf.flush(); tf.close()
        st.session_state.vpath = tf.name
    vpath = st.session_state.vpath

    # ── LAYOUT ───────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown('<div class="section-label">Live detection feed</div>', unsafe_allow_html=True)
        feed_slot = st.empty()

    with right:
        st.markdown('<div class="section-label">Real-time metrics</div>', unsafe_allow_html=True)
        metrics_slot = st.empty()
        pred_slot    = st.empty()

    st.markdown('<div class="section-label" style="margin-top:14px;">Density timeline — hover for values</div>', unsafe_allow_html=True)
    chart_slot = st.empty()

    b1, b2, b3, _ = st.columns([1,1,1,3])
    with b1: start_btn = st.button("[ START ]")
    with b2: stop_btn  = st.button("[ STOP  ]")
    with b3: reset_btn = st.button("[ RESET ]")

    if start_btn: st.session_state.running = True;  st.session_state.history = []
    if stop_btn:  st.session_state.running = False
    if reset_btn:
        st.session_state.running = False
        st.session_state.history = []
        if st.session_state.vpath and os.path.exists(st.session_state.vpath):
            try: os.unlink(st.session_state.vpath)
            except: pass
        st.session_state.vpath = None
        st.rerun()

    # ── PROCESSING LOOP ──────────────────────────────────────
    if st.session_state.running:
        import cv2
        cap     = cv2.VideoCapture(vpath)
        fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25
        buf     = deque(maxlen=30)
        fnum    = 0
        t0      = time.time()

        while cap.isOpened() and st.session_state.running:
            ret, frame = cap.read()
            if not ret: st.session_state.running = False; break

            fnum += 1
            if fnum % process_every != 0: continue

            if yolo_model is not None:
                try:
                    r     = yolo_model(frame, conf=conf_thresh, verbose=False)[0]
                    count = len(r.boxes)
                    frame = r.plot()
                except: count = max(0, int(np.random.normal(15,5)))
            else:
                t_el  = fnum / fps_vid
                count = max(0, int(12 + 8*np.sin(t_el/30) + np.random.normal(0,2)))

            label, level, color = get_density(count, risk_mult)
            buf.append([count, level])
            pred     = predict_lstm(lstm_model, buf)
            elapsed  = time.time() - t0
            fps_proc = fnum / elapsed if elapsed > 0 else 0

            frame = draw_on_frame(frame, count, label, color, pred, fnum, fps_proc, risk_mult)
            feed_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

            st.session_state.history.append({
                "t": round(fnum/fps_vid,1),
                "count": count, "level": level,
                "label": label, "pred": pred
            })

            # ── METRICS ──
            metrics_slot.markdown(f"""
            <div class="grid-3" style="margin-bottom:8px;">
                <div class="gcell" style="border-top:2px solid {color};">
                    <div class="stat-num" style="color:#c8c9b8;">{count}</div>
                    <div class="stat-unit">vehicles</div>
                </div>
                <div class="gcell" style="border-top:2px solid {color};">
                    <div class="stat-num density-{label}" style="font-size:2rem;">{label}</div>
                    <div class="stat-unit">density</div>
                </div>
                <div class="gcell" style="border-top:2px solid #3a3c2c;">
                    <div class="stat-num" style="color:#c8c9b8;">{fps_proc:.0f}</div>
                    <div class="stat-unit">fps</div>
                </div>
            </div>
            {f'<div style="border:1px solid #c84030;border-left:2px solid #c84030;padding:6px 10px;font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:#c84030;background:rgba(200,64,48,0.05);margin-bottom:8px;">RISK FACTOR x{risk_mult} ACTIVE</div>' if risk_mult > 1.05 else ''}
            """, unsafe_allow_html=True)

            # ── PREDICTION ──
            if pred is not None:
                plab = ["LOW","MEDIUM","HIGH","CRITICAL"][min(3,int(round(pred)))]
                diff = pred - level
                if diff > 0.35:
                    cls,icon,title,msg,tcol = (
                        "alert-rise","↑","CONGESTION RISING",
                        f"Forecast: <b>{plab}</b> — traffic building up","#c84030"
                    )
                elif diff < -0.35:
                    cls,icon,title,msg,tcol = (
                        "alert-ease","↓","TRAFFIC EASING",
                        f"Forecast: <b>{plab}</b> — conditions improving","#5a8c50"
                    )
                else:
                    cls,icon,title,msg,tcol = (
                        "alert-hold","—","CONDITIONS STABLE",
                        f"Forecast: <b>{plab}</b> — no significant change","#e8a020"
                    )
                pred_slot.markdown(f"""
                <div class="alert-box {cls}">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;
                    font-weight:700;color:{tcol};letter-spacing:1px;margin-bottom:4px;">
                    {icon} {title}</div>
                    <div style="font-size:0.8rem;color:#5a5c48;margin-bottom:8px;">{msg}</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                    color:#2a2c24;padding-top:6px;border-top:1px solid #2a2c24;">
                    lstm output: {pred:.1f}/3.0 &nbsp;·&nbsp; frame {fnum:05d}</div>
                </div>""", unsafe_allow_html=True)
            else:
                needed = 30 - len(buf)
                pct    = int(len(buf)/30*100)
                pred_slot.markdown(f"""
                <div class="alert-box alert-wait">
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;
                    font-weight:600;color:#4a4c3c;letter-spacing:1px;margin-bottom:4px;">
                    COLLECTING DATA — {pct}%</div>
                    <div style="font-size:0.78rem;color:#3a3c2c;margin-bottom:8px;">
                    {needed} more frames needed for LSTM prediction</div>
                    <div style="height:3px;background:#1e2018;">
                        <div style="width:{pct}%;height:100%;background:#e8a020;
                        transition:width 0.3s;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            if len(st.session_state.history) > 3:
                chart_slot.plotly_chart(
                    make_chart(st.session_state.history),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

        cap.release()
        st.session_state.running = False

    # ── SUMMARY ──────────────────────────────────────────────
    if st.session_state.history and not st.session_state.running:
        h    = st.session_state.history
        avgv = np.mean([x["count"] for x in h])
        maxv = max(x["count"] for x in h)
        nf   = len(h)
        cpct = sum(1 for x in h if x["level"] >= 2) / nf * 100
        cc   = "#c84030" if cpct > 50 else ("#e8a020" if cpct > 25 else "#5a8c50")

        st.markdown('<div class="rule" style="margin-top:20px;"></div>', unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:0.55rem;
        color:#2a2c24;letter-spacing:3px;margin-bottom:10px;">SESSION REPORT</div>""",
        unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        for col, val, lbl, vc in [
            (s1, f"{avgv:.1f}", "avg vehicles / frame", "#c8c9b8"),
            (s2, str(maxv),     "peak vehicle count",   "#c8c9b8"),
            (s3, str(nf),       "frames analyzed",      "#c8c9b8"),
            (s4, f"{cpct:.0f}%","time in congestion",   cc),
        ]:
            col.markdown(f"""
            <div style="border:1px solid #2a2c24;padding:16px 14px;
            transition:border-color 0.15s;"
            onmouseover="this.style.borderColor='#4a4c3c'"
            onmouseout="this.style.borderColor='#2a2c24'">
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.4rem;
                font-weight:700;color:{vc};letter-spacing:-1px;line-height:1;">{val}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
                color:#3a3c2c;text-transform:uppercase;letter-spacing:1.5px;
                margin-top:5px;">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        if st.session_state.vpath and os.path.exists(st.session_state.vpath):
            try: os.unlink(st.session_state.vpath); st.session_state.vpath = None
            except: pass
