"""
College Security Surveillance Dashboard
=========================================
Multi-page Streamlit application with:
  1. Live Detection     — real-time camera feed + anomaly overlay + identified student info
  2. Incident Records   — searchable incident history with evidence
  3. Analytics          — daily/weekly trends, severity breakdown, peak hours
  4. Student Management — rich UI for adding/viewing student data with photo cards
  5. Camera Management  — configure cameras
"""

import io
import os
import sys
import json
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import torch

# ---- Project imports ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.schema import init_db
from database import db as database
from models.autoencoder import ConvolutionalAutoencoder
from models.violence_classifier import ViolenceClassifier
from face_recognition_module.face_detector import FaceDetector
from face_recognition_module.face_matcher import FaceMatcher
from incidents.incident_manager import IncidentManager
from live.camera import CameraManager
from live.processor import LiveProcessor

# ---- Page Config ----
st.set_page_config(
    page_title="College Security Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# =====================================================================
#  Custom CSS for premium look
# =====================================================================
st.markdown("""
<style>
/* ---------- Global ---------- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}
h1, h2, h3 { color: #e2e8f0 !important; }
p, label, span { color: #cbd5e1; }

/* ---------- Student Card ---------- */
.student-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.student-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(99,102,241,0.25);
    border-color: rgba(99,102,241,0.4);
}
.student-name  { font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin: 0; }
.student-id    { font-size: 0.85rem; color: #818cf8; font-weight: 600; }
.student-dept  { font-size: 0.85rem; color: #94a3b8; }
.student-role  { display: inline-block; padding: 3px 10px; border-radius: 20px;
                 font-size: 0.75rem; font-weight: 600; }
.role-student  { background: rgba(34,211,238,0.15); color: #22d3ee; }
.role-faculty  { background: rgba(168,85,247,0.15); color: #a855f7; }
.role-staff    { background: rgba(251,191,36,0.15); color: #fbbf24; }
.role-other    { background: rgba(156,163,175,0.15); color: #9ca3af; }

.detail-label  { font-size: 0.75rem; color: #64748b; text-transform: uppercase;
                 letter-spacing: 0.05em; margin-bottom: 2px; }
.detail-value  { font-size: 0.95rem; color: #e2e8f0; margin-bottom: 8px; }

.incident-badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
                  font-size: 0.75rem; font-weight: 600; }
.badge-clean   { background: rgba(34,197,94,0.15); color: #22c55e; }
.badge-warn    { background: rgba(251,191,36,0.15); color: #fbbf24; }
.badge-danger  { background: rgba(239,68,68,0.15); color: #ef4444; }

/* ---------- Stat Card ---------- */
.stat-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
}
.stat-value { font-size: 2rem; font-weight: 800; color: #f8fafc; }
.stat-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; }

/* ---------- Alert Box ---------- */
.alert-box {
    border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
    border-left: 4px solid;
}
.alert-high   { background: rgba(239,68,68,0.10); border-color: #ef4444; }
.alert-medium { background: rgba(251,191,36,0.10); border-color: #fbbf24; }
.alert-low    { background: rgba(34,197,94,0.10); border-color: #22c55e; }
.alert-normal { background: rgba(99,102,241,0.10); border-color: #6366f1; }

/* ---------- Section Header ---------- */
.section-hdr {
    font-size: 1.1rem; font-weight: 700; color: #c7d2fe;
    border-bottom: 2px solid rgba(99,102,241,0.3);
    padding-bottom: 6px; margin-bottom: 14px;
}

/* ---------- Form / Inputs ---------- */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
#  Session State Helpers
# =====================================================================

def _init_state():
    defaults = {
        "processor": None,
        "camera": None,
        "live_running": False,
        "model_loaded": False,
        "incident_mgr": IncidentManager(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_model():
    if st.session_state.get("model_loaded"):
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvolutionalAutoencoder(input_channels=1, latent_dim=256)
    model_path = os.path.join(PROJECT_ROOT, "outputs", "trained_model.pth")
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        threshold = ckpt.get("threshold") or 0.005069
    else:
        threshold = 0.005069
    model.eval()
    st.session_state["processor"] = LiveProcessor(
        model=model, device=device, threshold=threshold,
        evidence_dir=os.path.join(PROJECT_ROOT, "evidence"),
    )
    st.session_state["model_loaded"] = True


# =====================================================================
#  Helper: render a student card  (HTML)
# =====================================================================

def _student_card_html(p: Dict, count: int) -> str:
    role_cls = {"Student": "role-student", "Faculty": "role-faculty",
                "Staff": "role-staff"}.get(p.get("role", ""), "role-other")

    if count == 0:
        badge = '<span class="incident-badge badge-clean">✓ No Incidents</span>'
    elif count < 3:
        badge = f'<span class="incident-badge badge-warn">⚠ {count} Incident{"s" if count>1 else ""}</span>'
    else:
        badge = f'<span class="incident-badge badge-danger">🔴 {count} Incidents — Repeat Offender</span>'

    return f"""
    <div class="student-card">
        <div style="display:flex;gap:16px;align-items:flex-start;">
            <div style="flex:1;">
                <p class="student-name">{p.get("name","—")}</p>
                <p class="student-id">{p.get("college_id","—")}</p>
                <span class="student-role {role_cls}">{p.get("role","—")}</span>
                <div style="margin-top:12px;">
                    <p class="detail-label">Department</p>
                    <p class="detail-value">{p.get("department","—") or "—"}</p>
                    <p class="detail-label">Phone</p>
                    <p class="detail-value">{p.get("phone","—") or "—"}</p>
                    <p class="detail-label">Email</p>
                    <p class="detail-value">{p.get("email","—") or "—"}</p>
                </div>
                {badge}
            </div>
        </div>
    </div>
    """


# =====================================================================
#  PAGE 1 — Live Detection
# =====================================================================

def page_live_detection():
    st.markdown('<h1 style="text-align:center;">📹 Live Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#94a3b8;">Real-time CCTV feed with anomaly detection, '
                'violence classification, face recognition &amp; exam-cheating monitoring</p>',
                unsafe_allow_html=True)

    _load_model()
    processor: LiveProcessor = st.session_state["processor"]

    # ---- Config sidebar ----
    col_cfg, col_feed = st.columns([1, 3])

    with col_cfg:
        st.markdown('<div class="section-hdr">📷 Camera Source</div>', unsafe_allow_html=True)
        source_type = st.radio("Source", ["Webcam", "Video File", "RTSP"], horizontal=True)
        if source_type == "Webcam":
            source = 0
        elif source_type == "Video File":
            uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
            if uploaded:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(uploaded.read()); tmp.flush()
                source = tmp.name
            else:
                source = None
        else:
            source = st.text_input("RTSP URL", "rtsp://")

        camera_id = st.text_input("Camera ID", "cam_exam_hall")
        detection_mode = st.selectbox("Detection Mode", [
            "General Violence", "Exam Cheating / Copying", "Ragging",
            "Unauthorized Access", "Late-Night Suspicious"])

        start = st.button("▶️  Start Detection", use_container_width=True, type="primary")
        stop = st.button("⏹️  Stop", use_container_width=True)

        # Show registered students for quick reference
        st.markdown('<div class="section-hdr" style="margin-top:20px;">👥 Registered Students</div>',
                    unsafe_allow_html=True)
        persons = database.get_all_persons()
        st.caption(f"{len(persons)} students registered")
        for p in persons[:8]:
            cnt = database.get_person_incident_count(p["id"])
            badge_color = "#22c55e" if cnt == 0 else "#fbbf24" if cnt < 3 else "#ef4444"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:{badge_color};display:inline-block;"></span>'
                f'<span style="color:#e2e8f0;font-size:0.9rem;">{p["name"]}</span>'
                f'<span style="color:#818cf8;font-size:0.75rem;">{p["college_id"]}</span>'
                f'</div>', unsafe_allow_html=True)

    with col_feed:
        status_ph = st.empty()
        frame_ph = st.empty()
        metrics_ph = st.empty()
        identified_ph = st.empty()

    # ---- Run detection ----
    if start and source is not None:
        cam = CameraManager(source=source, camera_id=camera_id)
        if cam.start():
            status_ph.markdown(
                '<div class="alert-box alert-normal">'
                '🟢 <strong>Camera connected</strong> — Processing frames…</div>',
                unsafe_allow_html=True)
            face_det = FaceDetector()
            face_match = FaceMatcher()
            frame_count = 0
            while cam.is_running() and frame_count < 3000:
                frame = cam.read()
                if frame is None:
                    time.sleep(0.03); continue

                result = processor.process_frame(frame, camera_id=camera_id)
                annotated = processor.get_latest_frame()
                frame_count += 1

                if annotated is not None and frame_count % 3 == 0:
                    frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                   caption=f"Frame {frame_count}", use_container_width=True)

                # Update metrics
                if frame_count % 8 == 0:
                    sev = result["severity"]
                    if sev == "High":
                        alert_cls, icon = "alert-high", "🔴"
                    elif sev == "Medium":
                        alert_cls, icon = "alert-medium", "🟡"
                    else:
                        alert_cls, icon = "alert-low", "🟢"

                    metrics_ph.markdown(f"""
                    <div style="display:flex;gap:12px;margin-top:10px;">
                        <div class="stat-card" style="flex:1;">
                            <div class="stat-value">{icon} {result["status"]}</div>
                            <div class="stat-label">Status</div>
                        </div>
                        <div class="stat-card" style="flex:1;">
                            <div class="stat-value">{sev}</div>
                            <div class="stat-label">Severity</div>
                        </div>
                        <div class="stat-card" style="flex:1;">
                            <div class="stat-value">{result["score"]:.6f}</div>
                            <div class="stat-label">Anomaly Score</div>
                        </div>
                        <div class="stat-card" style="flex:1;">
                            <div class="stat-value">{detection_mode.split("/")[0].strip()}</div>
                            <div class="stat-label">Mode</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Identify faces every 30 frames
                if frame_count % 30 == 0 and result["status"] == "ALERT":
                    faces = face_det.detect(frame)
                    if faces:
                        matches = face_match.match_faces([f["face_image"] for f in faces])
                        cards_html = '<div class="section-hdr">🔍 Identified Persons</div>'
                        cards_html += '<div style="display:flex;gap:10px;flex-wrap:wrap;">'
                        for i, m in enumerate(matches):
                            if m["matched"]:
                                p = database.get_person(m["person_id"])
                                if p:
                                    cnt = database.get_person_incident_count(p["id"])
                                    cards_html += f"""
                                    <div class="student-card" style="flex:1;min-width:200px;">
                                        <p class="student-name">{p["name"]}</p>
                                        <p class="student-id">{p["college_id"]}</p>
                                        <p class="student-dept">{p.get("department","")}</p>
                                        <p class="detail-label">Past Incidents: {cnt}</p>
                                        <p class="detail-label">Confidence: {m["confidence"]:.0%}</p>
                                    </div>"""
                            else:
                                cards_html += """
                                <div class="student-card" style="flex:1;min-width:200px;border-color:rgba(239,68,68,0.4);">
                                    <p class="student-name" style="color:#ef4444;">⚠️ Unknown Person</p>
                                    <p class="student-dept">Not in college database</p>
                                    <p class="detail-label">Image captured for records</p>
                                </div>"""
                        cards_html += '</div>'
                        identified_ph.markdown(cards_html, unsafe_allow_html=True)

                if stop:
                    break

            cam.stop()
            processor.violence_classifier.force_close()
            status_ph.markdown(
                '<div class="alert-box alert-normal">⏹️ <strong>Detection stopped</strong></div>',
                unsafe_allow_html=True)
        else:
            status_ph.error("❌ Could not open camera source")

    # ---- Session events ----
    st.divider()
    st.markdown('<div class="section-hdr">📋 Session Events</div>', unsafe_allow_html=True)
    events = processor.violence_classifier.get_all_events() if processor else []
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True)
    else:
        st.info("No events detected yet. Start live detection above.")


# =====================================================================
#  PAGE 2 — Incident Records
# =====================================================================

def page_records():
    st.markdown('<h1>📋 Incident Records &amp; Logs</h1>', unsafe_allow_html=True)

    mgr: IncidentManager = st.session_state["incident_mgr"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        severity_f = st.selectbox("Severity", ["All", "High", "Medium", "Low"])
    with col2:
        type_f = st.selectbox("Type", ["All", "physical_fight", "ragging",
                                        "unauthorized_access", "suspicious_activity",
                                        "exam_cheating", "other"])
    with col3:
        class_f = st.selectbox("Classification", ["All", "college_only",
                                                    "college_and_outsider", "outsiders_only"])
    with col4:
        days_f = st.slider("Last N days", 1, 90, 7)

    since = (datetime.now() - timedelta(days=days_f)).isoformat()
    incidents = database.get_incidents(
        limit=200,
        severity=severity_f if severity_f != "All" else None,
        incident_type=type_f if type_f != "All" else None,
        classification=class_f if class_f != "All" else None,
        since=since,
    )

    # Summary row
    st.markdown(f"""
    <div style="display:flex;gap:12px;margin:16px 0;">
        <div class="stat-card" style="flex:1;">
            <div class="stat-value">{len(incidents)}</div>
            <div class="stat-label">Total Incidents</div>
        </div>
        <div class="stat-card" style="flex:1;">
            <div class="stat-value">{sum(1 for i in incidents if i["severity"]=="High")}</div>
            <div class="stat-label">🔴 High Severity</div>
        </div>
        <div class="stat-card" style="flex:1;">
            <div class="stat-value">{sum(1 for i in incidents if i["classification"]=="college_and_outsider")}</div>
            <div class="stat-label">⚠️ College + Outsider</div>
        </div>
        <div class="stat-card" style="flex:1;">
            <div class="stat-value">{sum(1 for i in incidents if i.get("incident_type")=="exam_cheating")}</div>
            <div class="stat-label">📝 Exam Cheating</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not incidents:
        st.info("No incidents found for the selected filters.")
        return

    df = pd.DataFrame(incidents)
    display_cols = [c for c in ["id","incident_type","severity","classification",
                                 "start_time","end_time","duration_sec","anomaly_score","status"]
                    if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, height=400)

    # Detail view
    st.markdown('<div class="section-hdr" style="margin-top:20px;">🔎 Incident Detail</div>',
                unsafe_allow_html=True)
    sel_id = st.number_input("Enter Incident ID", min_value=1, step=1)
    if st.button("Load Detail"):
        detail = mgr.get_incident_detail(int(sel_id))
        if detail:
            st.json(detail)
            evidence_list = detail.get("evidence", [])
            imgs = [e for e in evidence_list if e["evidence_type"] == "image"]
            if imgs:
                st.markdown('<div class="section-hdr">📸 Evidence</div>', unsafe_allow_html=True)
                cols = st.columns(min(4, len(imgs)))
                for i, ev in enumerate(imgs):
                    if os.path.exists(ev["file_path"]):
                        with cols[i % 4]:
                            st.image(Image.open(ev["file_path"]), use_container_width=True)
            persons = detail.get("persons", [])
            if persons:
                st.markdown('<div class="section-hdr">👤 Persons Involved</div>', unsafe_allow_html=True)
                for p in persons:
                    if p.get("is_outsider"):
                        st.warning(f"**Unknown Person** — Image: {p.get('outsider_image','N/A')}")
                    else:
                        st.success(f"**{p.get('name','N/A')}** — {p.get('college_id','N/A')} | "
                                   f"{p.get('department','N/A')} | {p.get('role','N/A')}")
        else:
            st.warning("Incident not found.")

    st.divider()
    act1, act2 = st.columns(2)
    with act1:
        rid = st.number_input("Resolve ID", min_value=1, step=1, key="r")
        if st.button("✅ Resolve"):
            mgr.resolve_incident(int(rid)); st.success(f"#{rid} resolved")
    with act2:
        fid = st.number_input("False Alarm ID", min_value=1, step=1, key="f")
        if st.button("❌ False Alarm"):
            mgr.mark_false_alarm(int(fid)); st.success(f"#{fid} false alarm")

    if incidents:
        csv = pd.DataFrame(incidents).to_csv(index=False)
        st.download_button("📥 Export CSV", csv, "incidents.csv", "text/csv",
                           use_container_width=True)


# =====================================================================
#  PAGE 3 — Analytics
# =====================================================================

def page_analytics():
    st.markdown('<h1>📊 Analytics &amp; Graphs</h1>', unsafe_allow_html=True)

    days = st.slider("Analysis period (days)", 1, 90, 7)
    stats = database.get_incident_stats(days=days)

    st.markdown(f"""
    <div style="display:flex;gap:12px;margin:16px 0;">
        <div class="stat-card" style="flex:1;">
            <div class="stat-value">{stats["total"]}</div>
            <div class="stat-label">Total Incidents</div>
        </div>
        <div class="stat-card" style="flex:1;">
            <div class="stat-value">{stats["by_severity"].get("High",0)}</div>
            <div class="stat-label">🔴 High Severity</div>
        </div>
        <div class="stat-card" style="flex:1;">
            <div class="stat-value">{stats["by_classification"].get("college_and_outsider",0)}</div>
            <div class="stat-label">⚠️ College + Outsider</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Daily Incident Trend")
        by_day = stats.get("by_day", {})
        if by_day:
            df = pd.DataFrame(list(by_day.items()), columns=["Date", "Count"])
            fig = px.bar(df, x="Date", y="Count", color_discrete_sequence=["#818cf8"])
            fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data.")

    with c2:
        st.subheader("Severity Distribution")
        by_sev = stats.get("by_severity", {})
        if by_sev:
            df = pd.DataFrame(list(by_sev.items()), columns=["Severity", "Count"])
            fig = px.pie(df, names="Severity", values="Count",
                         color="Severity", color_discrete_map={"High":"#ef4444","Medium":"#fbbf24","Low":"#22c55e"})
            fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data.")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Incident Type Breakdown")
        by_type = stats.get("by_type", {})
        if by_type:
            mgr = st.session_state["incident_mgr"]
            labelled = {mgr.USE_CASE_LABELS.get(k, k): v for k, v in by_type.items()}
            df = pd.DataFrame(list(labelled.items()), columns=["Type", "Count"])
            fig = px.bar(df, x="Type", y="Count", color="Type")
            fig.update_layout(height=340, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data.")

    with c4:
        st.subheader("Peak Hours")
        by_hour = stats.get("by_hour", {})
        if by_hour:
            all_h = {h: 0 for h in range(24)}
            all_h.update({int(k): v for k, v in by_hour.items()})
            df = pd.DataFrame(list(all_h.items()), columns=["Hour", "Count"]).sort_values("Hour")
            fig = px.area(df, x="Hour", y="Count", color_discrete_sequence=["#a78bfa"])
            fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data.")

    st.divider()
    st.subheader("Incident Classification")
    by_cls = stats.get("by_classification", {})
    if by_cls:
        lbl = {"college_only": "College-Only", "college_and_outsider": "College + Outsider ⚠️",
               "outsiders_only": "Outsiders Only", "unclassified": "Unclassified"}
        df = pd.DataFrame([(lbl.get(k,k), v) for k,v in by_cls.items()], columns=["Classification","Count"])
        fig = px.bar(df, x="Classification", y="Count", color="Classification",
                     color_discrete_map={"College-Only":"#22c55e","College + Outsider ⚠️":"#ef4444",
                                         "Outsiders Only":"#fbbf24","Unclassified":"#6366f1"})
        fig.update_layout(height=300, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
#  PAGE 4 — Student Management (ENHANCED UI)
# =====================================================================

def page_persons():
    st.markdown('<h1>👤 Student &amp; Staff Management</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;">Add students and staff so the system can recognise them '
                'during live detection — including exam-cheating monitoring.</p>',
                unsafe_allow_html=True)

    persons = database.get_all_persons()

    # ---- Top stats ----
    total = len(persons)
    students = sum(1 for p in persons if p.get("role") == "Student")
    faculty = sum(1 for p in persons if p.get("role") == "Faculty")
    staff = sum(1 for p in persons if p.get("role") == "Staff")

    st.markdown(f"""
    <div style="display:flex;gap:12px;margin:16px 0;">
        <div class="stat-card" style="flex:1;">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Total Registered</div>
        </div>
        <div class="stat-card" style="flex:1;">
            <div class="stat-value" style="color:#22d3ee;">{students}</div>
            <div class="stat-label">Students</div>
        </div>
        <div class="stat-card" style="flex:1;">
            <div class="stat-value" style="color:#a855f7;">{faculty}</div>
            <div class="stat-label">Faculty</div>
        </div>
        <div class="stat-card" style="flex:1;">
            <div class="stat-value" style="color:#fbbf24;">{staff}</div>
            <div class="stat-label">Staff</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Tabs ----
    tab_add, tab_list, tab_search = st.tabs(["➕ Add New Person", "📋 All Registered", "🔍 Search"])

    # ====== TAB: Add New Person ======
    with tab_add:
        st.markdown('<div class="section-hdr">Register New Student / Staff</div>', unsafe_allow_html=True)

        col_form, col_preview = st.columns([2, 1])

        with col_form:
            with st.form("add_person", clear_on_submit=True):
                r1c1, r1c2 = st.columns(2)
                with r1c1:
                    college_id = st.text_input("🆔  College ID *", placeholder="STU2024001")
                with r1c2:
                    name = st.text_input("👤  Full Name *", placeholder="Priya Sharma")

                r2c1, r2c2 = st.columns(2)
                with r2c1:
                    department = st.text_input("🏛️  Department", placeholder="Computer Science")
                with r2c2:
                    role = st.selectbox("🎓  Role", ["Student", "Faculty", "Staff", "Other"])

                r3c1, r3c2 = st.columns(2)
                with r3c1:
                    phone = st.text_input("📱  Phone", placeholder="+91 9876543210")
                with r3c2:
                    email = st.text_input("📧  Email", placeholder="priya@college.edu")

                photo = st.file_uploader("📸  Upload Photo (for face recognition)",
                                         type=["jpg", "jpeg", "png"],
                                         help="Clear, front-facing photo for best recognition accuracy")

                st.markdown("---")
                submitted = st.form_submit_button("✅  Register Person", type="primary",
                                                   use_container_width=True)

                if submitted:
                    if not college_id or not name:
                        st.error("College ID and Full Name are required.")
                    elif database.get_person_by_college_id(college_id):
                        st.error(f"A person with ID '{college_id}' already exists.")
                    else:
                        photo_path = ""
                        face_embedding = None
                        if photo:
                            photo_dir = os.path.join(PROJECT_ROOT, "college_db", "faces")
                            os.makedirs(photo_dir, exist_ok=True)
                            photo_path = os.path.join(photo_dir, f"{college_id}.jpg")
                            img = Image.open(photo)
                            img.save(photo_path)
                            try:
                                frame = cv2.imread(photo_path)
                                det = FaceDetector()
                                detected = det.detect(frame)
                                if detected:
                                    matcher = FaceMatcher()
                                    face_embedding = matcher.compute_embedding(detected[0]["face_image"])
                                    st.success("✅ Face detected and embedding computed!")
                                else:
                                    st.warning("⚠️ No face detected in the photo — person added without face data.")
                            except Exception as e:
                                st.warning(f"Face processing error: {e}")

                        pid = database.add_person(college_id=college_id, name=name,
                                                  department=department, role=role,
                                                  phone=phone, email=email,
                                                  photo_path=photo_path,
                                                  face_embedding=face_embedding)
                        st.success(f"🎉 **{name}** registered successfully! (ID: {pid})")
                        st.balloons()

        with col_preview:
            st.markdown('<div class="section-hdr">Preview</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="student-card">
                <p class="student-name">New person will appear here</p>
                <p class="student-dept" style="margin-top:8px;">
                    Fill the form and click Register to add a student or staff member.
                    Their photo will be used for face recognition during live detection.
                </p>
                <p class="detail-label" style="margin-top:12px;">Use Cases</p>
                <p class="detail-value">• Exam monitoring (cheating/copying detection)<br>
                • Violence / ragging identification<br>
                • Unauthorized access tracking<br>
                • Incident history per person</p>
            </div>
            """, unsafe_allow_html=True)

    # ====== TAB: All Registered ======
    with tab_list:
        st.markdown('<div class="section-hdr">All Registered Persons</div>', unsafe_allow_html=True)

        if not persons:
            st.info("No persons registered yet. Use the **Add New Person** tab to get started.")
        else:
            # Display as card grid
            cols = st.columns(3)
            for idx, p in enumerate(persons):
                cnt = database.get_person_incident_count(p["id"])
                with cols[idx % 3]:
                    st.markdown(_student_card_html(p, cnt), unsafe_allow_html=True)

                    # Photo display
                    if p.get("photo_path") and os.path.exists(p["photo_path"]):
                        st.image(p["photo_path"], width=120)

                    # Deactivate button
                    if st.button(f"🗑️ Deactivate", key=f"del_{p['id']}"):
                        database.delete_person(p["id"])
                        st.rerun()

    # ====== TAB: Search ======
    with tab_search:
        st.markdown('<div class="section-hdr">Search Persons</div>', unsafe_allow_html=True)
        query = st.text_input("🔍  Search by name, ID, or department")
        if query:
            q_lower = query.lower()
            results = [p for p in persons if
                       q_lower in p.get("name", "").lower() or
                       q_lower in p.get("college_id", "").lower() or
                       q_lower in p.get("department", "").lower()]
            if results:
                for p in results:
                    cnt = database.get_person_incident_count(p["id"])
                    st.markdown(_student_card_html(p, cnt), unsafe_allow_html=True)
                    if p.get("photo_path") and os.path.exists(p["photo_path"]):
                        st.image(p["photo_path"], width=100)
            else:
                st.warning("No matching persons found.")


# =====================================================================
#  PAGE 5 — Camera Management
# =====================================================================

def page_cameras():
    st.markdown('<h1>📷 Camera Management</h1>', unsafe_allow_html=True)
    tab_list, tab_add = st.tabs(["📋 Cameras", "➕ Add Camera"])

    with tab_list:
        cameras = database.get_all_cameras()
        if cameras:
            st.dataframe(pd.DataFrame(cameras), use_container_width=True)
        else:
            st.info("No cameras configured yet.")

    with tab_add:
        with st.form("add_cam"):
            cam_id = st.text_input("Camera ID *", placeholder="cam_exam_hall_1")
            cam_name = st.text_input("Camera Name *", placeholder="Exam Hall A — Front")
            location = st.text_input("Location", placeholder="Building A, 2nd Floor")
            stream_url = st.text_input("Stream URL", placeholder="rtsp://192.168.1.100:554/stream")
            cam_type = st.selectbox("Type", ["webcam", "rtsp", "file"])
            if st.form_submit_button("Add Camera", type="primary"):
                if cam_id and cam_name:
                    database.add_camera(cam_id, cam_name, location, stream_url, cam_type)
                    st.success(f"✅ Camera '{cam_name}' added!")
                else:
                    st.error("Camera ID and Name are required.")


# =====================================================================
#  Main Navigation
# =====================================================================

def main():
    _init_state()

    st.sidebar.markdown("""
    <div style="text-align:center;padding:16px 0;">
        <span style="font-size:2.5rem;">🛡️</span>
        <h2 style="margin:4px 0;font-size:1.3rem;color:#c7d2fe;">College Security</h2>
        <p style="font-size:0.8rem;color:#64748b;margin:0;">Surveillance System v2.0</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("", [
        "📹 Live Detection",
        "📋 Incident Records",
        "📊 Analytics",
        "👤 Student Management",
        "📷 Camera Management",
    ], label_visibility="collapsed")

    st.sidebar.divider()
    device = "GPU 🟢" if torch.cuda.is_available() else "CPU 🔵"
    st.sidebar.caption(f"Device: {device}")
    st.sidebar.caption(f"Time: {datetime.now().strftime('%H:%M:%S')}")

    if page == "📹 Live Detection":
        page_live_detection()
    elif page == "📋 Incident Records":
        page_records()
    elif page == "📊 Analytics":
        page_analytics()
    elif page == "👤 Student Management":
        page_persons()
    elif page == "📷 Camera Management":
        page_cameras()


if __name__ == "__main__":
    main()
