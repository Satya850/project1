"""
Live Processor
===============
Real-time processing pipeline: Frame → Anomaly Detection → Violence
Classification → Face Recognition → Incident Recording → Evidence Capture.
"""

import cv2
import numpy as np
import os
import time
import threading
from collections import deque
from datetime import datetime
from typing import Optional, Dict, List, Callable

import torch

from models.autoencoder import ConvolutionalAutoencoder
from models.violence_classifier import ViolenceClassifier, ViolenceEvent
from face_recognition_module.face_detector import FaceDetector
from face_recognition_module.face_matcher import FaceMatcher
from database import db as database
from database.schema import init_db
from live.camera import CameraManager


EVIDENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evidence")


class LiveProcessor:
    """
    Orchestrates real-time anomaly detection, violence classification,
    face recognition, and incident management.
    """

    def __init__(
        self,
        model: ConvolutionalAutoencoder,
        device: torch.device,
        threshold: float = 0.005,
        frame_size: int = 64,
        evidence_dir: str = "evidence",
        video_clip_seconds: int = 15,
        on_incident: Optional[Callable] = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold if threshold is not None else 0.005
        self.frame_size = frame_size
        self.evidence_dir = evidence_dir
        self.video_clip_seconds = video_clip_seconds
        self.on_incident = on_incident  # callback for dashboard

        # Sub-modules
        self.violence_classifier = ViolenceClassifier(
            low_threshold=self.threshold,
            med_threshold=self.threshold * 1.5,
            high_threshold=self.threshold * 2.5,
        )
        self.face_detector = FaceDetector()
        self.face_matcher = FaceMatcher()

        # State
        self._running = False
        self._current_score = 0.0
        self._current_status = "Normal"
        self._current_severity = "None"
        self._frame_buffer: deque = deque(maxlen=450)  # ~15s @ 30fps
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_annotated: Optional[np.ndarray] = None
        self._stats = {"frames_processed": 0, "incidents_total": 0}

        # Ensure evidence directory exists
        os.makedirs(self.evidence_dir, exist_ok=True)

        # Ensure DB is initialised
        init_db()

    # ------------------------------------------------------------------
    #  Core pipeline
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, camera_id: str = "cam_0") -> Dict:
        """
        Process a single BGR frame through the full pipeline.

        Returns a status dict for dashboard consumption.
        """
        self._latest_frame = frame.copy()
        self._frame_buffer.append(frame.copy())
        self._stats["frames_processed"] += 1

        # 1) Anomaly score
        score = self._compute_anomaly_score(frame)
        self._current_score = score

        # 2) Violence classification + timestamp tracking
        timestamp = datetime.now().isoformat()
        finished_event = self.violence_classifier.update(score, timestamp)
        status = self.violence_classifier.get_current_status()
        self._current_status = status["status"]
        self._current_severity = status.get("severity", "None")

        # 3) If a violence event just ended → process incident
        if finished_event is not None:
            self._handle_violence_event(finished_event, frame, camera_id)

        # 4) Annotate frame for display
        annotated = self._annotate_frame(frame, score, status)
        self._latest_annotated = annotated

        return {
            "score": score,
            "status": self._current_status,
            "severity": self._current_severity,
            "timestamp": timestamp,
        }

    def _compute_anomaly_score(self, frame: np.ndarray) -> float:
        """Convert BGR frame to grayscale tensor, run through autoencoder."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.frame_size, self.frame_size))
        normalised = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalised).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstructed = self.model(tensor)
            error = ((tensor - reconstructed) ** 2).mean().item()
        return error

    # ------------------------------------------------------------------
    #  Incident handling
    # ------------------------------------------------------------------

    def _handle_violence_event(self, event: ViolenceEvent, frame: np.ndarray, camera_id: str):
        """Create DB incident, detect faces, classify, capture evidence."""
        # Determine incident type heuristic
        hour = datetime.now().hour
        if 22 <= hour or hour < 5:
            incident_type = "suspicious_activity"
        else:
            incident_type = "physical_fight"

        # Face detection on current frame
        faces = self.face_detector.detect(frame)
        face_images = [f["face_image"] for f in faces]

        # Face matching
        match_results = self.face_matcher.match_faces(face_images) if face_images else []
        classification = FaceMatcher.classify_incident(match_results)

        # --- Save evidence ---
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        incident_dir = os.path.join(self.evidence_dir, f"incident_{ts_slug}")
        os.makedirs(incident_dir, exist_ok=True)

        # Save face images
        face_paths = []
        for i, face_info in enumerate(faces):
            tag = "college" if match_results and i < len(match_results) and match_results[i]["matched"] else "unknown"
            path = os.path.join(incident_dir, f"face_{i}_{tag}.jpg")
            cv2.imwrite(path, face_info["face_image"])
            face_paths.append(path)

        # Save video clip from buffer
        clip_path = os.path.join(incident_dir, "clip.mp4")
        self._save_video_clip(clip_path, frame.shape[1], frame.shape[0])

        # --- Create DB records ---
        duration = 0
        try:
            start_dt = datetime.fromisoformat(event.start_time)
            end_dt = datetime.fromisoformat(event.end_time)
            duration = (end_dt - start_dt).total_seconds()
        except Exception:
            pass

        incident_id = database.create_incident(
            incident_type=incident_type,
            severity=event.severity,
            classification=classification,
            camera_id=camera_id,
            start_time=event.start_time,
            end_time=event.end_time,
            duration_sec=duration,
            video_clip_path=clip_path,
            anomaly_score=event.peak_score,
        )

        # Link persons
        for i, mr in enumerate(match_results):
            outsider_img = face_paths[i] if i < len(face_paths) else ""
            database.link_person_to_incident(
                incident_id=incident_id,
                person_id=mr["person_id"] if mr["matched"] else None,
                is_outsider=not mr["matched"],
                outsider_image=outsider_img,
                confidence=mr["confidence"],
            )

        # Add evidence records
        for fp in face_paths:
            database.add_evidence(incident_id, "image", fp, "Face capture")
        if os.path.exists(clip_path):
            database.add_evidence(incident_id, "video", clip_path, "Video clip")

        self._stats["incidents_total"] += 1

        # Callback for live dashboard
        if self.on_incident:
            self.on_incident({
                "incident_id": incident_id,
                "severity": event.severity,
                "classification": classification,
                "faces": len(faces),
                "matched": sum(1 for r in match_results if r["matched"]),
            })

        print(f"[Incident #{incident_id}] {event.severity} | {classification} | "
              f"{len(faces)} faces | {event.start_time} → {event.end_time}")

    def _save_video_clip(self, path: str, width: int, height: int):
        """Write buffered frames to an MP4 file."""
        if not self._frame_buffer:
            return
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, 15, (width, height))
            for f in self._frame_buffer:
                writer.write(f)
            writer.release()
        except Exception as e:
            print(f"[Evidence] Failed to save clip: {e}")

    # ------------------------------------------------------------------
    #  Annotation for dashboard
    # ------------------------------------------------------------------

    def _annotate_frame(self, frame: np.ndarray, score: float, status: Dict) -> np.ndarray:
        """Draw status overlay on frame for live display."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Status bar background
        colour = (0, 200, 0) if status["status"] == "Normal" else (0, 0, 255)
        cv2.rectangle(annotated, (0, 0), (w, 40), colour, -1)

        label = f"{status['status']}"
        if status["status"] == "ALERT":
            label += f" [{status.get('severity', '')}]"
        label += f"  Score: {score:.6f}"
        cv2.putText(annotated, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, ts, (w - 220, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    # ------------------------------------------------------------------
    #  Accessors
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> Optional[np.ndarray]:
        return self._latest_annotated

    def get_current_status(self) -> Dict:
        return {
            "status": self._current_status,
            "severity": self._current_severity,
            "score": self._current_score,
            "stats": self._stats.copy(),
            "events": self.violence_classifier.get_all_events(),
        }
