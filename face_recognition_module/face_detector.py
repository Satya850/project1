"""
Face Detector
==============
Detects faces in video frames using OpenCV's DNN face detector.
Falls back to Haar cascades if DNN model files aren't available.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


# Paths for the DNN face detector model (Caffe)
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_PROTOTXT = os.path.join(_MODEL_DIR, "deploy.prototxt")
_CAFFEMODEL = os.path.join(_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


class FaceDetector:
    """
    Detect faces in BGR frames.

    Uses OpenCV DNN SSD face detector when model files are present,
    otherwise falls back to Haar cascade classifier.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._net = None
        self._cascade = None
        self._use_dnn = False

        # Try loading DNN model
        if os.path.isfile(_PROTOTXT) and os.path.isfile(_CAFFEMODEL):
            try:
                self._net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)
                self._use_dnn = True
                print("[FaceDetector] Using DNN (SSD) face detector")
            except Exception:
                pass

        if not self._use_dnn:
            # Fallback: Haar cascade (always available in OpenCV)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            print("[FaceDetector] Using Haar cascade face detector (fallback)")

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Detect faces in a BGR frame.

        Returns list of dicts:
            [{"bbox": (x, y, w, h), "confidence": float, "face_image": np.ndarray}, ...]
        """
        if self._use_dnn:
            return self._detect_dnn(frame)
        return self._detect_haar(frame)

    # ------------------------------------------------------------------
    def _detect_dnn(self, frame: np.ndarray) -> List[dict]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            face_img = frame[y1:y2, x1:x2].copy()
            faces.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "confidence": confidence,
                "face_image": face_img,
            })
        return faces

    def _detect_haar(self, frame: np.ndarray) -> List[dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        rects = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = []
        for (x, y, w, h) in rects:
            face_img = frame[y: y + h, x: x + w].copy()
            faces.append({
                "bbox": (x, y, w, h),
                "confidence": 1.0,
                "face_image": face_img,
            })
        return faces
