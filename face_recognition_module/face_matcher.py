"""
Face Matcher
=============
Matches detected faces against the college person database.
Uses 128-d face embeddings via OpenCV DNN or a simple histogram
comparison as fallback.

Classifies incidents into:
  Case 1 – College-only
  Case 2 – College + Outsider  (HIGHEST PRIORITY)
  Case 3 – Outsiders-only
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Optional, Tuple

from database.db import get_all_face_embeddings


# Optional: OpenCV DNN face embedding model (OpenFace nn4)
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_EMBEDDING_MODEL = os.path.join(_MODEL_DIR, "openface_nn4.small2.v1.t7")


class FaceMatcher:
    """
    Match face images against college database embeddings.

    If the OpenFace DNN model is available, uses 128-d L2 distance.
    Otherwise falls back to histogram-based comparison.
    """

    def __init__(self, match_threshold: float = 0.6):
        self.match_threshold = match_threshold
        self._embedder = None
        self._use_dnn = False

        if os.path.isfile(_EMBEDDING_MODEL):
            try:
                self._embedder = cv2.dnn.readNetFromTorch(_EMBEDDING_MODEL)
                self._use_dnn = True
                print("[FaceMatcher] Using DNN face embedding model")
            except Exception:
                pass

        if not self._use_dnn:
            print("[FaceMatcher] Using histogram-based face matching (fallback)")

    # ------------------------------------------------------------------
    #  Embedding
    # ------------------------------------------------------------------

    def compute_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Compute a face embedding vector from a cropped face image."""
        if self._use_dnn and self._embedder is not None:
            blob = cv2.dnn.blobFromImage(
                face_image, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
            )
            self._embedder.setInput(blob)
            return self._embedder.forward().flatten().astype(np.float64)

        # Fallback: normalised colour histogram (64 bins)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        resized = cv2.resize(gray, (64, 64))
        hist = cv2.calcHist([resized], [0], None, [64], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)
        return hist.astype(np.float64)

    # ------------------------------------------------------------------
    #  Matching
    # ------------------------------------------------------------------

    def match_face(self, face_image: np.ndarray) -> Dict:
        """
        Match a single face against all college persons.

        Returns:
            {
                "matched": bool,
                "person_id": int | None,
                "person_name": str,
                "confidence": float,   # 0–1  (1 = perfect match)
            }
        """
        embedding = self.compute_embedding(face_image)
        db_entries = get_all_face_embeddings()  # [(id, name, emb), ...]

        best_match = {"matched": False, "person_id": None, "person_name": "Unknown", "confidence": 0.0}
        best_distance = float("inf")

        for person_id, name, db_emb in db_entries:
            # Ensure same shape
            if db_emb.shape != embedding.shape:
                continue
            distance = np.linalg.norm(embedding - db_emb)
            if distance < best_distance:
                best_distance = distance

                # Lower distance = better match; convert to confidence
                confidence = max(0.0, 1.0 - distance / 2.0)
                if confidence >= self.match_threshold:
                    best_match = {
                        "matched": True,
                        "person_id": person_id,
                        "person_name": name,
                        "confidence": round(confidence, 3),
                    }

        return best_match

    def match_faces(self, face_images: List[np.ndarray]) -> List[Dict]:
        """Match a list of face images, return match result per face."""
        return [self.match_face(img) for img in face_images]

    # ------------------------------------------------------------------
    #  Incident classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_incident(match_results: List[Dict]) -> str:
        """
        Classify the incident based on face matching results.

        Returns one of:
            'college_only'          – All faces matched
            'college_and_outsider'  – Mix of matched + unmatched (⚠️)
            'outsiders_only'        – No faces matched
            'unclassified'          – No faces detected
        """
        if not match_results:
            return "unclassified"

        matched = sum(1 for r in match_results if r["matched"])
        unmatched = len(match_results) - matched

        if matched > 0 and unmatched > 0:
            return "college_and_outsider"
        elif matched > 0:
            return "college_only"
        elif unmatched > 0:
            return "outsiders_only"
        return "unclassified"
