"""
Violence Classifier
====================
Maps anomaly‑detection scores to severity levels and tracks violence
start / end timestamps via a simple state machine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict


@dataclass
class ViolenceEvent:
    """Represents a single violence occurrence with start/end times."""
    start_time: str = ""
    end_time: str = ""
    peak_score: float = 0.0
    severity: str = "Low"
    frame_count: int = 0
    avg_score: float = 0.0


class ViolenceClassifier:
    """
    Classifies anomaly scores into severity levels and tracks temporal events.

    Thresholds (configurable):
        low_threshold   – score above this  →  Low violence
        med_threshold   – score above this  →  Medium violence
        high_threshold  – score above this  →  High violence

    The state machine:
        NORMAL  →  score ≥ low_threshold  →  VIOLENCE (start_time recorded)
        VIOLENCE → score < low_threshold for `cooldown_frames` consecutive frames →  NORMAL (end_time recorded)
    """

    def __init__(
        self,
        low_threshold: float = 0.004,
        med_threshold: float = 0.006,
        high_threshold: float = 0.010,
        cooldown_frames: int = 5,
    ):
        self.low_threshold = low_threshold
        self.med_threshold = med_threshold
        self.high_threshold = high_threshold
        self.cooldown_frames = cooldown_frames

        # State machine
        self._in_violence = False
        self._cooldown_counter = 0
        self._current_event: Optional[ViolenceEvent] = None
        self._score_accumulator: List[float] = []
        self.events: List[ViolenceEvent] = []

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------

    def classify_score(self, score: float) -> str:
        """Return severity label for a single anomaly score."""
        if score >= self.high_threshold:
            return "High"
        elif score >= self.med_threshold:
            return "Medium"
        elif score >= self.low_threshold:
            return "Low"
        return "Normal"

    def update(self, score: float, timestamp: Optional[str] = None) -> Optional[ViolenceEvent]:
        """
        Feed the next frame's anomaly score into the state machine.

        Returns a completed ViolenceEvent when a violence period ends,
        otherwise returns None.
        """
        ts = timestamp or datetime.now().isoformat()
        severity = self.classify_score(score)

        if severity != "Normal":
            # ---- Score is above violence threshold ----
            self._cooldown_counter = 0

            if not self._in_violence:
                # Transition: NORMAL → VIOLENCE
                self._in_violence = True
                self._current_event = ViolenceEvent(start_time=ts)
                self._score_accumulator = []

            # Accumulate stats for current event
            self._score_accumulator.append(score)
            self._current_event.frame_count += 1
            if score > self._current_event.peak_score:
                self._current_event.peak_score = score
                self._current_event.severity = severity

            return None

        else:
            # ---- Score is normal ----
            if self._in_violence:
                self._cooldown_counter += 1
                if self._cooldown_counter >= self.cooldown_frames:
                    # Transition: VIOLENCE → NORMAL  →  finalize event
                    return self._finalize_event(ts)
            return None

    def force_close(self, timestamp: Optional[str] = None) -> Optional[ViolenceEvent]:
        """Close any open event (e.g. when the stream ends)."""
        if self._in_violence and self._current_event:
            return self._finalize_event(timestamp or datetime.now().isoformat())
        return None

    # ------------------------------------------------------------------

    def _finalize_event(self, end_ts: str) -> ViolenceEvent:
        evt = self._current_event
        evt.end_time = end_ts
        evt.avg_score = (
            sum(self._score_accumulator) / len(self._score_accumulator)
            if self._score_accumulator
            else 0.0
        )
        self.events.append(evt)
        self._in_violence = False
        self._current_event = None
        self._score_accumulator = []
        self._cooldown_counter = 0
        return evt

    def get_current_status(self) -> Dict:
        """Real‑time status for dashboard display."""
        if self._in_violence and self._current_event:
            return {
                "status": "ALERT",
                "severity": self._current_event.severity,
                "since": self._current_event.start_time,
                "peak_score": self._current_event.peak_score,
                "frame_count": self._current_event.frame_count,
            }
        return {"status": "Normal", "severity": "None"}

    def get_all_events(self) -> List[Dict]:
        return [
            {
                "start": e.start_time,
                "end": e.end_time,
                "severity": e.severity,
                "peak_score": e.peak_score,
                "avg_score": e.avg_score,
                "frames": e.frame_count,
            }
            for e in self.events
        ]
