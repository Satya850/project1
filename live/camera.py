"""
Camera Manager
===============
Manages video sources: webcam, RTSP stream, or video file.
Provides a uniform frame‑producer interface used by the live processor.
"""

import cv2
import time
import threading
from typing import Optional, Tuple
import numpy as np


class CameraManager:
    """
    Wraps OpenCV VideoCapture with reconnection logic and FPS tracking.

    Usage:
        cam = CameraManager(source=0)          # webcam
        cam = CameraManager(source="rtsp://…") # RTSP
        cam = CameraManager(source="video.mp4")# file
        cam.start()
        while cam.is_running():
            frame = cam.read()
            ...
        cam.stop()
    """

    def __init__(
        self,
        source=0,
        camera_id: str = "cam_0",
        name: str = "Default Camera",
        max_reconnect: int = 5,
        target_fps: int = 15,
    ):
        self.source = source
        self.camera_id = camera_id
        self.name = name
        self.max_reconnect = max_reconnect
        self.target_fps = target_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = 0.0

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Open the video source and start the capture thread."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            print(f"[Camera {self.camera_id}] Failed to open source: {self.source}")
            return False

        self._running = True
        self._last_fps_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[Camera {self.camera_id}] Started — {self.source}")
        return True

    def stop(self):
        """Stop capture thread and release resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        print(f"[Camera {self.camera_id}] Stopped")

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    #  Frame access
    # ------------------------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        """Get the latest frame (thread‑safe)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_fps(self) -> float:
        return self._fps

    def get_frame_count(self) -> int:
        return self._frame_count

    def get_info(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source": str(self.source),
            "running": self._running,
            "fps": round(self._fps, 1),
            "frames": self._frame_count,
        }

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _capture_loop(self):
        reconnect_count = 0
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if reconnect_count >= self.max_reconnect:
                    print(f"[Camera {self.camera_id}] Max reconnect attempts reached")
                    self._running = False
                    break
                reconnect_count += 1
                print(f"[Camera {self.camera_id}] Reconnecting ({reconnect_count}/{self.max_reconnect})…")
                time.sleep(2)
                self._cap = cv2.VideoCapture(self.source)
                continue

            ret, frame = self._cap.read()
            if not ret:
                # End of video file or stream error
                if isinstance(self.source, str) and not self.source.startswith("rtsp"):
                    # Video file ended
                    self._running = False
                    break
                # Stream error — try reconnecting
                self._cap.release()
                self._cap = None
                continue

            reconnect_count = 0
            with self._lock:
                self._frame = frame

            self._frame_count += 1

            # FPS calculation
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._frame_count / elapsed if elapsed > 0 else 0
                self._frame_count = 0
                self._last_fps_time = now

            # Throttle to target FPS
            if frame_interval > 0:
                time.sleep(frame_interval)
