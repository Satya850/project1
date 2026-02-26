"""
Database Operations (CRUD)
===========================
All data‑access helpers for persons, cameras, incidents, and evidence.
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from .schema import get_connection, init_db


# ---------------------------------------------------------------------------
#  Persons
# ---------------------------------------------------------------------------

def add_person(
    college_id: str,
    name: str,
    department: str = "",
    role: str = "Student",
    phone: str = "",
    email: str = "",
    photo_path: str = "",
    face_embedding: Optional[np.ndarray] = None,
) -> int:
    """Add a new person to the database. Returns the person's row id."""
    conn = get_connection()
    embedding_blob = face_embedding.tobytes() if face_embedding is not None else None
    cursor = conn.execute(
        """INSERT INTO persons (college_id, name, department, role, phone, email, photo_path, face_embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (college_id, name, department, role, phone, email, photo_path, embedding_blob),
    )
    conn.commit()
    pid = cursor.lastrowid
    conn.close()
    return pid


def get_person(person_id: int) -> Optional[Dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_person_by_college_id(college_id: str) -> Optional[Dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM persons WHERE college_id = ?", (college_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_persons(active_only: bool = True) -> List[Dict]:
    conn = get_connection()
    query = "SELECT * FROM persons"
    if active_only:
        query += " WHERE is_active = 1"
    query += " ORDER BY name"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_face_embeddings() -> List[Tuple[int, str, np.ndarray]]:
    """Return (person_id, name, embedding) for every person that has an embedding."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, name, face_embedding FROM persons WHERE face_embedding IS NOT NULL AND is_active = 1"
    ).fetchall()
    conn.close()
    results = []
    for r in rows:
        emb = np.frombuffer(r["face_embedding"], dtype=np.float64)
        results.append((r["id"], r["name"], emb))
    return results


def update_person(person_id: int, **fields):
    conn = get_connection()
    if "face_embedding" in fields and fields["face_embedding"] is not None:
        fields["face_embedding"] = fields["face_embedding"].tobytes()
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [person_id]
    conn.execute(f"UPDATE persons SET {sets}, updated_at = datetime('now') WHERE id = ?", vals)
    conn.commit()
    conn.close()


def delete_person(person_id: int):
    conn = get_connection()
    conn.execute("UPDATE persons SET is_active = 0, updated_at = datetime('now') WHERE id = ?", (person_id,))
    conn.commit()
    conn.close()


def get_person_incident_count(person_id: int) -> int:
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM incident_persons WHERE person_id = ?", (person_id,)
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
#  Cameras
# ---------------------------------------------------------------------------

def add_camera(camera_id: str, name: str, location: str = "", stream_url: str = "",
               cam_type: str = "webcam") -> int:
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO cameras (camera_id, name, location, stream_url, cam_type) VALUES (?,?,?,?,?)",
        (camera_id, name, location, stream_url, cam_type),
    )
    conn.commit()
    cid = cursor.lastrowid
    conn.close()
    return cid


def get_all_cameras() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM cameras ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_camera_status(camera_id: str, status: str):
    conn = get_connection()
    conn.execute("UPDATE cameras SET status = ? WHERE camera_id = ?", (status, camera_id))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
#  Incidents
# ---------------------------------------------------------------------------

def create_incident(
    incident_type: str = "other",
    severity: str = "Medium",
    classification: str = "unclassified",
    camera_id: str = "",
    location: str = "",
    start_time: str = "",
    end_time: str = "",
    duration_sec: float = 0,
    video_clip_path: str = "",
    anomaly_score: float = 0,
    notes: str = "",
) -> int:
    conn = get_connection()
    if not start_time:
        start_time = datetime.now().isoformat()
    cursor = conn.execute(
        """INSERT INTO incidents
           (incident_type, severity, classification, camera_id, location,
            start_time, end_time, duration_sec, video_clip_path, anomaly_score, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (incident_type, severity, classification, camera_id, location,
         start_time, end_time, duration_sec, video_clip_path, anomaly_score, notes),
    )
    conn.commit()
    iid = cursor.lastrowid
    conn.close()
    return iid


def update_incident(incident_id: int, **fields):
    conn = get_connection()
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [incident_id]
    conn.execute(f"UPDATE incidents SET {sets} WHERE id = ?", vals)
    conn.commit()
    conn.close()


def get_incident(incident_id: int) -> Optional[Dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_incidents(
    limit: int = 100,
    severity: Optional[str] = None,
    incident_type: Optional[str] = None,
    classification: Optional[str] = None,
    since: Optional[str] = None,
) -> List[Dict]:
    conn = get_connection()
    query = "SELECT * FROM incidents WHERE 1=1"
    params: list = []
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if incident_type:
        query += " AND incident_type = ?"
        params.append(incident_type)
    if classification:
        query += " AND classification = ?"
        params.append(classification)
    if since:
        query += " AND start_time >= ?"
        params.append(since)
    query += " ORDER BY start_time DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_incident_with_persons(incident_id: int) -> Optional[Dict]:
    """Get incident details including involved persons."""
    conn = get_connection()
    incident = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    if not incident:
        conn.close()
        return None

    persons = conn.execute(
        """SELECT ip.*, p.college_id, p.name, p.department, p.role, p.photo_path
           FROM incident_persons ip
           LEFT JOIN persons p ON ip.person_id = p.id
           WHERE ip.incident_id = ?""",
        (incident_id,),
    ).fetchall()

    evidence_rows = conn.execute(
        "SELECT * FROM evidence WHERE incident_id = ?", (incident_id,)
    ).fetchall()

    conn.close()
    result = dict(incident)
    result["persons"] = [dict(p) for p in persons]
    result["evidence"] = [dict(e) for e in evidence_rows]
    return result


# ---------------------------------------------------------------------------
#  Incident ↔ Persons link
# ---------------------------------------------------------------------------

def link_person_to_incident(
    incident_id: int,
    person_id: Optional[int] = None,
    is_outsider: bool = False,
    outsider_image: str = "",
    confidence: float = 0.0,
):
    conn = get_connection()
    conn.execute(
        """INSERT INTO incident_persons (incident_id, person_id, is_outsider, outsider_image, confidence)
           VALUES (?,?,?,?,?)""",
        (incident_id, person_id, int(is_outsider), outsider_image, confidence),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
#  Evidence
# ---------------------------------------------------------------------------

def add_evidence(incident_id: int, evidence_type: str, file_path: str, description: str = "") -> int:
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO evidence (incident_id, evidence_type, file_path, description) VALUES (?,?,?,?)",
        (incident_id, evidence_type, file_path, description),
    )
    conn.commit()
    eid = cursor.lastrowid
    conn.close()
    return eid


# ---------------------------------------------------------------------------
#  Analytics helpers
# ---------------------------------------------------------------------------

def get_incident_stats(days: int = 7) -> Dict[str, Any]:
    """Aggregate counts for the analytics dashboard."""
    conn = get_connection()
    since = (datetime.now() - timedelta(days=days)).isoformat()

    total = conn.execute(
        "SELECT COUNT(*) as c FROM incidents WHERE start_time >= ?", (since,)
    ).fetchone()["c"]

    by_severity = conn.execute(
        "SELECT severity, COUNT(*) as c FROM incidents WHERE start_time >= ? GROUP BY severity",
        (since,),
    ).fetchall()

    by_type = conn.execute(
        "SELECT incident_type, COUNT(*) as c FROM incidents WHERE start_time >= ? GROUP BY incident_type",
        (since,),
    ).fetchall()

    by_day = conn.execute(
        """SELECT DATE(start_time) as day, COUNT(*) as c
           FROM incidents WHERE start_time >= ?
           GROUP BY DATE(start_time) ORDER BY day""",
        (since,),
    ).fetchall()

    by_hour = conn.execute(
        """SELECT CAST(strftime('%H', start_time) AS INTEGER) as hour, COUNT(*) as c
           FROM incidents WHERE start_time >= ?
           GROUP BY hour ORDER BY hour""",
        (since,),
    ).fetchall()

    by_classification = conn.execute(
        "SELECT classification, COUNT(*) as c FROM incidents WHERE start_time >= ? GROUP BY classification",
        (since,),
    ).fetchall()

    conn.close()
    return {
        "total": total,
        "by_severity": {r["severity"]: r["c"] for r in by_severity},
        "by_type": {r["incident_type"]: r["c"] for r in by_type},
        "by_day": {r["day"]: r["c"] for r in by_day},
        "by_hour": {r["hour"]: r["c"] for r in by_hour},
        "by_classification": {r["classification"]: r["c"] for r in by_classification},
    }
