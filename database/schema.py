"""
Database Schema & Initialization
=================================
SQLite database for the college security surveillance system.
Tables: persons, cameras, incidents, incident_persons, evidence
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "college_security.db")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get a database connection with row factory."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Optional[str] = None):
    """Create all tables if they don't exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS persons (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        college_id      TEXT UNIQUE NOT NULL,
        name            TEXT NOT NULL,
        department      TEXT DEFAULT '',
        role            TEXT CHECK(role IN ('Student','Faculty','Staff','Other')) DEFAULT 'Student',
        phone           TEXT DEFAULT '',
        email           TEXT DEFAULT '',
        photo_path      TEXT DEFAULT '',
        face_embedding  BLOB,
        is_active       INTEGER DEFAULT 1,
        created_at      TEXT DEFAULT (datetime('now')),
        updated_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS cameras (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id       TEXT UNIQUE NOT NULL,
        name            TEXT NOT NULL,
        location        TEXT DEFAULT '',
        stream_url      TEXT DEFAULT '',
        cam_type        TEXT CHECK(cam_type IN ('webcam','rtsp','file')) DEFAULT 'webcam',
        status          TEXT CHECK(status IN ('active','inactive','maintenance')) DEFAULT 'inactive',
        created_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS incidents (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_type   TEXT CHECK(incident_type IN (
                            'physical_fight','ragging','unauthorized_access',
                            'suspicious_activity','vandalism','exam_cheating','other'
                        )) DEFAULT 'other',
        severity        TEXT CHECK(severity IN ('Low','Medium','High')) DEFAULT 'Medium',
        classification  TEXT CHECK(classification IN (
                            'college_only','college_and_outsider','outsiders_only','unclassified'
                        )) DEFAULT 'unclassified',
        camera_id       TEXT,
        location        TEXT DEFAULT '',
        start_time      TEXT NOT NULL,
        end_time        TEXT,
        duration_sec    REAL DEFAULT 0,
        video_clip_path TEXT DEFAULT '',
        status          TEXT CHECK(status IN ('active','resolved','investigating','false_alarm')) DEFAULT 'active',
        notes           TEXT DEFAULT '',
        anomaly_score   REAL DEFAULT 0,
        created_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS incident_persons (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id     INTEGER NOT NULL,
        person_id       INTEGER,
        is_outsider     INTEGER DEFAULT 0,
        outsider_image  TEXT DEFAULT '',
        confidence      REAL DEFAULT 0,
        role_in_incident TEXT DEFAULT 'involved',
        FOREIGN KEY (incident_id) REFERENCES incidents(id),
        FOREIGN KEY (person_id) REFERENCES persons(id)
    );

    CREATE TABLE IF NOT EXISTS evidence (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id     INTEGER NOT NULL,
        evidence_type   TEXT CHECK(evidence_type IN ('image','video','snapshot')) DEFAULT 'image',
        file_path       TEXT NOT NULL,
        description     TEXT DEFAULT '',
        captured_at     TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (incident_id) REFERENCES incidents(id)
    );

    CREATE INDEX IF NOT EXISTS idx_incidents_start ON incidents(start_time);
    CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);
    CREATE INDEX IF NOT EXISTS idx_incidents_type ON incidents(incident_type);
    CREATE INDEX IF NOT EXISTS idx_incident_persons_incident ON incident_persons(incident_id);
    CREATE INDEX IF NOT EXISTS idx_incident_persons_person ON incident_persons(person_id);
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path or DB_PATH}")
