"""
Incident Manager
=================
High‑level helpers that combine database operations with business logic
for the college security use‑cases.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from database import db as database
from database.schema import init_db


class IncidentManager:
    """
    Provides convenience methods on top of the raw database layer:
    - tag incidents by use‑case (ragging, fight, unauthorized, late‑night)
    - repeat‑offender queries
    - recent alert summaries for the dashboard
    """

    USE_CASE_LABELS = {
        "ragging": "Ragging Detection",
        "physical_fight": "Physical Fight",
        "unauthorized_access": "Unauthorized Access",
        "suspicious_activity": "Late‑Night Suspicious Behaviour",
        "vandalism": "Vandalism",
        "exam_cheating": "Exam Cheating / Copying",
        "other": "Other",
    }

    def __init__(self):
        init_db()

    # ------------------------------------------------------------------
    #  Queries
    # ------------------------------------------------------------------

    def get_recent_incidents(self, limit: int = 50) -> List[Dict]:
        return database.get_incidents(limit=limit)

    def get_incident_detail(self, incident_id: int) -> Optional[Dict]:
        return database.get_incident_with_persons(incident_id)

    def get_active_alerts(self) -> List[Dict]:
        return database.get_incidents(limit=20, severity=None)

    def get_person_history(self, person_id: int) -> Dict:
        """Get a person's incident history and repeat‑offender status."""
        person = database.get_person(person_id)
        if not person:
            return {"error": "Person not found"}

        count = database.get_person_incident_count(person_id)
        return {
            "person": person,
            "incident_count": count,
            "repeat_offender": count >= 3,
        }

    def resolve_incident(self, incident_id: int, notes: str = ""):
        database.update_incident(incident_id, status="resolved", notes=notes)

    def mark_false_alarm(self, incident_id: int):
        database.update_incident(incident_id, status="false_alarm")

    # ------------------------------------------------------------------
    #  Analytics helpers
    # ------------------------------------------------------------------

    def get_dashboard_summary(self, days: int = 7) -> Dict:
        stats = database.get_incident_stats(days=days)
        return {
            "period_days": days,
            **stats,
        }

    def get_use_case_summary(self) -> Dict:
        """Return human‑readable use‑case labels with counts."""
        stats = database.get_incident_stats(days=30)
        by_type = stats.get("by_type", {})
        return {
            self.USE_CASE_LABELS.get(k, k): v
            for k, v in by_type.items()
        }
