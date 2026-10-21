import datetime
import logging
from typing import Dict, Any, List, Optional
from app.db.database import DatabaseManager

logger = logging.getLogger("neerai.governance")

class ControlledQualityImprovementEngine:
    """
    Manages the controlled quality improvement loop for NeerAI:
    User-Approved Feedback -> Privacy Sanitization -> Candidate Evaluation Cases ->
    Offline Regression Benchmark -> Human Architecture Approval -> Versioned Deployment.

    GOVERNANCE RULES:
    1. ZERO automated prompt changes based on unreviewed individual ratings.
    2. ZERO automatic fine-tuning or weight training on private fisherman conversations.
    3. Strict explicit consent required before feedback can be reviewed for evaluations.
    """
    def __init__(self):
        self.db = DatabaseManager()

    def get_governance_policy(self) -> Dict[str, Any]:
        """Returns the formal NeerAI AI safety & quality improvement policy."""
        return {
            "automated_prompt_mutation": "DISABLED",
            "automated_weight_fine_tuning": "DISABLED",
            "private_conversation_training": "PROHIBITED",
            "explicit_consent_enforcement": "STRICT",
            "quality_loop": [
                "1. User submits voluntary rating with explicit consent",
                "2. Feedback sanitized and stored with request correlation ID",
                "3. Problematic cases (ratings 1-2) isolated for evaluation candidate review",
                "4. Offline synthetic benchmark run to assess baseline score",
                "5. Candidate routing/prompt improvements tested against regression suite",
                "6. Human AI engineer reviews benchmark delta and approves changes",
                "7. Versioned release deployed with telemetry monitoring"
            ]
        }

    def inspect_feedback_candidates(self, max_rating: int = 3, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Retrieves consent-given feedback items with lower ratings (1-3)
        to identify queries needing improved routing or clarity in the offline benchmark.
        """
        with self.db.get_connection() as conn:
            rows = conn.execute("""
                SELECT feedback_id, correlation_id, message_id, rating, category, feedback_notes, created_at
                FROM user_feedback
                WHERE consent_given = 1 AND rating <= ?
                ORDER BY created_at DESC
                LIMIT ?;
            """, (max_rating, limit)).fetchall()
            candidates = []
            for r in rows:
                candidates.append({
                    "feedback_id": r["feedback_id"],
                    "correlation_id": r["correlation_id"] or "unlinked",
                    "rating": r["rating"],
                    "category": r["category"],
                    "notes": r["feedback_notes"] or "",
                    "candidate_for_benchmark": True,
                    "submitted_at": r["created_at"]
                })
            return candidates

    def get_quality_metrics_summary(self) -> Dict[str, Any]:
        """Provides an aggregated summary of recorded feedback and consent statistics."""
        with self.db.get_connection() as conn:
            row = conn.execute("""
                SELECT 
                    COUNT(*) as total_count,
                    AVG(rating) as average_rating,
                    SUM(CASE WHEN consent_given = 1 THEN 1 ELSE 0 END) as consent_count,
                    SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative_count
                FROM user_feedback;
            """).fetchone()

            total = row["total_count"] or 0
            avg_rating = round(row["average_rating"] or 0.0, 2)
            consented = row["consent_count"] or 0
            pos = row["positive_count"] or 0
            neg = row["negative_count"] or 0

            return {
                "total_feedback_submissions": total,
                "average_user_rating": avg_rating,
                "explicit_consent_rate": round(consented / total, 2) if total > 0 else 1.0,
                "positive_ratings_count": pos,
                "negative_ratings_count": neg,
                "governance_status": "CONTROLLED_HUMAN_IN_THE_LOOP"
            }
