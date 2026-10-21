import logging
from typing import Dict, Any, List, Optional
from app.db.database import DatabaseManager

logger = logging.getLogger("neerai.feedback.analyzer")

# Recognized Quality Problem Categories for Marine Intelligence
QUALITY_CATEGORIES = {
    "FACTUAL_CLAIM": "Reported inaccurate marine condition or wave measurement",
    "MISSING_PROVENANCE": "Missing source agency or observation timestamp",
    "WRONG_LANGUAGE": "Response produced in incorrect or unlocalized language",
    "IRRELEVANT_ROUTING": "Unnecessary or irrelevant specialist agents dispatched",
    "INCOMPLETE_ANSWER": "Answer failed to address core aspect of fisherman query",
    "STALE_DATA_UNDISCLOSED": "Expired or cached observations used without disclosure",
    "INTENT_MISINTERPRETATION": "User question intent misclassified by planner",
    "TECHNICAL_FAILURE": "Timeout, empty response, or unhandled agent error"
}

class FeedbackQualityAnalyzer:
    """
    Privacy-conscious feedback quality analyzer.
    Analyzes aggregate user feedback trends to identify systematic defects.

    SAFETY & GOVERNANCE INVARIANTS:
    1. Zero Fabrication: Never invents feedback records when real data is scarce.
    2. Feedback is treated strictly as an observational SIGNAL, never as ground-truth marine facts.
    3. Production prompts and routing policies are NEVER autonomously modified by user feedback.
    4. Personal identifiers are strictly excluded from trend reports.
    """
    def __init__(self):
        self.db = DatabaseManager()

    def classify_signal_category(self, notes: str, user_category: str) -> str:
        """Categorizes user feedback signal into one of the 8 formal problem types."""
        text = (notes or "").lower() + " " + (user_category or "").lower()

        if any(w in text for w in ["language", "bhasha", "hindi", "english", "wrong language", "translate"]):
            return "WRONG_LANGUAGE"
        if any(w in text for w in ["source", "timestamp", "kahan se", "who said", "provenance", "agency"]):
            return "MISSING_PROVENANCE"
        if any(w in text for w in ["stale", "purana", "old", "yesterday", "expired", "cache"]):
            return "STALE_DATA_UNDISCLOSED"
        if any(w in text for w in ["route", "unnecessary", "gis", "agent", "extra", "slow"]):
            return "IRRELEVANT_ROUTING"
        if any(w in text for w in ["intent", "galat samjha", "misunderstood", "didn't ask"]):
            return "INTENT_MISINTERPRETATION"
        if any(w in text for w in ["incomplete", "adhoora", "short", "half", "more detail"]):
            return "INCOMPLETE_ANSWER"
        if any(w in text for w in ["error", "fail", "timeout", "crash", "500", "broken"]):
            return "TECHNICAL_FAILURE"
        
        return "FACTUAL_CLAIM"

    def analyze_feedback_corpus(self, min_sample_size: int = 5) -> Dict[str, Any]:
        """
        Aggregates consent-granted feedback submissions to detect statistically significant quality issues.
        Returns honest status if data volume is below min_sample_size.
        """
        with self.db.get_connection() as conn:
            rows = conn.execute("""
                SELECT feedback_id, correlation_id, rating, category, feedback_notes, created_at
                FROM user_feedback
                WHERE consent_given = 1
                ORDER BY created_at DESC;
            """).fetchall()

        total = len(rows)
        if total == 0:
            return {
                "status": "NO_DATA",
                "sample_size": 0,
                "message": "No consent-granted user feedback submissions found in database. Zero synthetic records fabricated.",
                "actionable_trends": [],
                "recommendation": "Maintain verified deterministic baseline. Do not alter routing without real user evidence."
            }

        ratings = [r["rating"] for r in rows]
        avg_rating = round(sum(ratings) / total, 2)
        low_rated = [r for r in rows if r["rating"] <= 2]

        # Categorize low ratings
        category_counts: Dict[str, int] = {cat: 0 for cat in QUALITY_CATEGORIES}
        for r in low_rated:
            cat = self.classify_signal_category(r["feedback_notes"], r["category"])
            category_counts[cat] = category_counts.get(cat, 0) + 1

        actionable_trends = []
        if total >= min_sample_size:
            for cat, count in category_counts.items():
                frequency = round((count / total) * 100.0, 1)
                if frequency >= 20.0:  # >=20% incidence threshold
                    actionable_trends.append({
                        "category": cat,
                        "description": QUALITY_CATEGORIES[cat],
                        "occurrences": count,
                        "frequency_pct": frequency,
                        "requires_candidate_evaluation": True
                    })

        status = "ACTIONABLE" if actionable_trends else ("INSUFFICIENT_DATA" if total < min_sample_size else "NORMAL")
        recommendation = (
            f"Insufficient feedback volume (N={total} < {min_sample_size}) for automated statistical significance. Baseline preserved."
            if total < min_sample_size else
            ("Actionable trends detected; human engineer review recommended for candidate evaluation." if actionable_trends else "Quality within normal bounds.")
        )

        return {
            "status": status,
            "sample_size": total,
            "average_rating": avg_rating,
            "low_rating_count": len(low_rated),
            "low_rating_ratio": round(len(low_rated) / total, 2),
            "problem_category_distribution": category_counts,
            "actionable_trends": actionable_trends,
            "recommendation": recommendation,
            "governance_rule": "Feedback is treated strictly as an observational signal, never as ground-truth marine evidence."
        }
