from typing import Dict, Any

class RecommendationEngine:
    def recommend(self, context_data: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
        if risk.get("level") == "HIGH":
            return {
                "recommended": False,
                "area": None,
                "reasons": ["Risk level is too high for safe recommendation."]
            }
            
        # Mocking an offset for safe area recommendation
        return {
            "recommended": True,
            "area": {
                "latitude": 15.3493, # slight offset from user
                "longitude": 74.1740
            },
            "reasons": ["SST and Chlorophyll data suggest favorable conditions.", "Weather is calm in this sector."],
            "score": 85
        }
