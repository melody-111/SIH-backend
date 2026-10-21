from app.schemas.marine_data import EvidenceItem
from typing import Dict, Any, List

class EvidenceEngine:
    def extract_evidence(self, context_dict: Dict[str, Any]) -> List[EvidenceItem]:
        evidence_list = []
        
        if "weather" in context_dict and context_dict["weather"].get("data"):
            w = context_dict["weather"]
            evidence_list.append(
                EvidenceItem(
                    claim="Weather conditions captured",
                    source=w.get("sources", ["Unknown"])[0],
                    measurement="temperature/wind",
                    value=str(w["data"].get("temperature", "N/A")),
                    unit="C",
                    timestamp=w.get("timestamp", ""),
                    confidence=w.get("confidence", 0.0)
                )
            )
            
        if "sst" in context_dict and context_dict["sst"].get("data"):
            s = context_dict["sst"]
            evidence_list.append(
                EvidenceItem(
                    claim="SST Data Search",
                    source=s.get("sources", ["Unknown"])[0],
                    measurement="collections",
                    value=str(s["data"].get("collections_found", 0)),
                    unit="count",
                    timestamp=s.get("timestamp", ""),
                    confidence=s.get("confidence", 0.0)
                )
            )
            
        # Add more mappings as needed for other agents
        return evidence_list
