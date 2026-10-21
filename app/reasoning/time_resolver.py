import datetime
from typing import Dict, Any

class TimeResolver:
    def resolve_time(self, message: str) -> Dict[str, Any]:
        msg = message.lower()
        now = datetime.datetime.utcnow()
        
        if "kal" in msg or "tomorrow" in msg:
            target = now + datetime.timedelta(days=1)
        elif "parso" in msg:
            target = now + datetime.timedelta(days=2)
        else:
            target = now
            
        return {
            "human_readable": "Current/Specified Time",
            "utc_timestamp": target.isoformat(),
            "is_future": target > now
        }
