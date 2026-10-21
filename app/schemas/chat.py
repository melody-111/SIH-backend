from pydantic import BaseModel, model_validator
from typing import Optional, Any, Dict

class UserQuery(BaseModel):
    message: str = ""
    audio_base64: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location: Optional[Dict[str, Any]] = None
    language: str = "en"
    timestamp: Optional[str] = None
    conversation_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def extract_location(cls, data: Any) -> Any:
        if isinstance(data, dict):
            loc = data.get("location")
            if isinstance(loc, dict):
                if data.get("latitude") is None and "latitude" in loc:
                    data["latitude"] = loc["latitude"]
                elif data.get("latitude") is None and "lat" in loc:
                    data["latitude"] = loc["lat"]
                if data.get("longitude") is None and "longitude" in loc:
                    data["longitude"] = loc["longitude"]
                elif data.get("longitude") is None and "lon" in loc:
                    data["longitude"] = loc["lon"]
        return data
