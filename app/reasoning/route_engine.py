from typing import Dict, Any

class RouteEngine:
    def calculate_route(self, start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> Dict[str, Any]:
        return {
            "coordinates": [
                {"latitude": start_lat, "longitude": start_lon},
                {"latitude": end_lat, "longitude": end_lon}
            ],
            "estimated_time_mins": 45,
            "hazards_on_route": []
        }
