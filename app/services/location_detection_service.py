import httpx
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("neerai.location.detect")

class LocationDetectionService:
    """
    Real-time client geolocation detection service.
    Detects user city, state, country, and precise latitude/longitude
    based on network IP and cellular/broadband connection.
    Free, high-speed, and requires no proprietary API keys.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocationDetectionService, cls).__new__(cls)
            cls._instance._client = httpx.AsyncClient(timeout=4.0)
        return cls._instance

    async def detect_location(self, client_ip: Optional[str] = None) -> Dict[str, Any]:
        """
        Detects coordinates from IP address.
        If client_ip is loopback/private (127.0.0.1, ::1, 192.168.*),
        queries external IP directly to detect machine location.
        """
        is_local = (
            not client_ip
            or client_ip in ["127.0.0.1", "::1", "localhost"]
            or client_ip.startswith("192.168.")
            or client_ip.startswith("10.")
            or client_ip.startswith("172.")
        )

        url = "http://ip-api.com/json/" if is_local else f"http://ip-api.com/json/{client_ip}"

        try:
            res = await self._client.get(url)
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "success":
                    city = data.get("city", "Unknown City")
                    region = data.get("regionName", "")
                    country = data.get("country", "India")
                    lat = float(data.get("lat", 26.8467))
                    lon = float(data.get("lon", 80.9462))
                    disp_name = f"{city}, {region}, {country}".replace(", ,", ",").strip(", ")

                    return {
                        "status": "success",
                        "is_detected": True,
                        "city": city,
                        "region": region,
                        "country": country,
                        "display_name": disp_name,
                        "latitude": lat,
                        "longitude": lon,
                        "ip": data.get("query", client_ip or ""),
                        "isp": data.get("isp", ""),
                        "source": "IP Geolocation Network"
                    }
        except Exception as e:
            logger.warning(f"IP Geolocation error: {e}")

        # Graceful fallback: Lucknow (Northern Hub) or Goa Coast
        return {
            "status": "fallback",
            "is_detected": False,
            "city": "Lucknow",
            "region": "Uttar Pradesh",
            "country": "India",
            "display_name": "Lucknow, Uttar Pradesh, India (Default Anchor)",
            "latitude": 26.8467,
            "longitude": 80.9462,
            "ip": client_ip or "127.0.0.1",
            "source": "NeerAI Geographic Anchor"
        }
