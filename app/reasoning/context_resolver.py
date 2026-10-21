import re
from typing import Dict, Any, Optional, Tuple

KNOWN_COASTAL_PORTS: Dict[str, Tuple[float, float]] = {
    "goa": (15.2993, 73.8000),
    "panaji": (15.4989, 73.8278),
    "mormugao": (15.4167, 73.8000),
    "mumbai": (18.9220, 72.8347),
    "bombay": (18.9220, 72.8347),
    "kochi": (9.9312, 76.2673),
    "cochin": (9.9312, 76.2673),
    "chennai": (13.0827, 80.2707),
    "madras": (13.0827, 80.2707),
    "mangalore": (12.8700, 74.8800),
    "ratnagiri": (16.9902, 73.3120),
    "visakhapatnam": (17.6868, 83.2185),
    "vizag": (17.6868, 83.2185),
    "veraval": (20.9000, 70.3700),
    "paradip": (20.3160, 86.6110),
    "karwar": (14.8100, 74.1300),
    "kollam": (8.8932, 76.6141),
    "tuticorin": (8.7642, 78.1348),
    "thoothukudi": (8.7642, 78.1348),
    "puri": (19.8135, 85.8312),
    "digha": (21.6266, 87.5074),
    "porbandar": (21.6417, 69.6293),
    "kandla": (23.0000, 70.2167),
    "thiruvananthapuram": (8.5241, 76.9366),
    "trivandrum": (8.5241, 76.9366),
    "vizhinjam": (8.3804, 76.9930),
    "kanyakumari": (8.0883, 77.5385),
    "alappuzha": (9.4981, 76.3388),
    "kozhikode": (11.2588, 75.7804),
    "calicut": (11.2588, 75.7804),
    "kannur": (11.8745, 75.3704),
    "machilipatnam": (16.1875, 81.1389),
    "kakinada": (16.9891, 82.2475)
}

_context_memory: Dict[str, Dict[str, Any]] = {}

class ContextResolver:
    """
    Resolves geographic and conversational context for marine queries.
    Identifies explicit coastal ports, extracts coordinates, and handles spatial follow-ups.
    """
    def resolve_context(
        self,
        conversation_id: str,
        message: str,
        intent: str,
        current_lat: float,
        current_lon: float
    ) -> Dict[str, Any]:
        global _context_memory
        msg_lower = (message or "").lower()

        resolved_lat = current_lat
        resolved_lon = current_lon
        resolved_port: Optional[str] = None

        # 1. Extract potential location name using LocationIntelligenceService
        from app.services.location_intelligence_service import LocationIntelligenceService
        loc_service = LocationIntelligenceService()

        extracted_name = loc_service.extract_candidate_location_name(message)
        geocoded = None
        if extracted_name:
            if extracted_name.lower() in KNOWN_COASTAL_PORTS:
                p_lat, p_lon = KNOWN_COASTAL_PORTS[extracted_name.lower()]
                resolved_lat = p_lat
                resolved_lon = p_lon
                resolved_port = extracted_name.capitalize()
            else:
                from app.services.location_intelligence_service import _GEOCODE_CACHE
                if extracted_name.lower() in _GEOCODE_CACHE:
                    cached = _GEOCODE_CACHE[extracted_name.lower()]
                    resolved_lat = cached["latitude"]
                    resolved_lon = cached["longitude"]
                    resolved_port = cached["name"]

        # 2. Check for explicit coastal port keywords in message (Origin and Destination)
        ports_found = []
        for port_name, (p_lat, p_lon) in KNOWN_COASTAL_PORTS.items():
            match = re.search(r"\b" + re.escape(port_name) + r"\b", msg_lower)
            if match:
                ports_found.append((match.start(), port_name, p_lat, p_lon))

        ports_found.sort(key=lambda x: x[0])
        target_lat = None
        target_lon = None
        target_port = None

        if len(ports_found) >= 2:
            _, origin_name, o_lat, o_lon = ports_found[0]
            _, dest_name, d_lat, d_lon = ports_found[1]
            resolved_lat = o_lat
            resolved_lon = o_lon
            resolved_port = origin_name.capitalize()
            target_lat = d_lat
            target_lon = d_lon
            target_port = dest_name.capitalize()
        elif len(ports_found) == 1:
            _, p_name, p_lat, p_lon = ports_found[0]
            resolved_lat = p_lat
            resolved_lon = p_lon
            resolved_port = p_name.capitalize()

        # 3. If no explicit port in query, check conversational context for spatial follow-up
        is_context_followup = False
        if resolved_port is None and conversation_id in _context_memory:
            past = _context_memory[conversation_id]
            if (
                intent == "FOLLOW_UP" or
                "wahan" in msg_lower or
                "udhar" in msg_lower or
                "there" in msg_lower or
                "same place" in msg_lower
            ):
                resolved_lat = past.get("last_recommended_lat", current_lat)
                resolved_lon = past.get("last_recommended_lon", current_lon)
                resolved_port = past.get("port_name") or "Saved Location"
                is_context_followup = True

        # 3. Detect if coordinates are inland (far from sea)
        is_inland = False
        inland_distance_km = 0.0
        nearest_coast_name = "Goa Pelagic Coast (Arabian Sea)"
        nearest_coast_lat = 15.2993
        nearest_coast_lon = 73.8000

        min_dist = float("inf")
        for port_name, (p_lat, p_lon) in KNOWN_COASTAL_PORTS.items():
            # Haversine distance in km
            import math
            dlat = math.radians(p_lat - current_lat)
            dlon = math.radians(p_lon - current_lon)
            a = math.sin(dlat / 2)**2 + math.cos(math.radians(current_lat)) * math.cos(math.radians(p_lat)) * math.sin(dlon / 2)**2
            d = 6371.0 * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))
            if d < min_dist:
                min_dist = d
                nearest_coast_name = f"{port_name.capitalize()} Coast"
                nearest_coast_lat = p_lat
                nearest_coast_lon = p_lon

        # Check if coordinates are in maritime territory (e.g. Lakshadweep Sea: 8-14 N, 71-74.5 E or Andaman: 6-14 N, 92-94.5 E)
        is_island_or_marine = (
            (8.0 <= current_lat <= 14.0 and 71.0 <= current_lon <= 74.5) or
            (6.0 <= current_lat <= 14.0 and 92.0 <= current_lon <= 94.5)
        )

        # Check if inland within Indian landmass (e.g. Lucknow, Delhi, central India)
        is_peninsular_inland = (18.0 <= current_lat <= 32.0 and 74.5 <= current_lon <= 88.0)

        if not is_context_followup and not is_island_or_marine and resolved_port is None and (min_dist > 50.0 and is_peninsular_inland):
            is_inland = True
            inland_distance_km = round(min_dist, 1)


        # Save resolved coordinates to session context
        self.save_context(conversation_id, resolved_lat, resolved_lon, resolved_port)

        return {
            "latitude": resolved_lat,
            "longitude": resolved_lon,
            "port_name": resolved_port,
            "target_latitude": target_lat,
            "target_longitude": target_lon,
            "target_port": target_port,
            "is_inland": is_inland,
            "inland_distance_km": inland_distance_km,
            "nearest_coast_name": nearest_coast_name,
            "nearest_coast_lat": nearest_coast_lat,
            "nearest_coast_lon": nearest_coast_lon,
            "original_query": message,
            "intent": intent
        }

    def save_context(
        self,
        conversation_id: str,
        lat: float,
        lon: float,
        port_name: Optional[str] = None
    ):
        global _context_memory
        _context_memory[conversation_id] = {
            "last_recommended_lat": lat,
            "last_recommended_lon": lon,
            "port_name": port_name
        }

    async def resolve_context_async(
        self,
        conversation_id: str,
        message: str,
        intent: str,
        current_lat: float,
        current_lon: float
    ) -> Dict[str, Any]:
        """
        Asynchronously resolves context with real-time global geocoding support.
        Identifies any named location in query, tracks coordinates, handles nearby sea requests,
        and maintains multi-turn conversation memory.
        """
        from app.services.location_intelligence_service import LocationIntelligenceService
        loc_service = LocationIntelligenceService()

        # 1. Check if user asked about nearby sea/coast distance
        if loc_service.is_nearby_query(message):
            nearest_port_name = "Digha"
            nearest_dist = float("inf")
            n_lat, n_lon = 21.6266, 87.5074
            for p_name, (p_lat, p_lon) in KNOWN_COASTAL_PORTS.items():
                d = loc_service.haversine_km(current_lat, current_lon, p_lat, p_lon)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_port_name = p_name
                    n_lat, n_lon = p_lat, p_lon

            rev = await loc_service.reverse_geocode(current_lat, current_lon)
            user_loc_name = rev.get("name") if rev else f"Sector ({current_lat:.2f}°N, {current_lon:.2f}°E)"
            nearest_coast_name = f"{nearest_port_name.capitalize()} Coast"

            res = {
                "latitude": n_lat,
                "longitude": n_lon,
                "user_latitude": current_lat,
                "user_longitude": current_lon,
                "user_location_name": user_loc_name,
                "port_name": nearest_coast_name,
                "display_name": nearest_coast_name,
                "target_latitude": None,
                "target_longitude": None,
                "target_port": None,
                "is_inland": (nearest_dist > 50.0),
                "inland_distance_km": round(nearest_dist, 1),
                "nearest_coast_name": nearest_coast_name,
                "nearest_coast_lat": n_lat,
                "nearest_coast_lon": n_lon,
                "is_nearby_query": True,
                "original_query": message,
                "intent": intent
            }
            self.save_context(conversation_id, n_lat, n_lon, res["port_name"])
            return res

        # 2. Extract potential location name using LocationIntelligenceService
        extracted_name = loc_service.extract_candidate_location_name(message)
        if extracted_name:
            geo = await loc_service.geocode_location(extracted_name)
            if geo:
                res = self.resolve_context(conversation_id, message, intent, geo["latitude"], geo["longitude"])
                res["latitude"] = geo["latitude"]
                res["longitude"] = geo["longitude"]
                res["port_name"] = geo["name"]
                res["display_name"] = geo.get("display_name", geo["name"])
                res["is_coastal"] = geo.get("is_coastal", False)
                res["is_inland"] = not geo.get("is_coastal", False)
                res["is_nearby_query"] = False

                min_d = float("inf")
                c_name = "Coast"
                c_lat, c_lon = 15.2993, 73.8000
                for p_name, (p_lat, p_lon) in KNOWN_COASTAL_PORTS.items():
                    d = loc_service.haversine_km(geo["latitude"], geo["longitude"], p_lat, p_lon)
                    if d < min_d:
                        min_d = d
                        c_name = f"{p_name.capitalize()} Coast"
                        c_lat, c_lon = p_lat, p_lon
                res["nearest_coast_name"] = c_name
                res["nearest_coast_lat"] = c_lat
                res["nearest_coast_lon"] = c_lon
                res["inland_distance_km"] = round(min_d, 1)

                self.save_context(conversation_id, geo["latitude"], geo["longitude"], geo["name"])
                return res

        # 3. Default or conversational follow-up
        res = self.resolve_context(conversation_id, message, intent, current_lat, current_lon)
        res["is_nearby_query"] = False
        if not res.get("port_name") or res.get("port_name") == "Saved Location":
            rev = await loc_service.reverse_geocode(current_lat, current_lon)
            if rev:
                res["port_name"] = rev.get("name")
                res["display_name"] = rev.get("display_name")
                res["is_coastal"] = rev.get("is_coastal", not res.get("is_inland", False))
                res["is_inland"] = not res["is_coastal"]
                if "distance_to_sea_km" in rev:
                    res["inland_distance_km"] = rev["distance_to_sea_km"]
                self.save_context(conversation_id, current_lat, current_lon, res["port_name"])
        return res

    def clear_context(self, conversation_id: str):
        global _context_memory
        if conversation_id in _context_memory:
            del _context_memory[conversation_id]
