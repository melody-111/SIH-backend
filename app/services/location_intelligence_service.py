import re
import math
import logging
import httpx
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("neerai.location")

KNOWN_COASTAL_PORTS: Dict[str, Tuple[float, float, str]] = {
    "goa": (15.2993, 73.8000, "Goa Coast (Arabian Sea)"),
    "panaji": (15.4989, 73.8278, "Panaji Port, Goa"),
    "mormugao": (15.4167, 73.8000, "Mormugao Harbor, Goa"),
    "mumbai": (18.9220, 72.8347, "Mumbai Harbor, Maharashtra"),
    "bombay": (18.9220, 72.8347, "Mumbai Coast, Maharashtra"),
    "kochi": (9.9312, 76.2673, "Cochin Harbor, Kerala"),
    "cochin": (9.9312, 76.2673, "Cochin Harbor, Kerala"),
    "chennai": (13.0827, 80.2707, "Chennai Port, Tamil Nadu"),
    "madras": (13.0827, 80.2707, "Chennai Coast, Tamil Nadu"),
    "mangalore": (12.8700, 74.8800, "Mangalore Old Port, Karnataka"),
    "ratnagiri": (16.9902, 73.3120, "Ratnagiri Coast, Maharashtra"),
    "visakhapatnam": (17.6868, 83.2185, "Visakhapatnam Harbor, Andhra Pradesh"),
    "vizag": (17.6868, 83.2185, "Visakhapatnam Port, Andhra Pradesh"),
    "veraval": (20.9000, 70.3700, "Veraval Fishing Port, Gujarat"),
    "paradip": (20.3160, 86.6110, "Paradip Port, Odisha"),
    "karwar": (14.8100, 74.1300, "Karwar Port, Karnataka"),
    "kollam": (8.8932, 76.6141, "Kollam Port, Kerala"),
    "tuticorin": (8.7642, 78.1348, "Tuticorin Port, Tamil Nadu"),
    "thoothukudi": (8.7642, 78.1348, "Thoothukudi Port, Tamil Nadu"),
    "puri": (19.8135, 85.8312, "Puri Coast, Odisha"),
    "digha": (21.6266, 87.5074, "Digha Coast, West Bengal"),
    "porbandar": (21.6417, 69.6293, "Porbandar Harbor, Gujarat"),
    "kandla": (23.0000, 70.2167, "Kandla Port, Gujarat"),
    "kanyakumari": (8.0883, 77.5385, "Kanyakumari (Cape Comorin), Tamil Nadu"),
    "dwarka": (22.2442, 68.9685, "Dwarka Coastal Waters, Gujarat"),
    "alibaug": (18.6414, 72.8722, "Alibaug Coast, Maharashtra"),
    "puducherry": (11.9416, 79.8083, "Puducherry Port"),
    "pondicherry": (11.9416, 79.8083, "Puducherry Port"),
    "kolkata": (22.5726, 88.3639, "Kolkata Riverine Port, West Bengal"),
    "calcutta": (22.5726, 88.3639, "Kolkata Port, West Bengal"),
    "bhavnagar": (21.7645, 72.1519, "Bhavnagar Port, Gujarat"),
    "okha": (22.4667, 69.0667, "Okha Port, Gujarat"),
    "machilipatnam": (16.1875, 81.1389, "Machilipatnam, Andhra Pradesh"),
    "lakshadweep": (10.5667, 72.6417, "Lakshadweep Islands"),
    "kavaratti": (10.5667, 72.6417, "Kavaratti, Lakshadweep"),
    "port blair": (11.6234, 92.7265, "Port Blair, Andaman & Nicobar"),
    "andaman": (11.6234, 92.7265, "Andaman Islands")
}

KNOWN_GLOBAL_CITIES: Dict[str, Tuple[float, float, str, bool]] = {
    # Countries and Global Regions
    "usa": (38.8951, -77.0364, "United States (USA)", True),
    "us": (38.8951, -77.0364, "United States (USA)", True),
    "america": (38.8951, -77.0364, "United States (USA)", True),
    "united states": (38.8951, -77.0364, "United States (USA)", True),
    "united states of america": (38.8951, -77.0364, "United States (USA)", True),
    "uk": (51.5074, -0.1278, "United Kingdom (UK)", True),
    "united kingdom": (51.5074, -0.1278, "United Kingdom (UK)", True),
    "britain": (51.5074, -0.1278, "United Kingdom (UK)", True),
    "great britain": (51.5074, -0.1278, "United Kingdom (UK)", True),
    "england": (51.5074, -0.1278, "England, United Kingdom", True),
    "japan": (35.6762, 139.6503, "Tokyo, Japan", True),
    "australia": (-33.8688, 151.2093, "Sydney, Australia", True),
    "canada": (45.4215, -75.6972, "Ottawa, Canada", True),
    "russia": (55.7558, 37.6173, "Moscow, Russia", False),
    "china": (39.9042, 116.4074, "Beijing, China", True),
    "france": (48.8566, 2.3522, "Paris, France", False),
    "germany": (52.5200, 13.4050, "Berlin, Germany", False),
    "uae": (25.2048, 55.2708, "Dubai, United Arab Emirates", True),
    "united arab emirates": (25.2048, 55.2708, "Dubai, United Arab Emirates", True),
    "singapore": (1.3521, 103.8198, "Singapore", True),
    "sri lanka": (6.9271, 79.8612, "Colombo, Sri Lanka", True),
    "maldives": (4.1755, 73.5093, "Malé, Maldives", True),
    "bangladesh": (23.8103, 90.4125, "Dhaka, Bangladesh", True),
    "indonesia": (-6.2088, 106.8456, "Jakarta, Indonesia", True),
    "thailand": (13.7563, 100.5018, "Bangkok, Thailand", True),
    "malaysia": (3.1390, 101.6869, "Kuala Lumpur, Malaysia", True),
    "nepal": (27.7172, 85.3240, "Kathmandu, Nepal", False),
    "pakistan": (24.8607, 67.0011, "Karachi, Pakistan", True),

    # Major Global Metro Cities
    "london": (51.5074, -0.1278, "London, United Kingdom", True),
    "tokyo": (35.6762, 139.6503, "Tokyo, Japan", True),
    "paris": (48.8566, 2.3522, "Paris, France", False),
    "berlin": (52.5200, 13.4050, "Berlin, Germany", False),
    "new york": (40.7128, -74.0060, "New York, USA", True),
    "miami": (25.7617, -80.1918, "Miami, Florida, USA", True),
    "los angeles": (34.0522, -118.2437, "Los Angeles, California, USA", True),
    "san francisco": (37.7749, -122.4194, "San Francisco, California, USA", True),
    "chicago": (41.8781, -87.6298, "Chicago, Illinois, USA", False),
    "boston": (42.3601, -71.0589, "Boston, Massachusetts, USA", True),
    "seattle": (47.6062, -122.3321, "Seattle, Washington, USA", True),
    "hawaii": (21.3069, -157.8583, "Honolulu, Hawaii, USA", True),
    "sydney": (-33.8688, 151.2093, "Sydney, Australia", True),
    "melbourne": (-37.8136, 144.9631, "Melbourne, Australia", True),
    "dubai": (25.2048, 55.2708, "Dubai, United Arab Emirates", True),
    "toronto": (43.6532, -79.3832, "Toronto, Canada", True),
    "vancouver": (49.2827, -123.1207, "Vancouver, Canada", True),
    "moscow": (55.7558, 37.6173, "Moscow, Russia", False),
    "beijing": (39.9042, 116.4074, "Beijing, China", True),
    "shanghai": (31.2304, 121.4737, "Shanghai, China", True),
    "colombo": (6.9271, 79.8612, "Colombo, Sri Lanka", True),
    "male": (4.1755, 73.5093, "Malé, Maldives", True),
    "dhaka": (23.8103, 90.4125, "Dhaka, Bangladesh", True),
    "karachi": (24.8607, 67.0011, "Karachi, Pakistan", True),

    # Major Indian Cities (Inland and Coastal)
    "lucknow": (26.8467, 80.9462, "Lucknow, Uttar Pradesh, India", False),
    "delhi": (28.6139, 77.2090, "New Delhi, NCR, India", False),
    "new delhi": (28.6139, 77.2090, "New Delhi, NCR, India", False),
    "bengaluru": (12.9716, 77.5946, "Bengaluru, Karnataka, India", False),
    "bangalore": (12.9716, 77.5946, "Bengaluru, Karnataka, India", False),
    "hyderabad": (17.3850, 78.4867, "Hyderabad, Telangana, India", False),
    "jaipur": (26.9124, 75.7873, "Jaipur, Rajasthan, India", False),
    "bhopal": (23.2599, 77.4126, "Bhopal, Madhya Pradesh, India", False),
    "patna": (25.5941, 85.1376, "Patna, Bihar, India", False),
    "varanasi": (25.3176, 82.9739, "Varanasi, Uttar Pradesh, India", False),
    "kanpur": (26.4499, 80.3319, "Kanpur, Uttar Pradesh, India", False),
    "pune": (18.5204, 73.8567, "Pune, Maharashtra, India", False),
    "ahmedabad": (23.0225, 72.5714, "Ahmedabad, Gujarat, India", False),
    "surat": (21.1702, 72.8311, "Surat, Gujarat, India", True),
    "chandigarh": (30.7333, 76.7794, "Chandigarh, India", False),
    "srinagar": (34.0837, 74.7973, "Srinagar, Jammu & Kashmir, India", False),
    "guwahati": (26.1445, 91.7362, "Guwahati, Assam, India", False),
    "agra": (27.1767, 78.0081, "Agra, Uttar Pradesh, India", False),
    "amritsar": (31.6340, 74.8723, "Amritsar, Punjab, India", False),
    "dehradun": (30.3165, 78.0322, "Dehradun, Uttarakhand, India", False),
    "shimla": (31.1048, 77.1734, "Shimla, Himachal Pradesh, India", False),
    "ayodhya": (26.7922, 82.1998, "Ayodhya, Uttar Pradesh, India", False),
    "prayagraj": (25.4358, 81.8463, "Prayagraj, Uttar Pradesh, India", False),
    "allahabad": (25.4358, 81.8463, "Prayagraj, Uttar Pradesh, India", False),
    "indore": (22.7196, 75.8577, "Indore, Madhya Pradesh, India", False),
    "ranchi": (23.3441, 85.3096, "Ranchi, Jharkhand, India", False),
    "nagpur": (21.1458, 79.0882, "Nagpur, Maharashtra, India", False),
    "bhubaneswar": (20.2961, 85.8245, "Bhubaneswar, Odisha, India", False),
    "cuttack": (20.4625, 85.8828, "Cuttack, Odisha, India", False)
}

# In-memory geocoding caches to keep response times under 5ms
_GEOCODE_CACHE: Dict[str, Dict[str, Any]] = {}
_REVERSE_GEOCODE_CACHE: Dict[str, Dict[str, Any]] = {}


class LocationIntelligenceService:
    """
    Universal location intelligence engine for NeerAI.
    Combines real-time geocoding (Open-Meteo & Nominatim OSM), maritime shoreline detection,
    and smart place-name extraction from natural queries in English, Hindi, and Hinglish.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocationIntelligenceService, cls).__new__(cls)
            cls._instance._client = httpx.AsyncClient(timeout=4.0)
        return cls._instance

    @staticmethod
    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

    @staticmethod
    def is_nearby_query(query: str) -> bool:
        """
        Determines if the query is asking about the closest sea/coast from the user's location.
        """
        if not query:
            return False
        q = query.lower()
        terms = [
            "nearby", "near me", "closest sea", "nearest sea", "closest coast", "nearest coast",
            "paas me", "paas ka", "paas ki", "paas ke", "paas wala", "paas wali", "kareebi",
            "karibi", "najdeeki", "nazdeeki", "kitni door", "kitna door", "kitni dur", "kitna dur",
            "paas ka samundar", "nearby samundar", "nearby ocean", "closest ocean", "paas samundar",
            "kareeb samundar", "najdeek samundar"
        ]
        return any(t in q for t in terms)

    def extract_candidate_location_name(self, query: str) -> Optional[str]:
        """
        Extracts potential location or place names from query text in Hindi, Hinglish, and English.
        Handles patterns like:
        - 'london ke bare me batao' / 'phir usa ke bare me'
        - 'usa ka batao' / 'tokyo me cyclone hai kya'
        - 'weather in Delhi' / 'mausam in Mumbai'
        - 'Delhi ka mausam' / 'Kanyakumari mein hawa'
        - 'how is the sea near Dwarka' / 'tell me about London'
        """
        if not query:
            return None

        text = query.strip()
        text_lower = text.lower()

        # If user explicitly asked for "nearby" sea, don't mistakenly extract another place unless specified
        if self.is_nearby_query(text_lower):
            # If there's an explicit anchor city after "near" (e.g., "near goa"), handle it
            pass

        # 1. Check known coastal registry and global cities directly first (longest match first)
        all_known = sorted(list(KNOWN_GLOBAL_CITIES.keys()) + list(KNOWN_COASTAL_PORTS.keys()), key=len, reverse=True)
        for name in all_known:
            if re.search(r"\b" + re.escape(name) + r"\b", text_lower):
                return name

        # 2. Regex patterns for locations in English & Hindi/Hinglish
        patterns = [
            # 'phir usa ke bare me batao' / 'tokyo ke baarem' / 'boston ke bare me'
            r"(?:phir|aur|ab|and|then)?\s*([A-Za-z\u0900-\u097F]{2,25})\s+(?:ke\s+(?:baare|bare|baarem|barem|bareme)\s*(?:me|mein)?)\b",
            # 'usa ka batao' / 'london ka' / 'delhi me'
            r"(?:phir|aur|ab|and|then)?\s*([A-Za-z\u0900-\u097F]{2,25})\s+(?:ka|ki|ke|mein|me|pe|par|se)\s+(?:batao|bataiye|bata|kaisa|kaisi|kaise|check)\b",
            # 'delhi ka mausam' / 'london weather'
            r"([A-Za-z\u0900-\u097F]{2,25})\s+(?:ka|ki|ke|mein|me|pe|par|se|ke paas)\s+(?:koi\s+)?(?:mausam|weather|hawa|halat|haalat|lehrein|baarish|barish|temperature|cyclone|toofan|storm|disaster|bhookamp|earthquake|alert|warning|khatra|fishing|zone)",
            # 'weather in London' / 'cyclone in Miami'
            r"(?:cyclone|storm|disaster|earthquake|bhookamp|toofan|weather|mausam|alert|warning|fishing|zone)\s+(?:in|near|around|for|of|at|me|mein)\s+([A-Za-z\u0900-\u097F]{2,25})",
            # 'tell me about London' / 'what about USA'
            r"(?:tell me about|what about|how about|check|about)\s+([A-Za-z\u0900-\u097F]{2,25})",
            # 'in/at Paris'
            r"\b(?:in|at|around|near|for|of|to)\b\s+([A-Za-z\u0900-\u097F\s]{2,25}?)(?:\?|\.|,|$|\s+weather|\s+mausam|\s+today|\s+tomorrow|\s+aaj|\s+kal|\s+hawa|\s+lehrein|\s+cyclone|\s+toofan)",
            # 'weather of Paris'
            r"(?:weather|forecast|mausam)\s+(?:in|of|for|at)\s+([A-Za-z\u0900-\u097F]{2,25})",
            # 'Paris weather'
            r"([A-Za-z\u0900-\u097F]{2,25})\s+(?:weather|cyclone|forecast|mausam|disaster)",
            # 'kaisa hai London'
            r"(?:kaisa hai|kaisi hai)\s+([A-Za-z\u0900-\u097F]{2,25})"
        ]

        stopwords = {
            "the", "a", "an", "this", "that", "there", "here", "wahan", "vahan", "yahan", "yahin", "idhar", "kahan", "jahan", "today", "tomorrow",
            "aaj", "kal", "samundar", "sea", "ocean", "weather", "mausam", "wave", "waves", "wind",
            "hawa", "kaisa", "kaisi", "kaise", "what", "how", "tell", "show", "current", "conditions",
            "condition", "khatra", "safe", "help", "information", "report", "cyclone", "toofan", "disaster",
            "storm", "are", "is", "am", "be", "were", "was", "can", "could", "would", "should", "will",
            "who", "where", "why", "which", "state", "kya", "check", "about", "me", "mein", "par", "pe",
            "phir", "aur", "and", "then", "batao", "bataiye", "bata", "ka", "ki", "ke", "ko", "se", "door",
            "paas", "nearby", "kareebi", "karibi", "najdeeki", "upcomming", "upcoming", "any", "bhi"
        }

        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip().lower()
                candidate = re.sub(r"[^\w\s]", "", candidate).strip()
                words = candidate.split()
                if words and words[0] not in stopwords:
                    cleaned_candidate = " ".join([w for w in words if w not in stopwords])
                    if cleaned_candidate and len(cleaned_candidate) >= 2 and cleaned_candidate not in stopwords:
                        return cleaned_candidate

        return None

    async def geocode_location(self, place_name: str) -> Optional[Dict[str, Any]]:
        """
        Geocodes any city, port, island, or region globally.
        Checks known dictionary, in-memory cache, Open-Meteo Geocoding, and Nominatim OSM.
        """
        if not place_name:
            return None

        norm_name = place_name.lower().strip()

        # 1. Global Cities dictionary hit
        if norm_name in KNOWN_GLOBAL_CITIES:
            lat, lon, desc, is_coastal = KNOWN_GLOBAL_CITIES[norm_name]
            _, dist_to_sea = self.check_if_coastal(lat, lon)
            return {
                "name": norm_name.capitalize(),
                "display_name": desc,
                "latitude": lat,
                "longitude": lon,
                "is_coastal": is_coastal,
                "distance_to_sea_km": dist_to_sea,
                "country": desc.split(",")[-1].strip(),
                "source": "NeerAI Global City Registry"
            }

        # 2. Coastal ports dictionary hit
        if norm_name in KNOWN_COASTAL_PORTS:
            lat, lon, desc = KNOWN_COASTAL_PORTS[norm_name]
            return {
                "name": norm_name.capitalize(),
                "display_name": desc,
                "latitude": lat,
                "longitude": lon,
                "is_coastal": True,
                "country": "India",
                "source": "NeerAI Coastal Port Registry"
            }


        # 2. Cache hit
        if norm_name in _GEOCODE_CACHE:
            return _GEOCODE_CACHE[norm_name]

        # 3. Open-Meteo Geocoding API
        try:
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={norm_name}&count=1&language=en&format=json"
            res = await self._client.get(url)
            if res.status_code == 200:
                data = res.json()
                results = data.get("results")
                if results and len(results) > 0:
                    top = results[0]
                    lat = float(top["latitude"])
                    lon = float(top["longitude"])
                    name = top.get("name", place_name.capitalize())
                    country = top.get("country", "")
                    admin1 = top.get("admin1", "")
                    disp = f"{name}, {admin1}, {country}".replace(", ,", ",").strip(", ")

                    is_coastal, dist_to_sea = self.check_if_coastal(lat, lon)
                    item = {
                        "name": name,
                        "display_name": disp,
                        "latitude": lat,
                        "longitude": lon,
                        "is_coastal": is_coastal,
                        "distance_to_sea_km": dist_to_sea,
                        "country": country,
                        "source": "Open-Meteo Global Geocoding"
                    }
                    _GEOCODE_CACHE[norm_name] = item
                    return item
        except Exception as e:
            logger.warning(f"Open-Meteo Geocoding error for '{place_name}': {e}")

        # 4. Fallback: OpenStreetMap Nominatim
        try:
            headers = {"User-Agent": "NeerAI-Marine-Intelligence/1.0"}
            url = f"https://nominatim.openstreetmap.org/search?q={norm_name}&format=json&limit=1"
            res = await self._client.get(url, headers=headers)
            if res.status_code == 200:
                data = res.json()
                if data and len(data) > 0:
                    top = data[0]
                    lat = float(top["lat"])
                    lon = float(top["lon"])
                    disp = top.get("display_name", place_name.capitalize())
                    is_coastal, dist_to_sea = self.check_if_coastal(lat, lon)
                    item = {
                        "name": place_name.capitalize(),
                        "display_name": disp,
                        "latitude": lat,
                        "longitude": lon,
                        "is_coastal": is_coastal,
                        "distance_to_sea_km": dist_to_sea,
                        "country": "Identified by OpenStreetMap",
                        "source": "OpenStreetMap Nominatim"
                    }
                    _GEOCODE_CACHE[norm_name] = item
                    return item
        except Exception as e:
            logger.warning(f"Nominatim Geocoding error for '{place_name}': {e}")

        return None

    async def reverse_geocode(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Reverse-geocodes any latitude and longitude coordinates globally on Earth.
        Converts raw GPS/network coordinates into a clear human-readable locality,
        city, district, state, and country name.
        Uses multi-tier caching, known port anchors, and free zero-key reverse APIs.
        """
        cache_key = f"{round(lat, 3)},{round(lon, 3)}"
        if cache_key in _REVERSE_GEOCODE_CACHE:
            return _REVERSE_GEOCODE_CACHE[cache_key]

        # 1. Check known coastal ports and global cities if within 15 km
        for name, (p_lat, p_lon, desc, is_c) in KNOWN_GLOBAL_CITIES.items():
            if self.haversine_km(lat, lon, p_lat, p_lon) <= 15.0:
                is_coastal, dist_to_sea = self.check_if_coastal(lat, lon)
                item = {
                    "name": name.capitalize(),
                    "display_name": desc,
                    "city": name.capitalize(),
                    "region": desc.split(",")[-2].strip() if len(desc.split(",")) > 2 else "",
                    "country": desc.split(",")[-1].strip(),
                    "latitude": lat,
                    "longitude": lon,
                    "is_coastal": is_coastal,
                    "distance_to_sea_km": dist_to_sea,
                    "source": "NeerAI Known City Anchor"
                }
                _REVERSE_GEOCODE_CACHE[cache_key] = item
                return item

        for name, (p_lat, p_lon, desc) in KNOWN_COASTAL_PORTS.items():
            if self.haversine_km(lat, lon, p_lat, p_lon) <= 15.0:
                item = {
                    "name": name.capitalize(),
                    "display_name": desc,
                    "city": name.capitalize(),
                    "region": "Coastal",
                    "country": "India",
                    "latitude": lat,
                    "longitude": lon,
                    "is_coastal": True,
                    "distance_to_sea_km": 0.0,
                    "source": "NeerAI Coastal Port Registry"
                }
                _REVERSE_GEOCODE_CACHE[cache_key] = item
                return item

        # 2. BigDataCloud Free Global Reverse Geocoding API (0 key, ultra-fast)
        try:
            url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=en"
            res = await self._client.get(url)
            if res.status_code == 200:
                data = res.json()
                city = data.get("city") or data.get("locality") or ""
                locality = data.get("locality") or ""
                region = data.get("principalSubdivision") or ""
                country = data.get("countryName") or ""

                parts = [p for p in [locality, city, region, country] if p]
                # Deduplicate consecutive terms (e.g., "Lucknow, Lucknow")
                unique_parts = []
                for p in parts:
                    if not unique_parts or unique_parts[-1].lower() != p.lower():
                        unique_parts.append(p)
                display_name = ", ".join(unique_parts) if unique_parts else f"Sector ({lat:.2f}°N, {lon:.2f}°E)"
                short_name = city or locality or region or f"Sector ({lat:.2f}°N, {lon:.2f}°E)"

                is_coastal, dist_to_sea = self.check_if_coastal(lat, lon)
                item = {
                    "name": short_name,
                    "display_name": display_name,
                    "city": city or locality,
                    "region": region,
                    "country": country,
                    "latitude": lat,
                    "longitude": lon,
                    "is_coastal": is_coastal,
                    "distance_to_sea_km": dist_to_sea,
                    "source": "Global Reverse Geocoding Network"
                }
                _REVERSE_GEOCODE_CACHE[cache_key] = item
                return item
        except Exception as e:
            logger.warning(f"BigDataCloud reverse geocoding error ({lat}, {lon}): {e}")

        # 3. Fallback: OpenStreetMap Nominatim Reverse
        try:
            headers = {"User-Agent": "NeerAI-Marine-Intelligence/1.0"}
            url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
            res = await self._client.get(url, headers=headers)
            if res.status_code == 200:
                data = res.json()
                disp = data.get("display_name", f"Sector ({lat:.2f}°N, {lon:.2f}°E)")
                addr = data.get("address", {})
                short = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county") or "Local Sector"
                is_coastal, dist_to_sea = self.check_if_coastal(lat, lon)
                item = {
                    "name": short,
                    "display_name": disp,
                    "city": short,
                    "region": addr.get("state", ""),
                    "country": addr.get("country", ""),
                    "latitude": lat,
                    "longitude": lon,
                    "is_coastal": is_coastal,
                    "distance_to_sea_km": dist_to_sea,
                    "source": "OpenStreetMap Nominatim Reverse"
                }
                _REVERSE_GEOCODE_CACHE[cache_key] = item
                return item
        except Exception as e:
            logger.warning(f"Nominatim reverse geocoding error ({lat}, {lon}): {e}")

        is_coastal, dist_to_sea = self.check_if_coastal(lat, lon)
        return {
            "name": f"Coordinates ({lat:.2f}°N, {lon:.2f}°E)",
            "display_name": f"Sector ({lat:.4f}°N, {lon:.4f}°E)",
            "latitude": lat,
            "longitude": lon,
            "is_coastal": is_coastal,
            "distance_to_sea_km": dist_to_sea,
            "source": "Coordinate Boundary Engine"
        }

    def check_if_coastal(self, lat: float, lon: float) -> Tuple[bool, float]:
        """
        Determines whether coordinates are along the coast or offshore,
        or inland, by measuring proximity to marine shorelines.
        """
        # Distance to closest known port/coastal shoreline anchor
        min_dist = float("inf")
        for p_lat, p_lon, _ in KNOWN_COASTAL_PORTS.values():
            d = self.haversine_km(lat, lon, p_lat, p_lon)
            if d < min_dist:
                min_dist = d

        # Check if coordinates are directly in maritime/ocean zones
        # Arabian Sea, Bay of Bengal, Indian Ocean
        is_in_ocean = (
            (0.0 <= lat <= 24.0 and 65.0 <= lon <= 77.0) or # Arabian Sea
            (0.0 <= lat <= 23.0 and 78.0 <= lon <= 95.0)    # Bay of Bengal
        ) and min_dist <= 60.0

        is_coastal = (min_dist <= 45.0) or is_in_ocean
        return is_coastal, round(min_dist, 1)
