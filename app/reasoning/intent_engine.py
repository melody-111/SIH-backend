import re
from typing import Dict, Any, Tuple

class IntentEngine:
    """
    Classifies user natural-language queries into structured intents and operating modes
    (CHAT, AGENT, HYBRID) to plan minimal, relevant specialist agent executions.
    """
    def determine_intent(self, message: str) -> str:
        msg = (message or "").lower().strip()
        msg_norm = re.sub(r"[^\w\s]", " ", msg)
        msg_norm = " ".join(msg_norm.split())

        # 0. Conversational Greetings, Introductions, Feelings & Small Talk
        feelings_keywords = [
            "achha nahi lag raha", "achha nhi lag raha", "acha nahi lag raha", "theek nahi lag raha",
            "not feeling well", "feel sad", "feeling sad", "feeling down", "feeling low",
            "mood off", "upset hoon", "pareshan hoon", "thak gaya", "tired", "unwell",
            "lonely", "bura lag raha", "kuch theek nahi"
        ]
        if any(f in msg or f in msg_norm for f in feelings_keywords):
            return "CONVERSATIONAL"

        conv_words = [
            "hello", "hi", "hey", "namaste", "namaskar", "kaise ho", "kya haal", "kya haal hai",
            "how are you", "how r u", "how do you do", "good morning", "good evening", "good afternoon",
            "shubh prabhat", "pranam", "help", "madad", "who are you", "tum kaun ho", "aap kaun ho",
            "kya kar sakte ho", "what can you do", "what are your features", "thanks", "thank you",
            "shukriya", "dhanyawad", "bye", "alvida", "see you", "kya kar rahe ho", "kya kar rhe ho",
            "what are you doing", "aur batao", "sab theek", "sab badiya", "sab badhiya", "kya chal raha",
            "kya chal raha hai", "kya chal rha", "kya hal chal", "kya haal chaal", "kya scene", "kya scene hai",
            "bol bhai", "bhai suno", "kuch sunao", "kuch batao", "tum batao", "apna batao", "mast",
            "badhiya", "theek hai", "thik hai", "acha", "achha", "kya haal chaal", "kaisa chal raha",
            "joke", "kahani", "story", "chutkula", "kisne banaya", "who created you", "who made you"
        ]
        has_marine_data_request = any(w in msg_norm for w in [
            "weather", "mausam", "wave", "waves", "lehrein", "lehar", "wind", "hawa",
            "cyclone", "toofan", "fish", "fishing", "machhli", "machli", "pfz",
            "harbor", "port", "currents", "ocean currents", "chlorophyll", "sst",
            "salinity", "tide", "tides", "map", "gis", "route", "alert", "warning",
            "atmosphere", "disaster", "earthquake", "tsunami", "path", "timing",
            "boat", "naav", "ship", "sea", "ocean", "samundar", "samunder", "water",
            "temperature", "taapmaan", "barish", "baarish", "rain", "shoal", "corridor"
        ])
        is_greeting = any(
            msg_norm == w
            or msg_norm.startswith(w + " ")
            or msg_norm.endswith(" " + w)
            or f" {w} " in f" {msg_norm} "
            for w in conv_words
        )
        from app.services.location_intelligence_service import LocationIntelligenceService
        loc_service = LocationIntelligenceService()
        has_location_or_nearby = bool(loc_service.extract_candidate_location_name(message)) or loc_service.is_nearby_query(message)

        if is_greeting and not has_marine_data_request and not has_location_or_nearby:
            return "CONVERSATIONAL"

        # 1. Ambiguous Queries Requiring Clarification (no location, no subject)
        ambiguous_questions = [
            "is it good to go", "is it good", "should i go", "can i go", "kaisa hai", "kaisa h",
            "chalega", "kya main ja sakta hoon", "kya main ja sakta hu", "should we sail",
            "can we sail", "is it safe", "safe hai kya", "surakshit hai kya"
        ]
        has_specific_location = any(loc in msg_norm for loc in [
            "goa", "mumbai", "kochi", "cochin", "chennai", "vizag", "visakhapatnam",
            "thiruvananthapuram", "trivandrum", "vizhinjam", "porbandar", "digha",
            "kanyakumari", "harbor", "coast", "sea", "samundar", "there", "wahan",
            "kal", "tomorrow", "today", "aaj"
        ])
        has_hazard_clue = any(term in msg for term in [
            "cyclone", "hazard", "alert", "warning", "warns", "toofan", "storm", "khatra", "signal",
            "dangerous", "danger", "disaster", "tsunami", "bhookamp"
        ])
        if any(msg_norm == q or msg_norm.startswith(q + " ") for q in ambiguous_questions) and not has_specific_location and not has_hazard_clue:
            return "CLARIFICATION_NEEDED"

        # 2. General Explanatory & Educational Questions
        # Must take precedence before keyword hits like 'ocean' or 'waves' trigger sensor agent pipelines!
        explanatory_starters = [
            "explain", "what does", "what is", "how does", "tell us about", "kya hota hai",
            "samjhao", "why is", "how do", "meaning of", "define", "batayein ki"
        ]
        if any(msg.startswith(s) or (" " + s + " ") in msg for s in explanatory_starters):
            edu_topics = [
                "chlorophyll", "ocean productivity", "pfz concept", "upwelling", "tide cycle",
                "tides", "rule", "regulation", "ban", "protocol", "law", "fairway", "meaning",
                "ocean currents", "currents", "sea salty", "salinity", "marine ecosystem",
                "coriolis", "tsunami formation", "el nino", "la nina", "phytoplankton"
            ]
            if any(t in msg for t in edu_topics):
                return "GENERAL_EXPLANATORY"

        # 3. Hazards, Cyclones & Extreme Sea Alerts (High safety priority)
        if any(term in msg for term in [
            "cyclone", "hazard", "alert", "warning", "toofan", "storm", "khatra", "dangerous sea",
            "disaster", "earthquake", "tsunami", "bhookamp", "bhukamp", "bhookamp aane wala",
            "disaster aane wala", "danger", "upcoming disaster", "upcomming disaster", "deaster",
            "disaster aane", "toofan aane", "cyclone aane", "koi disaster", "koi toofan"
        ]):
            return "HAZARD"

        # 4. Follow-Up Inquiries (Spatial & Temporal relative queries)
        clean_msg = msg.replace("is there", "").replace("are there", "")
        if (
            any(term in clean_msg for term in ["wahan", "udhar", "vahan", "that area", "us jagah", "there"]) or
            msg.startswith("what about") or
            msg.startswith("and tomorrow") or
            msg.startswith("aur kal") or
            "compare" in msg or
            "kal kaisa" in msg or
            "how about" in msg
        ):
            return "FOLLOW_UP"

        # 5. Fishing Locations & PFZ
        if any(term in msg for term in ["fish", "machhli", "machli", "shikaar", "suggest fishing", "fishing area", "fishing jagah", "pfz", "best spot", "tuna", "catch", "fishing zone", "shoal", "machhliya"]):
            return "FISHING_LOCATION"

        # 6. Dedicated Ocean Variable Queries (e.g. SST, Currents)
        if any(term in msg for term in ["sea-surface temperature", "sea surface temperature", "sst", "ocean temperature", "current speed"]):
            return "OCEAN_INFO"

        # 7. Route & Navigation Planning (Path, Timing, Departure, Return, Avoidance)
        if any(term in msg for term in [
            "route", "rasta", "path", "navigate", "safe path", "distance to", "passage",
            "kis path", "kis raste", "kis time", "kab jaana", "kab jana", "kab nikalna",
            "kab wapas", "wapas aane", "wapas aana", "wapas kab", "lautne ka", "avoid",
            "timing", "kaise jaana", "kaise jana", "ocean path", "path in ocean", "3d map",
            "dual map", "kab tak rehna", "kab tak samundar"
        ]):
            return "ROUTE"

        # 8. Marine Weather, Waves & Atmosphere
        if any(term in msg for term in ["atmosphere", "wave", "lehrein", "wind", "hawa", "weather", "mausam", "rain", "baarish", "condition", "sea state", "temperature", "climate"]):
            return "MARINE_CONDITION"

        # 9. GIS & Map Requests
        if any(term in msg for term in ["map", "gis", "satellite image", "show layer"]):
            return "GIS_MAP"

        # 10. Nearby Sea / Coast Queries
        from app.services.location_intelligence_service import LocationIntelligenceService
        loc_service = LocationIntelligenceService()
        if loc_service.is_nearby_query(msg):
            return "MARINE_CONDITION"

        # 11. Explicit Location Inquiries (e.g., 'phir usa ke bare me', 'london ke bare me batao', 'tokyo ka batao')
        cand_loc = loc_service.extract_candidate_location_name(message)
        if cand_loc:
            return "MARINE_CONDITION"

        # 12. General Marine if explicit coastal or maritime terms exist
        has_marine_context = any(term in msg_norm for term in [
            "ocean", "marine", "sea", "samundar", "samunder", "coast", "coastal", "tatt",
            "port", "harbor", "boat", "ship", "vessel", "sail", "sailing", "shore", "beach",
            "goa", "mumbai", "kochi", "chennai", "vizag", "visakhapatnam", "thiruvananthapuram",
            "trivandrum", "paradip", "digha", "porbandar", "mangalore"
        ])
        if has_marine_context:
            return "GENERAL_MARINE"

        # Default: Pure natural conversation (Greetings, chit-chat, curiosity, life, empathy)
        return "CONVERSATIONAL"

    def determine_mode(self, message: str, intent: str) -> str:
        """
        Determines the orchestration mode:
        - CHAT: Greetings, small talk, feelings, general explanations, capabilities, clarification.
        - AGENT: Explicit marine observation/forecast/hazard queries.
        - HYBRID: Conversational query requesting live marine data.
        """
        if intent in ["CONVERSATIONAL", "GENERAL_EXPLANATORY", "CLARIFICATION_NEEDED"]:
            return "CHAT"

        msg = (message or "").lower()
        has_chat_opener = any(w in msg for w in ["hello", "hi", "namaste", "good morning", "please", "can you tell", "batao na", "kripya", "i am planning", "thinking of", "bhai"])
        has_marine_data = any(w in msg for w in ["weather", "wave", "lehrein", "wind", "cyclone", "fish", "goa", "mumbai", "kochi", "sea", "samundar", "harbor", "vizag", "thiruvananthapuram"])

        if has_chat_opener and has_marine_data:
            return "HYBRID"

        if intent in ["HAZARD", "FISHING_LOCATION", "OCEAN_INFO", "ROUTE", "MARINE_CONDITION", "GIS_MAP", "FOLLOW_UP", "GENERAL_MARINE"]:
            return "AGENT"

        return "CHAT"

    async def determine_intent_async(self, message: str, correlation_id: str = "") -> str:
        """
        Dynamically classifies intent using Fast LLM when configured,
        otherwise safely falls back to deterministic rule-based parsing.
        """
        from app.services.llm_service import MultiModelLLMService
        llm = MultiModelLLMService()
        prov_name, prov = llm.resolve_provider()
        if prov_name != "deterministic" and prov.is_available:
            try:
                res = await llm.classify_intent(message, correlation_id=correlation_id)
                valid_intents = {
                    "CONVERSATIONAL", "CLARIFICATION_NEEDED", "GENERAL_EXPLANATORY",
                    "HAZARD", "FOLLOW_UP", "FISHING_LOCATION", "OCEAN_INFO",
                    "MARINE_CONDITION", "ROUTE", "GIS_MAP", "GENERAL_MARINE"
                }
                if res.intent in valid_intents:
                    return res.intent
            except Exception:
                pass
        return self.determine_intent(message)

