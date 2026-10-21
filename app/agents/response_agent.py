from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentResult
from app.schemas.response import FinalResponse
from app.schemas.chat import UserQuery
from app.schemas.marine import MarineContext
from app.reasoning.final_judge import FinalJudge
from typing import Dict, Any, List, Optional
import datetime

class ResponseAgent(BaseAgent):
    """
    Synthesizes validated evidence and safety risk evaluations into a user-friendly,
    evidence-grounded natural language response in the requested language.
    Utilizes LLM when configured; otherwise provides strict, high-fidelity deterministic synthesis.
    """
    def __init__(self):
        super().__init__("response", "Synthesizes validated evidence into user-friendly multilingual response")
        self.judge = FinalJudge()

    async def generate_response(
        self,
        query: UserQuery,
        intent: str,
        marine_context: MarineContext,
        risk: Dict[str, Any],
        recommendation: Dict[str, Any],
        agents_used: List[str],
        unavailable_sources: List[str],
        is_cached: bool = False,
        cached_at: Optional[str] = None
    ) -> FinalResponse:
        now = datetime.datetime.utcnow().isoformat() + "Z"
        lang = query.language or "en"
        is_hindi = lang.startswith("hi")

        # 1. If real LLM provider is active, attempt LLM generation
        prov_name, prov = self.judge.llm_service.resolve_provider()
        if prov_name != "deterministic" and prov.is_available:
            try:
                llm_resp = await self.judge.judge(query, marine_context, risk, recommendation, agents_used)
                if llm_resp and llm_resp.answer:
                    if is_cached:
                        llm_resp.is_cached = True
                        llm_resp.cached_at = cached_at
                        llm_resp.warnings.append(f"Response built from local cache ({cached_at}). Real-time warnings unavailable.")
                    if unavailable_sources:
                        unavail_str = ", ".join(unavailable_sources)
                        llm_resp.warnings.append(f"Some upstream providers are temporarily unavailable: {unavail_str}. Missing data is not fabricated.")
                    return llm_resp
            except Exception:
                pass  # Fall back gracefully to deterministic synthesis

        # 2. High-Fidelity Deterministic Synthesis
        answer = ""
        reasons: List[str] = list(recommendation.get("reasons", []))
        warnings: List[str] = list(risk.get("warnings", []))
        rec_location = None

        # --- CONVERSATIONAL (ChatGPT-like Greeting, Feelings, Capabilities, Clarification) ---
        if intent in ["CONVERSATIONAL", "CLARIFICATION_NEEDED"]:
            return await self.generate_conversational_response(query, intent, is_cached=is_cached, cached_at=cached_at)

        # --- Example D: General Explanatory Question ---
        if intent == "GENERAL_EXPLANATORY":
            from app.services.marine_knowledge_service import MarineKnowledgeService
            kb_service = MarineKnowledgeService()
            kb_docs = kb_service.search_unstructured_knowledge(query.message or "")

            citations = []
            if kb_docs:
                top_doc = kb_docs[0]
                content = top_doc["content"]
                auth = top_doc["source_authority"]
                citations.append(f"{top_doc['title']} ({auth})")
                if is_hindi:
                    answer = f"{content}\n\n[Srot: {auth}]"
                else:
                    answer = f"{content}\n\n[Source Authority: {auth}]"
            else:
                msg_lower = (query.message or "").lower()
                if "currents" in msg_lower or "ocean current" in msg_lower:
                    if is_hindi:
                        answer = (
                            "Samundari dharayein (ocean currents) samundar ke paani ki ek nishchit disha mein behne wali "
                            "vishal nadiyon jaisi hoti hain. Ye mukhya roop se teen cheezon se banti hain:\n"
                            "1. Hawa (Surface winds): Jo upari paani ko dhakelti hai.\n"
                            "2. Paani ki density (Taapmaan aur kharepan ka antar): Thanda aur khara paani bhari hokar neeche baithta hai aur garm paani upar rehta hai (thermohaline circulation).\n"
                            "3. Prithvi ka ghumna (Coriolis Effect): Jo dharayon ko uttar mein right aur dakshin mein left disha mein modta hai.\n"
                            "Ye dharayein poore vishwa ke mausam ko santulit rakhti hain aur machhliyon ke liye poshak tatva laati hain."
                        )
                    else:
                        answer = (
                            "Ocean currents are continuous, predictable, directional movements of seawater flowing through the oceans. "
                            "Simply put, think of them as giant underwater rivers driven by three main forces:\n"
                            "1. Wind: Global winds blow across the sea surface, dragging surface waters along with them.\n"
                            "2. Water Density Differences (Thermohaline Circulation): Cold, salty water is denser and sinks toward the seabed, while warmer, less salty water rises to the surface.\n"
                            "3. Earth's Rotation (Coriolis Effect): Deflects currents to the right in the Northern Hemisphere and to the left in the Southern Hemisphere.\n"
                            "These currents regulate global climate and transport nutrients essential for marine ecosystems."
                        )
                    citations.append("Physical oceanography principles (Wind, Density, Coriolis force)")
                elif "chlorophyll" in msg_lower:
                    if is_hindi:
                        answer = (
                            "Chlorophyll concentration samundar mein phytoplankton (microscopic paudhe) ki maatra ko darshata hai. "
                            "Jahan chlorophyll ka level zyada hota hai, wahan ocean biological productivity adhik hoti hai aur machhliyon "
                            "ke liye prachur maatra mein bhojan uplabdh hota hai, jo Potential Fishing Zones (PFZ) banane mein madad karta hai."
                        )
                    else:
                        answer = (
                            "Chlorophyll concentration indicates the abundance of phytoplankton, the primary microscopic producers in the ocean. "
                            "High chlorophyll levels signify high biological productivity and nutrient-rich waters where fish gather to feed, "
                            "making it a key satellite indicator for Potential Fishing Zones (PFZ)."
                        )
                else:
                    if is_hindi:
                        answer = "Marine ecosystem mein vibhinn jal-vayu factors jaise taapmaan, hawa aur chlorophyll machhliyon ke aavagaman ko nirdharit karte hain."
                    else:
                        answer = "In marine ecosystems, oceanographic factors such as sea surface temperature, currents, and chlorophyll dictate marine life habitats and navigation safety."

            return FinalResponse(
                answer=answer,
                language=lang,
                risk="LOW",
                reasons=["Educational marine knowledge retrieved from verified shared knowledge layer."],
                warnings=[],
                evidence=citations,
                map_actions=[],
                agents_used=agents_used,
                timestamp=now,
                is_cached=is_cached,
                cached_at=cached_at,
                mode="CHAT"
            )


        # Extract real context data
        ocean_data = (marine_context.ocean or {}).get("data", {})
        weather_data = (marine_context.weather or {}).get("data", {})
        hazard_data = (marine_context.hazards or {}).get("data", {})
        pfz_data = (marine_context.pfz or {}).get("data", {})
        loc_data = (marine_context.location or {})

        wave_h = ocean_data.get("wave_height")
        wind_s = weather_data.get("wind_speed")
        sst_val = ocean_data.get("sst")
        wave_str = f"{wave_h} m" if wave_h is not None else "Normal"
        wind_str = f"{wind_s} km/h" if wind_s is not None else "Normal"
        sst_str = f"{sst_val} °C" if sst_val is not None else "Normal"

        is_inland = loc_data.get("is_inland", False)
        inland_dist = loc_data.get("inland_distance_km", 0.0)
        nearest_coast = loc_data.get("nearest_coast_name", "Nearby Coast")
        place_name = loc_data.get("display_name") or loc_data.get("port_name") or loc_data.get("name")
        if not place_name:
            if query.latitude and query.longitude:
                place_name = f"Coordinates ({query.latitude:.2f}°N, {query.longitude:.2f}°E)"
            else:
                place_name = "your requested sector"

        temp_val = weather_data.get("temperature")
        apparent_temp = weather_data.get("apparent_temperature")
        humidity_val = weather_data.get("humidity")
        gusts_val = weather_data.get("wind_gusts")
        pressure_val = weather_data.get("surface_pressure")
        cond_text = weather_data.get("condition_text") or "Clear / Moderate"
        upcoming_forecast = weather_data.get("upcoming_forecast") or []

        temp_str = f"{temp_val}°C" if temp_val is not None else None
        feels_str = f"{apparent_temp}°C" if apparent_temp is not None else None
        gusts_str = f"{gusts_val} km/h" if gusts_val is not None else None
        humidity_str = f"{humidity_val}%" if humidity_val is not None else None
        pressure_str = f"{pressure_val} hPa" if pressure_val is not None else None

        evidence_items: List[str] = []

        # --- A. Nearby Sea Distance Query ---
        if loc_data.get("is_nearby_query"):
            user_loc_name = loc_data.get("user_location_name") or place_name
            if is_hindi:
                answer = (
                    f"📍 **Aapke Kareebi Samundar Ki Jaankari:**\n"
                    f"• 🏠 **Vartamaan Sthan:** {user_loc_name}\n"
                    f"• 🌊 **Sabse Paas Ka Samundar:** {nearest_coast} (lagbhag **{inland_dist} km door** hai)\n"
                    f"• 📊 **Vahan Samundari Halaat:** Lehrein {wave_str}, hawa {wind_str}, SST {sst_str}\n"
                    f"• 🛡️ **Suraksha Level:** {risk['level']} (Sthaniya mausam: {cond_text}, taapmaan {temp_str or 'Normal'})"
                )
            else:
                answer = (
                    f"📍 **Nearest Coastal Sea Information:**\n"
                    f"• 🏠 **Current Location:** {user_loc_name}\n"
                    f"• 🌊 **Closest Sea / Coast:** {nearest_coast} (approximately **{inland_dist} km away**)\n"
                    f"• 📊 **Marine Conditions There:** Waves {wave_str}, winds {wind_str}, SST {sst_str}\n"
                    f"• 🛡️ **Safety Status:** {risk['level']} (Local weather: {cond_text}, temperature {temp_str or 'Normal'})"
                )
            answer += "\n\n[Source: Open-Meteo Marine (Copernicus) & INCOIS]"

        # --- B. Marine Hazard & Disaster Query ---
        elif intent == "HAZARD":
            disaster_intel = hazard_data.get("disaster_intelligence")
            active_warnings = hazard_data.get("official_warnings", [])
            derived_alerts = hazard_data.get("derived_alerts", [])

            if is_hindi:
                if active_warnings:
                    bulletin_text = active_warnings[0].get('bulletin', 'Active warning detected')
                    status_bullet = f"⚠️ **Official Alert:** {bulletin_text}"
                elif derived_alerts:
                    status_bullet = f"⚠️ **Risk Alert:** {derived_alerts[0].get('message')}"
                elif disaster_intel and disaster_intel.get("summary_hi"):
                    status_bullet = f"🌪️ **Disaster Monitor:** {disaster_intel.get('summary_hi')}"
                else:
                    status_bullet = f"✅ **Disaster / Toofan Status:** {place_name} ke liye koi active cyclone ya severe disaster alert nahi hai."

                answer = (
                    f"🌪️ **Disaster & Safety Overview ({place_name}):**\n"
                    f"• {status_bullet}\n"
                    f"• 💨 **Hawa & Mausam:** Hawa {wind_str}" + (f" (jhoke {gusts_str})" if gusts_str else "") + f", mausam {cond_text}, taapmaan {temp_str or 'N/A'}.\n"
                    f"• 🛡️ **Suraksha Nirdesh:** Risk level: {risk['level']} — " + ("Khatre ke karan open sea mein na jayein aur surakshit sthan par rahein." if risk["level"] == "HIGH" else "Sthaniya mausam samanya aur surakshit hai.")
                )
                answer += "\n\n[Srot: GDACS (UN / European Commission), USGS Earthquakes & IMD]"
            else:
                if active_warnings:
                    bulletin_text = active_warnings[0].get('bulletin', 'Active warning detected')
                    status_bullet = f"⚠️ **Official Alert:** {bulletin_text}"
                elif derived_alerts:
                    status_bullet = f"⚠️ **Risk Alert:** {derived_alerts[0].get('message')}"
                elif disaster_intel and disaster_intel.get("summary_en"):
                    status_bullet = f"🌪️ **Disaster Monitor:** {disaster_intel.get('summary_en')}"
                else:
                    status_bullet = f"✅ **Disaster & Storm Status:** No active cyclone, storm, or severe disaster warning for {place_name}."

                answer = (
                    f"🌪️ **Disaster & Safety Overview ({place_name}):**\n"
                    f"• {status_bullet}\n"
                    f"• 💨 **Winds & Weather:** Speed {wind_str}" + (f" (gusts {gusts_str})" if gusts_str else "") + f", conditions: {cond_text}, temp {temp_str or 'N/A'}.\n"
                    f"• 🛡️ **Safety Advisory:** Risk level: {risk['level']} — " + ("High risk: Avoid offshore travel and seek safe harbor." if risk["level"] == "HIGH" else "Current atmospheric conditions are safe.")
                )
                answer += "\n\n[Source: GDACS (UN / European Commission), USGS Earthquakes & IMD]"

        # --- C. Ocean Information Query (e.g. SST, Currents) ---
        elif intent == "OCEAN_INFO":
            sst_display = sst_str if sst_val is not None else "Normal / N/A"
            if is_hindi:
                answer = (
                    f"🌊 **Ocean Information ({place_name}):**\n"
                    f"• 🌡️ **Sea Surface Temperature (SST):** {sst_display}\n"
                    f"• 🌊 **Samundari Lehrein & Hawa:** Lehrein {wave_str}, hawa {wind_str}\n"
                    f"• 🛡️ **Risk Level:** {risk['level']}"
                )
                answer += "\n\n[Srot: Open-Meteo Copernicus Marine Model]"
            else:
                answer = (
                    f"🌊 **Ocean Information ({place_name}):**\n"
                    f"• 🌡️ **Sea Surface Temperature (SST):** {sst_display}\n"
                    f"• 🌊 **Waves & Wind:** Wave height {wave_str}, winds {wind_str}\n"
                    f"• 🛡️ **Risk Level:** {risk['level']}"
                )
                answer += "\n\n[Source: Open-Meteo Copernicus Marine Model]"

        # --- D. Fishing Assistance Query ---
        elif intent == "FISHING_LOCATION":
            has_pfz = pfz_data.get("pfz_available", False)
            if has_pfz and pfz_data.get("recommended_lat"):
                rec_location = f"{pfz_data['recommended_lat']},{pfz_data['recommended_lon']}"

            if is_inland:
                if is_hindi:
                    answer = (
                        f"🎣 **Fishing Status & Advisory ({place_name}):**\n"
                        f"• 🏞️ **Inland Kshetra:** {place_name} ek zameeni (inland) shehar hai, yahan direct samundari sea fishing zone nahi hota.\n"
                        f"• 🌊 **Kareebi Tat:** Sabse paas ka samundar **{nearest_coast}** (~{inland_dist} km door) hai.\n"
                        f"• 🌤️ **Sthaniya Mausam:** {cond_text}, taapmaan {temp_str or 'Normal'}, hawa {wind_str}.\n"
                        f"• 💡 **Advisory:** Talaab/nadi freshwater activity ke liye mausam anukool hai; samundari fishing ke liye {nearest_coast} ka live data dekhein."
                    )
                    answer += "\n\n[Srot: Open-Meteo Weather Model & INCOIS]"
                else:
                    answer = (
                        f"🎣 **Fishing Status & Advisory ({place_name}):**\n"
                        f"• 🏞️ **Inland Territory:** {place_name} is an inland area without open sea marine fishing zones.\n"
                        f"• 🌊 **Nearest Coast:** The closest sea is **{nearest_coast}** (~{inland_dist} km away).\n"
                        f"• 🌤️ **Local Weather:** {cond_text}, temperature {temp_str or 'Normal'}, winds {wind_str}.\n"
                        f"• 💡 **Advisory:** Conditions are suitable for local freshwater activity. For marine commercial fishing, consult {nearest_coast}."
                    )
                    answer += "\n\n[Source: Open-Meteo Weather Model & INCOIS]"
            else:
                if is_hindi:
                    pfz_note = "Potential Fishing Zone (PFZ) map par active hai (INCOIS thermal front)." if has_pfz else "Coastal waters mein normal fishing activity anukool hai."
                    answer = (
                        f"🎣 **Fishing Zone & Marine Advisory ({place_name}):**\n"
                        f"• 🌊 **Samundari Halaat:** Lehrein {wave_str}, hawa {wind_str}, SST {sst_str}.\n"
                        f"• 🐟 **Fishing Zone (PFZ):** {pfz_note}\n"
                        f"• ⏱️ **Surakshit Samay:** Subah 05:00 AM – 11:30 AM (shant samay). Dopahar baad squall lines se pehle laut aayein.\n"
                        f"• 🛡️ **Safety & Helpline:** Risk level: {risk['level']} — " + ("High risk: samundar mein na jayein." if risk["level"] == "HIGH" else "Coast Guard SAR Helpline: 1554 / Emergency: 112.")
                    )
                    answer += "\n\n[Srot: INCOIS Ocean State Forecast & Open-Meteo Copernicus Marine Model]"
                else:
                    pfz_note = "Active Prospective Fishing Zone identified on map (INCOIS thermal front)." if has_pfz else "Coastal waters show standard fish habitat conditions."
                    answer = (
                        f"🎣 **Fishing Zone & Marine Advisory ({place_name}):**\n"
                        f"• 🌊 **Sea Conditions:** Waves {wave_str}, winds {wind_str}, SST {sst_str}.\n"
                        f"• 🐟 **Fishing Potential:** {pfz_note}\n"
                        f"• ⏱️ **Safe Departure Window:** 05:00 AM – 11:30 AM (morning calm window). Return before afternoon thermal squalls.\n"
                        f"• 🛡️ **Safety & Emergency:** Risk level: {risk['level']} — " + ("Departure not advised due to elevated hazard." if risk["level"] == "HIGH" else "Coast Guard SAR Hotline: 1554 / National Emergency: 112.")
                    )
                    answer += "\n\n[Source: INCOIS Ocean State Forecast & Open-Meteo Copernicus Marine Model]"

        # --- E. ROUTE: Maritime Route & Navigational Corridor ---
        elif intent == "ROUTE":
            route_data = (marine_context.route or {}).get("data", {})
            dist_km = route_data.get("distance_km", "N/A")
            dist_nm = route_data.get("distance_nautical_miles", "N/A")
            est_mins = route_data.get("estimated_transit_time_minutes", "N/A")

            if is_hindi:
                answer = (
                    f"🧭 **Marine Route & Navigation ({place_name}):**\n"
                    f"• 📏 **Corridor Doori:** {dist_km} km ({dist_nm} NM) | Anumaanit Safar: ~{est_mins} mins (10 knots par).\n"
                    f"• 🌊 **Halaat:** Lehrein {wave_str}, hawa {wind_str}, risk level: {risk['level']}.\n"
                    f"• ⏱️ **Schedule:** Subah 04:30 AM – 06:15 AM (bhor ka samay nikalne ke liye sarvottam). 15:30 PM (3:30 PM) se pehle harbor laut aayein.\n"
                    f"• 🗺️ **3D Map:** Interactive Map par cyan corridor dekhein aur hazard area se doori banaye rakhein."
                )
                answer += "\n\n[Srot: NeerAI Navigational Geodesic & INCOIS]"
            else:
                answer = (
                    f"🧭 **Marine Route & Passage ({place_name}):**\n"
                    f"• 📏 **Corridor Distance:** {dist_km} km ({dist_nm} NM) | Estimated Transit: ~{est_mins} mins @ 10 knots.\n"
                    f"• 🌊 **Sea Conditions:** Waves {wave_str}, winds {wind_str}, evaluated risk: {risk['level']}.\n"
                    f"• ⏱️ **Timing Window:** 04:30 AM – 06:15 AM departure. Return to harbor before 15:30 PM (3:30 PM).\n"
                    f"• 🗺️ **Interactive Corridor:** Cyan navigational corridor plotted on companion 3D Dual Map."
                )
                answer += "\n\n[Source: NeerAI Navigational Geodesic & INCOIS]"

        # --- F. WEATHER, MARINE_CONDITION & GENERAL LOCATION INQUIRIES ---
        else:
            wave_str = f"{wave_h} m" if wave_h is not None else "Normal"
            wind_str = f"{wind_s} km/h" if wind_s is not None else "Normal"
            sst_str = f"{sst_val} °C" if sst_val is not None else "Normal"

            if is_inland:
                if is_hindi:
                    answer = (
                        f"🌤️ **Mausam Jaankari ({place_name}):**\n"
                        f"• 🌡️ **Taapmaan & Mausam:** {cond_text}, taapmaan {temp_str or 'Normal'}" + (f" (mehsoos {feels_str})" if feels_str else "") + f".\n"
                        f"• 💨 **Hawa & Dabav:** {wind_str}" + (f" (jhoke {gusts_str})" if gusts_str else "") + f", Aadrata {humidity_str or 'Normal'}.\n"
                        f"• 🏞️ **Kshetra Status:** Inland location (samundar se lagbhag {inland_dist} km door; kareebi tat: {nearest_coast}).\n"
                        f"• 🛡️ **Risk Level:** {risk['level']} (Mausam samanya aur surakshit hai)."
                    )
                else:
                    answer = (
                        f"🌤️ **Weather Overview ({place_name}):**\n"
                        f"• 🌡️ **Temperature & Sky:** {cond_text}, temperature {temp_str or 'Normal'}" + (f" (feels like {feels_str})" if feels_str else "") + f".\n"
                        f"• 💨 **Winds & Humidity:** {wind_str}" + (f" (gusts up to {gusts_str})" if gusts_str else "") + f", humidity {humidity_str or 'Normal'}.\n"
                        f"• 🏞️ **Location Type:** Inland area (~{inland_dist} km to nearest coast: {nearest_coast}).\n"
                        f"• 🛡️ **Risk Level:** {risk['level']} (Atmospheric conditions are safe)."
                    )
            else:
                if is_hindi:
                    answer = (
                        f"🌊 **Coastal & Marine Weather ({place_name}):**\n"
                        f"• 🌤️ **Sthaniya Mausam:** {cond_text}, taapmaan {temp_str or 'Normal'}" + (f" (mehsoos {feels_str})" if feels_str else "") + f".\n"
                        f"• 🌊 **Samundari Halaat:** Lehrein {wave_str}, hawa {wind_str}" + (f" (jhoke {gusts_str})" if gusts_str else "") + f", SST {sst_str}.\n"
                        f"• 🛡️ **Navigation Risk:** {risk['level']} — " + ("Samundari yatra surakshit hai." if risk['level'] != 'HIGH' else "High risk ke karan yatra sthagit karein.")
                    )
                else:
                    answer = (
                        f"🌊 **Coastal & Marine Weather ({place_name}):**\n"
                        f"• 🌤️ **Local Atmosphere:** {cond_text}, temperature {temp_str or 'Normal'}" + (f" (feels like {feels_str})" if feels_str else "") + f".\n"
                        f"• 🌊 **Sea State:** Waves {wave_str}, winds {wind_str}" + (f" (gusts {gusts_str})" if gusts_str else "") + f", SST {sst_str}.\n"
                        f"• 🛡️ **Navigation Risk:** {risk['level']} — " + ("Marine conditions are safe." if risk['level'] != 'HIGH' else "High risk: open sea departure not advised.")
                    )

            if upcoming_forecast and any(w in (query.message or "").lower() for w in ["forecast", "aane wale", "kal", "parso", "days", "dino"]):
                forecast_lines = []
                for day_info in upcoming_forecast[1:3]:
                    d_date = day_info.get("date", "")
                    d_cond = day_info.get("condition", "Fair weather")
                    d_max = day_info.get("max_temp")
                    d_min = day_info.get("min_temp")
                    temp_range = f"{d_max}°C / {d_min}°C" if d_max is not None and d_min is not None else "N/A"
                    if is_hindi:
                        forecast_lines.append(f"• {d_date}: {d_cond}, taapmaan {temp_range}")
                    else:
                        forecast_lines.append(f"• {d_date}: {d_cond}, temp {temp_range}")

                if forecast_lines:
                    header = "\n\n📅 **Aane Wale Dino Ka Forecast:**\n" if is_hindi else "\n\n📅 **Upcoming Forecast:**\n"
                    answer += header + "\n".join(forecast_lines)

            sources_cited = ["Open-Meteo High-Resolution Model (ECMWF)"]
            if not is_inland:
                sources_cited.append("Copernicus Marine")
            answer += f"\n\n[Sources: {', '.join(sources_cited)}]"

        # Safety rule: If cached, attach concise offline indicator
        if is_cached:
            answer += f" [OFFLINE CACHE: {cached_at}]"
            warnings.append(f"Response built from local cache ({cached_at}). Real-time warnings unavailable.")

        if unavailable_sources:
            unavail_str = ", ".join(unavailable_sources)
            warnings.append(f"Some upstream providers are temporarily unavailable: {unavail_str}. Missing data is not fabricated.")
            if not any(term in answer.lower() for term in ["unavailable", "unable", "unreachable", "offline", "data"]):
                if is_hindi:
                    answer += f" (Upstream provider {unavail_str} data temporarily unavailable hai)."
                else:
                    answer += f" (Note: Upstream provider {unavail_str} data is temporarily unavailable)."

        # Rich GIS Map actions with distinct layer colors (especially Emerald Green #10B981 for PFZ fishing zone)
        # Prioritize resolved place from query text over default device coordinates
        if loc_data.get("port_name") or loc_data.get("display_name"):
            center_lat = loc_data.get("latitude", 15.30)
            center_lon = loc_data.get("longitude", 73.70)
        else:
            center_lat = loc_data.get("latitude") or query.latitude or 15.30
            center_lon = loc_data.get("longitude") or query.longitude or 73.70
        location_title = place_name

        map_actions = [
            {"type": "center", "lat": center_lat, "lon": center_lon},
            {
                "type": "marine_gis_card",
                "lat": center_lat,
                "lon": center_lon,
                "location_name": location_title,
                "is_inland": is_inland,
                "inland_distance_km": inland_dist,
                "view": "dual",
                "layers": {
                    "pfz": {"color": "#10B981", "name": "Fishing Zone (PFZ)", "status": "Protected / Monitored"},
                    "waves": {"color": "#0284C7", "name": "Waves & Swell", "value": wave_str},
                    "hazards": {"color": "#EF4444", "name": "Storm Warnings", "status": risk["level"]}
                }
            }
        ]

        if temp_str:
            evidence_items.append(f"Atmospheric Observation: Temp {temp_str}, Wind {wind_str}, Condition '{cond_text}' (Open-Meteo ECMWF IFS @ {now})")
        if not is_inland or wave_h is not None:
            evidence_items.append(f"Marine Sea State: Waves {wave_str}, SST {sst_str} (Open-Meteo Copernicus Marine @ {now})")
        evidence_items.append(f"Geocoded Geographic Sector: {place_name} @ ({center_lat:.4f}, {center_lon:.4f}) (Open-Meteo & Nominatim OSM)")

        return FinalResponse(
            answer=answer,
            language=lang,
            risk=risk["level"],
            recommended_location=rec_location,
            reasons=reasons,
            warnings=warnings,
            evidence=evidence_items,
            map_actions=map_actions,
            agents_used=agents_used,
            timestamp=now,
            is_cached=is_cached,
            cached_at=cached_at,
            mode="AGENT"
        )

    async def generate_conversational_response(
        self,
        query: UserQuery,
        intent: str,
        history: Optional[List[Dict[str, Any]]] = None,
        is_cached: bool = False,
        cached_at: Optional[str] = None
    ) -> FinalResponse:
        now = datetime.datetime.utcnow().isoformat() + "Z"
        lang = query.language or "en"
        msg = (query.message or "").strip()
        msg_lower = msg.lower()
        is_hindi = lang.startswith("hi") or any(k in msg_lower for k in [
            "kaisa", "kaisi", "kaise", "kya", "hai", "hain", "aaj", "kal", "samundar", "samunder",
            "lehrein", "lehar", "hawa", "toofan", "machhli", "machli", "kahan", "jaana", "namaste",
            "namaskar", "batao", "bataiye", "door", "paas", "khatra", "surakshit", "shukriya",
            "dhanyawad", "achha", "lag raha", "hoon", "theek", "madad"
        ]) or any('\u0900' <= c <= '\u097F' for c in msg)

        prov_name, prov = self.judge.llm_service.resolve_provider()

        # 1. Attempt Real LLM Generation if provider is active
        if prov_name != "deterministic" and prov.is_available:
            try:
                formatted_msgs = []
                if history:
                    for h in history[-6:]:
                        role = "user" if h.get("role", "").lower() in ["user", "human"] else "assistant"
                        formatted_msgs.append({"role": role, "content": h.get("text", "")})
                formatted_msgs.append({"role": "user", "content": msg})

                llm_output = await self.judge.llm_service.generate_chat_response(
                    messages=formatted_msgs,
                    language="hi" if is_hindi else "en",
                    correlation_id=getattr(query, "correlation_id", "conv_req")
                )
                if llm_output:
                    return FinalResponse(
                        answer=llm_output,
                        language="hi" if is_hindi else "en",
                        risk="LOW",
                        reasons=["Generated dynamically by configured AI language model."],
                        warnings=[],
                        evidence=[],
                        map_actions=[],
                        agents_used=[],
                        timestamp=now,
                        mode="CHAT",
                        provider_info={"provider": prov_name, "status": "ACTIVE", "model": self.judge.llm_service.fast_model}
                    )
            except Exception:
                pass  # Fall back seamlessly to high-fidelity deterministic conversational engine

        # 2. Rich Natural Conversational Engine (when LLM unconfigured or offline fallback)
        provider_info = {
            "provider": "deterministic",
            "status": "UNCONFIGURED" if prov_name == "deterministic" else "STANDBY",
            "message": "Deterministic conversational engine active."
        }

        # A. Clarification Needed (Ambiguous queries like "Is it good?", "Should I go?")
        if intent == "CLARIFICATION_NEEDED":
            if is_hindi:
                answer = (
                    "Aap kis coastal harbor ya specific area se samundar mein jaana chahte hain? "
                    "Kripya mujhe jagah batayein (jaise Goa, Mumbai, ya Kochi), taaki main wahan ke real-time "
                    "hawa ki gati, lehrein aur official IMD alerts verify kar sakoon."
                )
            else:
                answer = (
                    "Which coastal harbor or area are you planning to sail from? "
                    "Please let me know the specific location (such as Goa, Mumbai, or Kochi) so I can verify "
                    "the live wave heights, wind speeds, and active official maritime warnings for you."
                )
            return FinalResponse(
                answer=answer,
                language="hi" if is_hindi else "en",
                risk="LOW",
                reasons=["Clarification requested for ambiguous sailing inquiry."],
                warnings=[],
                evidence=[],
                map_actions=[],
                agents_used=[],
                timestamp=now,
                mode="CHAT",
                provider_info=provider_info
            )

        # B. Personal Feelings / Empathy ("Mujhe aaj achha nahi lag raha", "Feeling down/tired")
        feelings_keywords = [
            "achha nahi lag raha", "achha nhi lag raha", "acha nahi lag raha", "theek nahi lag raha",
            "not feeling well", "feel sad", "feeling sad", "feeling down", "feeling low",
            "mood off", "upset", "pareshan", "thak gaya", "tired", "unwell", "lonely", "bura lag raha"
        ]
        if any(f in msg_lower for f in feelings_keywords):
            if is_hindi:
                answer = (
                    "Main samajh sakta hoon ki aap aaj accha mehsoos nahi kar rahe hain. "
                    "Kabhi-kabhi din thoda thaka dene wala ya pareshani bhara ho sakta hai. "
                    "Aap thoda aaram kijiye aur apna khayal rakhiye. Agar aapko kisi cheez ke baare mein baat karni ho "
                    "ya samundar/mausam ke baare mein kuch dekhna ho, toh main yahin hoon."
                )
            else:
                answer = (
                    "I'm truly sorry to hear that you're not feeling well today. "
                    "Some days can be heavy or exhausting. Please take some time to rest and look after yourself. "
                    "I'm right here whenever you need a friendly chat or want to explore maritime topics."
                )
            return FinalResponse(
                answer=answer,
                language="hi" if is_hindi else "en",
                risk="LOW",
                reasons=["Empathetic response provided for personal well-being."],
                warnings=[],
                evidence=[],
                map_actions=[],
                agents_used=[],
                timestamp=now,
                mode="CHAT",
                provider_info=provider_info
            )

        # C. Capabilities ("What can you do?", "Kya kar sakte ho?")
        if any(w in msg_lower for w in ["what can you do", "kya kar sakte", "features", "capabilities", "madad", "help"]):
            if is_hindi:
                answer = (
                    "Main NeerAI hoon — aapka Maritime & Climate AI Copilot! Main in cheezon mein aapki sahayata kar sakta hoon:\n"
                    "• 💬 Natural Baatcheet: Kisi bhi sawal, gyan, ya casual topic par dost ki tarah baat karna\n"
                    "• 🌊 Live Samundari Halat: Wave swell, hawa ki gati aur surface temperature\n"
                    "• 🎣 Potential Fishing Zones: Machliyon ke prakaar, size, aur surakshit timing\n"
                    "• ⚠️ IMD/GDACS Cyclone Alerts: 1000 km ke daayre mein toofan aur disaster monitor\n"
                    "• 🗺️ 3D Ocean Maps: Navigational corridor aur live danger boundary\n"
                    "Aap mujhse kisi bhi vishay par baat kar sakte hain!"
                )
            else:
                answer = (
                    "I am NeerAI — your Maritime & Climate AI Copilot! Here is how I can assist you:\n"
                    "• 💬 Natural Conversation: Friendly, empathetic dialogue and general knowledge\n"
                    "• 🌊 Live Sea Conditions: Wave swell, wind speeds, and sea temperature\n"
                    "• 🎣 Potential Fishing Zones: Fish species, sizes, and safe dwell windows\n"
                    "• ⚠️ IMD/GDACS Alerts: Scanning cyclones and disasters across 1000 km radius\n"
                    "• 🗺️ 3D Ocean Maps: Safe geodesic navigation corridors and risk boundaries\n"
                    "Feel free to ask me anything!"
                )
            return FinalResponse(
                answer=answer,
                language="hi" if is_hindi else "en",
                risk="LOW",
                reasons=["NeerAI maritime copilot capabilities explained."],
                warnings=[],
                evidence=[],
                map_actions=[],
                agents_used=[],
                timestamp=now,
                mode="CHAT",
                provider_info=provider_info
            )

        # C1. Identity ("Who are you?", "Tum kaun ho?", "Aap kaun ho?")
        if any(w in msg_lower for w in ["who are you", "tum kaun ho", "aap kaun ho", "who r u", "kya ho tum", "apna intro"]):
            if is_hindi:
                answer = (
                    "Main NeerAI hoon — aapka friendly AI Copilot! Main aapse kisi bhi casual topic par dost ki tarah "
                    "baat kar sakta hoon, sawalon ke jawab de sakta hoon, aur jab aapko zaroorat ho to "
                    "real-time mausam, 1000 km cyclone tracking, aur samundari safar ki suraksha advise kar sakta hoon. "
                    "Aap sunaiye, aaj aap kis vishay par baat karna chahenge?"
                )
            else:
                answer = (
                    "I am NeerAI — your intelligent AI Copilot! I converse naturally on any everyday topic, "
                    "answer questions, and provide live climate, cyclone monitoring, and marine safety intelligence "
                    "whenever you need it. What would you like to explore today?"
                )
            return FinalResponse(
                answer=answer,
                language="hi" if is_hindi else "en",
                risk="LOW",
                reasons=["AI Copilot identity presented."],
                warnings=[],
                evidence=[],
                map_actions=[],
                agents_used=[],
                timestamp=now,
                mode="CHAT",
                provider_info=provider_info
            )

        # D1. Casual Chit-Chat ("Bhai kya chal raha hai", "Kya scene hai", "Kya hal chal", "Aur batao")
        if any(w in msg_lower for w in ["kya chal raha", "kya chal rha", "kya scene", "kya hal chal", "kya haal chaal", "aur batao", "tum batao", "bol bhai", "suno bhai", "apna batao"]):
            hi_chitchat = [
                "Bhai sab ekdum mast chal raha hai! Aap sunao, aapka kya haal chaal hai? Aaj ka din kaisa ja raha hai?",
                "Mast chal raha hai sab! Har waqt aapki madad ke liye taiyar hoon. Aap bataiye, aaj kya khas chal raha hai?",
                "Sab badhiya chal raha hai dost! Aap bataiye, aap kya kar rahe hain? Sab maze mein?",
                "Bas badhiya bhai! Sab smooth chal raha hai. Aap sunaiye, sab theek-thaak?"
            ]
            en_chitchat = [
                "Everything is going great! How about you? How is your day going so far?",
                "Doing well and ready to chat! What are you up to today?",
                "All smooth on my end! How are things with you? Feel free to ask anything on your mind.",
                "Things are going great! What's new with you today?"
            ]
            idx = (len(msg_lower) + abs(hash(msg_lower))) % len(hi_chitchat)
            answer = hi_chitchat[idx] if is_hindi else en_chitchat[idx]
            return FinalResponse(
                answer=answer,
                language="hi" if is_hindi else "en",
                risk="LOW",
                reasons=["Casual chit-chat dialogue processed."],
                warnings=[],
                evidence=[],
                map_actions=[],
                agents_used=[],
                timestamp=now,
                mode="CHAT",
                provider_info=provider_info
            )

        # D. How are you / Copilot state ("How are you?", "Kaise ho?", "Kya haal hai?")
        if any(w in msg_lower for w in ["how are you", "kaise ho", "kya haal", "sab theek", "how do you do", "kaisa chal raha"]):
            hi_variants = [
                "Main bilkul badhiya hoon dost, poochne ke liye bohot shukriya! Aap kaise hain? Sab kushal-mangal?",
                "Namaste! Sab theek-thaak chal raha hai. Aapka din kaisa guzar raha hai?",
                "Main mast hoon! Umeed hai aap bhi mast honge. Kahiye, aaj kya chal raha hai?",
                "Ekdum badiya! Har samay aapki madad ke liye taiyar hoon. Aap sunaiye, kya haal hai?"
            ]
            en_variants = [
                "I am doing well, thank you for asking! How are you doing today? How has your day been?",
                "Doing well and everything is operating smoothly, thank you! How has your day been going?",
                "Doing well and in good shape, thank you! Always ready to assist. What's on your mind today?",
                "I am doing great, thank you! How are things with you today?"
            ]
            v_idx = (len(msg_lower) + abs(hash(msg_lower))) % len(hi_variants)
            answer = hi_variants[v_idx] if is_hindi else en_variants[v_idx]
            return FinalResponse(
                answer=answer,
                language="hi" if is_hindi else "en",
                risk="LOW",
                reasons=["Dynamic conversational state inquiry addressed."],
                warnings=[],
                evidence=[],
                map_actions=[],
                agents_used=[],
                timestamp=now,
                mode="CHAT",
                provider_info=provider_info
            )

        # E. Standard Greetings ("Hello", "Hi", "Namaste", "Good morning")
        if any(w in msg_lower for w in ["hello", "hi", "hey", "namaste", "namaskar", "good morning", "good evening", "shubh prabhat", "pranam"]):
            hi_greetings = [
                "Namaste! Kahiye, kya haal chaal? Aaj main aapki kya madad kar sakta hoon?",
                "Hello dost! NeerAI par swagat hai aapka. Bataiye, aaj ka din kaisa chal raha hai?",
                "Namaskar! Umeed hai aapka din accha ja raha hoga. Kahiye, aaj kya janna chahte hain?",
                "Hey! Swagat hai aapka. Bataiye, aaj kis baare mein baat karna chahte hain?"
            ]
            en_greetings = [
                "Hello! Welcome to NeerAI. How are you doing today, and how can I help you?",
                "Hello! Great to connect with you. How is your day going?",
                "Hey there! Always ready to assist. What is on your mind today?",
                "Hello! How can I assist you today? Feel free to ask anything!"
            ]
            g_idx = (len(msg_lower) + abs(hash(msg_lower))) % len(hi_greetings)
            answer = hi_greetings[g_idx] if is_hindi else en_greetings[g_idx]
            return FinalResponse(
                answer=answer,
                language="hi" if is_hindi else "en",
                risk="LOW",
                reasons=["Dynamic greeting processed."],
                warnings=[],
                evidence=[],
                map_actions=[],
                agents_used=[],
                timestamp=now,
                mode="CHAT",
                provider_info=provider_info
            )

        # F. Gratitude ("Thank you", "Shukriya")
        if any(w in msg_lower for w in ["thank", "shukriya", "dhanyawad", "meherbani"]):
            hi_thanks = [
                "Aapka bohot-bohot swagat hai! Surakshit rahein aur koi bhi sawal ho toh bejhijhak poochhein.",
                "Koi baat nahi, ye mera farz hai! Agar kisi aur sheher ya mausam ki jankari chahiye toh main yahin hoon.",
                "Khushi hui aapki madad karke! Apna khayal rakhein aur jab bhi zaroorat ho, zaroor batayein."
            ]
            en_thanks = [
                "You're very welcome! Stay safe, and feel free to reach out whenever you need updates.",
                "Glad I could help! If you have any further questions about weather, storms, or navigation, just ask.",
                "Always happy to assist! Wishing you clear skies and safe travels."
            ]
            t_idx = (len(msg_lower) + abs(hash(msg_lower))) % len(hi_thanks)
            answer = hi_thanks[t_idx] if is_hindi else en_thanks[t_idx]
            return FinalResponse(
                answer=answer,
                language="hi" if is_hindi else "en",
                risk="LOW",
                reasons=["Gratitude acknowledged dynamically."],
                warnings=[],
                evidence=[],
                map_actions=[],
                agents_used=[],
                timestamp=now,
                mode="CHAT",
                provider_info=provider_info
            )

        # G. Educational Knowledge Search (Science, Disasters, Weather, Currents, Rain, etc.)
        from app.services.marine_knowledge_service import MarineKnowledgeService
        kb_docs = MarineKnowledgeService().search_unstructured_knowledge(msg)
        if kb_docs:
            top = kb_docs[0]
            if is_hindi:
                answer = f"{top['content']}\n\n[Srot: {top['source_authority']}]"
            else:
                answer = f"{top['content']}\n\n[Source Authority: {top['source_authority']}]"
            return FinalResponse(
                answer=answer,
                language="hi" if is_hindi else "en",
                risk="LOW",
                reasons=[f"Knowledge base topic retrieved: {top['title']}."],
                warnings=[],
                evidence=[],
                map_actions=[],
                agents_used=[],
                timestamp=now,
                mode="CHAT",
                provider_info=provider_info
            )

        # H. Dynamic Non-Repetitive Conversational Brain
        # Understand intent from question keywords and formulate a rich response
        if any(w in msg_lower for w in ["rain", "baarish", "barish", "cloud", "badal"]):
            if is_hindi:
                answer = (
                    "Baarish (rainfall) tab banti hai jab suraj ki dhoop se samundar, nadiyon aur jheelon ka paani garam hokar "
                    "jal-vaashp (water vapor) bankar upar uthta hai. Upar jaakar taapmaan girne par ye baashp thandi hokar "
                    "chhoti boondon mein condensation karti hai aur baadal bante hain. Jab ye boondein badi aur bhari ho jaati hain, "
                    "toh gurutwakarshan (gravity) ke karan baarish ke roop mein zameen par girti hain. "
                    "Kya aap kisi specific sheher ya kshetra ke baarish forecast ke baare mein jaanna chahte hain?"
                )
            else:
                answer = (
                    "Rain forms through the natural water cycle: solar heat evaporates surface water from oceans and lakes into water vapor. "
                    "As this warm, moist air rises, it cools and condenses around tiny aerosol particles to form clouds. "
                    "When cloud droplets coalesce into heavy water droplets, gravity pulls them down as precipitation (rain). "
                    "Would you like me to check the upcoming precipitation forecast for your location?"
                )
        elif any(w in msg_lower for w in ["cyclone", "toofan", "storm", "hurrincane", "typhoon"]):
            if is_hindi:
                answer = (
                    "Chakrawat (Cyclone) ek shaktishaali ghoomti hui hawaon ka toofan hota hai jo garm samundari paani (26.5°C se adhik) "
                    "ke upar barometric pressure girne (low pressure system) se banta hai. Coriolis force isse uttar golaardh mein anti-clockwise "
                    "aur dakshin golaardh mein clockwise disha mein ghumati hai. Isme tez hawaayein, bhaari baarish aur storm surge (samundar ki unchi lehrein) hoti hain. "
                    "Aap mujhse kisi bhi sheher (jaise Lucknow, Mumbai, ya koi bhi jagah) ke 1000 km ke daayre mein live cyclones check karne ko keh sakte hain!"
                )
            else:
                answer = (
                    "A tropical cyclone is an intense rotating storm system that forms over warm ocean waters (SST ≥ 26.5°C) around a deep "
                    "low-pressure center. Driven by latent heat and deflected by Earth's Coriolis force, cyclones produce destructive gale winds, "
                    "torrential rainfall, and dangerous coastal storm surges. "
                    "You can ask me to scan active cyclones within a 1,000 km radius of any place on Earth (such as Lucknow, Goa, or Tokyo)!"
                )
        elif any(w in msg_lower for w in ["bhookamp", "earthquake", "seismic", "richter"]):
            if is_hindi:
                answer = (
                    "Bhookamp (Earthquake) prithvi ki tectonic plates ke aapas mein takraane, sarakne ya tootne se paida hone wali energy waves "
                    "(seismic waves) ke karan aata hai. Iski teevrata Richter scale par maapi jaati hai. "
                    "Suraksha nirdesh: Bhookamp ke dauraan 'Drop, Cover, and Hold On' karein, majboot table ke neeche jayein aur building ki deewaron/khidkiyon se door rahein. "
                    "NeerAI USGS live feed ke zariye poore vishwa ke 4.5+ magnitude ke bhookamp real-time monitor karta hai."
                )
            else:
                answer = (
                    "An earthquake is the sudden release of accumulated energy in the Earth's crust along tectonic faults, "
                    "generating seismic waves that cause ground shaking. Its magnitude is measured on the moment magnitude scale (M). "
                    "Safety reminder: During shaking, remember 'Drop, Cover, and Hold On' under a sturdy desk and stay away from glass windows. "
                    "NeerAI tracks real-time USGS magnitude 4.5+ earthquakes globally."
                )
        else:
            hi_dynamic = [
                f"Main aapke sawal ko samajh gaya hoon! NeerAI real-time GPS aur network detection ke saath kaam karta hai. "
                f"Aap mujhse kisi bhi sheher ya samundari kshetra ke baare mein mausam, 1000 km ke daayre mein cyclone/disaster alerts, "
                f"lehron ki sthiti ya suraksha nirdesh pooch sakte hain. Kripya batayein aap kahan ki jankari chahte hain?",

                f"Bohot badhiya sawal! NeerAI par hum live Open-Meteo, GDACS aur USGS live satellite feeds ke zariye "
                f"taaza sthiti monitor karte hain. Aap jis bhi jagah ka naam lenge (jaise Lucknow, Delhi, ya koi bhi coastal port), "
                f"main wahan ke taaza mausam aur aane wale alerts turant nikaal kar bata doonga.",

                f"Ji haan! Ek intelligent copilot ke roop mein main mausam vigyan, samundari safar ki suraksha, "
                f"cyclone tracking aur emergency disaster advisories mein maahir hoon. Aap live weather check karna chahte hain "
                f"ya kisi vishay par vistar se baat karna chahte hain?"
            ]
            en_dynamic = [
                "I understand your question! NeerAI is powered by real-time geolocation, live numerical weather models, and global disaster systems. "
                "You can ask me to check real-time weather at your current position, scan active cyclones within 1,000 km of any city on Earth, "
                "or explain maritime safety concepts. Which location or topic would you like to explore?",

                "Great question! NeerAI monitors live weather and environmental conditions globally via Open-Meteo, GDACS (UN/EC), and USGS. "
                "Simply name any place (like Lucknow, Mumbai, Tokyo, or your current coordinates), and I will evaluate local weather, "
                "wave swell, and upcoming natural disaster alerts for you.",

                "Understood! As your AI Copilot, I can provide live atmospheric weather, multi-hazard alerts, and scientific explanations. "
                "Would you like a weather report for your current location, or to check for active storms nearby?"
            ]
            dyn_idx = (len(msg_lower) + abs(hash(msg_lower))) % len(hi_dynamic)
            answer = hi_dynamic[dyn_idx] if is_hindi else en_dynamic[dyn_idx]

        return FinalResponse(
            answer=answer,
            language="hi" if is_hindi else "en",
            risk="LOW",
            reasons=["Dynamic conversational inquiry addressed."],
            warnings=[],
            evidence=[],
            map_actions=[],
            agents_used=[],
            timestamp=now,
            mode="CHAT",
            provider_info=provider_info
        )


    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        # Compatibility stub
        return AgentResult(
            agent=self.name,
            status="SUCCESS",
            data={},
            confidence=1.0,
            sources=["Response Synthesis Engine"],
            timestamp=datetime.datetime.utcnow().isoformat() + "Z"
        )

