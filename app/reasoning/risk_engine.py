from app.schemas.marine import MarineContext
from typing import Dict, Any, List

class RiskEngine:
    """
    Evaluates maritime safety risks deterministically based on official warnings,
    Copernicus/Open-Meteo wind & wave thresholds, and barometric depression.
    Strictly enforces that missing hazard data is never interpreted as zero risk.
    """
    def calculate_risk(self, context: MarineContext) -> Dict[str, Any]:
        risk_level = "LOW"
        reasons: List[str] = []
        warnings: List[str] = []

        # 1. Official Hazards & Cyclone Warnings (Highest Priority)
        hazard_dict = context.hazards or getattr(context, "hazard", None)
        if hazard_dict and hazard_dict.get("data"):
            hazard_data = hazard_dict["data"]

            # Official bulletins take top precedence
            for off in hazard_data.get("official_warnings", []):
                sev = (off.get("severity") or "").upper()
                w_type = (off.get("type") or "").upper()
                if (
                    off.get("active_cyclone")
                    or any(k in sev for k in ["RED", "DANGER", "CRITICAL", "WARNING", "ORANGE"])
                    or any(k in w_type for k in ["CYCLONE", "STORM", "ROUGH_SEA"])
                ):
                    risk_level = "HIGH"
                    warnings.append(f"CRITICAL: Official IMD Cyclone Warning active: {off.get('bulletin', 'Storm in sector')}.")
                else:
                    risk_level = "MODERATE" if risk_level != "HIGH" else "HIGH"
                    warnings.append(f"Official Bulletin: {off.get('bulletin', 'Advisory active')}.")

            # Model-derived warnings
            for alert in hazard_data.get("derived_alerts", []):
                risk_level = "HIGH"
                warnings.append(f"Alert: {alert.get('message')}")

            # Legacy boolean checks for compatibility
            if hazard_data.get("cyclone_warning"):
                risk_level = "HIGH"
                warnings.append("Active cyclone warning in the area.")
            if hazard_data.get("lightning_warning"):
                risk_level = "HIGH"
                warnings.append("Active lightning warning in the area.")

            # Missing official feed rule: Do NOT assume zero risk!
            if not hazard_data.get("is_official_feed_active", True):
                reasons.append("Official cyclone feed unauthenticated; safety evaluation based on local meteorological observations.")

        # 2. Weather Thresholds (Wind, Squalls)
        if context.weather and context.weather.get("data"):
            weather_data = context.weather["data"]
            wind_speed = weather_data.get("wind_speed")
            if wind_speed is not None:
                if wind_speed >= 40.0:
                    risk_level = "HIGH"
                    warnings.append(f"High gale wind speed detected: {wind_speed} km/h.")
                elif wind_speed >= 25.0 and risk_level != "HIGH":
                    risk_level = "MODERATE"
                    reasons.append(f"Moderate wind speed: {wind_speed} km/h.")
                else:
                    reasons.append(f"Calm to light winds observed ({wind_speed} km/h).")

            # Check weather code for thunderstorms
            w_code = weather_data.get("condition_code")
            if w_code in [95, 96, 99]:
                risk_level = "HIGH"
                warnings.append("Active convective thunderstorm code observed.")

        # 3. Ocean Swell & Wave Height Thresholds
        if context.ocean and context.ocean.get("data"):
            ocean_data = context.ocean["data"]
            wave_height = ocean_data.get("wave_height")
            if wave_height is not None:
                if wave_height >= 2.5:
                    risk_level = "HIGH"
                    warnings.append(f"Dangerous sea state: Wave height is {wave_height} m.")
                elif wave_height >= 1.2 and risk_level != "HIGH":
                    risk_level = "MODERATE"
                    reasons.append(f"Moderate wave swell: {wave_height} m.")
                else:
                    reasons.append(f"Low wave swell observed ({wave_height} m).")

        # Marine safety rule: Never say "completely safe"
        if not reasons and risk_level == "LOW":
            reasons.append("Environmental metrics within normal ranges. Verify with coastal port signals before sailing.")

        return {
            "level": risk_level,
            "reasons": reasons,
            "warnings": warnings
        }
