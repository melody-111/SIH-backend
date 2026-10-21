import httpx
from typing import Dict, Any

WMO_WEATHER_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rainfall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Severe thunderstorm with heavy hail"
}

class WeatherService:
    async def get_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Retrieves comprehensive real-time atmospheric weather and upcoming
        multi-day forecasts using Open-Meteo High-Resolution Global Numerical Weather API.
        """
        current_vars = [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "weather_code",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m"
        ]
        daily_vars = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "precipitation_sum",
            "weather_code",
            "wind_speed_10m_max"
        ]

        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            f"current={','.join(current_vars)}&"
            f"daily={','.join(daily_vars)}&"
            f"timezone=auto&forecast_days=3"
        )

        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                current = data.get("current", {})
                daily = data.get("daily", {})

                code = current.get("weather_code", 0)
                cond_text = WMO_WEATHER_MAP.get(code, "Clear / Moderate")

                # Extract upcoming forecast
                upcoming_forecast = []
                times = daily.get("time", [])
                t_max = daily.get("temperature_2m_max", [])
                t_min = daily.get("temperature_2m_min", [])
                p_prob = daily.get("precipitation_probability_max", [])
                p_sum = daily.get("precipitation_sum", [])
                w_codes = daily.get("weather_code", [])

                for i in range(len(times)):
                    w_code = w_codes[i] if i < len(w_codes) else 0
                    upcoming_forecast.append({
                        "date": times[i],
                        "max_temp": t_max[i] if i < len(t_max) else None,
                        "min_temp": t_min[i] if i < len(t_min) else None,
                        "precip_prob_pct": p_prob[i] if i < len(p_prob) else 0,
                        "precip_mm": p_sum[i] if i < len(p_sum) else 0.0,
                        "condition": WMO_WEATHER_MAP.get(w_code, "Fair weather")
                    })

                return {
                    "temperature": current.get("temperature_2m"),
                    "apparent_temperature": current.get("apparent_temperature"),
                    "humidity": current.get("relative_humidity_2m"),
                    "wind_speed": current.get("wind_speed_10m"),
                    "wind_gusts": current.get("wind_gusts_10m"),
                    "wind_direction": current.get("wind_direction_10m"),
                    "precipitation": current.get("precipitation"),
                    "surface_pressure": current.get("surface_pressure"),
                    "condition_code": code,
                    "condition_text": cond_text,
                    "upcoming_forecast": upcoming_forecast,
                    "status": "success",
                    "source": "Open-Meteo High-Resolution Weather Model (ECMWF IFS)"
                }
        except Exception as e:
            return {"status": "unavailable", "error": str(e), "source": "Open-Meteo API"}
