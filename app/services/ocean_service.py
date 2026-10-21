import httpx
from typing import Dict, Any

class OceanService:
    async def get_ocean_conditions(self, lat: float, lon: float) -> Dict[str, Any]:
        # Open-Meteo Marine API with real SST and wave parameters
        url = (
            f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}"
            f"&current=wave_height,wave_direction,wave_period,ocean_current_velocity,sea_surface_temperature"
        )
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                current = data.get("current", {})
                
                # If point is on land and wave_height is null, try offshore fallback
                wave_height = current.get("wave_height")
                sst = current.get("sea_surface_temperature")
                
                if wave_height is None:
                    # Probe slightly seaward (Arabian Sea / Bay of Bengal)
                    step = -0.08 if lon < 77.5 else 0.08
                    fallback_url = (
                        f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={round(lon + step, 4)}"
                        f"&current=wave_height,wave_direction,wave_period,ocean_current_velocity,sea_surface_temperature"
                    )
                    try:
                        fb_resp = await client.get(fallback_url)
                        if fb_resp.status_code == 200:
                            fb_curr = fb_resp.json().get("current", {})
                            wave_height = fb_curr.get("wave_height")
                            if sst is None:
                                sst = fb_curr.get("sea_surface_temperature")
                    except Exception:
                        pass

                return {
                    "wave_height": wave_height,
                    "wave_direction": current.get("wave_direction"),
                    "wave_period": current.get("wave_period"),
                    "current_velocity": current.get("ocean_current_velocity"),
                    "sst": sst,
                    "status": "success",
                    "source": "Open-Meteo Marine API"
                }
        except Exception as e:
            return {"status": "unavailable", "error": str(e), "source": "Open-Meteo Marine API"}

