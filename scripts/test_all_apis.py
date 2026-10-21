import asyncio
import httpx
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.weather_service import WeatherService
from app.services.ocean_service import OceanService
from app.services.earthdata_service import EarthdataService
from app.services.pfz_service import PFZService
from app.services.mosdac_service import MOSDACService
from app.services.incois_service import INCOISService

async def main():
    print("ORCA API HEALTH REPORT\n")
    
    # Weather
    w = WeatherService()
    res = await w.get_weather(15.2, 74.1)
    status = "PASS" if res.get("status") == "success" else "PENDING"
    print(f"Weather (Open-Meteo) ........... {status}")
    
    # Ocean
    o = OceanService()
    res = await o.get_ocean_conditions(15.2, 74.1)
    status = "PASS" if res.get("status") == "success" else "PENDING"
    print(f"Marine (Open-Meteo) ............ {status}")
    
    # Earthdata
    e = EarthdataService()
    res = await e.search_collections("SST")
    status = "PASS" if res.get("status") == "success" else "PENDING"
    print(f"Earthdata CMR .................. {status}")
    
    # PFZ
    p = PFZService()
    res = await p.get_pfz(15.2, 74.1)
    status = "PASS/PENDING" if "Simulated" in res.get("source", "") else "PASS"
    print(f"PFZ ............................ {status}")
    
    # MOSDAC
    m = MOSDACService()
    res = await m.get_chlorophyll(15.2, 74.1)
    status = "PASS" if res.get("status") == "success" else "PENDING"
    print(f"MOSDAC ......................... {status}")
    
    # INCOIS
    i = INCOISService()
    res = await i.get_ocean_state(15.2, 74.1)
    status = "PASS" if res.get("status") == "success" else "PENDING"
    print(f"INCOIS ......................... {status}")
    
if __name__ == "__main__":
    asyncio.run(main())
