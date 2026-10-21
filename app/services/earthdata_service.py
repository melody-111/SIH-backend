import httpx
from typing import Dict, Any

class EarthdataService:
    async def search_collections(self, keyword: str) -> Dict[str, Any]:
        # CMR API requires no auth for basic search
        url = f"https://cmr.earthdata.nasa.gov/search/collections.json?keyword={keyword}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                return {
                    "status": "success",
                    "data": data.get("feed", {}).get("entry", []),
                    "source": "NASA Earthdata CMR"
                }
        except Exception as e:
            return {"status": "unavailable", "error": str(e), "source": "NASA Earthdata CMR"}
