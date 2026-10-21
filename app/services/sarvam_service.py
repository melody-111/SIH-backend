from typing import Dict, Any

class SarvamService:
    async def translate(self, text: str, target_lang: str) -> Dict[str, Any]:
        return {
            "status": "PENDING_VERIFICATION",
            "error": "Requires SARVAM_API_KEY",
            "source": "Sarvam AI"
        }
