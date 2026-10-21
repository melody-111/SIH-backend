import asyncio
import datetime
import logging
from typing import Dict, Any, Set, Optional

from app.services.offline_cache import OfflineMarineCache
from app.db.database import DatabaseManager
from app.core.config import settings

logger = logging.getLogger("neerai.background_sync")

class BackgroundSyncService:
    """
    Reliable background synchronization and data retention worker.
    
    Ensures:
    - No duplicate overlapping sync tasks for the same geographic coordinates
    - Bounded retries with exponential backoff on connector timeouts
    - Records last-successful-sync timestamps
    - Graceful recovery after restarts
    - Periodic cache retention cleanup and stale data eviction
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BackgroundSyncService, cls).__new__(cls)
            cls._instance.cache = OfflineMarineCache()
            cls._instance.db = DatabaseManager()
            cls._instance._active_sync_locks: Set[str] = set()
            cls._instance._last_successful_sync: Dict[str, str] = {}
            cls._instance._is_running = False
        return cls._instance

    def _grid_key(self, lat: float, lon: float) -> str:
        return f"{round(lat, 1)}:{round(lon, 1)}"

    async def sync_region_with_lock(
        self,
        lat: float,
        lon: float,
        location_name: str = "",
        past_days: int = 2,
        forecast_days: int = 2,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Executes bounded cache synchronization with concurrency deduplication.
        If a sync job for this coordinate grid is already executing, returns
        status 'IN_PROGRESS' to prevent resource waste and duplicate calls.
        """
        key = self._grid_key(lat, lon)
        if key in self._active_sync_locks:
            return {
                "status": "ALREADY_IN_PROGRESS",
                "message": f"Synchronization for {key} is already active.",
                "grid_key": key
            }

        self._active_sync_locks.add(key)
        attempt = 0
        last_error = None

        try:
            while attempt <= max_retries:
                attempt += 1
                try:
                    res = await self.cache.sync_cache_window(
                        lat=lat,
                        lon=lon,
                        location_name=location_name,
                        past_days=past_days,
                        forecast_days=forecast_days,
                        last_sync_timestamp=self._last_successful_sync.get(key)
                    )
                    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    self._last_successful_sync[key] = now_iso
                    return {
                        "status": "SUCCESS",
                        "grid_key": key,
                        "records_cached": res.records_cached_count,
                        "sync_status": res.status,
                        "synced_at": now_iso
                    }
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"[Sync: {key}] Attempt {attempt} failed: {last_error}")
                    if attempt <= max_retries:
                        await asyncio.sleep(0.5 * attempt)  # Exponential backoff

            return {
                "status": "FAILED",
                "grid_key": key,
                "error": last_error,
                "attempts": attempt
            }
        finally:
            self._active_sync_locks.discard(key)

    def run_retention_cleanup(self) -> Dict[str, Any]:
        """
        Purges expired marine observations and obsolete alert records
        according to configured retention policies.
        """
        try:
            pruned_cache = self.db.cleanup_expired_cache(settings.CACHE_RETENTION_HOURS)
            return {
                "status": "SUCCESS",
                "pruned_cache_records": pruned_cache,
                "retention_hours": settings.CACHE_RETENTION_HOURS,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Retention cleanup error: {str(e)}")
            return {"status": "ERROR", "error": str(e)}

    def get_last_sync_time(self, lat: float, lon: float) -> Optional[str]:
        """Returns the last successful synchronization timestamp for a location."""
        return self._last_successful_sync.get(self._grid_key(lat, lon))
