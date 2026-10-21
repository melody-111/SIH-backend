import datetime
import math
from typing import Dict, Any, Optional, Tuple, List
from app.connectors.open_meteo_weather import OpenMeteoWeatherConnector
from app.connectors.open_meteo_marine import OpenMeteoMarineConnector
from app.services.official_warning_provider import OfficialWarningProvider
from app.schemas.hazard_alert import (
    CacheSyncResponse,
    OfficialWarningDetail,
    AlertStatus
)

CATEGORY_TTL_HOURS = {
    "OBSERVATION": 3.0,
    "WEATHER_FORECAST": 12.0,
    "OCEAN_FORECAST": 12.0,
    "OFFICIAL_WARNINGS": 6.0,
    "SATELLITE_CATALOG": 24.0
}

class OfflineMarineCache:
    """
    Offline-first marine data store supporting 2-day past + 2-day forecast windows,
    per-category TTL policies, incremental sync, and nearest-location lookups.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OfflineMarineCache, cls).__new__(cls)
            from app.db.database import DatabaseManager
            cls._instance.db = DatabaseManager()
            cls._instance._cache: Dict[str, Dict[str, Any]] = {}
            cls._instance._force_offline: bool = False
            cls._instance._weather_connector = OpenMeteoWeatherConnector()
            cls._instance._marine_connector = OpenMeteoMarineConnector()
            cls._instance._warning_provider = OfficialWarningProvider()
        return cls._instance

    def set_force_offline(self, enabled: bool):
        """Simulates device in airplane mode or disconnected at sea."""
        self._force_offline = enabled

    @property
    def is_forced_offline(self) -> bool:
        return self._force_offline

    def _make_key(self, lat: float, lon: float) -> str:
        # 0.1 degree grid resolution (~11 km)
        grid_lat = round(lat, 1)
        grid_lon = round(lon, 1)
        return f"{grid_lat}:{grid_lon}"

    def _calculate_haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def store_observation(
        self,
        lat: float,
        lon: float,
        data: Dict[str, Any],
        location_name: str = ""
    ):
        """Backwards compatible observation cache storage with DB persistence."""
        key = self._make_key(lat, lon)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._cache[key] = {
            "latitude": lat,
            "longitude": lon,
            "location_name": location_name,
            "cached_at": now_iso,
            "sync_timestamp": now_iso,
            "data": data,
            "categories": {
                "OBSERVATION": {
                    "data": data,
                    "retrieval_time": now_iso,
                    "ttl_hours": CATEGORY_TTL_HOURS["OBSERVATION"]
                }
            }
        }
        try:
            self.db.store_marine_cache(
                grid_key=key,
                latitude=lat,
                longitude=lon,
                location_name=location_name,
                data=data,
                ttl_hours=CATEGORY_TTL_HOURS["OBSERVATION"]
            )
        except Exception:
            pass

    async def sync_cache_window(
        self,
        lat: float,
        lon: float,
        location_name: str = "",
        past_days: int = 2,
        forecast_days: int = 2,
        last_sync_timestamp: Optional[str] = None
    ) -> CacheSyncResponse:
        """
        Synchronizes 2-day past and 2-day forecast marine data into offline storage.
        Supports incremental sync checking and handles partial provider failure.
        """
        key = self._make_key(lat, lon)
        now = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now.isoformat()

        # Incremental sync check
        existing = self._cache.get(key)
        if existing and last_sync_timestamp:
            existing_sync = existing.get("sync_timestamp")
            if existing_sync:
                try:
                    last_sync_dt = datetime.datetime.fromisoformat(last_sync_timestamp.replace("Z", "+00:00"))
                    existing_dt = datetime.datetime.fromisoformat(existing_sync.replace("Z", "+00:00"))
                    # If existing cache is newer than client's last sync and less than 1 hour old, it's already up-to-date
                    age_minutes = (now - existing_dt).total_seconds() / 60.0
                    if existing_dt >= last_sync_dt and age_minutes < 60.0:
                        cached_warnings = existing.get("categories", {}).get("OFFICIAL_WARNINGS", {}).get("warnings", [])
                        return CacheSyncResponse(
                            latitude=lat,
                            longitude=lon,
                            sync_timestamp=existing_sync,
                            cached_window_past_days=existing.get("past_days", past_days),
                            cached_window_forecast_days=existing.get("forecast_days", forecast_days),
                            records_cached_count=existing.get("records_count", 0),
                            status="UP_TO_DATE",
                            is_cached=True,
                            freshness_age_minutes=round(age_minutes, 1),
                            active_warnings=[OfficialWarningDetail(**w) for w in cached_warnings]
                        )
                except Exception:
                    pass

        # Execute parallel data retrieval
        weather_res = await self._weather_connector.fetch_weather(lat, lon, past_days=past_days, forecast_days=forecast_days)
        marine_res = await self._marine_connector.fetch_marine(lat, lon, past_days=past_days, forecast_days=forecast_days)
        warnings = self._warning_provider.get_warnings_for_location(lat, lon, region_name=location_name, include_expired=False)

        total_records = 0
        categories: Dict[str, Any] = {}
        partial_failures: List[str] = []

        if weather_res.success and weather_res.data:
            hourly_w = weather_res.data.get("hourly", {})
            times_w = hourly_w.get("time", [])
            total_records += len(times_w)
            categories["WEATHER_FORECAST"] = {
                "data": weather_res.data,
                "retrieval_time": now_iso,
                "provider": "Open-Meteo Weather / ECMWF",
                "record_count": len(times_w),
                "ttl_hours": CATEGORY_TTL_HOURS["WEATHER_FORECAST"]
            }
        else:
            partial_failures.append("Weather provider unavailable")

        if marine_res.success and marine_res.data:
            hourly_m = marine_res.data.get("hourly", {})
            times_m = hourly_m.get("time", [])
            total_records += len(times_m)
            categories["OCEAN_FORECAST"] = {
                "data": marine_res.data,
                "retrieval_time": now_iso,
                "provider": "Open-Meteo Marine / Copernicus Marine Service",
                "record_count": len(times_m),
                "ttl_hours": CATEGORY_TTL_HOURS["OCEAN_FORECAST"]
            }
        else:
            partial_failures.append("Marine wave provider unavailable")

        # Official Warnings
        categories["OFFICIAL_WARNINGS"] = {
            "warnings": [w.model_dump() for w in warnings],
            "retrieval_time": now_iso,
            "provider": "Official Warning Provider (IMD/INCOIS)",
            "count": len(warnings),
            "ttl_hours": CATEGORY_TTL_HOURS["OFFICIAL_WARNINGS"]
        }
        total_records += len(warnings)

        # PFZ Status - strictly report unavailability if unauthenticated
        categories["PFZ"] = {
            "status": "UNAVAILABLE_UNAUTHENTICATED",
            "message": "INCOIS official PFZ data requires government authorization. No unverified estimates generated.",
            "retrieval_time": now_iso
        }

        # Determine overall sync status
        if weather_res.success and marine_res.success:
            sync_status = "FRESH"
        elif weather_res.success or marine_res.success:
            sync_status = "PARTIAL"
        else:
            sync_status = "FAILED"

        # Store in cache
        cache_entry = {
            "latitude": lat,
            "longitude": lon,
            "location_name": location_name,
            "cached_at": now_iso,
            "sync_timestamp": now_iso,
            "past_days": past_days,
            "forecast_days": forecast_days,
            "records_count": total_records,
            "status": sync_status,
            "partial_failures": partial_failures,
            "categories": categories,
            "data": {
                "weather": weather_res.data if weather_res.success else None,
                "marine": marine_res.data if marine_res.success else None,
                "warnings": [w.model_dump() for w in warnings]
            }
        }
        self._cache[key] = cache_entry
        try:
            self.db.store_marine_cache(
                grid_key=key,
                latitude=lat,
                longitude=lon,
                location_name=location_name,
                data=cache_entry["data"],
                ttl_hours=12.0
            )
            self.db.record_offline_sync(
                sync_id=f"sync_{now.timestamp()}",
                latitude=lat,
                longitude=lon,
                location_name=location_name,
                past_days=past_days,
                forecast_days=forecast_days,
                records_count=total_records,
                status=sync_status
            )
        except Exception:
            pass

        return CacheSyncResponse(
            latitude=lat,
            longitude=lon,
            sync_timestamp=now_iso,
            cached_window_past_days=past_days,
            cached_window_forecast_days=forecast_days,
            records_cached_count=total_records,
            status=sync_status,
            is_cached=True,
            freshness_age_minutes=0.0,
            active_warnings=warnings
        )

    def find_nearest_cache(
        self,
        lat: float,
        lon: float,
        max_distance_km: float = 75.0
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        """Finds closest cached observation or forecast within max_distance_km from memory or DB."""
        if not self._cache:
            try:
                db_nearest = self.db.find_nearest_marine_cache(lat, lon, max_distance_km)
                if db_nearest:
                    return db_nearest
            except Exception:
                pass
            return None

        best_entry = None
        best_dist = float("inf")

        for key, entry in self._cache.items():
            c_lat = entry["latitude"]
            c_lon = entry["longitude"]
            dist = self._calculate_haversine(lat, lon, c_lat, c_lon)

            if dist < best_dist and dist <= max_distance_km:
                best_dist = dist
                best_entry = entry

        if best_entry:
            return best_entry, round(best_dist, 1)
        return None

    def get_latest_observation(
        self,
        lat: float,
        lon: float,
        max_distance_km: float = 75.0
    ) -> Optional[Dict[str, Any]]:
        """Finds closest cached observation entry within max_distance_km."""
        nearest = self.find_nearest_cache(lat, lon, max_distance_km)
        if nearest:
            entry, dist = nearest
            return entry
        return None

    def get_offline_data(
        self,
        lat: float,
        lon: float,
        max_distance_km: float = 75.0
    ) -> Dict[str, Any]:
        """
        Retrieves offline cached data for a requested location.
        If no suitable cached data exists, clearly reports unavailable offline.
        """
        nearest = self.find_nearest_cache(lat, lon, max_distance_km)
        if not nearest:
            return {
                "status": "UNAVAILABLE_OFFLINE",
                "message": "No marine data available offline for this coastal location. Please connect to internet to download 48-hour forecasts.",
                "is_cached": False,
                "is_live": False,
                "latitude": lat,
                "longitude": lon
            }

        entry, dist = nearest
        now = datetime.datetime.now(datetime.timezone.utc)
        cached_at_str = entry.get("sync_timestamp", entry.get("cached_at"))
        age_minutes = 0.0
        try:
            cached_dt = datetime.datetime.fromisoformat(cached_at_str.replace("Z", "+00:00"))
            age_minutes = max(0.0, (now - cached_dt).total_seconds() / 60.0)
        except Exception:
            pass

        # Determine staleness: > 3 hours = stale
        is_stale = age_minutes > (CATEGORY_TTL_HOURS["OBSERVATION"] * 60.0)

        # Process cached warnings: mark expired if valid_until < now
        cached_warnings = entry.get("categories", {}).get("OFFICIAL_WARNINGS", {}).get("warnings", [])
        processed_warnings = []
        for w_dict in cached_warnings:
            w_obj = OfficialWarningDetail(**w_dict)
            if not self._warning_provider.is_warning_valid(w_obj, now):
                w_dict["status"] = AlertStatus.EXPIRED.value
            processed_warnings.append(w_dict)

        return {
            "status": "CACHED_STALE" if is_stale else "CACHED_FRESH",
            "is_cached": True,
            "is_live": False,
            "is_stale": is_stale,
            "distance_km": dist,
            "cached_at": cached_at_str,
            "freshness_age_minutes": round(age_minutes, 1),
            "location_name": entry.get("location_name"),
            "past_days_available": entry.get("past_days", 0),
            "forecast_days_available": entry.get("forecast_days", 0),
            "data": entry.get("data", {}),
            "warnings": processed_warnings,
            "notes": f"Offline data retrieved from cache (age: {age_minutes:.1f}m, distance: {dist}km)."
        }

    def clear(self):
        """Clears cache store for test isolation."""
        self._cache.clear()
        try:
            with self.db.transaction() as conn:
                conn.execute("DELETE FROM marine_cached_observations;")
        except Exception:
            pass
