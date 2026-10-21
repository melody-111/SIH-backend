import asyncio
import datetime
from typing import Dict, Any, List, Optional, Tuple

from app.connectors.open_meteo_weather import OpenMeteoWeatherConnector
from app.connectors.open_meteo_marine import OpenMeteoMarineConnector
from app.connectors.nasa_earthdata import NASAEarthdataConnector
from app.connectors.incois_connector import INCOISConnector
from app.connectors.imd_connector import IMDWarningsConnector
from app.schemas.marine_normalized import (
    MarineQueryRequest,
    NormalizedMarineSnapshot,
    NormalizedMarineRecord,
    MarineWarningRecord,
    DataSourceStatus,
    MarineTask,
    DataType,
    QualityStatus
)

# Reference catalog of major Indian maritime hubs and coastal ports
KNOWN_COASTAL_LOCATIONS = {
    "mumbai": {"lat": 18.9220, "lon": 72.8347, "name": "Mumbai Port, Maharashtra"},
    "goa": {"lat": 15.3990, "lon": 73.7990, "name": "Mormugao / Panaji Coast, Goa"},
    "panaji": {"lat": 15.4989, "lon": 73.8278, "name": "Panaji Coast, Goa"},
    "kochi": {"lat": 9.9312, "lon": 76.2673, "name": "Kochi Harbor, Kerala"},
    "cochin": {"lat": 9.9312, "lon": 76.2673, "name": "Kochi Harbor, Kerala"},
    "chennai": {"lat": 13.0827, "lon": 80.2707, "name": "Chennai Port, Tamil Nadu"},
    "visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "name": "Visakhapatnam Port, Andhra Pradesh"},
    "vizag": {"lat": 17.6868, "lon": 83.2185, "name": "Visakhapatnam Port, Andhra Pradesh"},
    "porbandar": {"lat": 21.6417, "lon": 69.6293, "name": "Porbandar Coast, Gujarat"},
    "veraval": {"lat": 20.9000, "lon": 70.3667, "name": "Veraval Fishing Port, Gujarat"},
    "mangalore": {"lat": 12.9141, "lon": 74.8560, "name": "New Mangalore Port, Karnataka"},
    "digha": {"lat": 21.6266, "lon": 87.5074, "name": "Digha Coast, West Bengal"},
    "kanyakumari": {"lat": 8.0883, "lon": 77.5385, "name": "Cape Comorin / Kanyakumari, Tamil Nadu"}
}

class MarineDataService:
    """
    Central service for query-driven marine data retrieval, normalization,
    validation, and evidence generation for NeerAI.
    """
    def __init__(self):
        self.weather_connector = OpenMeteoWeatherConnector()
        self.marine_connector = OpenMeteoMarineConnector()
        self.satellite_connector = NASAEarthdataConnector()
        self.incois_connector = INCOISConnector()
        self.imd_connector = IMDWarningsConnector()

    def resolve_location(
        self,
        query: MarineQueryRequest
    ) -> Tuple[float, float, str, bool]:
        """
        Resolves location from query text, explicit location_name, or coordinates.
        Does not silently overwrite explicitly requested locations with phone coordinates.
        """
        question_text = (query.question or "").lower()
        explicit_loc = (query.location_name or "").lower().strip()

        # Check if an explicit coastal hub was named
        detected_hub = None
        for hub_key, hub_data in KNOWN_COASTAL_LOCATIONS.items():
            if hub_key in explicit_loc or hub_key in question_text:
                detected_hub = hub_data
                break

        if detected_hub:
            # If coordinates were provided but a different named location was explicitly requested
            is_ambiguous = False
            if query.latitude and query.longitude:
                # If distance is large (> 0.5 degrees), user asked about a different location than current device location
                lat_diff = abs(query.latitude - detected_hub["lat"])
                lon_diff = abs(query.longitude - detected_hub["lon"])
                if lat_diff > 0.5 or lon_diff > 0.5:
                    is_ambiguous = True

            return detected_hub["lat"], detected_hub["lon"], detected_hub["name"], is_ambiguous

        # Fallback to provided coordinates
        if query.latitude is not None and query.longitude is not None:
            return query.latitude, query.longitude, query.location_name or "Specified Coordinates", False

        # Default fallback to central Indian west coast (Goa) if nothing provided, flagged ambiguous
        return 15.30, 73.70, "Arabian Sea (Default Indian Coast)", True

    def infer_task(self, query: MarineQueryRequest) -> MarineTask:
        if query.intended_task != MarineTask.GENERAL:
            return query.intended_task

        text = (query.question or "").lower()
        if any(w in text for w in ["cyclone", "hazard", "alert", "warning", "toofan", "khatra", "storm"]):
            return MarineTask.HAZARDS_WARNINGS
        if any(w in text for w in ["fish", "machhli", "pfz", "shikaar", "catch", "zone", "jagah"]):
            return MarineTask.FISHING
        if any(w in text for w in ["wave", "lehrein", "wind", "hawa", "weather", "mausam", "rain", "baarish"]):
            return MarineTask.MARINE_WEATHER
        if any(w in text for w in ["route", "rasta", "navigate", "safe path", "distance"]):
            return MarineTask.NAVIGATION_ROUTE
        return MarineTask.GENERAL

    async def get_marine_snapshot(
        self,
        query: MarineQueryRequest
    ) -> NormalizedMarineSnapshot:
        lat, lon, loc_name, is_ambiguous = self.resolve_location(query)
        task = self.infer_task(query)
        retrieval_time = datetime.datetime.utcnow().isoformat() + "Z"
        target_time = query.start_time or retrieval_time

        weather_records: List[NormalizedMarineRecord] = []
        ocean_records: List[NormalizedMarineRecord] = []
        satellite_records: List[NormalizedMarineRecord] = []
        advisory_records: List[NormalizedMarineRecord] = []
        active_warnings: List[MarineWarningRecord] = []
        source_statuses: Dict[str, DataSourceStatus] = {}
        unavailable_sources: List[str] = []

        # Selective, query-driven task dispatching
        tasks_to_run = []
        run_weather = True  # Weather wind/pressure is foundational for all marine safety
        run_marine = True   # Ocean waves/currents is foundational for all maritime tasks
        run_satellite = task in [MarineTask.FISHING, MarineTask.GENERAL]
        run_incois = task in [MarineTask.FISHING, MarineTask.GENERAL]
        run_imd = task in [MarineTask.HAZARDS_WARNINGS, MarineTask.GENERAL]

        # 1. Fetch Weather
        weather_coro = self.weather_connector.fetch_weather(lat, lon, past_days=query.past_days, forecast_days=query.forecast_days)
        # 2. Fetch Marine
        marine_coro = self.marine_connector.fetch_marine(lat, lon, past_days=query.past_days, forecast_days=query.forecast_days)

        dispatch_map = {
            "weather": weather_coro,
            "marine": marine_coro
        }

        if run_satellite:
            dispatch_map["satellite"] = self.satellite_connector.fetch_satellite_metadata(lat, lon)
        if run_incois:
            dispatch_map["incois"] = self.incois_connector.fetch_advisory(lat, lon)
        if run_imd:
            dispatch_map["imd"] = self.imd_connector.fetch_warnings(lat, lon)

        # Execute parallel requests
        keys = list(dispatch_map.keys())
        raw_results = await asyncio.gather(*[dispatch_map[k] for k in keys], return_exceptions=True)

        for key, res in zip(keys, raw_results):
            if isinstance(res, Exception):
                unavailable_sources.append(key)
                source_statuses[key] = DataSourceStatus(
                    provider_name=key,
                    endpoint="",
                    status="ERROR",
                    last_checked=retrieval_time,
                    error_message=str(res)
                )
                continue

            # Record provider status
            source_statuses[key] = DataSourceStatus(
                provider_name=res.source,
                endpoint="",
                status="OPERATIONAL" if res.success else "UNAVAILABLE",
                requires_auth=("requires" in (res.error or "").lower() or "unconfigured" in (res.error or "").lower()),
                is_authenticated=res.success,
                last_checked=retrieval_time,
                response_time_ms=res.latency_ms,
                error_message=res.error
            )

            if not res.success:
                unavailable_sources.append(res.source)

            # Normalization per connector
            if key == "weather":
                w_recs, w_warns = self.weather_connector.normalize(res, lat, lon, loc_name)
                weather_records.extend(w_recs)
                active_warnings.extend(w_warns)
            elif key == "marine":
                m_recs, m_warns = self.marine_connector.normalize(res, lat, lon, loc_name)
                ocean_records.extend(m_recs)
                active_warnings.extend(m_warns)
            elif key == "satellite":
                s_recs = self.satellite_connector.normalize(res, lat, lon, loc_name)
                satellite_records.extend(s_recs)
            elif key == "incois":
                i_recs = self.incois_connector.normalize(res, lat, lon, loc_name)
                advisory_records.extend(i_recs)
            elif key == "imd":
                imd_warns = self.imd_connector.normalize(res, lat, lon)
                active_warnings.extend(imd_warns)

        # Construct Agent Evidence Items
        evidence_items = []
        for r in weather_records + ocean_records:
            if not r.is_missing and r.value is not None:
                evidence_items.append({
                    "claim": f"{r.variable_name.replace('_', ' ').capitalize()} is {r.value} {r.unit}".strip(),
                    "variable": r.variable_name,
                    "value": r.value,
                    "unit": r.unit,
                    "source": r.source_provider,
                    "data_type": r.data_type.value,
                    "timestamp": r.timestamp,
                    "confidence": 0.95 if r.quality_status == QualityStatus.VALID else 0.70
                })

        for sat in satellite_records:
            if not sat.is_missing and sat.value is not None:
                evidence_items.append({
                    "claim": f"Satellite data available: {sat.value}",
                    "variable": sat.variable_name,
                    "value": sat.value,
                    "unit": sat.unit,
                    "source": sat.source_provider,
                    "data_type": sat.data_type.value,
                    "timestamp": sat.timestamp,
                    "confidence": 0.90
                })

        return NormalizedMarineSnapshot(
            latitude=lat,
            longitude=lon,
            location_name=loc_name,
            is_location_ambiguous=is_ambiguous,
            target_time=target_time,
            retrieval_time=retrieval_time,
            weather_records=weather_records,
            ocean_records=ocean_records,
            satellite_records=satellite_records,
            advisory_records=advisory_records,
            active_warnings=active_warnings,
            source_statuses=source_statuses,
            evidence_items=evidence_items,
            unavailable_sources=unavailable_sources
        )

    async def check_all_sources_health(self) -> Dict[str, DataSourceStatus]:
        """
        Probes live health and availability of all external marine sources.
        """
        now = datetime.datetime.utcnow().isoformat() + "Z"
        test_lat, test_lon = 15.30, 73.70
        statuses: Dict[str, DataSourceStatus] = {}

        # 1. Weather
        w_res = await self.weather_connector.fetch_weather(test_lat, test_lon, forecast_days=1)
        statuses["open_meteo_weather"] = DataSourceStatus(
            provider_name="Open-Meteo Weather API",
            endpoint=self.weather_connector.base_url,
            status="OPERATIONAL" if w_res.success else "DEGRADED",
            requires_auth=False,
            is_authenticated=True,
            last_checked=now,
            response_time_ms=w_res.latency_ms,
            error_message=w_res.error
        )

        # 2. Marine
        m_res = await self.marine_connector.fetch_marine(test_lat, test_lon, forecast_days=1)
        statuses["open_meteo_marine"] = DataSourceStatus(
            provider_name="Open-Meteo Marine API (Copernicus/ECMWF)",
            endpoint=self.marine_connector.base_url,
            status="OPERATIONAL" if m_res.success else "DEGRADED",
            requires_auth=False,
            is_authenticated=True,
            last_checked=now,
            response_time_ms=m_res.latency_ms,
            error_message=m_res.error
        )

        # 3. Earthdata
        e_res = await self.satellite_connector.fetch_satellite_metadata(test_lat, test_lon)
        statuses["nasa_earthdata_cmr"] = DataSourceStatus(
            provider_name="NASA Earthdata CMR",
            endpoint=self.satellite_connector.base_url,
            status="OPERATIONAL" if e_res.success else "DEGRADED",
            requires_auth=False,
            is_authenticated=True,
            last_checked=now,
            response_time_ms=e_res.latency_ms,
            error_message=e_res.error
        )

        # 4. INCOIS
        i_res = await self.incois_connector.fetch_advisory(test_lat, test_lon)
        statuses["incois"] = DataSourceStatus(
            provider_name=self.incois_connector.name,
            endpoint="https://incois.gov.in",
            status="SIMULATED" if i_res.success else "REQUIRES_AUTH",
            requires_auth=True,
            is_authenticated=i_res.success,
            last_checked=now,
            response_time_ms=i_res.latency_ms,
            error_message=i_res.error
        )

        # 5. IMD
        imd_res = await self.imd_connector.fetch_warnings(test_lat, test_lon)
        statuses["imd"] = DataSourceStatus(
            provider_name=self.imd_connector.name,
            endpoint="https://mausam.imd.gov.in",
            status="SIMULATED" if imd_res.success else "REQUIRES_AUTH",
            requires_auth=True,
            is_authenticated=imd_res.success,
            last_checked=now,
            response_time_ms=imd_res.latency_ms,
            error_message=imd_res.error
        )

        return statuses
