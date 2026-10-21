import datetime
from typing import Dict, Any, List, Tuple, Optional
from app.connectors.base_connector import BaseConnector, ConnectorResult
from app.schemas.marine_normalized import (
    NormalizedMarineRecord,
    MarineWarningRecord,
    DataType,
    QualityStatus,
    WarningSeverity,
    ConnectorStatus
)

class OpenMeteoMarineConnector(BaseConnector):
    def __init__(self):
        super().__init__(
            name="Open-Meteo Marine",
            base_url="https://marine-api.open-meteo.com/v1/marine"
        )

    async def fetch_marine(
        self,
        lat: float,
        lon: float,
        past_days: int = 0,
        forecast_days: int = 2
    ) -> ConnectorResult:
        current_vars = [
            "wave_height",
            "wave_direction",
            "wave_period",
            "wind_wave_height",
            "wind_wave_direction",
            "wind_wave_period",
            "swell_wave_height",
            "swell_wave_direction",
            "swell_wave_period",
            "ocean_current_velocity",
            "ocean_current_direction",
            "sea_surface_temperature"
        ]
        hourly_vars = [
            "wave_height",
            "wave_direction",
            "wave_period",
            "swell_wave_height",
            "ocean_current_velocity",
            "sea_surface_temperature"
        ]

        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ",".join(current_vars),
            "hourly": ",".join(hourly_vars),
            "timezone": "UTC",
            "past_days": past_days,
            "forecast_days": forecast_days
        }

        result = await self.get_with_retry(self.base_url, params=params)
        
        # Intelligent Coastal Land-Boundary Handling:
        # If the coordinate is on land/harbor, wave_height is returned as None by Copernicus models.
        # If so, probe adjacent offshore candidate points (e.g. stepping slightly seaward).
        if result.success and result.data:
            current = result.data.get("current", {})
            if current.get("wave_height") is None:
                # Determine seaward direction based on Indian coastlines
                # West coast of India (lon < 77.5): water is to the West (negative lon)
                # East coast of India (lon >= 77.5): water is to the East (positive lon)
                lon_step = -0.08 if lon < 77.5 else 0.08
                offshore_lon = round(lon + lon_step, 4)
                
                offshore_params = dict(params)
                offshore_params["longitude"] = offshore_lon
                offshore_result = await self.get_with_retry(self.base_url, params=offshore_params)
                
                if offshore_result.success and offshore_result.data:
                    off_curr = offshore_result.data.get("current", {})
                    if off_curr.get("wave_height") is not None:
                        # Attach offshore metadata so normalizer can document attribution honestly
                        result.data["_offshore_fallback"] = {
                            "latitude": lat,
                            "longitude": offshore_lon,
                            "current": off_curr,
                            "hourly": offshore_result.data.get("hourly")
                        }

        if result.success:
            result.connector_status = ConnectorStatus.LIVE_VERIFIED.value

        return result

    def normalize(
        self,
        raw_result: ConnectorResult,
        lat: float,
        lon: float,
        location_name: str = ""
    ) -> Tuple[List[NormalizedMarineRecord], List[MarineWarningRecord]]:
        if not raw_result.success or not raw_result.data:
            return [], []

        data = raw_result.data
        retrieval_time = datetime.datetime.utcnow().isoformat() + "Z"
        records: List[NormalizedMarineRecord] = []
        warnings: List[MarineWarningRecord] = []

        current = data.get("current", {})
        curr_time = current.get("time") or retrieval_time
        if not curr_time.endswith("Z") and not "+" in curr_time:
            curr_time += ":00Z" if len(curr_time) == 16 else "Z"

        has_offshore_fallback = "_offshore_fallback" in data
        offshore_info = data.get("_offshore_fallback", {})

        var_specs = [
            ("wave_height", "significant_wave_height", "m"),
            ("wave_direction", "wave_direction", "°"),
            ("wave_period", "wave_period", "s"),
            ("swell_wave_height", "swell_wave_height", "m"),
            ("swell_wave_period", "swell_wave_period", "s"),
            ("ocean_current_velocity", "ocean_current_velocity", "km/h"),
            ("ocean_current_direction", "ocean_current_direction", "°"),
            ("sea_surface_temperature", "sea_surface_temperature", "°C")
        ]

        for api_key, std_name, unit in var_specs:
            val = current.get(api_key)
            note = None
            is_derived = False

            if val is None and has_offshore_fallback:
                val = offshore_info.get("current", {}).get(api_key)
                if val is not None:
                    is_derived = True
                    note = f"Observed from nearest maritime grid ({lat}, {offshore_info['longitude']}) because land-point model cell was null."

            records.append(
                NormalizedMarineRecord(
                    latitude=lat,
                    longitude=lon,
                    location_name=location_name or None,
                    variable_name=std_name,
                    value=val,
                    unit=unit,
                    data_type=DataType.DERIVED_ESTIMATE if is_derived else DataType.OBSERVATION,
                    timestamp=curr_time,
                    retrieval_timestamp=retrieval_time,
                    source_provider="Open-Meteo Marine (Copernicus/ECMWF)",
                    forecast_horizon_hours=0.0,
                    quality_status=QualityStatus.VALID if val is not None else QualityStatus.UNAVAILABLE,
                    is_missing=(val is None),
                    notes=note
                )
            )

        # 2. Derive Marine Warnings from Real Ocean Data
        wave_height = current.get("wave_height")
        if wave_height is None and has_offshore_fallback:
            wave_height = offshore_info.get("current", {}).get("wave_height")

        current_vel = current.get("ocean_current_velocity")
        if current_vel is None and has_offshore_fallback:
            current_vel = offshore_info.get("current", {}).get("ocean_current_velocity")

        if wave_height is not None:
            if wave_height >= 3.5:
                warnings.append(
                    MarineWarningRecord(
                        warning_type="ROUGH_SEAS_DANGER",
                        severity=WarningSeverity.DANGER,
                        message=f"Dangerous sea conditions: Significant wave height observed at {wave_height} m. Do not venture into open sea.",
                        source="Open-Meteo Marine Model",
                        is_official=False,
                        valid_from=curr_time,
                        affected_coordinates={"latitude": lat, "longitude": lon}
                    )
                )
            elif wave_height >= 2.0:
                warnings.append(
                    MarineWarningRecord(
                        warning_type="HIGH_WAVE_ADVISORY",
                        severity=WarningSeverity.WARNING,
                        message=f"High wave advisory: Wave height is {wave_height} m. Small fishing boats should exercise caution.",
                        source="Open-Meteo Marine Model",
                        is_official=False,
                        valid_from=curr_time,
                        affected_coordinates={"latitude": lat, "longitude": lon}
                    )
                )

        if current_vel is not None and current_vel >= 3.5:
            warnings.append(
                MarineWarningRecord(
                    warning_type="STRONG_OCEAN_CURRENT",
                    severity=WarningSeverity.ADVISORY,
                    message=f"Strong surface current: {current_vel} km/h detected in area.",
                    source="Open-Meteo Marine Model",
                    is_official=False,
                    valid_from=curr_time,
                    affected_coordinates={"latitude": lat, "longitude": lon}
                )
            )

        return records, warnings
