import datetime
from typing import Dict, Any, List, Tuple
from app.connectors.base_connector import BaseConnector, ConnectorResult
from app.schemas.marine_normalized import (
    NormalizedMarineRecord,
    MarineWarningRecord,
    DataType,
    QualityStatus,
    WarningSeverity,
    ConnectorStatus
)

WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

class OpenMeteoWeatherConnector(BaseConnector):
    def __init__(self):
        super().__init__(
            name="Open-Meteo Weather",
            base_url="https://api.open-meteo.com/v1/forecast"
        )

    async def fetch_weather(
        self,
        lat: float,
        lon: float,
        past_days: int = 0,
        forecast_days: int = 2
    ) -> ConnectorResult:
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
        hourly_vars = [
            "temperature_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "precipitation",
            "weather_code",
            "surface_pressure"
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

        res = await self.get_with_retry(self.base_url, params=params)
        if res.success:
            res.connector_status = ConnectorStatus.LIVE_VERIFIED.value
        return res

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

        # 1. Normalize Current Observations
        var_specs = [
            ("temperature_2m", "air_temperature", "°C"),
            ("relative_humidity_2m", "relative_humidity", "%"),
            ("wind_speed_10m", "wind_speed", "km/h"),
            ("wind_direction_10m", "wind_direction", "°"),
            ("wind_gusts_10m", "wind_gusts", "km/h"),
            ("precipitation", "precipitation", "mm"),
            ("surface_pressure", "surface_pressure", "hPa")
        ]

        for api_key, std_name, unit in var_specs:
            val = current.get(api_key)
            records.append(
                NormalizedMarineRecord(
                    latitude=lat,
                    longitude=lon,
                    location_name=location_name or None,
                    variable_name=std_name,
                    value=val,
                    unit=unit,
                    data_type=DataType.OBSERVATION,
                    timestamp=curr_time,
                    retrieval_timestamp=retrieval_time,
                    source_provider="Open-Meteo Weather API",
                    forecast_horizon_hours=0.0,
                    quality_status=QualityStatus.VALID if val is not None else QualityStatus.UNAVAILABLE,
                    is_missing=(val is None)
                )
            )

        # Weather condition description
        w_code = current.get("weather_code")
        cond_text = WMO_WEATHER_CODES.get(w_code, f"Code {w_code}" if w_code is not None else None)
        records.append(
            NormalizedMarineRecord(
                latitude=lat,
                longitude=lon,
                location_name=location_name or None,
                variable_name="weather_condition",
                value=cond_text,
                unit="text",
                data_type=DataType.OBSERVATION,
                timestamp=curr_time,
                retrieval_timestamp=retrieval_time,
                source_provider="Open-Meteo Weather API",
                forecast_horizon_hours=0.0,
                quality_status=QualityStatus.VALID if cond_text is not None else QualityStatus.UNAVAILABLE,
                is_missing=(cond_text is None),
                notes=f"WMO Code: {w_code}" if w_code is not None else None
            )
        )

        # 2. Derive Meteorological Warnings from Real Data
        wind_speed = current.get("wind_speed_10m")
        wind_gusts = current.get("wind_gusts_10m")
        surface_pressure = current.get("surface_pressure")

        if wind_speed is not None and wind_speed >= 45.0:
            warnings.append(
                MarineWarningRecord(
                    warning_type="GALE_WIND_WARNING",
                    severity=WarningSeverity.DANGER if wind_speed > 60.0 else WarningSeverity.WARNING,
                    message=f"Severe maritime gale warning: Wind speed observed at {wind_speed} km/h.",
                    source="Open-Meteo Observations",
                    is_official=False,
                    valid_from=curr_time,
                    affected_coordinates={"latitude": lat, "longitude": lon}
                )
            )
        elif wind_speed is not None and wind_speed >= 30.0:
            warnings.append(
                MarineWarningRecord(
                    warning_type="HIGH_WIND_ADVISORY",
                    severity=WarningSeverity.ADVISORY,
                    message=f"Moderate-to-high wind advisory: Wind speed is {wind_speed} km/h. Small craft exercise caution.",
                    source="Open-Meteo Observations",
                    is_official=False,
                    valid_from=curr_time,
                    affected_coordinates={"latitude": lat, "longitude": lon}
                )
            )

        if wind_gusts is not None and wind_gusts >= 55.0:
            warnings.append(
                MarineWarningRecord(
                    warning_type="SQUALL_GUST_ALERT",
                    severity=WarningSeverity.WARNING,
                    message=f"Dangerous squalls detected: Wind gusts reaching {wind_gusts} km/h.",
                    source="Open-Meteo Observations",
                    is_official=False,
                    valid_from=curr_time,
                    affected_coordinates={"latitude": lat, "longitude": lon}
                )
            )

        if w_code in [95, 96, 99]:
            warnings.append(
                MarineWarningRecord(
                    warning_type="THUNDERSTORM_ALERT",
                    severity=WarningSeverity.DANGER if w_code in [96, 99] else WarningSeverity.WARNING,
                    message=f"Active marine thunderstorm: {WMO_WEATHER_CODES.get(w_code)}. High risk of lightning and sudden squalls.",
                    source="Open-Meteo Observations",
                    is_official=False,
                    valid_from=curr_time,
                    affected_coordinates={"latitude": lat, "longitude": lon}
                )
            )

        if surface_pressure is not None and surface_pressure < 998.0:
            warnings.append(
                MarineWarningRecord(
                    warning_type="BAROMETRIC_DEPRESSION",
                    severity=WarningSeverity.WARNING,
                    message=f"Barometric pressure drop detected ({surface_pressure} hPa), indicating potential cyclonic disturbance.",
                    source="Open-Meteo Observations",
                    is_official=False,
                    valid_from=curr_time,
                    affected_coordinates={"latitude": lat, "longitude": lon}
                )
            )

        return records, warnings
