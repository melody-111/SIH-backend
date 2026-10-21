import datetime
from typing import Dict, Any, List
from app.connectors.base_connector import BaseConnector, ConnectorResult
from app.schemas.marine_normalized import (
    NormalizedMarineRecord,
    DataType,
    QualityStatus,
    ConnectorStatus
)

class NASAEarthdataConnector(BaseConnector):
    """
    Connector for NASA Earthdata Common Metadata Repository (CMR) & OceanColor (https://oceandata.sci.gsfc.nasa.gov/).
    Provides collection-level satellite metadata discovery for SST (MODIS/VIIRS) and Chlorophyll.
    Status: METADATA_ONLY. Direct Level-3 HDF/NetCDF raster pixel ingestion requires Earthdata Login credentials.
    """
    def __init__(self):
        super().__init__(
            name="NASA Earthdata CMR",
            base_url="https://cmr.earthdata.nasa.gov/search/collections.json"
        )
        self.status = ConnectorStatus.METADATA_ONLY
        self.ocean_color_url = "https://oceandata.sci.gsfc.nasa.gov/"

    async def fetch_satellite_metadata(
        self,
        lat: float,
        lon: float,
        keyword: str = "MODIS SST"
    ) -> ConnectorResult:
        # CMR point query format: point=lon,lat
        params = {
            "keyword": keyword,
            "point": f"{lon},{lat}",
            "page_size": 3
        }
        res = await self.get_with_retry(self.base_url, params=params)
        res.connector_status = ConnectorStatus.METADATA_ONLY.value
        return res

    def normalize(
        self,
        raw_result: ConnectorResult,
        lat: float,
        lon: float,
        location_name: str = ""
    ) -> List[NormalizedMarineRecord]:
        retrieval_time = datetime.datetime.utcnow().isoformat() + "Z"
        records: List[NormalizedMarineRecord] = []

        if not raw_result.success or not raw_result.data:
            records.append(
                NormalizedMarineRecord(
                    latitude=lat,
                    longitude=lon,
                    location_name=location_name or None,
                    variable_name="satellite_observation_catalog",
                    value=None,
                    unit="metadata",
                    data_type=DataType.SATELLITE_CATALOG,
                    timestamp=retrieval_time,
                    retrieval_timestamp=retrieval_time,
                    source_provider="NASA Earthdata CMR",
                    quality_status=QualityStatus.UNAVAILABLE,
                    is_missing=True,
                    notes=raw_result.error or "No satellite collections discoverable"
                )
            )
            return records

        entries = raw_result.data.get("feed", {}).get("entry", [])
        if not entries:
            records.append(
                NormalizedMarineRecord(
                    latitude=lat,
                    longitude=lon,
                    location_name=location_name or None,
                    variable_name="satellite_observation_catalog",
                    value=0,
                    unit="count",
                    data_type=DataType.SATELLITE_CATALOG,
                    timestamp=retrieval_time,
                    retrieval_timestamp=retrieval_time,
                    source_provider="NASA Earthdata CMR",
                    quality_status=QualityStatus.VALID,
                    is_missing=False,
                    notes="No active satellite swath matching point at query time."
                )
            )
            return records

        for entry in entries:
            title = entry.get("title", "Unknown Satellite Dataset")
            time_start = entry.get("time_start", retrieval_time)
            short_name = entry.get("short_name", "MODIS")
            records.append(
                NormalizedMarineRecord(
                    latitude=lat,
                    longitude=lon,
                    location_name=location_name or None,
                    variable_name="satellite_collection",
                    value=title,
                    unit="dataset",
                    data_type=DataType.SATELLITE_CATALOG,
                    timestamp=time_start,
                    retrieval_timestamp=retrieval_time,
                    source_provider=f"NASA Earthdata ({short_name})",
                    quality_status=QualityStatus.VALID,
                    is_missing=False,
                    notes=f"ID: {entry.get('id', '')}"
                )
            )

        return records
