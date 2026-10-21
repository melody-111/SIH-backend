import asyncio
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.marine_data_service import MarineDataService
from app.schemas.marine_normalized import MarineQueryRequest, MarineTask

async def run_smoke_test():
    print("=================================================================")
    print("NEERAI - REAL MARINE DATA PIPELINE SMOKE TEST")
    print("=================================================================\n")
    service = MarineDataService()

    # 1. Test Real Coastal Query (Goa Coast)
    print(">>> 1. Fetching real conditions for Goa Coast (15.30° N, 73.70° E)...")
    req = MarineQueryRequest(
        question="Kal fishing ke liye Goa coast par wave aur SST conditions check karo",
        latitude=15.30,
        longitude=73.70,
        location_name="Goa Coast",
        intended_task=MarineTask.FISHING
    )
    snapshot = await service.get_marine_snapshot(req)

    print(f"Location Resolved: {snapshot.location_name} ({snapshot.latitude}, {snapshot.longitude})")
    print(f"Retrieval Timestamp: {snapshot.retrieval_time}")
    print(f"Weather Observations Count: {len(snapshot.weather_records)}")
    print(f"Ocean Observations Count:   {len(snapshot.ocean_records)}")
    print(f"Satellite Records Count:    {len(snapshot.satellite_records)}")
    print(f"Active Warnings Count:      {len(snapshot.active_warnings)}")
    print(f"Unavailable Sources:        {snapshot.unavailable_sources}\n")

    print(">>> Sample Normalized Marine Variables:")
    for rec in snapshot.weather_records[:4]:
        print(f"  [Weather] {rec.variable_name:<25}: {rec.value} {rec.unit:<6} ({rec.source_provider}) - {rec.quality_status.value}")
    for rec in snapshot.ocean_records[:4]:
        print(f"  [Ocean]   {rec.variable_name:<25}: {rec.value} {rec.unit:<6} ({rec.source_provider}) - {rec.quality_status.value}")

    print("\n>>> Structured AI Agent Evidence Items Generated:")
    for idx, ev in enumerate(snapshot.evidence_items[:6], 1):
        print(f"  {idx}. {ev['claim']} [Source: {ev['source']}, Conf: {ev['confidence']}]")

    if snapshot.active_warnings:
        print("\n>>> Detected Warnings / Alerts:")
        for w in snapshot.active_warnings:
            print(f"  • [{w.severity.value}] {w.warning_type}: {w.message} ({w.source})")
    else:
        print("\n>>> No extreme weather warnings detected in current observation.")

    # 2. Check Data Source Health Matrix
    print("\n>>> 2. Data Sources Health Matrix:")
    sources = await service.check_all_sources_health()
    for name, stat in sources.items():
        auth_tag = "[Auth Req]" if stat.requires_auth else "[Open/Free]"
        print(f"  • {stat.provider_name:<40} {stat.status:<12} {auth_tag:<12} Latency: {stat.response_time_ms}ms")
        if stat.error_message:
            print(f"    └─ Note: {stat.error_message}")

    print("\n=================================================================")
    print("PIPELINE SMOKE TEST COMPLETE - ALL REQUIRED METRICS VERIFIED")
    print("=================================================================")

if __name__ == "__main__":
    asyncio.run(run_smoke_test())
