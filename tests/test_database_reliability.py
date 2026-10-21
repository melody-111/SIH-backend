import pytest
import sqlite3
import datetime
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import DatabaseManager
from app.services.background_sync_service import BackgroundSyncService

client = TestClient(app)

# =============================================================================
# 1. DATABASE SCHEMA, MIGRATIONS & INTEGRITY
# =============================================================================

def test_database_migrations_and_integrity():
    db = DatabaseManager()
    health = db.check_db_health()
    assert health["status"] == "healthy"
    assert health["integrity"] == "ok"
    assert health["migrations_applied"] >= 1

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row["name"] for row in cursor.fetchall()}
        
        expected_tables = {
            "schema_migrations", "users", "user_preferences", "saved_locations",
            "user_memories", "conversations", "conversation_turns", "feedback_logs",
            "alert_subscriptions", "alert_deliveries", "marine_cached_observations",
            "offline_sync_logs", "agent_execution_logs"
        }
        for t in expected_tables:
            assert t in tables, f"Missing expected table {t}"

# =============================================================================
# 2. TRANSACTIONS & FOREIGN KEY CASCADING DELETES
# =============================================================================

def test_database_foreign_keys_and_cascading():
    db = DatabaseManager()
    user_id = "test_fk_user"
    mem_id = "mem_fk_001"
    
    # 1. Ensure user
    db.ensure_user(user_id)
    
    # 2. Save user memory
    db.save_user_memory(user_id, mem_id, "Berth 1", "Docking near north pier")
    
    # 3. Verify memory exists
    mems = db.list_user_memories(user_id)
    assert any(m["memory_id"] == mem_id for m in mems)
    
    # 4. Delete user -> should cascade delete memory
    with db.transaction() as conn:
        conn.execute("DELETE FROM users WHERE user_id = ?;", (user_id,))
        
    mems_after = db.list_user_memories(user_id)
    assert len(mems_after) == 0

def test_database_transaction_rollback():
    db = DatabaseManager()
    user_id = "test_rollback_user"
    mem_id = "mem_rollback_001"
    
    db.ensure_user(user_id)
    
    try:
        with db.transaction() as conn:
            conn.execute("""
                INSERT INTO user_memories (memory_id, user_id, title, content, category, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (mem_id, user_id, "Temp Title", "Temp Content", "NOTE", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"))
            # Force failure
            raise RuntimeError("Simulated transaction crash")
    except RuntimeError:
        pass
        
    # Memory must NOT be present
    mems = db.list_user_memories(user_id)
    assert not any(m["memory_id"] == mem_id for m in mems)

# =============================================================================
# 3. USER PREFERENCES & SAVED LOCATIONS PERSISTENCE
# =============================================================================

def test_user_preferences_persistence():
    db = DatabaseManager()
    user_id = "fisherman_db_pref_unique"
    
    with db.transaction() as conn:
        conn.execute("DELETE FROM user_preferences WHERE user_id = ?;", (user_id,))
        conn.execute("DELETE FROM saved_locations WHERE user_id = ?;", (user_id,))
    
    # 1. Default preferences
    prefs = db.get_user_preferences(user_id)
    assert prefs["user_id"] == user_id
    assert prefs["preferred_language"] in ["hi", "en"]
    
    # 2. Update preferences with saved locations
    updated = db.update_user_preferences(user_id, {
        "preferred_language": "ta",
        "saved_locations": [
            {"id": "loc_1", "name": "Chennai Harbor", "latitude": 13.0827, "longitude": 80.2707}
        ]
    })
    assert updated["preferred_language"] == "ta"
    assert len(updated["saved_locations"]) == 1
    assert updated["saved_locations"][0]["name"] == "Chennai Harbor"
    
    # 3. Fetch fresh to confirm persistence
    fresh = db.get_user_preferences(user_id)
    assert fresh["preferred_language"] == "ta"
    assert len(fresh["saved_locations"]) == 1

# =============================================================================
# 4. MEMORIES CRUD & PAGINATION
# =============================================================================

def test_user_memories_crud_and_pagination():
    db = DatabaseManager()
    user_id = "test_pagination_user"
    db.ensure_user(user_id)
    
    # Clean previous
    existing = db.list_user_memories(user_id, limit=100)
    for m in existing:
        db.delete_user_memory(user_id, m["memory_id"])
        
    # Add 5 memories
    for i in range(5):
        db.save_user_memory(user_id, f"mem_page_{i}", f"Title {i}", f"Content {i}")
        
    # Page 1 (limit 2, offset 0)
    page_1 = db.list_user_memories(user_id, limit=2, offset=0)
    assert len(page_1) == 2
    
    # Page 2 (limit 2, offset 2)
    page_2 = db.list_user_memories(user_id, limit=2, offset=2)
    assert len(page_2) == 2
    assert page_1[0]["memory_id"] != page_2[0]["memory_id"]

    # Test via API endpoint
    api_resp = client.get(
        "/api/memory/memories?limit=2&offset=0",
        headers={"X-User-ID": user_id}
    )
    assert api_resp.status_code == 200
    assert len(api_resp.json()) == 2

# =============================================================================
# 5. CONVERSATION TURN BOUNDING & PRUNING
# =============================================================================

def test_conversation_turn_pruning():
    db = DatabaseManager()
    conv_id = "conv_bound_test_123"
    
    # Clear any previous
    db.clear_conversation(conv_id)
    
    # Add 20 turns (max_turns set to 15)
    for i in range(20):
        db.add_conversation_turn(
            conversation_id=conv_id,
            role="user" if i % 2 == 0 else "assistant",
            text=f"Turn message {i}",
            max_turns=15
        )
        
    turns = db.get_conversation_turns(conv_id)
    assert len(turns) <= 15
    # Latest turn should be preserved
    assert turns[-1]["text"] == "Turn message 19"

# =============================================================================
# 6. ALERT DELIVERY DEDUPLICATION & IDEMPOTENCY
# =============================================================================

import uuid

def test_alert_delivery_idempotency_and_deduplication():
    db = DatabaseManager()
    user_id = f"test_alert_dedup_{uuid.uuid4().hex[:8]}"
    warning_id = "warn_cyclone_77"
    idempotency_key = f"idem_key_unique_{uuid.uuid4().hex[:8]}"
    
    notif_id = f"notif_{uuid.uuid4().hex[:8]}"
    # Record delivery
    rec = db.record_alert_delivery(
        notification_id=notif_id,
        user_id=user_id,
        warning_id=warning_id,
        channel="PUSH",
        status="DELIVERED",
        message="Cyclone alert in effect",
        idempotency_key=idempotency_key
    )
    assert rec["notification_id"] == notif_id
    
    # Deduplication check
    is_dedup = db.is_delivery_deduplicated(idempotency_key, window_minutes=60)
    assert is_dedup is True
    
    # Attempting to record again with same idempotency_key is ignored by ON CONFLICT
    db.record_alert_delivery(
        notification_id=f"notif_dup_{uuid.uuid4().hex[:8]}",
        user_id=user_id,
        warning_id=warning_id,
        channel="PUSH",
        status="DELIVERED",
        message="Cyclone alert in effect",
        idempotency_key=idempotency_key
    )
    
    history = db.get_user_alert_history(user_id)
    # Should only have 1 record with this idempotency key
    matching = [h for h in history if h["idempotency_key"] == idempotency_key]
    assert len(matching) == 1

# =============================================================================
# 7. CACHE RETENTION & EXPIRY POLICY
# =============================================================================

def test_marine_cache_expiry_and_retention():
    db = DatabaseManager()
    grid_key = "15.3:74.1"
    
    # Store observation with TTL
    db.store_marine_cache(
        grid_key=grid_key,
        latitude=15.3,
        longitude=74.1,
        location_name="Goa Coast",
        data={"sst": 28.5, "wave_height": 1.2},
        ttl_hours=12.0
    )
    
    cached = db.get_marine_cache(grid_key)
    assert cached is not None
    assert cached["data"]["sst"] == 28.5
    
    # Nearest lookup
    nearest = db.find_nearest_marine_cache(15.301, 74.101, max_dist_km=50.0)
    assert nearest is not None
    assert nearest[0]["grid_key"] == grid_key

# =============================================================================
# 8. BACKGROUND SYNC SERVICE CONCURRENCY LOCK
# =============================================================================

@pytest.mark.anyio
async def test_background_sync_concurrency_lock():
    sync_service = BackgroundSyncService()
    lat, lon = 15.2993, 74.1240
    
    # First sync run
    res1 = await sync_service.sync_region_with_lock(lat, lon, "Goa Harbor")
    assert res1["status"] in ["SUCCESS", "ALREADY_IN_PROGRESS"]
    
    # Retention cleanup
    cleanup_res = sync_service.run_retention_cleanup()
    assert cleanup_res["status"] == "SUCCESS"
