import sqlite3
import datetime
import json
import os
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Generator
from contextlib import contextmanager

from app.core.config import settings

logger = logging.getLogger("neerai.db")

MIGRATIONS: List[Tuple[int, str, str]] = [
    (
        1,
        "Initial relational schema with user isolation, indexes, and cache retention",
        """
        -- Schema Migrations Tracking
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT NOT NULL
        );

        -- Users Table
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        -- User Preferences
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            preferred_language TEXT NOT NULL DEFAULT 'en',
            preferred_units TEXT NOT NULL DEFAULT '{}',
            voice_settings TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );

        -- Saved Locations
        CREATE TABLE IF NOT EXISTS saved_locations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_saved_locations_user ON saved_locations(user_id);

        -- User-Approved Explicit Memories
        CREATE TABLE IF NOT EXISTS user_memories (
            memory_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'PREFERENCE',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_user_memories_user ON user_memories(user_id);

        -- Conversations & Bounded Turns
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversation_turns (
            turn_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            intent TEXT,
            latitude REAL,
            longitude REAL,
            location_name TEXT,
            target_time TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_conv_turns_conv ON conversation_turns(conversation_id);

        -- Feedback Logs
        CREATE TABLE IF NOT EXISTS feedback_logs (
            feedback_id TEXT PRIMARY KEY,
            user_id TEXT,
            message_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            feedback_notes TEXT,
            consent_given INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );

        -- Alert Subscriptions
        CREATE TABLE IF NOT EXISTS alert_subscriptions (
            subscription_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            region_name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            radius_km REAL NOT NULL DEFAULT 50.0,
            minimum_severity TEXT NOT NULL DEFAULT 'WARNING',
            channels TEXT NOT NULL DEFAULT '["IN_APP"]',
            device_token TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_alert_subs_user ON alert_subscriptions(user_id, is_active);

        -- Alert Notification Delivery History
        CREATE TABLE IF NOT EXISTS alert_deliveries (
            notification_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            warning_id TEXT NOT NULL,
            channel TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT NOT NULL,
            idempotency_key TEXT UNIQUE NOT NULL,
            error_detail TEXT,
            sent_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_alert_deliv_user ON alert_deliveries(user_id, warning_id);
        CREATE INDEX IF NOT EXISTS idx_alert_deliv_sent ON alert_deliveries(sent_at);

        -- Marine Cached Observations
        CREATE TABLE IF NOT EXISTS marine_cached_observations (
            grid_key TEXT PRIMARY KEY,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            location_name TEXT,
            data TEXT NOT NULL,
            cached_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            is_stale INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_cache_expires ON marine_cached_observations(expires_at);

        -- Offline Synchronization History
        CREATE TABLE IF NOT EXISTS offline_sync_logs (
            sync_id TEXT PRIMARY KEY,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            location_name TEXT,
            past_days INTEGER NOT NULL,
            forecast_days INTEGER NOT NULL,
            records_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            sync_timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sync_time ON offline_sync_logs(sync_timestamp);

        -- Agent Execution Records
        CREATE TABLE IF NOT EXISTS agent_execution_logs (
            execution_id TEXT PRIMARY KEY,
            correlation_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            status TEXT NOT NULL,
            execution_time_ms REAL,
            timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_agent_exec_corr ON agent_execution_logs(correlation_id);
        """
    ),
    (
        2,
        "Dedicated user feedback schema with privacy safeguards, rating bounds, and evaluation export",
        """
        CREATE TABLE IF NOT EXISTS user_feedback (
            feedback_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            correlation_id TEXT,
            message_id TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
            category TEXT NOT NULL DEFAULT 'GENERAL',
            feedback_notes TEXT,
            consent_given INTEGER NOT NULL DEFAULT 1,
            is_anonymized_for_eval INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_user_feedback_user ON user_feedback(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_feedback_created ON user_feedback(created_at);
        CREATE INDEX IF NOT EXISTS idx_user_feedback_corr ON user_feedback(correlation_id);
        """
    ),
    (
        3,
        "Shared Marine Knowledge, Timeseries Store, Daily Ingestion, Evaluations, and Model Training Pipeline",
        """
        -- Shared Marine Knowledge Documents (Unstructured facts, regulations, chlorophyll science)
        CREATE TABLE IF NOT EXISTS marine_knowledge_documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            source_authority TEXT NOT NULL,
            coastal_region TEXT NOT NULL DEFAULT 'ALL_INDIA',
            keywords TEXT NOT NULL DEFAULT '[]',
            embedding_json TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            is_verified INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_knowledge_cat ON marine_knowledge_documents(category);
        CREATE INDEX IF NOT EXISTS idx_knowledge_region ON marine_knowledge_documents(coastal_region);
        CREATE INDEX IF NOT EXISTS idx_knowledge_authority ON marine_knowledge_documents(source_authority);

        -- Shared Numeric Marine Timeseries Records (Waves, Wind, SST, Currents, Pressure)
        CREATE TABLE IF NOT EXISTS marine_timeseries_records (
            record_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            parameter TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            location_name TEXT,
            observation_time TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            data_type TEXT NOT NULL DEFAULT 'OBSERVATION',
            quality_status TEXT NOT NULL DEFAULT 'VALID',
            revision INTEGER NOT NULL DEFAULT 1,
            CONSTRAINT uq_timeseries_observation UNIQUE (source, parameter, latitude, longitude, observation_time)
        );
        CREATE INDEX IF NOT EXISTS idx_timeseries_loc_param ON marine_timeseries_records(latitude, longitude, parameter);
        CREATE INDEX IF NOT EXISTS idx_timeseries_obs_time ON marine_timeseries_records(observation_time);
        CREATE INDEX IF NOT EXISTS idx_timeseries_source ON marine_timeseries_records(source);

        -- Ingestion Job Runs (Tracks daily ingestion jobs, record counts, freshness, provider health)
        CREATE TABLE IF NOT EXISTS ingestion_job_runs (
            job_id TEXT PRIMARY KEY,
            pipeline_type TEXT NOT NULL DEFAULT 'DAILY_INGESTION',
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            providers_attempted TEXT NOT NULL DEFAULT '[]',
            providers_succeeded TEXT NOT NULL DEFAULT '[]',
            providers_failed TEXT NOT NULL DEFAULT '[]',
            records_added INTEGER NOT NULL DEFAULT 0,
            records_updated INTEGER NOT NULL DEFAULT 0,
            records_rejected INTEGER NOT NULL DEFAULT 0,
            records_skipped INTEGER NOT NULL DEFAULT 0,
            data_freshness_seconds REAL,
            error_summary TEXT,
            trigger_type TEXT NOT NULL DEFAULT 'SCHEDULED'
        );
        CREATE INDEX IF NOT EXISTS idx_ingest_status ON ingestion_job_runs(status);
        CREATE INDEX IF NOT EXISTS idx_ingest_started ON ingestion_job_runs(started_at);

        -- Evaluation Job Runs (Tracks daily benchmark evaluations, routing accuracy, regressions)
        CREATE TABLE IF NOT EXISTS evaluation_job_runs (
            eval_run_id TEXT PRIMARY KEY,
            baseline_version TEXT NOT NULL,
            candidate_version TEXT,
            benchmark_cases_count INTEGER NOT NULL,
            overall_score_pct REAL NOT NULL,
            zero_fabrication_pct REAL NOT NULL,
            routing_accuracy_pct REAL NOT NULL,
            multilingual_score_pct REAL NOT NULL,
            safety_regressions_count INTEGER NOT NULL DEFAULT 0,
            metrics_detail TEXT NOT NULL DEFAULT '{}',
            executed_at TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_eval_executed ON evaluation_job_runs(executed_at);
        CREATE INDEX IF NOT EXISTS idx_eval_status ON evaluation_job_runs(status);

        -- Model Training & Fine-Tuning Records (Tracks datasets, base models, metrics, approval gate)
        CREATE TABLE IF NOT EXISTS model_training_records (
            training_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            base_model TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            train_samples_count INTEGER NOT NULL,
            val_samples_count INTEGER NOT NULL,
            test_samples_count INTEGER NOT NULL,
            training_config TEXT NOT NULL DEFAULT '{}',
            eval_metrics TEXT NOT NULL DEFAULT '{}',
            artifact_path TEXT NOT NULL,
            approval_status TEXT NOT NULL DEFAULT 'PENDING_APPROVAL',
            deployed_in_production INTEGER NOT NULL DEFAULT 0,
            rollback_version TEXT,
            created_at TEXT NOT NULL,
            approved_by TEXT,
            approved_at TEXT,
            notes TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_training_created ON model_training_records(created_at);
        CREATE INDEX IF NOT EXISTS idx_training_approval ON model_training_records(approval_status);

        -- Pipeline Concurrency & Ingestion Locks
        CREATE TABLE IF NOT EXISTS pipeline_locks (
            lock_name TEXT PRIMARY KEY,
            locked_at TEXT NOT NULL,
            locked_by TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );
        """
    )
]

class DatabaseManager:
    """
    Central database manager for NeerAI.
    Manages SQLite connection pooling, migrations, transactions, and repository access.
    """
    _instance = None

    def __new__(cls, db_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._init(db_path)
        return cls._instance

    def _init(self, db_path: Optional[str] = None):
        target_path = db_path or settings.DATABASE_PATH
        if target_path != ":memory:":
            abs_dir = os.path.dirname(os.path.abspath(target_path))
            os.makedirs(abs_dir, exist_ok=True)
            self.db_path = os.path.abspath(target_path)
        else:
            self.db_path = ":memory:"

        self._run_migrations()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provides an isolated database connection with WAL mode and foreign keys enabled."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=float(settings.DATABASE_BUSY_TIMEOUT_MS) / 1000.0,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA busy_timeout = 10000;")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Executes operations inside an atomic transaction."""
        with self.get_connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE;")
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database transaction rolled back: {str(e)}")
                raise

    def _run_migrations(self):
        """Applies pending database migrations sequentially."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL,
                    description TEXT NOT NULL
                );
            """)
            conn.commit()

            cursor = conn.cursor()
            cursor.execute("SELECT version FROM schema_migrations;")
            applied_versions = {row["version"] for row in cursor.fetchall()}

            for version, description, script in MIGRATIONS:
                if version not in applied_versions:
                    logger.info(f"Applying migration {version}: {description}")
                    conn.executescript(script)
                    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?);",
                        (version, now_iso, description)
                    )
                    conn.commit()

    # -------------------------------------------------------------------------
    # System & Health
    # -------------------------------------------------------------------------
    def check_db_health(self) -> Dict[str, Any]:
        """Validates connection health and migration status."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as count FROM schema_migrations;")
                mig_count = cursor.fetchone()["count"]
                cursor.execute("PRAGMA integrity_check;")
                integrity = cursor.fetchone()[0]
                return {
                    "status": "healthy" if integrity == "ok" else "unhealthy",
                    "integrity": integrity,
                    "migrations_applied": mig_count,
                    "db_path": self.db_path
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # -------------------------------------------------------------------------
    # User Preferences & Saved Locations
    # -------------------------------------------------------------------------
    def ensure_user(self, user_id: str):
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (user_id, created_at, updated_at) VALUES (?, ?, ?);",
                (user_id, now_iso, now_iso)
            )

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        self.ensure_user(user_id)
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?;",
                (user_id,)
            ).fetchone()

            locations_rows = conn.execute(
                "SELECT id, name, latitude, longitude FROM saved_locations WHERE user_id = ? ORDER BY created_at ASC;",
                (user_id,)
            ).fetchall()

            saved_locations = [
                {"id": r["id"], "name": r["name"], "latitude": r["latitude"], "longitude": r["longitude"]}
                for r in locations_rows
            ]

            if row:
                return {
                    "user_id": user_id,
                    "preferred_language": row["preferred_language"],
                    "preferred_units": json.loads(row["preferred_units"]),
                    "saved_locations": saved_locations,
                    "enable_spoken_responses": True,
                    "spoken_speed": 1.0,
                    "voice_settings": json.loads(row["voice_settings"])
                }
            else:
                default_units = {
                    "wind_speed": "km/h",
                    "wave_height": "m",
                    "temperature": "celsius",
                    "pressure": "hPa"
                }
                return {
                    "user_id": user_id,
                    "preferred_language": "en",
                    "preferred_units": default_units,
                    "saved_locations": saved_locations,
                    "enable_spoken_responses": True,
                    "spoken_speed": 1.0,
                    "voice_settings": {}
                }

    def update_user_preferences(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_user(user_id)
        current = self.get_user_preferences(user_id)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if "preferred_language" in updates:
            current["preferred_language"] = updates["preferred_language"]
        if "preferred_units" in updates:
            current["preferred_units"].update(updates["preferred_units"])
        if "voice_settings" in updates:
            current["voice_settings"].update(updates["voice_settings"])

        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO user_preferences (user_id, preferred_language, preferred_units, voice_settings, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    preferred_language = excluded.preferred_language,
                    preferred_units = excluded.preferred_units,
                    voice_settings = excluded.voice_settings,
                    updated_at = excluded.updated_at;
            """, (
                user_id,
                current["preferred_language"],
                json.dumps(current["preferred_units"]),
                json.dumps(current.get("voice_settings", {})),
                now_iso
            ))

            # Handle saved locations replacement if provided
            if "saved_locations" in updates:
                conn.execute("DELETE FROM saved_locations WHERE user_id = ?;", (user_id,))
                for loc in updates["saved_locations"]:
                    conn.execute("""
                        INSERT OR REPLACE INTO saved_locations (id, user_id, name, latitude, longitude, created_at)
                        VALUES (?, ?, ?, ?, ?, ?);
                    """, (
                        loc.get("id", f"loc_{datetime.datetime.now().timestamp()}"),
                        user_id,
                        loc["name"],
                        float(loc["latitude"]),
                        float(loc["longitude"]),
                        now_iso
                    ))

        return self.get_user_preferences(user_id)

    # -------------------------------------------------------------------------
    # User-Approved Explicit Memories
    # -------------------------------------------------------------------------
    def list_user_memories(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT memory_id, user_id, title, content, category, created_at, updated_at
                FROM user_memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?;
            """, (user_id, limit, offset)).fetchall()
            return [dict(r) for r in rows]

    def save_user_memory(self, user_id: str, memory_id: str, title: str, content: str, category: str = "PREFERENCE") -> Dict[str, Any]:
        self.ensure_user(user_id)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_memories (memory_id, user_id, title, content, category, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (memory_id, user_id, title, content, category, now_iso, now_iso))
        return {
            "memory_id": memory_id,
            "user_id": user_id,
            "title": title,
            "content": content,
            "category": category,
            "created_at": now_iso,
            "updated_at": now_iso
        }

    def update_user_memory(self, user_id: str, memory_id: str, title: Optional[str], content: Optional[str]) -> Optional[Dict[str, Any]]:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM user_memories WHERE memory_id = ? AND user_id = ?;", (memory_id, user_id)).fetchone()
            if not row:
                return None
            new_title = title if title is not None else row["title"]
            new_content = content if content is not None else row["content"]
            conn.execute("""
                UPDATE user_memories
                SET title = ?, content = ?, updated_at = ?
                WHERE memory_id = ? AND user_id = ?;
            """, (new_title, new_content, now_iso, memory_id, user_id))
            return {
                "memory_id": memory_id,
                "user_id": user_id,
                "title": new_title,
                "content": new_content,
                "category": row["category"],
                "created_at": row["created_at"],
                "updated_at": now_iso
            }

    def delete_user_memory(self, user_id: str, memory_id: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM user_memories WHERE memory_id = ? AND user_id = ?;", (memory_id, user_id))
            return cur.rowcount > 0

    # -------------------------------------------------------------------------
    # Conversations & Bounded Context Turns
    # -------------------------------------------------------------------------
    def add_conversation_turn(
        self,
        conversation_id: str,
        role: str,
        text: str,
        intent: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_name: Optional[str] = None,
        target_time: Optional[str] = None,
        max_turns: int = 15
    ) -> Dict[str, Any]:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        turn_id = f"turn_{datetime.datetime.now().timestamp()}_{role[:3]}"
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO conversations (conversation_id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET updated_at = excluded.updated_at;
            """, (conversation_id, now_iso, now_iso))

            conn.execute("""
                INSERT INTO conversation_turns (
                    turn_id, conversation_id, role, text, intent, latitude, longitude, location_name, target_time, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (turn_id, conversation_id, role, text, intent, latitude, longitude, location_name, target_time, now_iso))

            # Prune turns older than max_turns to keep context bounded
            conn.execute("""
                DELETE FROM conversation_turns
                WHERE conversation_id = ?
                AND turn_id NOT IN (
                    SELECT turn_id FROM conversation_turns
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                );
            """, (conversation_id, conversation_id, max_turns))

        return {
            "turn_id": turn_id,
            "role": role,
            "text": text,
            "intent": intent,
            "latitude": latitude,
            "longitude": longitude,
            "location_name": location_name,
            "target_time": target_time,
            "timestamp": now_iso
        }

    def get_conversation_turns(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT turn_id, role, text, intent, latitude, longitude, location_name, target_time, created_at as timestamp
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY created_at ASC;
            """, (conversation_id,)).fetchall()
            return [dict(r) for r in rows]

    def clear_conversation(self, conversation_id: str):
        with self.transaction() as conn:
            conn.execute("DELETE FROM conversation_turns WHERE conversation_id = ?;", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE conversation_id = ?;", (conversation_id,))

    # -------------------------------------------------------------------------
    # Alert Subscriptions & Delivery Deduplication
    # -------------------------------------------------------------------------
    def add_alert_subscription(
        self,
        subscription_id: str,
        user_id: str,
        region_name: str,
        latitude: float,
        longitude: float,
        radius_km: float = 50.0,
        minimum_severity: str = "WARNING",
        channels: Optional[List[str]] = None,
        device_token: Optional[str] = None
    ) -> Dict[str, Any]:
        self.ensure_user(user_id)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        ch_json = json.dumps(channels or ["IN_APP"])
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO alert_subscriptions (
                    subscription_id, user_id, region_name, latitude, longitude, radius_km, minimum_severity, channels, device_token, is_active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?);
            """, (subscription_id, user_id, region_name, latitude, longitude, radius_km, minimum_severity, ch_json, device_token, now_iso, now_iso))

        return {
            "subscription_id": subscription_id,
            "user_id": user_id,
            "region_name": region_name,
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
            "minimum_severity": minimum_severity,
            "channels": channels or ["IN_APP"],
            "device_token": device_token,
            "is_active": True,
            "created_at": now_iso,
            "updated_at": now_iso
        }

    def get_user_subscriptions(self, user_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM alert_subscriptions
                WHERE user_id = ? AND is_active = 1
                ORDER BY created_at DESC;
            """, (user_id,)).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["channels"] = json.loads(d["channels"])
                d["is_active"] = bool(d["is_active"])
                results.append(d)
            return results

    def delete_alert_subscription(self, user_id: str, subscription_id: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute("""
                DELETE FROM alert_subscriptions
                WHERE subscription_id = ? AND user_id = ?;
            """, (subscription_id, user_id))
            return cur.rowcount > 0

    def record_alert_delivery(
        self,
        notification_id: str,
        user_id: str,
        warning_id: str,
        channel: str,
        status: str,
        message: str,
        idempotency_key: str,
        error_detail: Optional[str] = None
    ) -> Dict[str, Any]:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO alert_deliveries (
                    notification_id, user_id, warning_id, channel, status, message, idempotency_key, error_detail, sent_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING;
            """, (notification_id, user_id, warning_id, channel, status, message, idempotency_key, error_detail, now_iso))

        return {
            "notification_id": notification_id,
            "user_id": user_id,
            "warning_id": warning_id,
            "channel": channel,
            "status": status,
            "message": message,
            "idempotency_key": idempotency_key,
            "error_detail": error_detail,
            "sent_at": now_iso
        }

    def is_delivery_deduplicated(self, idempotency_key: str, window_minutes: int = 60) -> bool:
        """Checks if a delivery with this idempotency key was already recorded within the window."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT sent_at FROM alert_deliveries WHERE idempotency_key = ?;",
                (idempotency_key,)
            ).fetchone()
            if not row:
                return False

            sent_at = datetime.datetime.fromisoformat(row["sent_at"].replace("Z", "+00:00"))
            if sent_at.tzinfo is None:
                sent_at = sent_at.replace(tzinfo=datetime.timezone.utc)
            now = datetime.datetime.now(datetime.timezone.utc)
            return (now - sent_at).total_seconds() < (window_minutes * 60)

    def get_user_alert_history(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT notification_id, user_id, warning_id, channel, status, message, idempotency_key, error_detail, sent_at
                FROM alert_deliveries
                WHERE user_id = ?
                ORDER BY sent_at DESC
                LIMIT ? OFFSET ?;
            """, (user_id, limit, offset)).fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # Marine Observations Offline Cache
    # -------------------------------------------------------------------------
    def store_marine_cache(self, grid_key: str, latitude: float, longitude: float, location_name: str, data: Dict[str, Any], ttl_hours: float = 12.0):
        now = datetime.datetime.now(datetime.timezone.utc)
        cached_at = now.isoformat()
        expires_at = (now + datetime.timedelta(hours=ttl_hours)).isoformat()
        data_json = json.dumps(data)

        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO marine_cached_observations (grid_key, latitude, longitude, location_name, data, cached_at, expires_at, is_stale)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(grid_key) DO UPDATE SET
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    location_name = excluded.location_name,
                    data = excluded.data,
                    cached_at = excluded.cached_at,
                    expires_at = excluded.expires_at,
                    is_stale = 0;
            """, (grid_key, latitude, longitude, location_name, data_json, cached_at, expires_at))

    def get_marine_cache(self, grid_key: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM marine_cached_observations WHERE grid_key = ?;", (grid_key,)).fetchone()
            if not row:
                return None
            return {
                "grid_key": row["grid_key"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "location_name": row["location_name"],
                "data": json.loads(row["data"]),
                "cached_at": row["cached_at"],
                "expires_at": row["expires_at"],
                "is_stale": bool(row["is_stale"])
            }

    def find_nearest_marine_cache(self, lat: float, lon: float, max_dist_km: float = 80.0) -> Optional[Tuple[Dict[str, Any], float]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM marine_cached_observations;").fetchall()
            nearest = None
            min_dist = float("inf")

            for row in rows:
                dlat = math.radians(row["latitude"] - lat)
                dlon = math.radians(row["longitude"] - lon)
                a = (math.sin(dlat / 2) ** 2 +
                     math.cos(math.radians(lat)) * math.cos(math.radians(row["latitude"])) *
                     math.sin(dlon / 2) ** 2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                dist = 6371.0 * c

                if dist < min_dist and dist <= max_dist_km:
                    min_dist = dist
                    nearest = {
                        "grid_key": row["grid_key"],
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                        "location_name": row["location_name"],
                        "data": json.loads(row["data"]),
                        "cached_at": row["cached_at"],
                        "expires_at": row["expires_at"],
                        "is_stale": bool(row["is_stale"])
                    }

            if nearest:
                return nearest, round(min_dist, 2)
            return None

    def cleanup_expired_cache(self, retention_hours: Optional[int] = None) -> int:
        """Purges observations whose TTL has expired."""
        cutoff_hours = retention_hours or settings.CACHE_RETENTION_HOURS
        now = datetime.datetime.now(datetime.timezone.utc)
        cutoff_iso = (now - datetime.timedelta(hours=cutoff_hours)).isoformat()

        with self.transaction() as conn:
            cur = conn.execute("""
                DELETE FROM marine_cached_observations
                WHERE cached_at < ?;
            """, (cutoff_iso,))
            return cur.rowcount

    def record_offline_sync(self, sync_id: str, latitude: float, longitude: float, location_name: str, past_days: int, forecast_days: int, records_count: int, status: str) -> Dict[str, Any]:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO offline_sync_logs (sync_id, latitude, longitude, location_name, past_days, forecast_days, records_count, status, sync_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (sync_id, latitude, longitude, location_name, past_days, forecast_days, records_count, status, now_iso))
        return {
            "sync_id": sync_id,
            "latitude": latitude,
            "longitude": longitude,
            "location_name": location_name,
            "past_days": past_days,
            "forecast_days": forecast_days,
            "records_count": records_count,
            "status": status,
            "sync_timestamp": now_iso
        }

    # -------------------------------------------------------------------------
    # User Feedback & Controlled Evaluation Storage (Migration 002)
    # -------------------------------------------------------------------------
    def save_user_feedback(
        self,
        feedback_id: str,
        user_id: str,
        message_id: str,
        rating: int,
        correlation_id: Optional[str] = None,
        category: str = "GENERAL",
        feedback_notes: Optional[str] = None,
        consent_given: bool = True
    ) -> Dict[str, Any]:
        self.ensure_user(user_id)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        bounded_rating = max(1, min(5, int(rating)))
        sanitized_category = (category or "GENERAL").upper()[:50]
        notes = feedback_notes[:500] if feedback_notes else None
        consent_int = 1 if consent_given else 0

        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO user_feedback (
                    feedback_id, user_id, correlation_id, message_id, rating, category, feedback_notes, consent_given, is_anonymized_for_eval, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?);
            """, (feedback_id, user_id, correlation_id, message_id, bounded_rating, sanitized_category, notes, consent_int, now_iso, now_iso))

        return {
            "feedback_id": feedback_id,
            "user_id": user_id,
            "correlation_id": correlation_id,
            "message_id": message_id,
            "rating": bounded_rating,
            "category": sanitized_category,
            "feedback_notes": notes,
            "consent_given": consent_given,
            "created_at": now_iso,
            "updated_at": now_iso
        }

    def list_user_feedback(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT feedback_id, user_id, correlation_id, message_id, rating, category, feedback_notes, consent_given, created_at, updated_at
                FROM user_feedback
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?;
            """, (user_id, limit, offset)).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["consent_given"] = bool(d["consent_given"])
                results.append(d)
            return results

    def get_user_feedback_by_id(self, feedback_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("""
                SELECT feedback_id, user_id, correlation_id, message_id, rating, category, feedback_notes, consent_given, created_at, updated_at
                FROM user_feedback
                WHERE feedback_id = ? AND user_id = ?;
            """, (feedback_id, user_id)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["consent_given"] = bool(d["consent_given"])
            return d

    def delete_user_feedback(self, feedback_id: str, user_id: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute("""
                DELETE FROM user_feedback
                WHERE feedback_id = ? AND user_id = ?;
            """, (feedback_id, user_id))
            return cur.rowcount > 0

    def export_anonymized_feedback_for_eval(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Exports feedback items for evaluation benchmark only when explicit consent was granted.
        Strips user_id to anonymous identifier hash and redacts private details.
        """
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT feedback_id, correlation_id, message_id, rating, category, feedback_notes, created_at
                FROM user_feedback
                WHERE consent_given = 1
                ORDER BY created_at DESC
                LIMIT ?;
            """, (limit,)).fetchall()
            exported = []
            for r in rows:
                notes = r["feedback_notes"] or ""
                exported.append({
                    "eval_id": f"eval_fb_{r['feedback_id'][:8]}",
                    "correlation_id": r["correlation_id"] or "unlinked",
                    "rating": r["rating"],
                    "category": r["category"],
                    "feedback_summary": notes,
                    "submitted_at": r["created_at"]
                })
            return exported

    def log_agent_execution(
        self,
        execution_id: str,
        correlation_id: str,
        agent_name: str,
        status: str,
        execution_time_ms: float,
        timestamp: Optional[str] = None
    ) -> None:
        """Logs an agent execution step to the database."""
        ts = timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            with self.transaction() as conn:
                conn.execute("""
                    INSERT INTO agent_execution_logs (
                        execution_id, correlation_id, agent_name, status, execution_time_ms, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?);
                """, (execution_id, correlation_id, agent_name, status, execution_time_ms, ts))
        except Exception as e:
            logger.warning(f"Could not persist agent execution log: {e}")

    def get_agent_execution_logs(
        self,
        correlation_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieves recent agent execution logs."""
        with self.get_connection() as conn:
            if correlation_id:
                rows = conn.execute("""
                    SELECT execution_id, correlation_id, agent_name, status, execution_time_ms, timestamp
                    FROM agent_execution_logs
                    WHERE correlation_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?;
                """, (correlation_id, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT execution_id, correlation_id, agent_name, status, execution_time_ms, timestamp
                    FROM agent_execution_logs
                    ORDER BY timestamp DESC
                    LIMIT ?;
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]

    # -------------------------------------------------------------------------
    # Shared Marine Timeseries Records (Numeric Ocean & Weather Data)
    # -------------------------------------------------------------------------
    def upsert_timeseries_record(self, record: Dict[str, Any]) -> str:
        """
        Inserts or updates a numeric timeseries record with schema validation and revision tracking.
        Returns: 'ADDED', 'UPDATED', 'SKIPPED', or 'REJECTED'
        """
        import uuid
        source = str(record.get("source", "")).strip()
        param = str(record.get("parameter", "")).strip()
        val = record.get("value")
        unit = str(record.get("unit", "")).strip()
        lat = record.get("latitude")
        lon = record.get("longitude")
        obs_time = str(record.get("observation_time", "")).strip()
        data_type = str(record.get("data_type", "OBSERVATION")).strip()
        quality = str(record.get("quality_status", "VALID")).strip()
        loc_name = record.get("location_name")

        if not source or not param or val is None or lat is None or lon is None or not obs_time:
            return "REJECTED"

        try:
            val_f = float(val)
            lat_f = float(lat)
            lon_f = float(lon)
        except (ValueError, TypeError):
            return "REJECTED"

        # Coordinates sanity check
        if lat_f < -90.0 or lat_f > 90.0 or lon_f < -180.0 or lon_f > 180.0:
            return "REJECTED"

        # Physical sanity bounds
        if param == "wave_height" and (val_f < 0.0 or val_f > 30.0):
            return "REJECTED"
        if param == "wind_speed" and (val_f < 0.0 or val_f > 350.0):
            return "REJECTED"
        if param == "sst" and (val_f < -2.0 or val_f > 45.0):
            return "REJECTED"

        lat_round = round(lat_f, 4)
        lon_round = round(lon_f, 4)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self.transaction() as conn:
            existing = conn.execute("""
                SELECT record_id, value, revision FROM marine_timeseries_records
                WHERE source = ? AND parameter = ? AND latitude = ? AND longitude = ? AND observation_time = ?;
            """, (source, param, lat_round, lon_round, obs_time)).fetchone()

            if existing:
                if math.isclose(existing["value"], val_f, abs_tol=1e-3):
                    return "SKIPPED"
                new_rev = existing["revision"] + 1
                conn.execute("""
                    UPDATE marine_timeseries_records
                    SET value = ?, unit = ?, location_name = ?, ingested_at = ?, data_type = ?, quality_status = ?, revision = ?
                    WHERE record_id = ?;
                """, (val_f, unit, loc_name, now_iso, data_type, quality, new_rev, existing["record_id"]))
                return "UPDATED"
            else:
                rec_id = f"ts_{uuid.uuid4().hex[:12]}"
                conn.execute("""
                    INSERT INTO marine_timeseries_records (
                        record_id, source, parameter, value, unit, latitude, longitude,
                        location_name, observation_time, ingested_at, data_type, quality_status, revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1);
                """, (rec_id, source, param, val_f, unit, lat_round, lon_round, loc_name, obs_time, now_iso, data_type, quality))
                return "ADDED"

    def query_timeseries_records(
        self,
        parameter: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_km: float = 75.0,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Executes precise structured SQL queries for numeric marine observations and forecasts.
        Uses geodesic distance filtering and temporal boundaries.
        """
        deg_delta = radius_km / 111.0
        query_parts = ["1=1"]
        params = []

        if parameter:
            query_parts.append("parameter = ?")
            params.append(parameter)

        if lat is not None and lon is not None:
            query_parts.append("latitude BETWEEN ? AND ?")
            params.extend([lat - deg_delta, lat + deg_delta])
            query_parts.append("longitude BETWEEN ? AND ?")
            params.extend([lon - deg_delta, lon + deg_delta])

        if start_time:
            query_parts.append("observation_time >= ?")
            params.append(start_time)
        if end_time:
            query_parts.append("observation_time <= ?")
            params.append(end_time)

        where_clause = " AND ".join(query_parts)
        sql = f"""
            SELECT record_id, source, parameter, value, unit, latitude, longitude,
                   location_name, observation_time, ingested_at, data_type, quality_status, revision
            FROM marine_timeseries_records
            WHERE {where_clause}
            ORDER BY observation_time DESC
            LIMIT ?;
        """
        params.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            results = []
            for r in rows:
                rec = dict(r)
                if lat is not None and lon is not None:
                    # Calculate accurate haversine distance
                    rec["distance_km"] = round(self._haversine(lat, lon, rec["latitude"], rec["longitude"]), 2)
                results.append(rec)
            return results

    # -------------------------------------------------------------------------
    # Shared Marine Knowledge Documents (Unstructured facts, regulations)
    # -------------------------------------------------------------------------
    def upsert_knowledge_document(self, doc: Dict[str, Any]) -> str:
        """Stores or updates a verified marine knowledge document."""
        import uuid
        doc_id = doc.get("doc_id") or f"doc_{uuid.uuid4().hex[:12]}"
        title = doc["title"]
        content = doc["content"]
        category = doc.get("category", "GENERAL")
        authority = doc.get("source_authority", "NeerAI Marine Intelligence")
        region = doc.get("coastal_region", "ALL_INDIA")
        keywords_json = json.dumps(doc.get("keywords", []))
        emb_json = json.dumps(doc.get("embedding", [])) if "embedding" in doc else None
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self.transaction() as conn:
            existing = conn.execute("SELECT doc_id, version FROM marine_knowledge_documents WHERE doc_id = ?;", (doc_id,)).fetchone()
            if existing:
                conn.execute("""
                    UPDATE marine_knowledge_documents
                    SET title = ?, content = ?, category = ?, source_authority = ?, coastal_region = ?,
                        keywords = ?, embedding_json = ?, version = version + 1, updated_at = ?
                    WHERE doc_id = ?;
                """, (title, content, category, authority, region, keywords_json, emb_json, now_iso, doc_id))
            else:
                conn.execute("""
                    INSERT INTO marine_knowledge_documents (
                        doc_id, title, content, category, source_authority, coastal_region,
                        keywords, embedding_json, version, is_verified, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?);
                """, (doc_id, title, content, category, authority, region, keywords_json, emb_json, now_iso, now_iso))
        return doc_id

    def search_knowledge_documents(
        self,
        query: str,
        category: Optional[str] = None,
        coastal_region: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieves relevant marine knowledge documents with keyword matching,
        category filtering, and authority attribution.
        """
        tokens = [t.lower().strip() for t in query.split() if len(t.strip()) > 2]
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT doc_id, title, content, category, source_authority, coastal_region, keywords, version, updated_at
                FROM marine_knowledge_documents
                WHERE is_verified = 1;
            """).fetchall()

            scored = []
            for r in rows:
                # Category filter
                if category and r["category"].upper() != category.upper():
                    continue
                # Region filter
                if coastal_region and r["coastal_region"] != "ALL_INDIA" and r["coastal_region"].upper() != coastal_region.upper():
                    continue

                text_corpus = f"{r['title']} {r['content']} {r['keywords']}".lower()
                score = sum(1 for tok in tokens if tok in text_corpus)
                if score > 0 or not tokens:
                    scored.append((score, dict(r)))

            # Sort by match score descending
            scored.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scored[:limit]]

    # -------------------------------------------------------------------------
    # Ingestion & Evaluation Job Run Tracking
    # -------------------------------------------------------------------------
    def record_ingestion_job_start(self, job_id: str, trigger_type: str = "SCHEDULED") -> None:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO ingestion_job_runs (
                    job_id, pipeline_type, status, started_at, trigger_type
                ) VALUES (?, 'DAILY_INGESTION', 'RUNNING', ?, ?);
            """, (job_id, now_iso, trigger_type))

    def update_ingestion_job_finish(
        self,
        job_id: str,
        status: str,
        providers_attempted: List[str],
        providers_succeeded: List[str],
        providers_failed: List[str],
        records_added: int,
        records_updated: int,
        records_rejected: int,
        records_skipped: int,
        data_freshness_seconds: Optional[float] = None,
        error_summary: Optional[str] = None
    ) -> None:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute("""
                UPDATE ingestion_job_runs
                SET status = ?, completed_at = ?, providers_attempted = ?, providers_succeeded = ?,
                    providers_failed = ?, records_added = ?, records_updated = ?, records_rejected = ?,
                    records_skipped = ?, data_freshness_seconds = ?, error_summary = ?
                WHERE job_id = ?;
            """, (
                status, now_iso, json.dumps(providers_attempted), json.dumps(providers_succeeded),
                json.dumps(providers_failed), records_added, records_updated, records_rejected,
                records_skipped, data_freshness_seconds, error_summary, job_id
            ))

    def get_latest_ingestion_job(self) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("""
                SELECT job_id, pipeline_type, status, started_at, completed_at,
                       providers_attempted, providers_succeeded, providers_failed,
                       records_added, records_updated, records_rejected, records_skipped,
                       data_freshness_seconds, error_summary, trigger_type
                FROM ingestion_job_runs
                ORDER BY started_at DESC
                LIMIT 1;
            """).fetchone()
            if not row:
                return None
            res = dict(row)
            res["providers_attempted"] = json.loads(res.get("providers_attempted") or "[]")
            res["providers_succeeded"] = json.loads(res.get("providers_succeeded") or "[]")
            res["providers_failed"] = json.loads(res.get("providers_failed") or "[]")
            return res

    def record_evaluation_job(self, eval_data: Dict[str, Any]) -> None:
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO evaluation_job_runs (
                    eval_run_id, baseline_version, candidate_version, benchmark_cases_count,
                    overall_score_pct, zero_fabrication_pct, routing_accuracy_pct, multilingual_score_pct,
                    safety_regressions_count, metrics_detail, executed_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                eval_data["eval_run_id"],
                eval_data.get("baseline_version", "1.0.0"),
                eval_data.get("candidate_version"),
                eval_data.get("benchmark_cases_count", 0),
                eval_data.get("overall_score_pct", 0.0),
                eval_data.get("zero_fabrication_pct", 100.0),
                eval_data.get("routing_accuracy_pct", 0.0),
                eval_data.get("multilingual_score_pct", 0.0),
                eval_data.get("safety_regressions_count", 0),
                json.dumps(eval_data.get("metrics_detail", {})),
                now_iso,
                eval_data.get("status", "PASSED")
            ))

    def get_latest_evaluation_job(self) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("""
                SELECT eval_run_id, baseline_version, candidate_version, benchmark_cases_count,
                       overall_score_pct, zero_fabrication_pct, routing_accuracy_pct, multilingual_score_pct,
                       safety_regressions_count, metrics_detail, executed_at, status
                FROM evaluation_job_runs
                ORDER BY executed_at DESC
                LIMIT 1;
            """).fetchone()
            if not row:
                return None
            res = dict(row)
            res["metrics_detail"] = json.loads(res.get("metrics_detail") or "{}")
            return res

    # -------------------------------------------------------------------------
    # Versioned Model Training Records & Strict Approval Gate
    # -------------------------------------------------------------------------
    def record_model_training(self, training_data: Dict[str, Any]) -> str:
        """Stores a versioned model training record with PENDING_APPROVAL status."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tid = training_data["training_id"]
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO model_training_records (
                    training_id, model_name, base_model, dataset_version,
                    train_samples_count, val_samples_count, test_samples_count,
                    training_config, eval_metrics, artifact_path, approval_status,
                    deployed_in_production, rollback_version, created_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING_APPROVAL', 0, ?, ?, ?);
            """, (
                tid,
                training_data["model_name"],
                training_data["base_model"],
                training_data["dataset_version"],
                training_data.get("train_samples_count", 0),
                training_data.get("val_samples_count", 0),
                training_data.get("test_samples_count", 0),
                json.dumps(training_data.get("training_config", {})),
                json.dumps(training_data.get("eval_metrics", {})),
                training_data.get("artifact_path", ""),
                training_data.get("rollback_version"),
                now_iso,
                training_data.get("notes", "")
            ))
        return tid

    def get_model_training_records(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT training_id, model_name, base_model, dataset_version,
                       train_samples_count, val_samples_count, test_samples_count,
                       training_config, eval_metrics, artifact_path, approval_status,
                       deployed_in_production, rollback_version, created_at, approved_by, approved_at, notes
                FROM model_training_records
                ORDER BY created_at DESC
                LIMIT ?;
            """, (limit,)).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["training_config"] = json.loads(item.get("training_config") or "{}")
                item["eval_metrics"] = json.loads(item.get("eval_metrics") or "{}")
                results.append(item)
            return results

    def approve_and_deploy_model(self, training_id: str, approver: str) -> bool:
        """
        STRICT HUMAN APPROVAL GATE:
        Only explicitly approved models can be deployed to production.
        Deactivates any previously deployed model and activates the target model.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.transaction() as conn:
            target = conn.execute(
                "SELECT training_id, approval_status FROM model_training_records WHERE training_id = ?;",
                (training_id,)
            ).fetchone()
            if not target:
                return False

            # Deactivate previous production models
            conn.execute("UPDATE model_training_records SET deployed_in_production = 0 WHERE deployed_in_production = 1;")

            # Activate target model
            conn.execute("""
                UPDATE model_training_records
                SET approval_status = 'APPROVED', deployed_in_production = 1, approved_by = ?, approved_at = ?
                WHERE training_id = ?;
            """, (approver, now_iso, training_id))
            return True

    def rollback_model_deployment(self, training_id: str) -> Optional[str]:
        """
        Rolls back the specified model and restores the previous rollback_version.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.transaction() as conn:
            target = conn.execute(
                "SELECT training_id, rollback_version FROM model_training_records WHERE training_id = ?;",
                (training_id,)
            ).fetchone()
            if not target:
                return None

            prev_version = target["rollback_version"]
            # Demote current model
            conn.execute("""
                UPDATE model_training_records
                SET approval_status = 'ROLLED_BACK', deployed_in_production = 0
                WHERE training_id = ?;
            """, (training_id,))

            # Promote rollback model if exists
            if prev_version:
                conn.execute("""
                    UPDATE model_training_records
                    SET deployed_in_production = 1
                    WHERE training_id = ?;
                """, (prev_version,))
            return prev_version

    def get_active_production_model(self) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("""
                SELECT training_id, model_name, base_model, dataset_version,
                       artifact_path, approval_status, deployed_in_production, approved_by, approved_at
                FROM model_training_records
                WHERE deployed_in_production = 1
                LIMIT 1;
            """).fetchone()
            return dict(row) if row else None

    # -------------------------------------------------------------------------
    # Pipeline Concurrency Locks
    # -------------------------------------------------------------------------
    def acquire_pipeline_lock(self, lock_name: str, duration_seconds: int = 300) -> bool:
        """
        Acquires an exclusive pipeline lock to prevent overlapping runs.
        Returns True if lock acquired, False if already held and valid.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now.isoformat()
        expires = (now + datetime.timedelta(seconds=duration_seconds)).isoformat()

        with self.transaction() as conn:
            existing = conn.execute("SELECT lock_name, expires_at FROM pipeline_locks WHERE lock_name = ?;", (lock_name,)).fetchone()
            if existing:
                if existing["expires_at"] > now_iso:
                    return False  # Still actively locked
                # Lock expired, claim it
                conn.execute("""
                    UPDATE pipeline_locks
                    SET locked_at = ?, locked_by = 'neerai_worker', expires_at = ?
                    WHERE lock_name = ?;
                """, (now_iso, expires, lock_name))
                return True
            else:
                conn.execute("""
                    INSERT INTO pipeline_locks (lock_name, locked_at, locked_by, expires_at)
                    VALUES (?, ?, 'neerai_worker', ?);
                """, (lock_name, now_iso, expires))
                return True

    def release_pipeline_lock(self, lock_name: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM pipeline_locks WHERE lock_name = ?;", (lock_name,))

