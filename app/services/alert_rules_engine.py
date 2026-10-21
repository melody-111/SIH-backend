import datetime
import hashlib
import math
import uuid
from typing import Dict, Any, List, Optional, Tuple
from app.schemas.hazard_alert import (
    AlertSubscription,
    AlertSubscriptionRequest,
    OfficialWarningDetail,
    AlertNotificationRecord,
    AlertStatus,
    DeliveryStatus
)
from app.services.official_warning_provider import OfficialWarningProvider
from app.services.notification_service import NotificationDeliveryService
from app.core.config import settings

SEVERITY_RANKS = {
    "INFO": 1,
    "ADVISORY": 2,
    "WARNING": 3,
    "DANGER": 4
}

class AlertRulesEngine:
    """
    Configurable alert rules and deduplication engine.
    
    Ensures:
    - Alerts only trigger for matching geography and active validity windows
    - Deduplicates unchanged alerts within configurable time window
    - Immediately alerts on updated, escalated, or cancelled warning states
    - Provides multi-tenant subscription isolation
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AlertRulesEngine, cls).__new__(cls)
            from app.db.database import DatabaseManager
            cls._instance.db = DatabaseManager()
            cls._instance._subscriptions: Dict[str, AlertSubscription] = {}
            cls._instance._user_subscriptions: Dict[str, List[str]] = {}
            cls._instance._delivery_cache: Dict[str, Dict[str, Any]] = {}  # key: f"{user_id}:{warning_id}" -> {sent_at, version, status}
            cls._instance._warning_provider = OfficialWarningProvider()
            cls._instance._notification_service = NotificationDeliveryService()
        return cls._instance

    def _calculate_haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def generate_idempotency_key(
        self,
        user_id: str,
        warning_id: str,
        version: int,
        status: AlertStatus
    ) -> str:
        """Deterministic idempotency key preventing duplicate dispatch."""
        raw = f"{user_id}:{warning_id}:{version}:{status.value}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def add_subscription(self, req: AlertSubscriptionRequest) -> AlertSubscription:
        """Registers a user subscription for alert notifications."""
        sub_id = f"sub_{uuid.uuid4().hex[:10]}"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        sub = AlertSubscription(
            subscription_id=sub_id,
            user_id=req.user_id,
            device_token=req.device_token,
            latitude=req.latitude,
            longitude=req.longitude,
            radius_km=req.radius_km,
            min_severity=req.min_severity.upper(),
            enable_push=req.enable_push,
            enable_in_app=req.enable_in_app,
            created_at=now_iso
        )
        self._subscriptions[sub_id] = sub
        if req.user_id not in self._user_subscriptions:
            self._user_subscriptions[req.user_id] = []
        self._user_subscriptions[req.user_id].append(sub_id)

        try:
            self.db.add_alert_subscription(
                subscription_id=sub_id,
                user_id=req.user_id,
                region_name=req.region_name or "Coastal Waters",
                latitude=req.latitude,
                longitude=req.longitude,
                radius_km=req.radius_km,
                minimum_severity=req.min_severity.upper(),
                channels=["PUSH"] if req.enable_push else ["IN_APP"],
                device_token=req.device_token
            )
        except Exception:
            pass

        return sub

    def get_user_subscriptions(self, user_id: str) -> List[AlertSubscription]:
        """User-isolated subscription lookup with persistent SQLite database recovery."""
        sub_ids = self._user_subscriptions.get(user_id, [])
        records = [self._subscriptions[sid] for sid in sub_ids if sid in self._subscriptions]
        
        # If in-memory registry is empty, restore from database
        if not records:
            try:
                db_subs = self.db.get_user_subscriptions(user_id)
                for s in db_subs:
                    sid = s["subscription_id"]
                    sub_obj = AlertSubscription(
                        subscription_id=sid,
                        user_id=s["user_id"],
                        device_token=s.get("device_token"),
                        latitude=s["latitude"],
                        longitude=s["longitude"],
                        radius_km=s["radius_km"],
                        min_severity=s.get("minimum_severity", "WARNING"),
                        enable_push="PUSH" in s.get("channels", []),
                        enable_in_app="IN_APP" in s.get("channels", []),
                        created_at=s["created_at"]
                    )
                    self._subscriptions[sid] = sub_obj
                    if sid not in sub_ids:
                        sub_ids.append(sid)
                self._user_subscriptions[user_id] = sub_ids
                records = [self._subscriptions[sid] for sid in sub_ids if sid in self._subscriptions]
            except Exception:
                pass

        return records

    def delete_subscription(self, user_id: str, subscription_id: str) -> bool:
        """Deletes a subscription verifying user ownership."""
        sub = self._subscriptions.get(subscription_id)
        if not sub or sub.user_id != user_id:
            return False
        
        del self._subscriptions[subscription_id]
        if user_id in self._user_subscriptions:
            self._user_subscriptions[user_id] = [sid for sid in self._user_subscriptions[user_id] if sid != subscription_id]
        try:
            self.db.delete_alert_subscription(user_id, subscription_id)
        except Exception:
            pass
        return True

    def matches_subscription(self, sub: AlertSubscription, warning: OfficialWarningDetail) -> bool:
        """Checks if warning matches geography and severity threshold."""
        # 1. Severity check
        sub_rank = SEVERITY_RANKS.get(sub.min_severity, 2)
        warn_rank = SEVERITY_RANKS.get(warning.normalized_severity, 2)
        if warn_rank < sub_rank:
            return False

        # 2. Geographic coverage check
        if warning.affected_coordinates:
            w_lat = warning.affected_coordinates.get("latitude")
            w_lon = warning.affected_coordinates.get("longitude")
            if w_lat is not None and w_lon is not None:
                dist = self._calculate_haversine(sub.latitude, sub.longitude, w_lat, w_lon)
                # Match if within combined radius
                if dist <= (sub.radius_km + warning.coverage_radius_km):
                    return True

        return False

    async def process_warning_dispatch(
        self,
        warning: OfficialWarningDetail,
        target_user_id: Optional[str] = None
    ) -> List[AlertNotificationRecord]:
        """
        Evaluates a warning against subscriptions and dispatches notifications
        with strict deduplication and state change handling.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        results: List[AlertNotificationRecord] = []
        version = self._warning_provider.get_bulletin_version(warning.warning_id)

        # Check validity window
        is_valid = self._warning_provider.is_warning_valid(warning, now)
        if not is_valid and warning.status == AlertStatus.ACTIVE:
            warning = warning.model_copy(update={"status": AlertStatus.EXPIRED})

        # Find target subscriptions
        candidate_subs: List[AlertSubscription] = []
        if target_user_id:
            candidate_subs = self.get_user_subscriptions(target_user_id)
        else:
            candidate_subs = list(self._subscriptions.values())

        for sub in candidate_subs:
            # Check geographical & severity match
            if not self.matches_subscription(sub, warning):
                continue

            cache_key = f"{sub.user_id}:{warning.warning_id}"
            last_dispatch = self._delivery_cache.get(cache_key)
            idempotency_key = self.generate_idempotency_key(sub.user_id, warning.warning_id, version, warning.status)

            # Check if warning is expired
            if warning.status == AlertStatus.EXPIRED:
                rec = self._notification_service.record_suppression(
                    user_id=sub.user_id,
                    warning_id=warning.warning_id,
                    idempotency_key=idempotency_key,
                    reason=DeliveryStatus.SUPPRESSED_EXPIRED,
                    message=f"Alert {warning.warning_id} is expired and was suppressed."
                )
                results.append(rec)
                continue

            # Deduplication Check
            if last_dispatch:
                last_dt = last_dispatch["sent_at"]
                last_ver = last_dispatch["version"]
                last_status = last_dispatch["status"]
                delta_minutes = (now - last_dt).total_seconds() / 60.0

                # If same version and same status and within deduplication window, suppress
                if (last_ver == version and 
                    last_status == warning.status and 
                    delta_minutes < settings.ALERT_DEDUPLICATION_WINDOW_MINUTES):
                    rec = self._notification_service.record_suppression(
                        user_id=sub.user_id,
                        warning_id=warning.warning_id,
                        idempotency_key=idempotency_key,
                        reason=DeliveryStatus.SUPPRESSED_DUPLICATE,
                        message=f"Duplicate alert suppressed for warning {warning.warning_id} (last sent {delta_minutes:.1f}m ago)."
                    )
                    results.append(rec)
                    continue

            # Format Message
            prefix = ""
            if warning.status == AlertStatus.ESCALATED:
                prefix = f"🚨 ESCALATED [{warning.source_severity}]: "
            elif warning.status == AlertStatus.UPDATED:
                prefix = f"⚠️ UPDATED [{warning.source_severity}]: "
            elif warning.status == AlertStatus.CANCELLED:
                prefix = f"✅ CANCELLED [{warning.source_severity}]: "
            else:
                prefix = f"⚠️ OFFICIAL [{warning.source_severity}]: "

            message_text = f"{prefix}{warning.bulletin_text}"

            # Dispatch in-app if enabled
            if sub.enable_in_app:
                inapp_rec = await self._notification_service.deliver_in_app(
                    user_id=sub.user_id,
                    warning=warning,
                    idempotency_key=idempotency_key,
                    message=message_text
                )
                results.append(inapp_rec)

            # Dispatch push if enabled
            if sub.enable_push:
                push_rec = await self._notification_service.deliver_push(
                    user_id=sub.user_id,
                    device_token=sub.device_token,
                    warning=warning,
                    idempotency_key=idempotency_key,
                    message=message_text
                )
                results.append(push_rec)

            # Update Delivery Cache on confirmed attempt
            self._delivery_cache[cache_key] = {
                "sent_at": now,
                "version": version,
                "status": warning.status
            }

        return results

    async def evaluate_all_active_warnings(self) -> List[AlertNotificationRecord]:
        """Evaluates all registered active warnings across all subscriptions."""
        all_warnings = self._warning_provider.get_all_warnings(active_only=False)
        total_records: List[AlertNotificationRecord] = []
        for w in all_warnings:
            records = await self.process_warning_dispatch(w)
            total_records.extend(records)
        return total_records

    def clear(self):
        """Resets engine for clean testing."""
        self._subscriptions.clear()
        self._user_subscriptions.clear()
        self._delivery_cache.clear()
