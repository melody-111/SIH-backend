from fastapi import APIRouter, Query, Header, HTTPException, Body
from typing import Optional, List, Dict, Any
from app.schemas.hazard_alert import (
    OfficialWarningDetail,
    AlertSubscriptionRequest,
    AlertSubscription,
    AlertNotificationRecord
)
from app.services.official_warning_provider import OfficialWarningProvider
from app.services.alert_rules_engine import AlertRulesEngine
from app.services.notification_service import NotificationDeliveryService

router = APIRouter(prefix="/alerts", tags=["alerts"])

warning_provider = OfficialWarningProvider()
rules_engine = AlertRulesEngine()
notification_service = NotificationDeliveryService()

def _resolve_user_id(header_user: Optional[str], param_user: Optional[str]) -> str:
    uid = header_user or param_user
    if not uid:
        raise HTTPException(status_code=401, detail="Authentication required: Provide user_id or X-User-ID header.")
    return uid.strip()

@router.get("/active", response_model=List[OfficialWarningDetail])
async def get_active_alerts(
    latitude: float = Query(..., ge=-90.0, le=90.0),
    longitude: float = Query(..., ge=-180.0, le=180.0),
    region_name: Optional[str] = Query(None)
):
    """
    Returns active official alerts matching geographical coverage and validity period.
    """
    return warning_provider.get_warnings_for_location(
        lat=latitude,
        lon=longitude,
        region_name=region_name,
        include_expired=False
    )

@router.post("/subscribe", response_model=AlertSubscription)
async def subscribe_alerts(
    request: AlertSubscriptionRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Registers or updates an alert subscription with geographic coverage and minimum severity.
    Enforces user isolation.
    """
    if x_user_id and x_user_id != request.user_id:
        raise HTTPException(status_code=403, detail="User mismatch: Cannot register subscription for another user.")

    return rules_engine.add_subscription(request)

@router.get("/subscriptions", response_model=List[AlertSubscription])
async def get_subscriptions(
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Retrieves subscriptions belonging strictly to the authenticated user.
    """
    uid = _resolve_user_id(x_user_id, user_id)
    return rules_engine.get_user_subscriptions(uid)

@router.delete("/subscriptions/{subscription_id}")
async def delete_subscription(
    subscription_id: str,
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Deletes an alert subscription with strict ownership validation.
    """
    uid = _resolve_user_id(x_user_id, user_id)
    deleted = rules_engine.delete_subscription(uid, subscription_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Subscription not found or not owned by user.")
    return {"status": "success", "message": f"Subscription {subscription_id} removed."}

@router.get("/history", response_model=List[AlertNotificationRecord])
async def get_alert_history(
    user_id: Optional[str] = Query(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    limit: int = Query(50, ge=1, le=100, description="Max notification records (1-100)"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Retrieves alert notification delivery history with tenant isolation and pagination.
    """
    uid = _resolve_user_id(x_user_id, user_id)
    records = notification_service.get_user_history(uid)
    return records[offset:offset + limit]

@router.post("/test-deliver", response_model=List[AlertNotificationRecord])
async def test_deliver_alerts(
    payload: Dict[str, Any] = Body(...),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """
    Simulates alert evaluation and delivery in sandbox mode without sending live emergency alerts.
    """
    uid = _resolve_user_id(x_user_id, payload.get("user_id"))
    warning_id = payload.get("warning_id")

    if warning_id:
        # Evaluate specific warning
        all_warnings = warning_provider.get_all_warnings(active_only=False)
        target = next((w for w in all_warnings if w.warning_id == warning_id), None)
        if not target:
            raise HTTPException(status_code=404, detail=f"Warning {warning_id} not found.")
        return await rules_engine.process_warning_dispatch(target, target_user_id=uid)
    else:
        # Evaluate all active warnings for this user
        all_warnings = warning_provider.get_all_warnings(active_only=True)
        results = []
        for w in all_warnings:
            records = await rules_engine.process_warning_dispatch(w, target_user_id=uid)
            results.extend(records)
        return results
