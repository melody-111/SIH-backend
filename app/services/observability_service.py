import datetime
import math
import re
import threading
import time
from collections import deque
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from app.core.config import settings
from app.db.database import DatabaseManager

class SafeErrorCategory(str, Enum):
    NONE = "NONE"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    UPSTREAM_5XX = "UPSTREAM_5XX"
    PARSE_ERROR = "PARSE_ERROR"
    AUTH_ERROR = "AUTH_ERROR"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    CLIENT_DISCONNECT = "CLIENT_DISCONNECT"
    INTERNAL_ERROR = "INTERNAL_ERROR"

def redact_sensitive(text: str) -> str:
    """Masks API keys, tokens, and authorization headers."""
    if not text:
        return text
    s = re.sub(r"(sk-[a-zA-Z0-9_-]{15,})", r"sk-[REDACTED]", text)
    s = re.sub(r"(AIza[a-zA-Z0-9_-]{15,})", r"AIza[REDACTED]", s)
    s = re.sub(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{15,}", r"\1[REDACTED]", s)
    s = re.sub(r"(password=)[^&\s]+", r"\1[REDACTED]", s, flags=re.IGNORECASE)
    return s

def sanitize_location(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    """Coarsens location to 2 decimal places (~1.1km) to avoid exact tracking."""
    if lat is None or lon is None:
        return None
    return f"{round(lat, 2)},{round(lon, 2)}"

class RequestTrace(BaseModel):
    correlation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    intent: str = "UNKNOWN"
    query_length: int = 0
    coarse_location: Optional[str] = None
    agents_planned: List[str] = Field(default_factory=list)
    agents_completed: List[str] = Field(default_factory=list)
    agents_failed: List[str] = Field(default_factory=list)
    agents_timed_out: List[str] = Field(default_factory=list)
    unavailable_sources: List[str] = Field(default_factory=list)
    is_cached: bool = False
    cache_status: str = "miss"  # fresh, stale, miss
    model_calls_count: int = 0
    tokens_used: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    estimated_cost_usd: float = 0.0
    duration_ms: float = 0.0
    status: str = "SUCCESS"  # SUCCESS, PARTIAL_FAILURE, TIMEOUT, CANCELLED, FAILED
    error_category: SafeErrorCategory = SafeErrorCategory.NONE
    error_summary: Optional[str] = None

class ObservabilityService:
    """
    Lightweight, production-safe observability and telemetry system for NeerAI.
    Captures end-to-end request traces, agent executions, safe error classifications,
    percentile latencies, cache metrics, and token usage without logging sensitive user text.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ObservabilityService, cls).__new__(cls)
                cls._instance._init_service()
            return cls._instance

    def _init_service(self):
        self._traces: deque[RequestTrace] = deque(maxlen=500)
        self._active_requests: int = 0
        self._active_lock = threading.Lock()
        self._db = DatabaseManager()

    def increment_active_requests(self) -> int:
        with self._active_lock:
            self._active_requests += 1
            return self._active_requests

    def decrement_active_requests(self) -> int:
        with self._active_lock:
            self._active_requests = max(0, self._active_requests - 1)
            return self._active_requests

    @property
    def active_requests_count(self) -> int:
        with self._active_lock:
            return self._active_requests

    def record_agent_execution(
        self,
        execution_id: str,
        correlation_id: str,
        agent_name: str,
        status: str,
        execution_time_ms: float,
        timestamp: Optional[str] = None
    ) -> None:
        """Persists agent execution metric into SQLite and telemetry stream."""
        try:
            self._db.log_agent_execution(
                execution_id=execution_id,
                correlation_id=correlation_id,
                agent_name=agent_name,
                status=status,
                execution_time_ms=execution_time_ms,
                timestamp=timestamp
            )
        except Exception:
            pass

    def record_trace(self, trace: RequestTrace) -> None:
        """Appends a completed request trace into the in-memory ring buffer."""
        # Clean sensitive text in summary
        if trace.error_summary:
            trace.error_summary = redact_sensitive(trace.error_summary)
        self._traces.append(trace)

    def get_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns recent request traces."""
        traces_list = list(self._traces)
        sliced = traces_list[-limit:] if limit > 0 else traces_list
        return [t.model_dump() for t in reversed(sliced)]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Computes real-time telemetry metrics including median (p50),
        p90, p95 latencies, error breakdown, and token usage.
        """
        traces = list(self._traces)
        total = len(traces)

        if total == 0:
            return {
                "total_requests": 0,
                "active_requests": self.active_requests_count,
                "successful_requests": 0,
                "partial_failure_requests": 0,
                "timed_out_requests": 0,
                "cancelled_requests": 0,
                "failed_requests": 0,
                "error_rate_percent": 0.0,
                "latency_ms": {
                    "min": 0.0,
                    "median_p50": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "max": 0.0,
                    "average": 0.0
                },
                "cache_metrics": {
                    "hits": 0,
                    "misses": 0,
                    "stale_served": 0,
                    "hit_ratio_percent": 0.0
                },
                "token_usage": {
                    "total_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0
                },
                "error_categories": {}
            }

        successes = sum(1 for t in traces if t.status == "SUCCESS")
        partials = sum(1 for t in traces if t.status == "PARTIAL_FAILURE")
        timeouts = sum(1 for t in traces if t.status == "TIMEOUT")
        cancelled = sum(1 for t in traces if t.status == "CANCELLED")
        failed = sum(1 for t in traces if t.status in ["FAILED", "ERROR"])
        non_success = total - successes

        # Latency calculation
        durations = sorted([t.duration_ms for t in traces])
        min_lat = round(durations[0], 2)
        max_lat = round(durations[-1], 2)
        avg_lat = round(sum(durations) / total, 2)
        
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            idx = int(math.ceil(p * len(data))) - 1
            return round(data[max(0, min(idx, len(data) - 1))], 2)

        p50 = percentile(durations, 0.50)
        p90 = percentile(durations, 0.90)
        p95 = percentile(durations, 0.95)

        # Cache metrics
        hits = sum(1 for t in traces if t.is_cached or t.cache_status in ["fresh", "stale"])
        stale = sum(1 for t in traces if t.cache_status == "stale")
        misses = sum(1 for t in traces if not t.is_cached and t.cache_status == "miss")
        hit_ratio = round((hits / total) * 100.0, 2)

        # Token usage
        calls = sum(t.model_calls_count for t in traces)
        p_tok = sum(t.tokens_used.get("prompt_tokens", 0) for t in traces)
        c_tok = sum(t.tokens_used.get("completion_tokens", 0) for t in traces)
        tot_tok = sum(t.tokens_used.get("total_tokens", 0) for t in traces)
        tot_cost = round(sum(t.estimated_cost_usd for t in traces), 6)

        # Errors breakdown
        err_cat_counts: Dict[str, int] = {}
        for t in traces:
            cat = t.error_category.value if isinstance(t.error_category, SafeErrorCategory) else str(t.error_category)
            if cat != SafeErrorCategory.NONE.value:
                err_cat_counts[cat] = err_cat_counts.get(cat, 0) + 1

        return {
            "total_requests": total,
            "active_requests": self.active_requests_count,
            "successful_requests": successes,
            "partial_failure_requests": partials,
            "timed_out_requests": timeouts,
            "cancelled_requests": cancelled,
            "failed_requests": failed,
            "error_rate_percent": round((non_success / total) * 100.0, 2),
            "latency_ms": {
                "min": min_lat,
                "median_p50": p50,
                "p90": p90,
                "p95": p95,
                "max": max_lat,
                "average": avg_lat
            },
            "cache_metrics": {
                "hits": hits,
                "misses": misses,
                "stale_served": stale,
                "hit_ratio_percent": hit_ratio
            },
            "token_usage": {
                "total_calls": calls,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": tot_tok,
                "estimated_cost_usd": tot_cost
            },
            "error_categories": err_cat_counts
        }

    def clear_metrics(self) -> None:
        """Resets in-memory metrics buffers (useful for clean benchmark runs)."""
        self._traces.clear()
        with self._active_lock:
            self._active_requests = 0
