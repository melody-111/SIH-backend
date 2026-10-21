import asyncio
import time
from typing import Dict, Any, Optional
import httpx
from app.core.config import settings

class ConnectorResult:
    def __init__(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        source: str = "",
        error: Optional[str] = None,
        status_code: Optional[int] = None,
        latency_ms: float = 0.0,
        is_rate_limited: bool = False,
        connector_status: Optional[str] = None
    ):
        self.success = success
        self.data = data or {}
        self.source = source
        self.error = error
        self.status_code = status_code
        self.latency_ms = latency_ms
        self.is_rate_limited = is_rate_limited
        self.connector_status = connector_status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "source": self.source,
            "data": self.data,
            "error": self.error,
            "status_code": self.status_code,
            "latency_ms": round(self.latency_ms, 2),
            "is_rate_limited": self.is_rate_limited,
            "connector_status": self.connector_status
        }

class BaseConnector:
    """
    Base asynchronous HTTP connector with timeouts, bounded exponential backoff retries,
    and structured error capturing.
    """
    def __init__(self, name: str, base_url: str, default_timeout: Optional[float] = None):
        self.name = name
        self.base_url = base_url
        self.timeout = default_timeout or settings.CONNECTOR_TIMEOUT_SECONDS
        self.max_retries = settings.CONNECTOR_MAX_RETRIES

    async def get_with_retry(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> ConnectorResult:
        start_time = time.perf_counter()
        last_error = ""
        status_code = None

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.get(url, params=params, headers=headers)
                    latency = (time.perf_counter() - start_time) * 1000.0
                    status_code = resp.status_code

                    if resp.status_code == 200:
                        try:
                            json_data = resp.json()
                            return ConnectorResult(
                                success=True,
                                data=json_data,
                                source=self.name,
                                status_code=200,
                                latency_ms=latency
                            )
                        except Exception as parse_err:
                            return ConnectorResult(
                                success=False,
                                source=self.name,
                                error=f"Malformed JSON response: {str(parse_err)}",
                                status_code=200,
                                latency_ms=latency
                            )
                    elif resp.status_code == 429:
                        return ConnectorResult(
                            success=False,
                            source=self.name,
                            error="Rate limit exceeded (HTTP 429)",
                            status_code=429,
                            latency_ms=latency,
                            is_rate_limited=True
                        )
                    elif resp.status_code >= 500 and attempt < self.max_retries:
                        # Server error, transient retry
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
                    else:
                        return ConnectorResult(
                            success=False,
                            source=self.name,
                            error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                            status_code=resp.status_code,
                            latency_ms=latency
                        )
            except httpx.TimeoutException:
                last_error = f"Request timed out after {self.timeout}s"
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
            except httpx.ConnectError:
                last_error = "Connection failed to provider endpoint"
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
            except Exception as e:
                last_error = f"Network or transport error: {type(e).__name__}"
                break

        latency = (time.perf_counter() - start_time) * 1000.0
        return ConnectorResult(
            success=False,
            source=self.name,
            error=last_error,
            status_code=status_code,
            latency_ms=latency
        )
