import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple, Type, TypeVar
import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.schemas.llm import (
    TaskType,
    LLMTelemetryRecord,
    IntentClassificationOutput,
    QueryPlanOutput,
    SynthesizedAdvisoryOutput
)

logger = logging.getLogger("neerai.llm")

T = TypeVar("T", bound=BaseModel)

def redact_secrets(text: str) -> str:
    """Removes sensitive keys, tokens, and passwords from log strings."""
    if not text:
        return text
    # Mask OpenAI, Anthropic, Gemini, or generic keys
    redacted = re.sub(r"(sk-[a-zA-Z0-9_-]{20,})", r"sk-[REDACTED]", text)
    redacted = re.sub(r"(AIza[a-zA-Z0-9_-]{20,})", r"AIza[REDACTED]", redacted)
    redacted = re.sub(r"(Bearer\s+)[a-zA-Z0-9_\-\.]{20,}", r"\1[REDACTED]", redacted)
    return redacted

class BaseLLMProvider:
    """Base interface for all LLM providers in NeerAI."""
    def __init__(self, name: str):
        self.name = name

    @property
    def is_available(self) -> bool:
        raise NotImplementedError

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout_seconds: float = 8.0
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        """
        Executes completion and returns:
        (raw_text, token_usage_dict, estimated_cost_usd)
        """
        raise NotImplementedError

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        timeout_seconds: float = 8.0,
        response_format: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        """
        Executes multi-turn chat completion and returns:
        (raw_text, token_usage_dict, estimated_cost_usd)
        """
        raise NotImplementedError

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API integration supporting gpt-4o, gpt-4o-mini, etc."""
    def __init__(self):
        super().__init__("openai")
        self.api_key = settings.OPENAI_API_KEY or settings.LLM_API_KEY
        self._client = None
        if self.is_available:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)

    @property
    def is_available(self) -> bool:
        key = self.api_key
        return bool(key and key.strip() and key.strip() != "dummy" and not key.startswith("mock_"))

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout_seconds: float = 8.0
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        if not self._client:
            raise RuntimeError("OpenAI client not configured or key is dummy/missing.")

        response = await asyncio.wait_for(
            self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            ),
            timeout=timeout_seconds
        )
        content = response.choices[0].message.content or "{}"
        usage = None
        cost = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            # Approximate cost for gpt-4o-mini ($0.15/1M in, $0.60/1M out)
            cost = round((response.usage.prompt_tokens * 0.00000015) + (response.usage.completion_tokens * 0.00000060), 6)
        return content, usage, cost

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        timeout_seconds: float = 8.0,
        response_format: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        if not self._client:
            raise RuntimeError("OpenAI client not configured or key is dummy/missing.")
        formatted_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            formatted_messages.append({
                "role": m.get("role", "user"),
                "content": m.get("content") or m.get("text", "")
            })
        
        kwargs = {
            "model": model or "gpt-4o-mini",
            "messages": formatted_messages,
            "temperature": temperature
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
            
        response = await asyncio.wait_for(
            self._client.chat.completions.create(**kwargs),
            timeout=timeout_seconds
        )
        content = response.choices[0].message.content or ""
        usage = None
        cost = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            cost = round((response.usage.prompt_tokens * 0.00000015) + (response.usage.completion_tokens * 0.00000060), 6)
        return content, usage, cost

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API integration via HTTP."""
    def __init__(self):
        super().__init__("anthropic")
        self.api_key = settings.ANTHROPIC_API_KEY

    @property
    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip() and self.api_key != "dummy")

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout_seconds: float = 8.0
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        if not self.is_available:
            raise RuntimeError("Anthropic API key is not configured.")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": model or "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [{}])[0].get("text", "{}")
            usage = data.get("usage", {})
            tokens = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            }
            return content, tokens, None

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        timeout_seconds: float = 8.0,
        response_format: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        if not self.is_available:
            raise RuntimeError("Anthropic API key is not configured.")
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        anthropic_msgs = []
        for m in messages:
            anthropic_msgs.append({
                "role": "user" if m.get("role") in ["user", "USER"] else "assistant",
                "content": m.get("content") or m.get("text", "")
            })
        payload = {
            "model": model or "claude-3-haiku-20240307",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": anthropic_msgs,
            "temperature": temperature
        }
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("content", [{}])[0].get("text", "")
            usage = data.get("usage", {})
            tokens = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            }
            return content, tokens, None

class GeminiProvider(BaseLLMProvider):
    """Google Gemini API integration via HTTP."""
    def __init__(self):
        super().__init__("gemini")
        self.api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")

    @property
    def is_available(self) -> bool:
        key = self.api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        return bool(key and key.strip() and key.strip() != "dummy" and not key.startswith("mock_"))

    async def check_diagnostics(self) -> Dict[str, Any]:
        """
        Validates configured Gemini credentials without exposing the key.
        Performs a minimal health ping if key is configured.
        """
        key = self.api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        if not self.is_available:
            return {
                "provider": "gemini",
                "status": "UNCONFIGURED",
                "is_available": False,
                "message": "GEMINI_API_KEY is not configured or empty.",
                "action": "Set GEMINI_API_KEY in backend/.env to enable Google Gemini models."
            }

        masked_key = key[:4] + "..." + key[-4:] if len(key) > 8 else "[REDACTED]"
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return {
                        "provider": "gemini",
                        "status": "READY",
                        "is_available": True,
                        "masked_key": masked_key,
                        "message": "Google Gemini API key verified and operational."
                    }
                else:
                    return {
                        "provider": "gemini",
                        "status": "AUTH_FAILED",
                        "is_available": False,
                        "masked_key": masked_key,
                        "status_code": resp.status_code,
                        "message": f"Gemini API returned HTTP {resp.status_code}: invalid credentials or quota exceeded."
                    }
        except Exception as e:
            return {
                "provider": "gemini",
                "status": "NETWORK_UNAVAILABLE",
                "is_available": False,
                "masked_key": masked_key,
                "message": f"Unable to reach Google Gemini API endpoint: {redact_secrets(str(e))}"
            }

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout_seconds: float = 8.0
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        if not self.is_available:
            raise RuntimeError("Gemini API key is not configured.")

        # Ensure model is a recognized Gemini model name
        gemini_model = model or "gemini-1.5-flash"
        if not gemini_model.startswith("gemini-"):
            gemini_model = "gemini-1.5-flash"

        key = self.api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={key}"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": temperature
            }
        }
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])
            content = parts[0].get("text", "{}")
            # Strip markdown code fence if returned
            if content.strip().startswith("```"):
                clean_c = re.sub(r"^```[a-zA-Z]*\n?", "", content.strip())
                clean_c = re.sub(r"\n?```$", "", clean_c.strip())
                content = clean_c.strip()

            usage_meta = data.get("usageMetadata", {})
            tokens = {
                "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                "total_tokens": usage_meta.get("totalTokenCount", 0)
            }
            return content, tokens, None

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        timeout_seconds: float = 8.0,
        response_format: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        if not self.is_available:
            raise RuntimeError("Gemini API key is not configured.")

        # Ensure model is a recognized Gemini model name
        gemini_model = model or "gemini-1.5-flash"
        if not gemini_model.startswith("gemini-"):
            gemini_model = "gemini-1.5-flash"

        key = self.api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={key}"
        
        contents = []
        last_role = None
        for m in messages:
            role = "user" if m.get("role", "").lower() in ["user", "human"] else "model"
            text_part = (m.get("content") or m.get("text", "")).strip()
            if not text_part:
                continue
            if contents and last_role == role:
                contents[-1]["parts"][0]["text"] += f"\n\n{text_part}"
            else:
                contents.append({
                    "role": role,
                    "parts": [{"text": text_part}]
                })
                last_role = role

        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
        elif contents[0]["role"] != "user":
            contents[0]["role"] = "user"
            
        gen_config = {"temperature": temperature}
        if response_format == "json":
            gen_config["response_mime_type"] = "application/json"
            
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": gen_config
        }
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])
            content = parts[0].get("text", "")
            usage_meta = data.get("usageMetadata", {})

            tokens = {
                "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                "total_tokens": usage_meta.get("totalTokenCount", 0)
            }
            return content, tokens, None

class OllamaProvider(BaseLLMProvider):
    """
    Local, open-source Ollama integration (Llama-3, Mistral, Qwen, etc.).
    100% free, runs locally or on LAN with zero external network fees.
    """
    def __init__(self):
        super().__init__("ollama")
        self.base_url = (getattr(settings, "OLLAMA_BASE_URL", None) or "http://localhost:11434").rstrip("/")
        self.default_model = getattr(settings, "OLLAMA_MODEL", None) or "llama3"
        self._last_reachable = None
        self._last_checked_at = 0.0

    @property
    def is_available(self) -> bool:
        pref = settings.LLM_PROVIDER.lower()
        if pref != "ollama" and os.getenv("ENABLE_OLLAMA", "").lower() not in ("1", "true", "yes"):
            return False
        if self._last_reachable is False and (time.time() - self._last_checked_at) < 30.0:
            return False
        return True

    async def check_diagnostics(self) -> Dict[str, Any]:
        """
        Inspects whether Ollama is reachable and whether the configured model is installed.
        Returns explicit status, error details, and installation / pull commands if missing.
        """
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                self._last_checked_at = time.time()
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    has_model = any(self.default_model in m for m in models)
                    self._last_reachable = True
                    if has_model:
                        return {
                            "provider": "ollama",
                            "status": "READY",
                            "is_available": True,
                            "endpoint": self.base_url,
                            "active_model": self.default_model,
                            "installed_models": models,
                            "message": f"Ollama is running locally and model '{self.default_model}' is installed and ready."
                        }
                    else:
                        return {
                            "provider": "ollama",
                            "status": "MODEL_MISSING",
                            "is_available": False,
                            "endpoint": self.base_url,
                            "active_model": self.default_model,
                            "installed_models": models,
                            "message": f"Ollama is running, but configured model '{self.default_model}' is not pulled.",
                            "action": f"Run 'ollama pull {self.default_model}' to download model weights."
                        }
                else:
                    self._last_reachable = False
                    return {
                        "provider": "ollama",
                        "status": "HTTP_ERROR",
                        "is_available": False,
                        "endpoint": self.base_url,
                        "status_code": resp.status_code,
                        "message": f"Ollama daemon returned HTTP {resp.status_code}."
                    }
        except Exception:
            self._last_checked_at = time.time()
            self._last_reachable = False
            return {
                "provider": "ollama",
                "status": "UNREACHABLE",
                "is_available": False,
                "endpoint": self.base_url,
                "message": f"Ollama daemon is not reachable at {self.base_url}. Service is not running or not installed.",
                "action": "Install Ollama from https://ollama.com or 'brew install ollama', then launch with 'ollama serve'."
            }

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout_seconds: float = 8.0
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        url = f"{self.base_url}/v1/chat/completions"
        target_model = self.default_model
        if model and not model.startswith("gpt-") and not model.startswith("claude-") and not model.startswith("gemini-"):
            target_model = model

        payload = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            content = choices[0].get("message", {}).get("content", "{}") if choices else "{}"
            usage = data.get("usage", {})
            tokens = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
            return content, tokens, 0.0

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        timeout_seconds: float = 8.0,
        response_format: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        url = f"{self.base_url}/v1/chat/completions"
        target_model = self.default_model
        if model and not model.startswith("gpt-") and not model.startswith("claude-") and not model.startswith("gemini-"):
            target_model = model

        formatted_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            formatted_messages.append({
                "role": "user" if m.get("role") in ["user", "USER"] else "assistant",
                "content": m.get("content") or m.get("text", "")
            })

        payload = {
            "model": target_model,
            "messages": formatted_messages,
            "temperature": temperature
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = data.get("usage", {})
            tokens = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
            return content, tokens, 0.0

class DeterministicFallbackProvider(BaseLLMProvider):
    """
    Offline, local, zero-network deterministic provider.
    Guarantees zero paid API calls during offline evals and provides verified safety fallbacks.
    """
    def __init__(self):
        super().__init__("deterministic")

    @property
    def is_available(self) -> bool:
        return True

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: str,
        model: str,
        temperature: float = 0.1,
        timeout_seconds: float = 8.0
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        return json.dumps({
            "status": "deterministic_fallback",
            "message": "Fallback completed safely without external LLM."
        }), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0

    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        timeout_seconds: float = 8.0,
        response_format: Optional[str] = None
    ) -> Tuple[str, Optional[Dict[str, int]], Optional[float]]:
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0

class MultiModelLLMService:
    """
    Central multi-model LLM service for NeerAI.
    Manages provider selection, dynamic model routing (fast vs reasoning),
    strict Pydantic validation, safe retries, and deterministic fallbacks.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MultiModelLLMService, cls).__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self):
        self.providers: Dict[str, BaseLLMProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "gemini": GeminiProvider(),
            "ollama": OllamaProvider(),
            "deterministic": DeterministicFallbackProvider()
        }
        self.telemetry_history: deque = deque(maxlen=100)
        self.fast_model = settings.FAST_LLM_MODEL
        self.reasoning_model = settings.REASONING_LLM_MODEL
        self.timeout = settings.LLM_REQUEST_TIMEOUT_SECONDS
        self.max_retries = settings.LLM_MAX_RETRIES

    def resolve_provider(self) -> Tuple[str, BaseLLMProvider]:
        """Resolves configured provider or falls back to first available."""
        pref = settings.LLM_PROVIDER.lower()
        if pref in self.providers and self.providers[pref].is_available:
            return pref, self.providers[pref]

        # Auto-discovery prioritizing Gemini
        if self.providers["gemini"].is_available:
            return "gemini", self.providers["gemini"]
        if self.providers["openai"].is_available:
            return "openai", self.providers["openai"]
        if self.providers["anthropic"].is_available:
            return "anthropic", self.providers["anthropic"]
        if self.providers["ollama"].is_available:
            return "ollama", self.providers["ollama"]

        return "deterministic", self.providers["deterministic"]

    def _record_telemetry(
        self,
        task: str,
        provider: str,
        model: str,
        status: str,
        duration_ms: float,
        tokens_used: Optional[Dict[str, int]] = None,
        cost_usd: Optional[float] = None,
        correlation_id: str = "unlinked",
        error: Optional[str] = None
    ) -> LLMTelemetryRecord:
        record = LLMTelemetryRecord(
            task=task,
            provider=provider,
            model=model,
            status=status,
            duration_ms=round(duration_ms, 2),
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            correlation_id=correlation_id,
            error=redact_secrets(error) if error else None
        )
        self.telemetry_history.append(record)
        return record

    def get_telemetry_history(self) -> List[Dict[str, Any]]:
        return [t.model_dump() for t in self.telemetry_history]

    async def get_provider_registry_status(self) -> Dict[str, Any]:
        """
        Evaluates real runtime health across all supported model providers
        without exposing credentials. Accurately reports active vs fallback states.
        """
        results = {}

        # Gemini
        gemini_prov: GeminiProvider = self.providers.get("gemini")
        if gemini_prov:
            results["gemini"] = await gemini_prov.check_diagnostics()

        # Ollama
        ollama_prov: OllamaProvider = self.providers.get("ollama")
        if ollama_prov:
            results["ollama"] = await ollama_prov.check_diagnostics()

        # OpenAI
        openai_prov: OpenAIProvider = self.providers.get("openai")
        if openai_prov:
            results["openai"] = {
                "provider": "openai",
                "status": "READY" if openai_prov.is_available else "UNCONFIGURED",
                "is_available": openai_prov.is_available,
                "message": "OpenAI API configured" if openai_prov.is_available else "OPENAI_API_KEY is not configured or empty."
            }

        # Deterministic Fallback
        results["deterministic"] = {
            "provider": "deterministic",
            "status": "READY",
            "is_available": True,
            "message": "Deterministic safety rules engine is operational (zero external network dependency)."
        }

        active_name, _ = self.resolve_provider()
        ready_count = sum(1 for p in results.values() if p.get("is_available"))

        return {
            "active_provider": active_name,
            "available_providers_count": ready_count,
            "collaboration_mode": "MULTI_MODEL_HYBRID" if ready_count > 1 else ("DETERMINISTIC_SAFE_MODE" if active_name == "deterministic" else "SINGLE_PROVIDER_ACTIVE"),
            "providers": results
        }

    def _clean_json_output(self, raw_text: str) -> str:
        """Strips markdown code blocks and whitespace from LLM output."""
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    # -------------------------------------------------------------------------
    # Task 1: Intent Classification (Fast Model)
    # -------------------------------------------------------------------------
    async def classify_intent(
        self,
        message: str,
        correlation_id: str = "intent_req"
    ) -> IntentClassificationOutput:
        from app.reasoning.intent_engine import IntentEngine
        rule_engine = IntentEngine()
        deterministic_intent = rule_engine.determine_intent(message)

        provider_name, provider = self.resolve_provider()
        model_name = self.fast_model

        if provider_name == "deterministic" or not provider.is_available:
            self._record_telemetry(
                task=TaskType.INTENT_CLASSIFICATION.value,
                provider=provider_name,
                model=model_name,
                status="FALLBACK",
                duration_ms=0.1,
                correlation_id=correlation_id
            )
            return IntentClassificationOutput(
                intent=deterministic_intent,
                confidence=1.0,
                reasoning="Deterministic rule classification used (no external LLM configured)."
            )

        system_prompt = """You are an intent classification engine for NeerAI marine assistant.
Classify the user's message into one of these intents:
- GENERAL_EXPLANATORY: Educational marine questions (e.g. what is chlorophyll)
- HAZARD: Cyclones, severe alerts, storm warnings, dangerous seas
- FOLLOW_UP: Spatial or temporal follow-ups (e.g. what about tomorrow, wahan kaisa hai)
- FISHING_LOCATION: Fishing spots, Machhli catching areas, PFZ
- OCEAN_INFO: Sea Surface Temperature (SST), ocean currents
- MARINE_CONDITION: Wave height, wind speed, marine weather
- ROUTE: Navigational passage, distance between harbors
- GENERAL_MARINE: Other general ocean queries

Output JSON ONLY conforming to schema:
{
  "intent": "INTENT_NAME",
  "confidence": 0.95,
  "reasoning": "Brief justification"
}"""
        start = time.perf_counter()
        try:
            raw_text, tokens, cost = await provider.generate_completion(
                prompt=f"User Message: {message}",
                system_prompt=system_prompt,
                model=model_name,
                temperature=0.0,
                timeout_seconds=self.timeout
            )
            clean = self._clean_json_output(raw_text)
            parsed = json.loads(clean)
            validated = IntentClassificationOutput(**parsed)
            duration_ms = (time.perf_counter() - start) * 1000.0

            self._record_telemetry(
                task=TaskType.INTENT_CLASSIFICATION.value,
                provider=provider_name,
                model=model_name,
                status="SUCCESS",
                duration_ms=duration_ms,
                tokens_used=tokens,
                cost_usd=cost,
                correlation_id=correlation_id
            )
            return validated
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._record_telemetry(
                task=TaskType.INTENT_CLASSIFICATION.value,
                provider=provider_name,
                model=model_name,
                status="FALLBACK",
                duration_ms=duration_ms,
                correlation_id=correlation_id,
                error=str(e)
            )
            # Safe deterministic fallback
            return IntentClassificationOutput(
                intent=deterministic_intent,
                confidence=0.9,
                reasoning=f"Fallback to deterministic engine due to provider error: {redact_secrets(str(e))}"
            )

    # -------------------------------------------------------------------------
    # Task 2: Multilingual Advisory Synthesis (Reasoning Model)
    # -------------------------------------------------------------------------
    async def synthesize_advisory(
        self,
        query_message: str,
        language: str,
        marine_context_dict: Dict[str, Any],
        risk_level: str,
        recommendation_dict: Dict[str, Any],
        correlation_id: str = "syn_req"
    ) -> Optional[SynthesizedAdvisoryOutput]:
        """
        Synthesizes structured advisory via Reasoning LLM when available.
        Returns None if provider is unavailable or fails, signaling callers
        to use the strict deterministic synthesis in ResponseAgent.
        """
        provider_name, provider = self.resolve_provider()
        model_name = self.reasoning_model

        if provider_name == "deterministic" or not provider.is_available:
            self._record_telemetry(
                task=TaskType.FINAL_SYNTHESIS.value,
                provider=provider_name,
                model=model_name,
                status="FALLBACK",
                duration_ms=0.1,
                correlation_id=correlation_id
            )
            return None

        system_prompt = """You are NeerAI, an expert Indian Marine and Ocean Intelligence assistant for artisanal and commercial fishermen.
Rules:
1. Synthesize ONE clear natural-language advisory in the EXACT user language requested.
2. NEVER fabricate GPS coordinates or official storm warnings.
3. If official PFZ data or satellite coverage is unavailable, clearly state it is unavailable.
4. Output JSON ONLY conforming to schema:
{
  "answer": "Clear conversational advisory text...",
  "language": "hi or en",
  "risk_assessment": "LOW | MODERATE | HIGH | UNKNOWN",
  "recommended_coordinates": null or "lat,lon",
  "safety_guidance": ["Safety point 1", "Safety point 2"],
  "caveats": ["Caveat or data source limitations"],
  "map_action_type": "center"
}"""

        prompt_payload = json.dumps({
            "user_query": query_message,
            "language": language,
            "risk_level": risk_level,
            "recommendation": recommendation_dict,
            "marine_data": marine_context_dict
        }, default=str)

        start = time.perf_counter()
        try:
            raw_text, tokens, cost = await provider.generate_completion(
                prompt=f"Evidence & Query: {prompt_payload}",
                system_prompt=system_prompt,
                model=model_name,
                temperature=0.2,
                timeout_seconds=self.timeout
            )
            clean = self._clean_json_output(raw_text)
            parsed = json.loads(clean)
            validated = SynthesizedAdvisoryOutput(**parsed)
            duration_ms = (time.perf_counter() - start) * 1000.0

            self._record_telemetry(
                task=TaskType.FINAL_SYNTHESIS.value,
                provider=provider_name,
                model=model_name,
                status="SUCCESS",
                duration_ms=duration_ms,
                tokens_used=tokens,
                cost_usd=cost,
                correlation_id=correlation_id
            )
            return validated
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._record_telemetry(
                task=TaskType.FINAL_SYNTHESIS.value,
                provider=provider_name,
                model=model_name,
                status="FALLBACK",
                correlation_id=correlation_id,
                error=str(e)
            )
            return None

    # -------------------------------------------------------------------------
    # Task 3: Natural Conversational Chat (Fast / General Model)
    # -------------------------------------------------------------------------
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        language: str = "en",
        correlation_id: str = "chat_req"
    ) -> Optional[str]:
        """
        Generates natural, conversational completion using active LLM provider (Gemini/Ollama/OpenAI).
        Returns None when no external LLM is configured or available, signaling fallback.
        """
        provider_name, provider = self.resolve_provider()
        model_name = self.fast_model

        if provider_name == "deterministic" or not provider.is_available:
            self._record_telemetry(
                task="CHAT_COMPLETION",
                provider=provider_name,
                model=model_name,
                status="FALLBACK",
                duration_ms=0.1,
                correlation_id=correlation_id
            )
            return None

        default_system = (
            "You are NeerAI, an intelligent, empathetic, and friendly Maritime AI Copilot for fishermen, navigators, and coastal communities. "
            "You converse naturally, warmly, and helpfully in English, Hindi, and Hinglish.\n"
            "Rules:\n"
            "1. Answer the user's specific query conversationally and directly. Do not repeat robotic greetings on every message.\n"
            "2. Match the user's language and tone (English, Hindi, or Hinglish).\n"
            "3. If the user expresses personal feelings, fatigue, or sadness (e.g. 'Mujhe aaj achha nahi lag raha'), respond empathetically and supportively as a caring copilot.\n"
            "4. NEVER invent or fabricate real-world wave heights, current wind, SST, or storm alerts.\n"
            "5. If asked about ocean science (e.g. ocean currents, salinity), explain clearly and intuitively.\n"
            "6. For small talk or greetings, be concise, warm, and helpful."
        )
        sys_prompt = system_prompt or default_system
        start = time.perf_counter()
        try:
            content, tokens, cost = await provider.generate_chat_completion(
                messages=messages,
                system_prompt=sys_prompt,
                model=model_name,
                temperature=0.6,
                timeout_seconds=self.timeout
            )
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._record_telemetry(
                task="CHAT_COMPLETION",
                provider=provider_name,
                model=model_name,
                status="SUCCESS",
                duration_ms=duration_ms,
                tokens_used=tokens,
                cost_usd=cost,
                correlation_id=correlation_id
            )
            return content.strip() if content else None
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._record_telemetry(
                task="CHAT_COMPLETION",
                provider=provider_name,
                model=model_name,
                status="FALLBACK",
                duration_ms=duration_ms,
                correlation_id=correlation_id,
                error=str(e)
            )
            return None

