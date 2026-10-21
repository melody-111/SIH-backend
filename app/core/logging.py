import re
import logging
import json
from typing import Any, Dict

# Regex patterns for detecting and scrubbing sensitive information
SENSITIVE_PATTERNS = [
    # Authorization header / Bearer tokens
    (re.compile(r"(Bearer\s+)[A-Za-z0-9\-\._~+/]+=*", re.IGNORECASE), r"\1[REDACTED_TOKEN]"),
    # Generic tokens, keys, secrets, passwords
    (re.compile(r"((?:api[_-]?key|access[_-]?token|password|secret|auth)\s*[:=]\s*['\"]?)([^'\"\,\s\n]+)(['\"]?)", re.IGNORECASE), r"\1[REDACTED]\3"),
    # Base64 encoded audio or data URIs
    (re.compile(r"(data:audio/[a-z0-9]+;base64,)[A-Za-z0-9+/=]+", re.IGNORECASE), r"\1[REDACTED_AUDIO_BLOB]"),
    # Long base64 payloads (>100 characters)
    (re.compile(r"\"audio_base64\":\s*\"[A-Za-z0-9+/=]{100,}\"", re.IGNORECASE), r'"audio_base64": "[REDACTED_AUDIO_PAYLOAD]"'),
]

class SensitiveDataRedactor(logging.Filter):
    """
    Logging filter that intercepts log records and scrubs API keys,
    bearer tokens, credentials, and raw audio payloads.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.redact_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self.redact_value(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self.redact_value(v) for v in record.args)
        return True

    @classmethod
    def redact_text(cls, text: str) -> str:
        if not isinstance(text, str):
            return text
        sanitized = text
        for pattern, replacement in SENSITIVE_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    @classmethod
    def redact_value(cls, val: Any) -> Any:
        if isinstance(val, str):
            return cls.redact_text(val)
        return val

def setup_secure_logging(log_level: str = "INFO"):
    """Configures root and neerai loggers with the SensitiveDataRedactor filter."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    redactor = SensitiveDataRedactor()
    
    # Attach filter to existing handlers
    for handler in root_logger.handlers:
        handler.addFilter(redactor)
        
    neerai_logger = logging.getLogger("neerai")
    neerai_logger.setLevel(level)
    neerai_logger.addFilter(redactor)
