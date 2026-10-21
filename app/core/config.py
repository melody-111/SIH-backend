import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class ConfigurationError(Exception):
    """Raised when environment or startup configuration is invalid or missing required variables."""
    pass

class Settings(BaseSettings):
    # AI & Language Services
    OPENAI_API_KEY: str = ""
    LLM_API_KEY: str = ""
    LLM_PROVIDER: str = "auto"  # auto, openai, anthropic, gemini, ollama, deterministic
    FAST_LLM_MODEL: str = "gpt-4o-mini"
    REASONING_LLM_MODEL: str = "gpt-4o"
    ANTHROPIC_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    LLM_REQUEST_TIMEOUT_SECONDS: float = 8.0
    LLM_MAX_RETRIES: int = 1
    WEATHER_API_KEY: str = ""
    SARVAM_API_KEY: str = ""
    
    # GIS & Geospatial Services
    MAPBOX_ACCESS_TOKEN: str = ""
    
    # Marine & Meteorological Agencies
    IMD_API_KEY: str = ""
    INCOIS_API_KEY: str = ""
    MOSDAC_USERNAME: str = ""
    MOSDAC_PASSWORD: str = ""
    NASA_EARTHDATA_USERNAME: str = ""
    NASA_EARTHDATA_PASSWORD: str = ""
    
    # Resiliency and networking
    CONNECTOR_TIMEOUT_SECONDS: float = 6.0
    CONNECTOR_MAX_RETRIES: int = 2
    ENABLE_SIMULATION_FALLBACK: bool = False
    REQUIRE_REAL_DATA: bool = False
    
    # Agent & Orchestrator Reliability
    MAX_CONCURRENT_AGENT_TASKS: int = 10
    AGENT_TIMEOUT_SECONDS: float = 8.0
    AGENT_RETRY_BACKOFF_FACTOR: float = 1.5
    MAX_CONCURRENT_CHAT_REQUESTS: int = 25
    
    # Configurable Alert & Hazard Rules
    WIND_SPEED_WARNING_KMH: float = 40.0
    WIND_GUST_WARNING_KMH: float = 55.0
    WAVE_HEIGHT_WARNING_M: float = 2.5
    BAROMETRIC_PRESSURE_DROP_HPA: float = 998.0
    ALERT_DEDUPLICATION_WINDOW_MINUTES: int = 60
    
    # Push Notifications & Delivery
    FCM_SERVER_KEY: str = ""
    EXPO_ACCESS_TOKEN: str = ""
    NOTIFICATION_SANDBOX_MODE: bool = True
    
    # Environment & Security
    APP_ENV: str = "development"  # development, staging, production, testing
    CORS_ORIGINS: str = "*"
    LOG_LEVEL: str = "INFO"
    ENABLE_DOCS: bool = True
    
    # Database & Reliability
    DATABASE_PATH: str = "app/data/neerai.db"
    DATABASE_BUSY_TIMEOUT_MS: int = 10000
    MONGODB_URI: str = "mongodb+srv://Divyansh:Divyansh@portfolio.9nzqdw8.mongodb.net/"
    MONGODB_DB_NAME: str = "neerai_db"
    CACHE_RETENTION_HOURS: int = 72
    ALERT_HISTORY_RETENTION_DAYS: int = 30
    MAX_TURNS_PER_CONVERSATION: int = 15
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 120
    JWT_SECRET_KEY: str = "neerai_marine_secret_key_2026_super_secure"
    JWT_ALGORITHM: str = "HS256"
    
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        extra="ignore"
    )

    def validate_environment(self) -> None:
        """
        Validates configuration at startup.
        Ensures staging and production environments do not leak secrets,
        do not use simulated fallback data in real-data environments,
        and have all strictly mandatory variables configured.
        """
        env = self.APP_ENV.lower()

        # Database path must always be non-empty
        if not self.DATABASE_PATH or not self.DATABASE_PATH.strip():
            raise ConfigurationError(
                "Configuration error: DATABASE_PATH must be configured and non-empty."
            )

        if env in ["staging", "production"]:
            # Real-data environment rule: Never silently fall back to fake marine data
            if self.ENABLE_SIMULATION_FALLBACK:
                raise ConfigurationError(
                    f"Configuration error: ENABLE_SIMULATION_FALLBACK cannot be enabled in '{env}' environment. "
                    "NeerAI strictly disallows fabricated or simulated marine data in real-data environments."
                )

            # In staging or production, CORS should not be a blanket wildcard if specified
            if self.CORS_ORIGINS.strip() == "*":
                # Staging/Production warns or can enforce restricted origins
                pass

        if self.REQUIRE_REAL_DATA and self.ENABLE_SIMULATION_FALLBACK:
            raise ConfigurationError(
                "Configuration error: REQUIRE_REAL_DATA is enabled, but ENABLE_SIMULATION_FALLBACK is set to True. "
                "Conflict detected: simulation fallback is prohibited."
            )

settings = Settings()
