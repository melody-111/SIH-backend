import datetime
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("neerai.config.agent")

class ConfigurationError(Exception):
    """Raised when an agent system configuration is invalid or incomplete."""
    pass

class AgentMetadataConfig(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    required_inputs: List[str] = Field(default_factory=list)
    is_essential: bool = True

class ModelSelectionConfig(BaseModel):
    provider: str = Field(default="auto")
    fast_model: str = Field(default="gpt-4o-mini")
    reasoning_model: str = Field(default="gpt-4o")
    timeout_seconds: float = Field(default=8.0, gt=0.0, le=60.0)
    max_retries: int = Field(default=1, ge=0, le=5)
    allow_deterministic_fallback: bool = Field(default=True)

class PromptTemplateConfig(BaseModel):
    template_name: str
    system_prompt: str = Field(..., min_length=10)
    schema_name: str

class AgentSystemConfig(BaseModel):
    """
    Versioned, immutable configuration defining agent responsibilities,
    task-to-agent routing, model selection, prompt templates, and safety invariants.
    """
    version: str = Field(..., pattern=r"^[0-9]+\.[0-9]+(\.[0-9]+)?(-[a-zA-Z0-9_.-]+)?$")
    description: str = Field(..., min_length=5)
    author: str = Field(default="neerai_systems_architect")
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    agents: Dict[str, AgentMetadataConfig]
    routing_policies: Dict[str, List[str]]
    model_selection: ModelSelectionConfig
    prompt_templates: Dict[str, PromptTemplateConfig]
    safety_invariants: Dict[str, bool] = Field(default_factory=lambda: {
        "forbid_fabricated_pfz": True,
        "official_warnings_highest_priority": True,
        "strictly_require_fallback_on_error": True,
        "reject_out_of_bounds_coords": True
    })

    @field_validator("routing_policies")
    @classmethod
    def validate_routing_policies(cls, v: Dict[str, List[str]], values) -> Dict[str, List[str]]:
        # Mandatory intents must have defined routing policies
        mandatory_intents = {"HAZARD", "MARINE_CONDITION", "FISHING_LOCATION", "GENERAL_EXPLANATORY", "ROUTE"}
        missing = mandatory_intents - set(v.keys())
        if missing:
            raise ValueError(f"Configuration missing mandatory intent routing policies: {missing}")

        # Safety rule: HAZARD must contain official hazard agent
        if "hazard" not in v.get("HAZARD", []):
            raise ValueError("Safety Invariant Violation: HAZARD routing policy must include 'hazard' agent.")

        # Safety rule: GENERAL_EXPLANATORY must not dispatch live sensors
        if len(v.get("GENERAL_EXPLANATORY", [])) != 0:
            raise ValueError("Efficiency Invariant Violation: GENERAL_EXPLANATORY must have 0 live sensor agents.")

        return v

    @field_validator("safety_invariants")
    @classmethod
    def validate_safety_invariants(cls, v: Dict[str, bool]) -> Dict[str, bool]:
        if not v.get("forbid_fabricated_pfz", False):
            raise ValueError("Safety Invariant Violation: forbid_fabricated_pfz cannot be disabled.")
        if not v.get("official_warnings_highest_priority", False):
            raise ValueError("Safety Invariant Violation: official_warnings_highest_priority cannot be disabled.")
        return v

# -----------------------------------------------------------------------------
# Known-Good Baseline Configuration (v1.0.0)
# -----------------------------------------------------------------------------
DEFAULT_AGENTS_METADATA = {
    "weather": AgentMetadataConfig(name="weather", description="Retrieves coastal wind, air temp, and precipitation", required_inputs=["latitude", "longitude"]),
    "ocean": AgentMetadataConfig(name="ocean", description="Retrieves wave height, swell period, and sea surface temperature", required_inputs=["latitude", "longitude"]),
    "hazard": AgentMetadataConfig(name="hazard", description="Checks official IMD bulletins and coastal warnings", required_inputs=["latitude", "longitude"]),
    "pfz": AgentMetadataConfig(name="pfz", description="Checks INCOIS Potential Fishing Zones", required_inputs=["latitude", "longitude"]),
    "sst": AgentMetadataConfig(name="sst", description="Satellite Sea Surface Temperature observation", required_inputs=["latitude", "longitude"]),
    "chlorophyll": AgentMetadataConfig(name="chlorophyll", description="Ocean color phytoplankton productivity", required_inputs=["latitude", "longitude"]),
    "gis": AgentMetadataConfig(name="gis", description="Marine GIS nearest ports and boundaries", required_inputs=["latitude", "longitude"]),
    "route": AgentMetadataConfig(name="route", description="Geodesic navigational corridor calculation", required_inputs=["latitude", "longitude"]),
    "satellite": AgentMetadataConfig(name="satellite", description="NASA MODIS satellite imagery catalog", required_inputs=["latitude", "longitude"])
}

DEFAULT_ROUTING_POLICIES_V1_0 = {
    "GENERAL_EXPLANATORY": [],
    "OCEAN_INFO": ["ocean"],
    "HAZARD": ["hazard", "weather", "ocean"],
    "FISHING_LOCATION": ["weather", "ocean", "sst", "pfz", "hazard", "gis"],
    "ROUTE": ["route", "gis", "ocean", "hazard"],
    "MARINE_CONDITION": ["weather", "ocean", "hazard"],
    "FOLLOW_UP": ["weather", "ocean", "hazard"],
    "GENERAL_MARINE": ["weather", "ocean", "hazard"]
}

# Candidate Optimized Routing Policies (v1.1.0-adaptive)
# Improvement: Eliminates redundant GIS calls on fishing queries; adds gale checks to route corridors
OPTIMIZED_ROUTING_POLICIES_V1_1 = {
    "GENERAL_EXPLANATORY": [],
    "OCEAN_INFO": ["ocean"],
    "HAZARD": ["hazard", "weather", "ocean"],
    # Optimized: Removed 'gis' from fishing location to reduce unnecessary external agent overhead
    "FISHING_LOCATION": ["weather", "ocean", "sst", "pfz", "hazard"],
    # Optimized: Added 'weather' to route corridors to check for gale-force winds along passage
    "ROUTE": ["route", "gis", "ocean", "weather", "hazard"],
    "MARINE_CONDITION": ["weather", "ocean", "hazard"],
    "FOLLOW_UP": ["weather", "ocean", "hazard"],
    "GENERAL_MARINE": ["weather", "ocean", "hazard"]
}

DEFAULT_PROMPT_TEMPLATES = {
    "intent_classification": PromptTemplateConfig(
        template_name="intent_classification",
        system_prompt="Classify user natural language marine query into one of: GENERAL_EXPLANATORY, HAZARD, FOLLOW_UP, FISHING_LOCATION, OCEAN_INFO, MARINE_CONDITION, ROUTE, GENERAL_MARINE. Output valid JSON.",
        schema_name="IntentClassificationOutput"
    ),
    "advisory_synthesis": PromptTemplateConfig(
        template_name="advisory_synthesis",
        system_prompt="You are NeerAI, an expert Indian Marine AI Assistant. Never fabricate GPS coordinates or official warnings. Output JSON conforming to schema.",
        schema_name="SynthesizedAdvisoryOutput"
    )
}

CONFIG_V1_0_0 = AgentSystemConfig(
    version="1.0.0",
    description="NeerAI Baseline Verified Agent Configuration (Milestone 11 Known-Good)",
    agents=DEFAULT_AGENTS_METADATA,
    routing_policies=DEFAULT_ROUTING_POLICIES_V1_0,
    model_selection=ModelSelectionConfig(),
    prompt_templates=DEFAULT_PROMPT_TEMPLATES
)

CONFIG_V1_1_0_ADAPTIVE = AgentSystemConfig(
    version="1.1.0-adaptive",
    description="Adaptive Agent Routing Configuration: Pruned redundant GIS calls from fishing queries, added weather gale checks to route passages",
    agents=DEFAULT_AGENTS_METADATA,
    routing_policies=OPTIMIZED_ROUTING_POLICIES_V1_1,
    model_selection=ModelSelectionConfig(timeout_seconds=6.0, max_retries=1),
    prompt_templates=DEFAULT_PROMPT_TEMPLATES
)

class AgentConfigManager:
    """
    Central manager for versioned agent system configurations.
    Enforces validation, registry tracking, atomic activation, and safe rollback.
    Guarantees secrets are never stored in configurations.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentConfigManager, cls).__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        self._registry: Dict[str, AgentSystemConfig] = {}
        self._version_history: List[str] = []
        # Register known-good baseline
        self.register_config(CONFIG_V1_0_0, activate=True)
        # Register candidate adaptive configuration
        self.register_config(CONFIG_V1_1_0_ADAPTIVE, activate=False)

    def register_config(self, config: AgentSystemConfig, activate: bool = False) -> AgentSystemConfig:
        """Registers a validated configuration. Rejects invalid configs via Pydantic."""
        self._registry[config.version] = config
        if activate:
            self._active_version = config.version
            self._version_history.append(config.version)
            logger.info(f"AgentSystemConfig activated: {config.version}")
        return config

    def get_active_config(self) -> AgentSystemConfig:
        """Returns currently active verified configuration."""
        return self._registry[self._active_version]

    def get_active_version(self) -> str:
        return self._active_version

    def activate_version(self, version: str) -> AgentSystemConfig:
        """Atomically activates a registered version."""
        if version not in self._registry:
            raise ConfigurationError(f"Cannot activate unknown configuration version: {version}")
        self._active_version = version
        self._version_history.append(version)
        logger.info(f"Switched active agent configuration to: {version}")
        return self.get_active_config()

    def rollback(self, target_version: Optional[str] = None) -> AgentSystemConfig:
        """
        Rolls back to the previously active configuration or a specified target version.
        Guarantees instant reversibility.
        """
        if target_version:
            return self.activate_version(target_version)

        if len(self._version_history) >= 2:
            # Pop current version and reactivate previous
            self._version_history.pop()
            prev = self._version_history[-1]
            self._active_version = prev
            logger.warning(f"Rolled back agent configuration to previous version: {prev}")
            return self.get_active_config()

        # Fallback to known-good baseline
        self._active_version = "1.0.0"
        return self.get_active_config()

    def validate_and_register_dict(self, data: Dict[str, Any], activate: bool = False) -> AgentSystemConfig:
        """Validates raw dict payload against Pydantic schema before registration."""
        try:
            config = AgentSystemConfig(**data)
            return self.register_config(config, activate=activate)
        except Exception as e:
            raise ConfigurationError(f"Agent configuration rejected: {str(e)}")

    def export_config(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Exports sanitized configuration without any secrets."""
        v = version or self._active_version
        if v not in self._registry:
            raise ConfigurationError(f"Version {v} not found in registry.")
        return self._registry[v].model_dump()

    def reset_to_baseline(self):
        """Restores baseline v1.0.0 configuration for clean test isolation."""
        self._active_version = "1.0.0"
        self._version_history = ["1.0.0"]
