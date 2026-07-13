from pydantic import BaseModel, ConfigDict, Field


class RouterConfig(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    policy:                str   = "cost_aware"
    escalation_threshold:  float = Field(0.70, ge=0.0, le=1.0)
    retry_budget:          int   = Field(3, ge=1, le=5)
    rag_token_threshold:   int   = Field(30, ge=1, le=500)
    fast_model:            str   = "gpt-4o-mini"
    reasoning_model:       str   = "claude-sonnet-4-6"
    local_model:           str   = "llama3.2"
    local_provider:        str   = "ollama"   # "ollama" | "lm_studio" | "vllm"
    local_base_url:        str   = "http://localhost:11434"
    observability_db_path: str   = ".alrf/routing.db"
    classifier:            str   = "heuristic"  # "heuristic" | "llm"
    llm_classifier_model:  str   = "llama3.2"
    llm_classifier_url:    str   = "http://localhost:11434"
    cache_enabled:         bool  = True
    cache_threshold:       float = Field(0.95, ge=0.0, le=1.0)
    cache_ttl_seconds:     int   = Field(3600, ge=1)
    cache_max_entries:     int   = Field(1000, ge=1, le=100_000)
    otel_endpoint:         str | None = None
