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
