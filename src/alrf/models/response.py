from pydantic import BaseModel, ConfigDict


class RouterResult(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    answer:         str
    route:          str          # "local" | "fast" | "reasoning" | "rag+fast" | "rag+reasoning"
    provider:       str
    model:          str
    cost_usd:       float | None  # None for local/Ollama
    latency_ms:     int
    confidence:     float
    escalated:      bool
    rag_used:       bool
    decision_trace: list[dict]   # serialized steps
    cached:         bool = False
