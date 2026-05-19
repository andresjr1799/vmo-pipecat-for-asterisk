"""
Envelope serialization/deserialization for the agentic bus (§9.5.2).

All messages use JSON UTF-8. Pydantic v2 validates both outbound and inbound.

Schemas:
  vmo.agentic.request/1     — vmo-pipecat → backend agentic (outbound)
  vmo.agentic.response/1    — backend → vmo-pipecat (inbound, streaming or full)
  vmo.agentic.cancel/1      — vmo-pipecat → backend (barge-in / call ended)
  vmo.agentic.tool_result/1 — vmo-pipecat → backend (tool execution result)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Outbound: request (vmo-pipecat → backend) ─────────────────────────────────

class AgenticRequestEnvelope(BaseModel):
    """Envelope published to `outbound.destination` for each LLM turn."""

    schema_v: str = Field("vmo.agentic.request/1", alias="schema")
    correlation_id: str
    reply_to: str
    identity: Dict[str, str]
    session: Dict[str, Any] = Field(default_factory=dict)
    input: Dict[str, Any]
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    deadlines: Dict[str, Any] = Field(default_factory=dict)
    ts: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    model_config = {"populate_by_name": True}

    def to_bytes(self) -> bytes:
        return self.model_dump_json(by_alias=True).encode()


# ── Inbound: response (backend → vmo-pipecat) ─────────────────────────────────

class AgenticResponseEnvelope(BaseModel):
    """Single message from the backend for a given turn.

    type values:
      chunk      — streaming token(s); text field carries the text
      tool_call  — LLM requests a function call; tool_call field carries name+args
      final      — non-streaming mode full response
      end        — end-of-streaming sentinel with usage summary
      error      — backend error
    """

    schema_v: str = Field(..., alias="schema")
    correlation_id: str
    type: str                      # chunk | tool_call | final | end | error
    seq: Optional[int] = None
    text: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

    model_config = {"populate_by_name": True}

    @property
    def is_terminal(self) -> bool:
        return self.type in ("final", "end", "error")


# ── Outbound: cancel (vmo-pipecat → backend) ──────────────────────────────────

class AgenticCancelEnvelope(BaseModel):
    schema_v: str = Field("vmo.agentic.cancel/1", alias="schema")
    correlation_id: str
    reason: str   # barge_in | call_ended | timeout
    ts: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    model_config = {"populate_by_name": True}

    def to_bytes(self) -> bytes:
        return self.model_dump_json(by_alias=True).encode()


# ── Outbound: tool_result (vmo-pipecat → backend) ─────────────────────────────

class AgenticToolResultEnvelope(BaseModel):
    schema_v: str = Field("vmo.agentic.tool_result/1", alias="schema")
    correlation_id: str
    tool_call_id: str
    result: Dict[str, Any]
    ts: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())

    model_config = {"populate_by_name": True}

    def to_bytes(self) -> bytes:
        return self.model_dump_json(by_alias=True).encode()


# ── Builder helpers ────────────────────────────────────────────────────────────

def build_request(
    correlation_id: str,
    identity_dict: Dict[str, str],
    reply_to: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    session: Dict[str, Any],
    preferences: Dict[str, Any],
    deadlines: Dict[str, Any],
) -> AgenticRequestEnvelope:
    return AgenticRequestEnvelope(
        correlation_id=correlation_id,
        reply_to=reply_to,
        identity=identity_dict,
        session=session,
        input={"messages": messages, "history_strategy": "client_managed"},
        tools=tools,
        preferences=preferences,
        deadlines=deadlines,
    )


def parse_response(raw: bytes) -> AgenticResponseEnvelope:
    """Parse and validate an inbound response message."""
    return AgenticResponseEnvelope.model_validate_json(raw)
