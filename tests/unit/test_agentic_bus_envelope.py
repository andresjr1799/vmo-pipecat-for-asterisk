"""
Unit tests — agentic bus envelope serialization (Phase 11).

Verifies:
  - AgenticRequestEnvelope builds correctly and serializes to JSON
  - AgenticResponseEnvelope parses correctly (chunk, final, tool_call, end, error)
  - AgenticCancelEnvelope and AgenticToolResultEnvelope serialize correctly
  - parse_response() raises on invalid JSON
  - is_terminal property works for all types
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from vmo_pipecat.bus.envelope import (
    AgenticCancelEnvelope,
    AgenticRequestEnvelope,
    AgenticResponseEnvelope,
    AgenticToolResultEnvelope,
    build_request,
    parse_response,
)


# ── Request envelope ───────────────────────────────────────────────────────────

class TestRequestEnvelope:

    def test_schema_field_in_json(self):
        env = AgenticRequestEnvelope(
            correlation_id="cid-1",
            reply_to="agentic.responses.acme",
            identity={"vmo_call_id": "c1", "tenant_id": "acme"},
            input={"messages": []},
        )
        data = json.loads(env.to_bytes())
        assert data["schema"] == "vmo.agentic.request/1"

    def test_correlation_id_preserved(self):
        env = AgenticRequestEnvelope(
            correlation_id="cid-abc",
            reply_to="dest",
            identity={},
            input={"messages": []},
        )
        data = json.loads(env.to_bytes())
        assert data["correlation_id"] == "cid-abc"

    def test_to_bytes_returns_valid_json(self):
        env = AgenticRequestEnvelope(
            correlation_id="cid-1",
            reply_to="dest",
            identity={"tenant_id": "acme"},
            input={"messages": [{"role": "user", "content": "Hola"}]},
            tools=[{"type": "function", "function": {"name": "transfer_call"}}],
        )
        raw = env.to_bytes()
        assert isinstance(raw, bytes)
        parsed = json.loads(raw)
        assert parsed["tools"][0]["function"]["name"] == "transfer_call"

    def test_ts_is_iso_format(self):
        env = AgenticRequestEnvelope(
            correlation_id="cid-ts",
            reply_to="d",
            identity={},
            input={"messages": []},
        )
        data = json.loads(env.to_bytes())
        ts = data["ts"]
        # Should be parseable as ISO datetime
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_build_request_helper(self):
        env = build_request(
            correlation_id="c1:t1",
            identity_dict={"vmo_call_id": "c1", "tenant_id": "acme"},
            reply_to="agentic.responses.acme",
            messages=[{"role": "system", "content": "You are an agent."}],
            tools=[],
            session={"thread_id": "c1"},
            preferences={"streaming": "tokens"},
            deadlines={"first_token_ms": 4000},
        )
        data = json.loads(env.to_bytes())
        assert data["input"]["messages"][0]["role"] == "system"
        assert data["session"]["thread_id"] == "c1"
        assert data["preferences"]["streaming"] == "tokens"
        assert data["deadlines"]["first_token_ms"] == 4000
        assert data["input"]["history_strategy"] == "client_managed"


# ── Response envelope ──────────────────────────────────────────────────────────

class TestResponseEnvelope:

    def _make_chunk(self, text: str, seq: int = 1) -> bytes:
        return json.dumps({
            "schema": "vmo.agentic.response/1",
            "correlation_id": "cid-1:t1",
            "type": "chunk",
            "seq": seq,
            "text": text,
        }).encode()

    def _make_final(self, text: str) -> bytes:
        return json.dumps({
            "schema": "vmo.agentic.response/1",
            "correlation_id": "cid-1:t1",
            "type": "final",
            "text": text,
            "usage": {"tokens_in": 50, "tokens_out": 20},
        }).encode()

    def _make_end(self) -> bytes:
        return json.dumps({
            "schema": "vmo.agentic.response/1",
            "correlation_id": "cid-1:t1",
            "type": "end",
            "seq": 5,
            "usage": {"tokens_in": 50, "tokens_out": 20, "model": "gpt-4o"},
        }).encode()

    def _make_tool_call(self) -> bytes:
        return json.dumps({
            "schema": "vmo.agentic.response/1",
            "correlation_id": "cid-1:t1",
            "type": "tool_call",
            "seq": 3,
            "tool_call": {"id": "tc_1", "name": "transfer_call", "arguments": {"target": "9000"}},
        }).encode()

    def _make_error(self) -> bytes:
        return json.dumps({
            "schema": "vmo.agentic.response/1",
            "correlation_id": "cid-1:t1",
            "type": "error",
            "error": {"code": "backend_unavailable", "message": "Service down"},
        }).encode()

    def test_parse_chunk(self):
        env = parse_response(self._make_chunk("Hola "))
        assert env.type == "chunk"
        assert env.text == "Hola "
        assert env.seq == 1
        assert not env.is_terminal

    def test_parse_final(self):
        env = parse_response(self._make_final("Respuesta completa."))
        assert env.type == "final"
        assert env.text == "Respuesta completa."
        assert env.is_terminal

    def test_parse_end(self):
        env = parse_response(self._make_end())
        assert env.type == "end"
        assert env.usage["tokens_in"] == 50
        assert env.is_terminal

    def test_parse_tool_call(self):
        env = parse_response(self._make_tool_call())
        assert env.type == "tool_call"
        assert env.tool_call["name"] == "transfer_call"
        assert env.tool_call["arguments"]["target"] == "9000"
        assert not env.is_terminal

    def test_parse_error(self):
        env = parse_response(self._make_error())
        assert env.type == "error"
        assert env.error["code"] == "backend_unavailable"
        assert env.is_terminal

    def test_parse_invalid_json_raises(self):
        with pytest.raises(Exception):
            parse_response(b"not json at all")

    def test_is_terminal_for_chunk_is_false(self):
        env = parse_response(self._make_chunk("hi"))
        assert not env.is_terminal

    def test_is_terminal_for_tool_call_is_false(self):
        env = parse_response(self._make_tool_call())
        assert not env.is_terminal


# ── Cancel envelope ────────────────────────────────────────────────────────────

class TestCancelEnvelope:

    def test_schema_and_reason(self):
        env = AgenticCancelEnvelope(correlation_id="cid-1:t1", reason="barge_in")
        data = json.loads(env.to_bytes())
        assert data["schema"] == "vmo.agentic.cancel/1"
        assert data["reason"] == "barge_in"
        assert data["correlation_id"] == "cid-1:t1"

    def test_call_ended_reason(self):
        env = AgenticCancelEnvelope(correlation_id="cid-2:t2", reason="call_ended")
        data = json.loads(env.to_bytes())
        assert data["reason"] == "call_ended"


# ── Tool result envelope ───────────────────────────────────────────────────────

class TestToolResultEnvelope:

    def test_schema_and_fields(self):
        env = AgenticToolResultEnvelope(
            correlation_id="cid-1:t1",
            tool_call_id="tc_1",
            result={"status": "ok", "latency_ms": 182},
        )
        data = json.loads(env.to_bytes())
        assert data["schema"] == "vmo.agentic.tool_result/1"
        assert data["tool_call_id"] == "tc_1"
        assert data["result"]["status"] == "ok"
        assert data["result"]["latency_ms"] == 182
