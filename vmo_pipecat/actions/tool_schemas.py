"""
JSON-Schema definitions for LLM function-calling tools (§4.2).

Compatible with OpenAI function calling format used by PipeCat's
OpenAILLMContext.  Only tools listed in context.tools are injected.
"""

from __future__ import annotations

# ── Individual tool schemas ────────────────────────────────────────────────────

TRANSFER_CALL: dict = {
    "type": "function",
    "function": {
        "name": "transfer_call",
        "description": (
            "Transfer the current call to another extension or agent. "
            "Use when the caller requests human assistance or a different department."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Destination extension number or PSTN number to transfer to.",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason for the transfer (e.g. 'escalation', 'billing').",
                },
            },
            "required": ["target"],
        },
    },
}

HANGUP_CALL: dict = {
    "type": "function",
    "function": {
        "name": "hangup_call",
        "description": (
            "Hang up the current call. "
            "Use only when the conversation is complete and the caller agrees to end the call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for hanging up (e.g. 'goodbye', 'resolved').",
                },
            },
            "required": [],
        },
    },
}

PLAY_AUDIO_FILE: dict = {
    "type": "function",
    "function": {
        "name": "play_audio_file",
        "description": (
            "Play a pre-recorded audio file to the caller. "
            "The uri must use Asterisk's media URI format: "
            "'sound:welcome', 'recording:intro', or 'file:/path/to/file'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "Asterisk media URI (e.g. 'sound:welcome', 'recording:terms').",
                },
            },
            "required": ["uri"],
        },
    },
}

SEND_DTMF: dict = {
    "type": "function",
    "function": {
        "name": "send_dtmf",
        "description": (
            "Send DTMF digit(s) into the call. "
            "Useful for navigating IVR menus on outbound legs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "digits": {
                    "type": "string",
                    "description": "DTMF digits to send (e.g. '1234', '#').",
                },
            },
            "required": ["digits"],
        },
    },
}

# ── Registry ───────────────────────────────────────────────────────────────────

ALL_SCHEMAS: dict[str, dict] = {
    "transfer_call": TRANSFER_CALL,
    "hangup_call": HANGUP_CALL,
    "play_audio_file": PLAY_AUDIO_FILE,
    "send_dtmf": SEND_DTMF,
}


def schemas_for_tools(tool_names: list[str]) -> list[dict]:
    """Return OpenAI-format tool schemas for the requested tool names."""
    return [ALL_SCHEMAS[name] for name in tool_names if name in ALL_SCHEMAS]
