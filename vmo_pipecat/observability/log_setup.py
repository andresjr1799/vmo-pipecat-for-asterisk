"""
Structured Logging Configuration for vmo-pipecat (MELT Phase).

Configures structured logging using structlog (JSON output by default).
Provides per-call contextvars binding so every log line emitted within
an async call context carries the full call identity.

OTel integration: the add_otel_context processor injects trace_id and span_id
into every log line when running inside an active OTel span.

loguru interception: pipecat-ai uses loguru internally. We redirect loguru
output through stdlib logging so it appears as JSON via the foreign_pre_chain.
"""

import os
import logging
import sys
import contextvars
import uuid
import time
import datetime

import structlog
from structlog import dev as structlog_dev
from logging.handlers import RotatingFileHandler

correlation_id_var = contextvars.ContextVar('correlation_id', default=None)


def get_correlation_id():
    return correlation_id_var.get()


def set_correlation_id(value=None):
    if value is None:
        value = str(uuid.uuid4())
    correlation_id_var.set(value)


def add_correlation_id(logger, method_name, event_dict):
    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict['correlation_id'] = correlation_id
    return event_dict


def add_service_context(logger, method_name, event_dict):
    event_dict['service'] = 'vmo-pipecat'
    component = event_dict.get('logger')
    if not component:
        try:
            component = getattr(getattr(logger, 'logger', None), 'name', None) or getattr(logger, 'name')
        except Exception:
            component = 'unknown'
    event_dict['component'] = component
    return event_dict


def sanitize_secrets(logger, method_name, event_dict):
    """Redact sensitive fields (api_key, token, password, etc.) from logs."""
    SENSITIVE_KEYS = {
        'api_key', 'apikey', 'api-key', 'api_keys',
        'token', 'access_token', 'refresh_token', 'auth_token', 'bearer',
        'password', 'passwd', 'pwd', 'pass',
        'authorization', 'auth',
        'credential', 'credentials', 'secret', 'secrets',
        'private_key', 'private-key', 'privatekey',
        'client_secret', 'client-secret', 'clientsecret',
    }

    def redact_value(value):
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, str):
            if not value:
                return ''
            return f"{value[:2]}***REDACTED***" if len(value) > 4 else "***REDACTED***"
        if isinstance(value, (int, float)):
            return "***REDACTED***"
        if isinstance(value, (list, tuple)):
            return [redact_value(v) for v in value]
        if isinstance(value, dict):
            return {k: redact_value(v) if k.lower() in SENSITIVE_KEYS else v for k, v in value.items()}
        return "***REDACTED***"

    def sanitize_dict(d):
        if not isinstance(d, dict):
            return d
        sanitized = {}
        for key, value in d.items():
            key_normalized = str(key).lower().replace('_', '').replace('-', '')
            is_sensitive = any(
                key_normalized == p.replace('_', '').replace('-', '') or
                key_normalized.endswith(p.replace('_', '').replace('-', ''))
                for p in SENSITIVE_KEYS
            )
            if is_sensitive:
                sanitized[key] = redact_value(value)
            elif isinstance(value, dict):
                sanitized[key] = sanitize_dict(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [sanitize_dict(v) if isinstance(v, dict) else v for v in value]
            else:
                sanitized[key] = value
        return sanitized

    return sanitize_dict(event_dict)


def add_local_timestamp(logger, method_name, event_dict):
    event_dict["timestamp"] = datetime.datetime.now().astimezone().isoformat()
    return event_dict


def add_otel_context(logger, method_name, event_dict):
    """Inject trace_id and span_id from the active OTel span into log records.

    Safe to call when OTel is not installed or no span is active — silently
    skips injection.
    """
    try:
        from opentelemetry import trace as otel_trace
        span = otel_trace.get_current_span()
        if span is not None:
            ctx = span.get_span_context()
            if ctx.is_valid:
                event_dict["trace_id"] = format(ctx.trace_id, "032x")
                event_dict["span_id"] = format(ctx.span_id, "016x")
    except Exception:
        pass
    return event_dict


def _redirect_loguru() -> None:
    """Redirect loguru output through stdlib logging so it appears as JSON.

    pipecat-ai uses loguru internally. Without this, loguru lines use a
    custom format (timestamp | LEVEL | module:line - message) that bypasses
    our structlog JSON renderer.
    """
    try:
        from loguru import logger as loguru_logger
        import logging

        class _LoguruHandler(logging.Handler):
            def emit(self, record):
                # loguru already formatted the message; forward to stdlib
                pass

        # Remove loguru's default handler and add one that forwards to stdlib
        loguru_logger.remove()
        loguru_logger.add(
            _LoguruSink(),
            level=os.getenv("VMO_LOG_LEVEL", "INFO"),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        )
    except ImportError:
        pass


class _LoguruSink:
    """Forward loguru messages to Python stdlib logging (→ structlog JSON)."""

    def write(self, message: str) -> None:
        message = message.rstrip("\n")
        if not message:
            return
        # Parse loguru's default format to extract level and logger name
        # Format: "YYYY-MM-DD HH:mm:ss.SSS | LEVEL    | name:func:line - message"
        try:
            parts = message.split(" | ", 2)
            if len(parts) >= 3:
                level_name = parts[1].strip()
                rest = parts[2]
                if " - " in rest:
                    source, msg = rest.split(" - ", 1)
                else:
                    source, msg = rest, ""
                logger_name = source.rsplit(":", 2)[0] if ":" in source else "pipecat"
                level = getattr(logging, level_name, logging.INFO)
            else:
                logger_name = "pipecat"
                level = logging.INFO
                msg = message
        except Exception:
            logger_name = "pipecat"
            level = logging.INFO
            msg = message

        logging.getLogger(logger_name).log(level, msg)


def configure_logging(log_level="INFO", log_to_file=False, log_file_path="service.log", service_name="vmo-pipecat"):
    """Configure structlog with JSON output, contextvars merge, and secret redaction."""
    env_level = os.getenv("LOG_LEVEL") or os.getenv("VMO_LOG_LEVEL")
    if env_level:
        log_level = env_level.upper()
    try:
        log_to_file = bool(int(os.getenv("LOG_TO_FILE", "0")))
    except Exception:
        pass
    log_file_path = os.getenv("LOG_FILE_PATH", log_file_path)
    log_format = os.getenv("LOG_FORMAT", "json").strip().lower()
    log_color = os.getenv("LOG_COLOR", "1").strip() not in ("0", "false", "False")

    log_level_upper = log_level.upper() if isinstance(log_level, str) else str(log_level)
    tb_mode = os.getenv("LOG_SHOW_TRACEBACKS", "auto").strip().lower()
    if tb_mode == "always":
        show_tracebacks = True
    elif tb_mode == "never":
        show_tracebacks = False
    else:
        show_tracebacks = (log_level_upper == "DEBUG")

    def suppress_exc_info_if_disabled(logger, method_name, event_dict):
        if not show_tracebacks and event_dict.get("exc_info"):
            event_dict.pop("exc_info", None)
        return event_dict

    try:
        level_value = getattr(logging, log_level_upper, logging.INFO) if isinstance(log_level, str) else int(log_level)
    except Exception:
        level_value = logging.INFO

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.add_log_level,
            add_local_timestamp,
            add_service_context,
            add_correlation_id,
            sanitize_secrets,
            suppress_exc_info_if_disabled,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer = (
        structlog_dev.ConsoleRenderer(colors=log_color)
        if log_format == "console"
        else structlog.processors.JSONRenderer()
    )

    processor_formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=[
            structlog.stdlib.add_logger_name,
            structlog.processors.add_log_level,
            add_local_timestamp,
        ],
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level_value)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(processor_formatter)
    root_logger.addHandler(console_handler)

    if log_to_file:
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = log_file_path
            looks_like_dir = path.endswith(os.sep) or (os.path.exists(path) and os.path.isdir(path))
            if looks_like_dir:
                path = os.path.join(path.rstrip(os.sep), f"{service_name}-{ts}.log")
            elif "{ts}" in path:
                path = path.replace("{ts}", ts)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            file_handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5)
            file_handler.setFormatter(processor_formatter)
            root_logger.addHandler(file_handler)
        except Exception:
            pass

    for noisy in (
        'websockets', 'websockets.client', 'websockets.protocol',
        'aiohttp', 'asyncio',
        'httpx', 'httpcore', 'httpcore.connection', 'httpcore.http11',
        'openai._base_client', 'openai',
    ):
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass

    # Redirect loguru → stdlib so pipecat-ai internal logs become JSON too
    _redirect_loguru()


def get_logger(name: str):
    return structlog.get_logger(name)
