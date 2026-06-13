"""구조화 로깅(JSON) — 운영 표준.

한 줄 = 한 JSON 이벤트(타임스탬프·레벨·이벤트명 + 커스텀 필드).
로그 수집기(ELK/Datadog/CloudWatch 등)에서 그대로 파싱·검색·집계 가능.
한글은 ensure_ascii=False 로 그대로 출력.
"""
from __future__ import annotations
import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        # logger.info(..., extra={"fields": {...}}) 로 넘어온 커스텀 필드 병합
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    # uvicorn 로거도 같은 JSON 포맷으로 일원화(중복 방지)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers[:] = [handler]
        lg.propagate = False

    # 서드파티 노이즈 억제: 접근로그는 우리 미들웨어(http_request)로 대체,
    # ES 클라이언트의 요청별 INFO 로그는 WARNING 이상만 남김
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)

    return logging.getLogger("irsearch")
