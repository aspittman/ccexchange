import json
import logging
from datetime import datetime, timezone
from pathlib import Path


class AuditLog:
    def __init__(self, path: str = "logs/events.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ccexchange")

    def write(self, event: str, **details):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        self.logger.info("%s %s", event, details)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str, sort_keys=True) + "\n")
        except Exception:
            self.logger.exception("Optional audit recording failed")
