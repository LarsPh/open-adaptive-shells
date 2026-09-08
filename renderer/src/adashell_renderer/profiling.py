from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator


@dataclass(slots=True)
class ProfileBook:
    stage_ms: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, float] = field(default_factory=dict)

    def add_time_ms(self, key: str, elapsed_ms: float) -> None:
        self.stage_ms[key] = self.stage_ms.get(key, 0.0) + float(elapsed_ms)

    def set_counter(self, key: str, value: float) -> None:
        self.counters[key] = float(value)

    @contextmanager
    def time(self, key: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add_time_ms(key, (time.perf_counter() - start) * 1000.0)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "stage_ms": dict(sorted(self.stage_ms.items())),
            "counters": dict(sorted(self.counters.items())),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)
