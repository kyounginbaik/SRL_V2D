"""Atomic publication of a reference and the lower learner evaluated with it."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReferenceLearnerUpdate:
    learner: Any
    # None means a lower-only update at the unchanged reference/environment.
    environment: Any | None
