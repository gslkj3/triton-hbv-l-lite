"""Compiler-certified candidate identity and directly transformed state."""
from dataclasses import dataclass
from .state import ResponseState


@dataclass(frozen=True)
class Candidate:
    identity: str
    state: tuple[ResponseState, ...]
    legality_evidence: str

    def __post_init__(self):
        if not self.identity or not self.state or not self.legality_evidence:
            raise ValueError('candidate needs state and compiler legality evidence')
