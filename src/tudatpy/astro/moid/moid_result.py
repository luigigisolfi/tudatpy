from dataclasses import dataclass


@dataclass
class MOIDResult:
    moid: float
    candidates: list
