"""Solver results independent of SCIP models and compatibility dictionaries."""
from dataclasses import dataclass
from copy import deepcopy


@dataclass(frozen=True)
class EndpointResult:
    primal: float | None = None
    dual: float | None = None
    time: float | None = None
    end: int = -1

    @classmethod
    def from_dict(cls, result):
        return cls(**{name: result[name] for name in cls.__dataclass_fields__ if name in result})

    def to_dict(self):
        return {name: value for name in self.__dataclass_fields__
                if (value := getattr(self, name)) is not None}


@dataclass(frozen=True)
class SolverResult:
    lower: EndpointResult
    upper: EndpointResult
    width: float
    relative_gap: float
    dgps: dict | None = None

    def legacy_bounds(self):
        return self.lower.to_dict(), self.upper.to_dict(), self.width, self.relative_gap

    def to_legacy(self):
        bounds = self.legacy_bounds()
        return bounds if self.dgps is None else (bounds, deepcopy(self.dgps))
