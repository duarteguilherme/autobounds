"""Minimal backend contract: validate support, then solve an immutable model."""
from typing import Protocol, TypeVar
from ..model import MathematicalModel

Result = TypeVar('Result', covariant=True)


class Backend(Protocol[Result]):
    def supports(self, model: MathematicalModel) -> bool:
        """Whether the backend can represent this model without relaxation."""
        ...

    def solve(self, model: MathematicalModel, **options) -> Result:
        """Solve a supported model or reject it explicitly."""
        ...
