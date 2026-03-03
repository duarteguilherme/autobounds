from .causalProblem import causalProblem


class Bounder(causalProblem):
    """
    Backward-compatible single-problem bounds solver.

    PR1 intentionally keeps behavior identical to causalProblem while
    introducing a dedicated type that will later own single-problem logic.
    """

    pass
