"""Sharp symbolic bounds for Y_1B from observational and factorial data.

Install the symbolic extra, then run:
    python examples/symbolic_factorial.py
"""

from itertools import product
from time import perf_counter

from autobounds import DAG, SymbolicProblem


def build_problem():
    """Allow arbitrary dependence among (A, B, Y00, Y01, Y10, Y11)."""
    graph = DAG("A -> Y, B -> Y, U -> A, U -> B, U -> Y", unob="U")
    problem = SymbolicProblem(graph)
    problem.load_data("A,B,Y")
    for a, b in product(range(2), repeat=2):
        problem.load_data("Y", do=f"A={a},B={b}")
    problem.set_estimand(problem.p("Y(A=1)=1"))
    return problem


if __name__ == "__main__":
    model = build_problem().write_program()
    start = perf_counter()
    bounds = model.solve()
    elapsed = perf_counter() - start
    print(f"{len(model.target)} response types; solved in {elapsed:.3f} s")
    print(f"Lower bound: maximum of {len(bounds.lower_expressions)} expressions")
    for expression in bounds.lower_expressions:
        print(f"  {expression}")
    print(f"Upper bound: minimum of {len(bounds.upper_expressions)} expressions")
    for expression in bounds.upper_expressions:
        print(f"  {expression}")
