"""Symbolic projection checked against the full causal response-type LP."""
from itertools import product
import random

import pandas as pd
import pytest
import sympy as sp

pytest.importorskip('cdd.gmp')

from autobounds import Bounder, DAG
from tests.homogeneous_iv_oracle import homogeneous_iv_bounds
from autobounds.backends.scip import ScipBackend


def full_model(values, homogeneous=True):
    a, n, w, g0, g1, b0, b1 = [values[k] for k in
                                 ('a', 'n', 'w', 'g0', 'g1', 'b0', 'b1')]
    table = [[1-a-b0, b0, a-g0, g0], [n-g1, g1, 1-n-b1, b1]]
    p = Bounder(DAG('Z -> D, D -> Y, Z -> Y, U -> D, U -> Y', unob='U'))
    p.add_assumption(p.p('D(Z=0)=1&D(Z=1)=0'))
    p.load_data(pd.DataFrame([(z, d, y, table[z][2*d+y]*[1-w, w][z])
                             for z, d, y in product(range(2), repeat=3)],
                            columns=['Z', 'D', 'Y', 'prob']))
    ate = p.p('Y(D=1)=1') - p.p('Y(D=0)=1')
    if homogeneous:
        for d0, d1, mass in [(0, 0, n), (0, 1, 1-a-n), (1, 1, a)]:
            event = f'D(Z=0)={d0}&D(Z=1)={d1}'
            delta = p.p('Y(D=1)=1&'+event) - p.p('Y(D=0)=1&'+event)
            p.add_constraint(delta - mass*ate)
    p.set_estimand(ate)
    return p.compile()


def assert_numeric_agreement(values):
    model = full_model(values)
    # Root probabilities have been substituted: even homogeneity is affine.
    assert model.is_linear
    result = ScipBackend().solve(model, verbose=False, maxtime=10)
    try:
        expected = homogeneous_iv_bounds().evaluate(values)
    except ValueError as exc:
        assert 'incompatible' in str(exc)
        assert result.lower.end == result.upper.end == 0
    else:
        for endpoint, value in zip((result.lower, result.upper), expected):
            assert endpoint.primal == pytest.approx(value, abs=2e-6)
            assert endpoint.dual == pytest.approx(value, abs=2e-6)


def test_exact_farkas_certificates():
    b = homogeneous_iv_bounds()
    assert len(b.inequalities) == 37
    for expression, ray in zip(b.inequalities, b.certificates):
        assert all(v >= 0 for v in ray)
        assert all(sum(v*row[j] for v, row in zip(ray, b.matrix)) == 0 for j in (0, 1))
        assert sp.expand(expression - sum(v*r for v, r in zip(ray, b.rhs))) == 0


@pytest.mark.parametrize('seed', range(12))
def test_symbolic_bounds_match_full_graph_lp(seed):
    rng = random.Random(seed)
    a = rng.randrange(1, 7)/8
    n = rng.randrange(1, 8-int(a*8))/8
    values = dict(a=a, n=n, w=rng.randrange(1, 8)/8,
                  g0=a*rng.randrange(9)/8, g1=n*rng.randrange(9)/8,
                  b0=(1-a)*rng.randrange(9)/8, b1=(1-n)*rng.randrange(9)/8)
    assert_numeric_agreement(values)


@pytest.mark.parametrize('a,n', [(0, 0), (0, .5), (.5, 0), (.5, .5), (1, 0), (0, 1)])
def test_absent_principal_strata_are_vacuous(a, n):
    assert_numeric_agreement(dict(a=a, n=n, w=.25, g0=a*.75, g1=n*.25,
                                  b0=(1-a)*.25, b1=(1-n)*.75))


def test_homogeneity_sharpens_bounds_without_imposing_exclusion():
    values = dict(a=.25, n=.25, w=.5, g0=.25, g1=.25, b0=0, b1=.75)
    bounds = homogeneous_iv_bounds().evaluate(values)
    assert bounds == pytest.approx((0, .5))
    unrestricted = ScipBackend().solve(full_model(values, homogeneous=False), verbose=False)
    assert bounds[0] > unrestricted.lower.dual
    assert bounds[1] < unrestricted.upper.dual


def test_invalid_data_are_rejected():
    values = dict(a=.75, n=.75, w=.5, g0=0, g1=0, b0=0, b1=0)
    with pytest.raises(ValueError, match='Invalid probabilities'):
        homogeneous_iv_bounds().evaluate(values)


def test_compact_documented_formulas_match_projection():
    rng = random.Random(917)
    bounds = homogeneous_iv_bounds()
    for _ in range(40):
        a, n, w = .25, .375, rng.randrange(1, 8)/8
        c, r = 1-a-n, 1-w
        g0, g1 = a*rng.randrange(9)/8, n*rng.randrange(9)/8
        b0, b1 = (1-a)*rng.randrange(9)/8, (1-n)*rng.randrange(9)/8
        lu, uu = r*max(0, b0-c), r*min(n, b0)
        lv, uv = w*max(0, b1-c), w*min(a, b1)
        k = r*b0-w*b1
        lower = max(-1, (-w*g1-uu)/n, (lv+r*g0-a)/a,
                    (lu-uv-k-w*c)/c, (lu-a+r*g0-k-w*c)/(a+c),
                    (-w*g1-uv-k-w*c)/(n+c), -w*g1-a+r*g0-k-w*c)
        upper = min(1, (n-w*g1-lu)/n, (uv+r*g0)/a,
                    (uu-lv-k+r*c)/c, (uu+r*g0-k+r*c)/(a+c),
                    (n-w*g1-lv-k+r*c)/(n+c), n-w*g1+r*g0-k+r*c)
        assert bounds.evaluate(dict(a=a, n=n, w=w, g0=g0, g1=g1, b0=b0, b1=b1)) == pytest.approx((lower, upper))
