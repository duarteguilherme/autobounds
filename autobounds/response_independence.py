"""Sparse marginal independence constraints for independent latent roots.

A district need not be a single unrestricted common cause. Response vectors
with disjoint latent parents are independent, including vectors of several
responses. These are necessary restrictions for general latent networks, not
a complete characterization of their latent compatibility inequalities.
"""
from itertools import combinations
from fractions import Fraction


def _known_sums(program):
    """Extract exact equalities fixing a sum, without using sampling metadata."""
    from .model import Polynomial
    sums = {}
    for row in program.constraints:
        if not row or row[-1] != ['==']:
            continue
        polynomial = Polynomial.from_tokens(row[:-1], program.parameters)
        terms = [t for t in polynomial.terms if t.variables]
        if not terms or any(len(t.variables) != 1 for t in terms):
            continue
        coefficient = terms[0].coefficient
        if any(t.coefficient != coefficient for t in terms):
            continue
        support = frozenset(t.variables[0] for t in terms)
        sums[support] = -polynomial.constant / coefficient
    return sorted(sums.items(), key=lambda item: (-len(item[0]), sorted(item[0])))


def independence_templates(canonical):
    """Index maximal independent response blocks once, before data replay.

    Enumerating subsets is exponential in district size; the existing full
    response table is already at least that large. Complete districts take a
    fast path. Taking maximal blocks avoids adding their implied sub-blocks.
    """
    dag = canonical.dag
    if any(dag.find_parents(u) for u in dag.U):
        raise ValueError('Response independence requires root latent variables.')
    parents = {v: dag.find_parents_u(v) for v in dag.V}
    owner = {token: v for v, tokens in canonical.iso_params.items()
             for token in tokens}
    templates = []
    for component in canonical.c_comp:
        nodes = tuple(sorted(component))
        independent = {v: {w for w in nodes if w != v
                           and not parents[v].intersection(parents[w])}
                       for v in nodes}
        if not any(independent.values()):
            continue
        partitions = set()
        for size in range(1, len(nodes)):
            for subset in combinations(nodes, size):
                right = set.intersection(*(independent[v] for v in subset))
                if not right:
                    continue
                left = set.intersection(*(independent[v] for v in right))
                partitions.add(tuple(sorted((tuple(sorted(left)),
                                             tuple(sorted(right))))))
        rows = [(name, {owner[t]: t for t in name.split('.')})
                for name in canonical.parameters
                if owner[name.split('.')[0]] in component]
        for left, right in sorted(partitions):
            cells = {}
            for name, tokens in rows:
                a = tuple(tokens[v] for v in left)
                b = tuple(tokens[v] for v in right)
                cells.setdefault((a, b), []).append(name)
            a_values = sorted({a for a, b in cells})
            b_values = sorted({b for a, b in cells})
            if len(a_values) == 1 or len(b_values) == 1:
                continue
            # Normalization implies the omitted last row and column, including
            # on zero-probability boundaries. No division is used.
            left_sums = tuple(tuple(name for b in b_values for name in cells[a, b])
                              for a in a_values[:-1])
            right_sums = tuple(tuple(name for a in a_values for name in cells[a, b])
                               for b in b_values[:-1])
            joint = tuple(tuple(tuple(cells[a, b]) for b in b_values[:-1])
                          for a in a_values[:-1])
            templates.append((tuple(name for name, _ in rows),
                              left_sums, right_sums, joint))
    return tuple(templates)


def append_independences(program, templates, excluded):
    """Add marginal definitions and only (r-1)(c-1) product equations.

    Auxiliary marginals prevent expansion of products of large sums. Reuse
    each distinct marginal across partitions; SCIP can fix observed marginals
    during presolve, reducing their product equations to linear constraints.
    """
    used = set(program.parameters)
    marginals = {}
    normalized = set()
    known_sums = _known_sums(program) if templates else ()

    def active(names):
        return tuple(sorted(n for n in names if n not in excluded))

    def marginal(names):
        names = active(names)
        if not names:
            return None
        # A root marginal is often a disjoint union of observed joint cells.
        # Substitute its exact known sum before introducing any bilinearity.
        remaining, value = set(names), Fraction(0)
        for support, fixed in known_sums:
            if support <= remaining:
                remaining.difference_update(support)
                value += fixed
                if not remaining:
                    return str(value)
        if len(names) == 1:
            return names[0]
        if names not in marginals:
            name = f'_ab_indep_{len(marginals)}'
            while name in used:
                name += '_'
            used.add(name)
            program.parameters.append(name)
            program.constraints.append([[n] for n in names] +
                                       [['-1', name], ['==']])
            marginals[names] = name
        return marginals[names]

    for component, left, right, cells in templates:
        support = active(component)
        if support not in normalized:
            normalization = [[n] for n in support] + [['-1'], ['==']]
            # Normalization is also necessary when no estimand was set yet.
            already_normalized = any(s == frozenset(support) and value == 1
                                     for s, value in known_sums)
            if not already_normalized and normalization not in program.constraints:
                program.constraints.append(normalization)
            normalized.add(support)
        left_names = [marginal(s) for s in left]
        right_names = [marginal(s) for s in right]
        for i, a in enumerate(left_names):
            for j, b in enumerate(right_names):
                if a is None or b is None:
                    continue  # the joint cell is zero as well
                row = [[n] for n in active(cells[i][j])]
                program.constraints.append(row + [['-1', a, b], ['==']])
