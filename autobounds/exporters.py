"""Mathematical-model serializers, independent of solver execution."""
from contextlib import nullcontext
import io
import math
import os

from .polynomial import decimal_token


def write_pip(model, filename, sense='max'):
    """Export a bound numerical model; PIP/SCIP are floating-point formats."""
    if model.parameters:
        raise ValueError('Bind symbolic parameters before numerical PIP export')
    if sense not in ('max', 'min'):
        raise ValueError("sense must be 'max' or 'min'")
    if isinstance(filename, (str, os.PathLike)):
        destination = open(filename, 'w')
    elif isinstance(filename, io.StringIO):
        destination = nullcontext(filename)
    else:
        raise TypeError('Expected a path or StringIO')

    def number(value):
        try:
            return decimal_token(value)
        except ValueError:
            result = float(value)
            if not math.isfinite(result):
                raise ValueError('Coefficient cannot be represented numerically')
            return repr(result)

    def expression(polynomial):
        result = []
        for term in polynomial.terms:
            if not term.variables:
                result.append(number(term.coefficient))
            else:
                product = ' * '.join(term.variables)
                result.append(product if term.coefficient == 1
                              else number(term.coefficient) + ' ' + product)
        return ' + '.join(result) or '0'

    with destination as stream:
        stream.write(('MAXIMIZE' if sense == 'max' else 'MINIMIZE') + '\n')
        stream.write('  obj: ' + expression(model.objective) + '\n\nSUBJECT TO\n')
        for i, constraint in enumerate(model.constraints):
            relation = '=' if constraint.relation == '==' else constraint.relation
            stream.write(f'a{i}: {expression(constraint.expression)} {relation} 0\n')
        stream.write('\nBOUNDS\n')
        for variable in model.variables:
            lower, upper, name = variable.lower, variable.upper, variable.name
            if lower is None and upper is None:
                stream.write(f'  {name} free\n')
            elif lower is None:
                stream.write(f'  -inf <= {name} <= {number(upper)}\n')
            elif upper is None:
                stream.write(f'  {number(lower)} <= {name} <= +inf\n')
            else:
                stream.write(f'  {number(lower)} <= {name} <= {number(upper)}\n')
        stream.write('\nEND')
