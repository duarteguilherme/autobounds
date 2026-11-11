"""Exact numeric serialization shared by models and adapters."""
from fractions import Fraction


def decimal_token(value):
    """Serialize a terminating rational exactly; reject repeating decimals."""
    value = Fraction(value)
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError('Coefficient does not have a finite decimal expansion')
    places = max(twos, fives)
    numerator = value.numerator * 2 ** (places - twos) * 5 ** (places - fives)
    sign = '-' if numerator < 0 else ''
    digits = str(abs(numerator)).zfill(places + 1)
    if not places:
        return sign + digits
    return (sign + digits[:-places] + '.' + digits[-places:]).rstrip('0').rstrip('.')
