"""Compatibility helpers for legacy imports and bounds plotting."""
from .backends.scip_logs import (
    solve_scip, tofloat, get_parse_file_vals_scip, parse_whole_file_scip, parse_particular_bound_scip, get_final_bound_scip, check_process_end_scip, parse_bounds_scip
)
from functools import reduce

get_symb_func = {
        '==': lambda a,b: a== b,
        '<=': lambda a,b: a<= b,
        '>=': lambda a,b: a>= b,
        '<': lambda a,b: a < b,
        '>': lambda a,b: a > b,
        }

fix_symbol_pip = lambda a: '=' if a == '==' else a


def pip_join_expr(expr, params):
    """Serialize a product, multiplying numeric factors instead of joining them."""
    from .polynomial import collect_terms, decimal_token
    terms = collect_terms([expr, ['==']], set(params))
    if not terms:
        return '0'
    variables, coefficient = next(iter(terms.items()))
    if not variables:
        return decimal_token(coefficient)
    product = ' * '.join(variables)
    return product if coefficient == 1 else decimal_token(coefficient) + ' ' + product


def mult_params_scip(k, par_dict):
    """Multiply SCIP factors, with the empty product equal to one."""
    return reduce(lambda a, b: a * b, 
    [ par_dict[r] if r in par_dict.keys() else float(r)  
        for r in k ], 1)


def change_constraint_parameter_value(constraint, parameter, value):
    """Substitute every occurrence of a parameter, including powers."""
    from fractions import Fraction
    from .polynomial import collect_terms, write_constraint
    parameters = set()
    for term in constraint[:-1]:
        for token in term:
            try:
                Fraction(str(token))
            except ValueError:
                parameters.add(token)
    terms = collect_terms(constraint, parameters, {parameter: Fraction(str(value))})
    return write_constraint(terms, constraint[-1][0])


def plot_bounds(traj,
                lb_theory=None,
                ub_theory=None,
                xlab_name='Seconds',
                ylab_name='ATE',
                size=20,
                family=None
                ):
    from .optional import require_optional
    require_optional('plotnine', 'plot')
    from plotnine import (ggplot,
                          geom_errorbar,
                          geom_ribbon,
                          geom_line,
                          geom_hline,
                          theme_classic,
                          aes,
                          element_line,
                          xlim,
                          element_text,
                          theme,
                          xlab,
                          ylab)
    import pandas as pd
    import numpy as np
    # Color Palette
    red = '#A51C30'
    lightred = '#E16273'
    darkblue = '#0C2C5C'
    lightblue = '#4C6C9C'
    grey70 = "#b3b3b3"
    grey30 = "#4d4d4d"
    p = ggplot(traj) + \
        geom_ribbon(aes(x='seconds',
                        ymin='lb.dual',
                        ymax='ub.dual'
                        ),
                    color=None,
                    fill=lightred
                    ) + \
        geom_line(aes(x='seconds',
                      y='ub.dual'
                      ),
                  color=red,
                  lineend='square',
                  size=3
                  ) + \
        geom_line(aes(x='seconds',
                      y='lb.dual'),
                  color=red,
                  lineend='square',
                  size=3
                  ) + \
        geom_ribbon(aes(x='seconds',
                        ymin='lb.prim',
                        ymax='ub.prim'
                        ),
                    fill=lightblue,
                    ) + \
        geom_line(aes(x='seconds',
                      y='ub.prim'
                      ),
                  color='darkblue',
                  lineend='square',
                  size=1
                  ) + \
        geom_line(aes(x='seconds',
                      y='lb.prim'
                      ),
                  color=darkblue,
                  lineend='square',
                  size=1
                  )
    if lb_theory is not None and ub_theory is not None:
        # Define the plot data first
        # Only use the last result to generate error_bar
        last_row_id = np.argmax(traj['seconds'] == max(traj['seconds']))
        bound_data = pd.DataFrame({'seconds': 1.1 * max(traj['seconds']),
                                   'lb': lb_theory.iloc[last_row_id],
                                   'ub': ub_theory.iloc[last_row_id]},
                                  index=[0])
        p = p + geom_errorbar(aes(x='seconds',
                                  ymin='lb',
                                  ymax='ub'
                                  ),
                              width=.05 * max(traj['seconds']),
                              data=bound_data,
                              size=2)
    # Final Plot
    p = p + xlab(xlab_name) + ylab(ylab_name) +\
        geom_hline(yintercept=0, linetype='dashed') + \
        theme_classic(base_size=size) + \
        theme(axis_text=element_text(colour=grey30, size=12),
              axis_line=element_line(color=grey70,
                                     lineend='square',
                                     size=1.5
                                     ),
              axis_ticks=element_line(color=grey70,
                                      size=1.5
                                      ),
              text=element_text(family=family)
              ) + \
        xlim(0, 1.15 * max(traj['seconds']))
    return p
    
