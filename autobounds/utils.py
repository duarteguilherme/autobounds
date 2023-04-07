from pyscipopt import quicksum

def prepare_ate(var, do):
    """ Prepare ATE expr for simulations
    """
    expr1  = { 'sign': 1,'var': (var, 1),'do': (do, 1) }
    expr2  = { 'sign': -1,'var': (var, 1),'do': (do, 0) }
    return (expr1, expr2)

def get_probability_from_model(m, intervention = {}, overlap = False):
    data = m.draw_sample(intervention = intervention)
    columns_to_group = [x for x in data.columns  if x not in m.u_data.keys() ]
    data = (data
            .drop(m.u_data.keys(), axis = 1)
            .assign(P = 1)
            .groupby(columns_to_group)
            .agg('count')
            .assign(P = lambda v: v['P'] / N)
            .reset_index()
            )
    if overlap == True and len(data) < (2**len(m.V)):
        print("*" * 30)
        print("Be Careful! It violates overlap")
        print("*" * 30)
        print(data)
    return data

def get_estimand_value(m, e):
    prob_table = get_probability_from_model(m, 
                intervention = dict([e['do']]))
    filter_v = dict([e['var']])
    filtered = (prob_table
            .loc[(prob_table[list(filter_v)] == pd.Series(filter_v)).all(axis=1)])
    if len(filtered) > 1:
        vars = [e['var'][0]] + [e['do'][0]]
        filtered = filtered.groupby(vars).sum()
    if len(filtered) == 0:
        return 0
    return e['sign']*filtered.P.iloc[0]

def get_c_estimand_value(m, estimand):
    value = 0
    for e in estimand:
        value += get_estimand_value(m, e)
    return value

def simulate_model(dag):
    model = SCM()
    model.from_dag(dag)
    model.sample_u(N)
    return model


def parse_estimand(program, estimand):
    return quicksum(
            [ e['sign'] * quicksum(
                program.get_expr(**{ key: '='.join([ str(part) for part in e[key] ])
                            for key in ['var','do'] }) )
                                for e in estimand ])


def introduce_prob_into_progr(program, prob_table):
    prob_data = prob_table.T.to_dict().values()
    for p in prob_data:
        program.program.addCons(
            quicksum(program.get_expr(var = 
                    ','.join(
                        [ tupl[0] + '=' + str(int(tupl[1])) for tupl in list(p.items())[:-1] ]))
                ) ==
            list(p.items())[-1][1]
        )


