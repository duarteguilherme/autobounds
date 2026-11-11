"""SCIP worker output, log parsing and process monitoring."""
import json
import sys
import time
from multiprocessing.connection import wait as wait_for_processes
import numpy as np

def solve_scip(model, filename, verbose = False, dgps_filename = None):
    """ To be used with scip """
    model.redirectOutput()
    sys.stdout = open(filename, 'w', buffering = 1)
    model.optimize()
    if dgps_filename is None:
        return
    dgps = {"status": "no_solution", "values": {}}
    try:
        # PySCIPOpt can return a wrapper even when SCIP has no solution.
        # An empty variable list would otherwise make that look like a valid DGP.
        sol = model.getBestSol() if model.getNSols() > 0 else None
        if sol is not None:
            var_values = {
                v.name: float(model.getSolVal(sol, v))
                for v in model.getVars()
            }
            dgps = {"status": "ok", "values": var_values}
    except Exception:
        dgps = {"status": "error", "values": {}}
    with open(dgps_filename, "w") as f:
        json.dump(dgps, f)


def tofloat(num):
    try:
        num = float(num)
    except:
        pass
    num = float(np.nan) if num == '--' else num # Fixing problem with scip parser
    return num


def get_parse_file_vals_scip(res):
    return {
                                    'time': tofloat(res[0].strip().split('s')[0].split('L')[-1].split('*')[-1].split('R')[-1]),
                                    'dual': tofloat(res[1].strip()),
                                    'primal': tofloat(res[2].strip())
             }


def parse_whole_file_scip(filename):
    """ Read any of ".lower.log" or
    ".upper.log" and it returns
    data

    This is a variant of parse_particular_bound_scip -- It works by parsing the whole file at the end
    """
    with open(filename) as f:
        data = f.readlines()
    datarows = [ x for x in data if len(x.split('|')) > 5 ]
    if len(datarows) > 0:
        getn = lambda l: l[0:1] + l[-4:-2]
        return [ get_parse_file_vals_scip(getn(k.split('|')))
                for k in datarows ] + [ get_final_bound_scip(filename) ]
    else:
        return None


def parse_particular_bound_scip(filename, n_bound):
    """ Read any of ".lower.log" or
    ".upper.log" and it returns
    data

    Every time it reads the whole file.
    However, it tracks the line from the last reading (n_bound).
    If there are new lines, it returns those lines.
    """
    with open(filename) as f:
        data = f.readlines()
    datarows = [ x
            for x in data if len(x.split('|')) > 5 ]
    if len(datarows) > n_bound:
        res = datarows[-1].split('|')
        res = res[0:1] + res[-4:-2]
        output = (len(datarows), [{
                                'time': tofloat(res[0].strip().split('s')[0].split('L')[-1].split('*')[-1].split('R')[-1]),
                                'dual': tofloat(res[1].strip()),
                                'primal': tofloat(res[2].strip()) }])
        if  res[1].strip() == 'cutoff':
            output[1][0]['dual'] = float('nan')
        return output
    else:
        return (n_bound, [])


def get_final_bound_scip(filename):
    with open(filename,'r') as f:
        data = f.readlines()
    result = {}
    result['primal'] = float([ k for k in data if k.startswith('Primal Bound')][-1]
            .split(':')[1].split('(')[0].strip())
    result['dual'] = float([ k for k in data if k.startswith('Dual Bound')][-1]
            .split(':')[1].split('(')[0].strip())
    result['time'] = float([ k for k in data if k.startswith('Solving Time')][-1]
            .split(':')[1].split('s')[0].strip())
    return result


def check_process_end_scip(p, filename, verbose = True):
    # The final status may be printed before the summary and optional DGP
    # file are written. Only consume final results once the worker exits.
    if p.is_alive():
        return -1
    if p.exitcode != 0:
        raise RuntimeError(f"SCIP worker exited with code {p.exitcode}.")
    with open(filename,'r') as f:
        data = f.readlines()
    if any([x for x in data if "[optimal solution found]" in x ]):
        if verbose:
            print("Problem is finished! Returning final values")
        return 1
    if any([x for x in data if 'problem is solved [infeasible]' in x ]):
        if verbose:
            print("Problem is infeasible. Returning without solutions")
        return 0
    else:
        raise RuntimeError(f"SCIP worker exited without a recognized final status in {filename}.")


def parse_bounds_scip(p_lower, p_upper, filelower = None, fileupper = None, output = None, epsilon = -10, theta = 0.01, maxtime = None, verbose = True):
    """Monitor bounds, waking on worker exit or a short progress interval.

    Process sentinels avoid delaying completed solves. The progress interval
    still permits early stopping on bounds while both workers are running.
    """
    init_time = time.monotonic()
    i = {'primal': np.nan, 'dual': np.nan, 'time': np.nan}
    j = {'primal': np.nan, 'dual': np.nan, 'time': np.nan}
    total_lower, total_upper = [ ], []
    n_lower, n_upper = 0, 0
    current_theta, current_epsilon = 9999, 9999
    if output is not None:
        with open(output, 'w') as f:
            f.write(f"bound,primal,dual,time\n")
    while True:
        n_lower, partial_lower = parse_particular_bound_scip(filelower, n_lower)
        n_upper, partial_upper = parse_particular_bound_scip(fileupper, n_upper)
        total_lower += partial_lower
        total_upper += partial_upper
        if len(partial_lower) > 0:
            for i in partial_lower:
                if verbose:
                    print(f"LOWER BOUND: # -- Primal: {i['primal']} / Dual: {i['dual']} / Time: {i['time']} ##")
                if output is not None:
                    with open(output, 'a') as f:
                        f.write(f"lb,{i['primal']},{i['dual']},{i['time']}\n")
        if len(partial_upper) > 0:
            for j in partial_upper:
                if verbose:
                    print(f"UPPER BOUND: # -- Primal: {j['primal']} / Dual: {j['dual']} / Time: {j['time']} ##")
                if output is not None:
                    with open(output, 'a') as f:
                        f.write(f"ub,{j['primal']},{j['dual']},{j['time']}\n")
        end_lower = check_process_end_scip(p_lower, filelower, verbose)
        end_upper = check_process_end_scip(p_upper, fileupper, verbose)
        if len(total_lower) > 0 and len(total_upper) > 0:
            total_upper[-1]['dual'] = np.inf if np.isnan(total_upper[-1]['dual']) else total_upper[-1]['dual'] # Fixing the problem with nan inside dual
            total_lower[-1]['dual'] = -np.inf if np.isnan(total_lower[-1]['dual']) else total_lower[-1]['dual']
            try:
                current_theta = total_upper[-1]['dual'] - total_lower[-1]['dual']
                gamma = abs(total_upper[-1]['primal'] - total_lower[-1]['primal'])
                current_epsilon = current_theta/gamma - 1 if gamma != 0 else 99999999
            except:
                current_theta, gamma, current_epsilon = np.nan, np.nan, np.nan
            if verbose:
                print(f"CURRENT THRESHOLDS: # -- Theta: {current_theta} / Epsilon: {current_epsilon} ##")
            if current_theta <  theta or current_epsilon < epsilon:
                p_lower.terminate()
                p_upper.terminate()
                break
        if end_lower != -1 and end_upper != -1:
            break
        wait_timeout = 0.1
        if maxtime is not None:
            remaining = maxtime - (time.monotonic() - init_time)
            if remaining <= 0:
                p_lower.terminate()
                p_upper.terminate()
                break
            wait_timeout = min(wait_timeout, remaining)
        active = [p.sentinel for p in (p_lower, p_upper) if p.is_alive()]
        if active:
            wait_for_processes(active, timeout=wait_timeout)
    # Checking bounds if problem is finished
    if end_lower == 1 or end_upper == 1:
        if end_lower == 1:
            i = get_final_bound_scip(filelower)
        if end_upper == 1:
            j = get_final_bound_scip(fileupper)
        try:
            current_theta = j['dual'] - i['dual']
            gamma = abs(j['primal'] - i['primal'])
            current_epsilon = current_theta/gamma - 1 if gamma != 0 else 99999999
        except:
            current_theta, gamma, current_epsilon = -1, -1, -1
    else:
        if end_lower == 0 and end_upper == 0:
            i, j, current_theta, current_epsilon = {}, {},-1,-1
    i['end'] = end_lower
    j['end'] = end_upper
    if output is not None:
        with open(output, 'a') as f:
            f.write(f"lb,{i.get('primal', np.nan)},{i.get('dual', np.nan)},{i.get('time', np.nan)}\n")
            f.write(f"ub,{j.get('primal', np.nan)},{j.get('dual', np.nan)},{j.get('time', np.nan)}\n")
    return (i, j, current_theta, current_epsilon)
