from .canonicalModel import canonicalModel
from .Q import Query, Q, sub_list
from .Program import Program
from .DAG import DAG
from .Parser import Parser
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product
from functools import reduce
from scipy.optimize import newton
import scipy
from numpy import log
import statsmodels.stats.proportion
import inspect
import statsmodels.api as sm
from tqdm import tqdm
import warnings


def generate_posterior_beta(result, randomize = True):
    """
    Given a result from a regression, 
    generate a posterior beta
    """
    coef_mean = result.params
    coef_cov = result.cov_params()
    if randomize:
        coef_sampled = np.random.multivariate_normal(coef_mean.flatten(), coef_cov).reshape(coef_mean.shape)
    else:
        coef_sampled = coef_mean.reshape(coef_mean.shape)
    return coef_sampled

def generate_mn_sample(coef_sampled, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        logits_sampled = X @ coef_sampled  # Compute logits
        probs_sampled = np.exp(logits_sampled)
        probs_sampled /= (1 + probs_sampled.sum(axis=1, keepdims=True))  # Normalize to probabilities
        # Add probability for reference category (first category as baseline)
        probs_sampled = np.column_stack([1 - probs_sampled.sum(axis=1), probs_sampled])
        return probs_sampled.reshape(-1)


class respect_to:
    """ Class to be used as a context manager 
    to respect to a causal Problem 
    """
    def __init__(self, problem):
        self.problem = problem
        self.globals = inspect.currentframe().f_back.f_globals
        self._names = (
            "p",
            "E",
            "add_assumption",
            "set_estimand",
            "set_ate",
            "solve",
            "load_data",
            "read_data",
            "generate_samples",
            "is_active",
            "calculate_ci",
        )
        self._previous = {}
        self._created = set()

    def __enter__(self):
        bindings = {
            "p": self.problem.p,
            "E": self.problem.E,
            "add_assumption": self.problem.add_assumption,
            "set_estimand": self.problem.set_estimand,
            "set_ate": self.problem.set_ate,
            "solve": self.problem.solve,
            "load_data": self.problem.load_data,
            "read_data": self.problem.read_data,
            "generate_samples": self.problem.generate_samples,
            "is_active": self.problem.is_active,
            "calculate_ci": self.problem.calculate_ci,
        }
        for name in self._names:
            if name in self.globals:
                self._previous[name] = self.globals[name]
            else:
                self._created.add(name)
            self.globals[name] = bindings[name]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self._names:
            if name in self._previous:
                self.globals[name] = self._previous[name]
            elif name in self._created and name in self.globals:
                del self.globals[name]

def get_summary_from_raw(datam):
    """
    Gets a data set and returns a summary
    """
    nrow = datam.shape[0]
    cols = list(datam.columns)
    datam = deepcopy(datam)
    datam['prob'] = 1/nrow
    return (
        datam.groupby(cols)
        .sum()
        .reset_index()
    )

def multiply_matrix_gaussian(q, mu, sigma_inv):
    if len(q) != len(mu):
        " Q and mu have different sizes"
    len_proc = len(q)
    q_minus_mu = [ mu[i] - q[i]      for i in range(len_proc) ] 
    sum_result = Query(0)
    for i in range(len_proc):
        for j in range(len_proc):
            sum_result += q_minus_mu[i] * sigma_inv[i,j] * q_minus_mu[j]
    return sum_result

def solve_gaussian(nr, o, alpha, index = 'qp'):
    """ alpha is the level of confidence...
    nr is the number of rows
    p is the population distribution we are trying to find
    K is the number of pieces of data
    obs is the observed data 
    """
    if index == 'qp':
        print("Make sure that this dataset is the first to be introduced. For other datasets, remember to introduce the argument data_name")
    query_vectorize = np.vectorize(lambda a: Query(a))
    mu = np.array([o[:-1]])
    params = [ Query(index + '_' + str(i)) for i,f in enumerate(o[:-1]) ]
    mu_diag = np.diag(np.array(o[:-1]))
    sigma = (mu_diag - np.matmul(mu.transpose(),mu)) / nr
    sigma_inv_query = query_vectorize(np.linalg.pinv(sigma))
    mu_query = [ Query(i) for i in o[:-1] ]
    lh_side = multiply_matrix_gaussian(params, mu_query, sigma_inv_query)
    k = len(o)
    rh_side = Query(scipy.stats.chi2.ppf( 1- alpha, k - 1))
    res = lh_side - rh_side
    return (index,k, res)


def solve_kl_p(ns, K, o, alpha):
    """ alpha is the level of confidence...
    ns is the number of sample
    p is the population distribution we are trying to find
    K is the number of pieces of data
    o is the observed data 
    """
    KL = lambda p: o * log(o / p) + (1 - o) * log((1 - o) / (1 - p)) 
    thresh = log(2 * K / alpha) / ns
    optim_func = lambda p: KL(p) - thresh
    eps = 1e-12
    if o == 0:
        return np.array(
                statsmodels
                .stats
                .proportion.proportion_confint(0, ns, alpha = alpha/K, method = 'agresti_coull'))
    elif o == 1:
        return np.array(
                statsmodels
                .stats
                .proportion.proportion_confint(ns, ns, alpha = alpha/K, method = 'agresti_coull'))
    else:
        # Use bounded roots to avoid evaluating log terms outside (0, 1).
        lower_left = eps
        lower_right = max(o - eps, lower_left + eps)
        upper_left = min(o + eps, 1 - 2 * eps)
        upper_right = 1 - eps

        try:
            lb = scipy.optimize.brentq(optim_func, lower_left, lower_right)
        except ValueError:
            lb = 0.0
        try:
            ub = scipy.optimize.brentq(optim_func, upper_left, upper_right)
        except ValueError:
            ub = 1.0
        return np.array([lb, ub])

def get_dirichlet_sample(backbone, all_data, row, covariates):
    """
    Generate Dirichlet samples based on the provided data.

    Args:
    - backbone: DataFrame containing the backbone data
    - all_data: DataFrame containing all data
    - row: Current row data
    - covariates: List of covariates to match the data
    - n: Number of times to calculate the Dirichlet samples (default = 1000)

    Returns:
    - dirichlet_samples: Dirichlet samples generated from the matched data
    """
    if covariates is None:
        prov = pd.merge(backbone, all_data, how = 'left').fillna(0)
    else:
        prov = pd.merge(backbone,
            all_data[
            (all_data[covariates].values == row[covariates].values).all(axis=1)
            ], how = 'left').fillna(0)
    counts = prov['count'].values + 1
    dirichlet_samples = np.random.dirichlet(counts)
    return dirichlet_samples

# Simplifiers 
### 1) First nodes
def simplify_first_nodes(problem, dag, datam, cond): 
    """ 
    Firstly, all first nodes are collected from dag.
    Secondly, if data is complete for those nodes,
    they must be set to zero.
    """
    if len(cond) > 0: # Simplifier 1 cannot handle conditional data
        return None
    data_count = datam.drop('prob', axis = 1).nunique()
    complete_data = [ i  for i, j in dict(data_count).items() 
            if problem.number_values[i] == j ]
    if any([k for k in data_count == 1]): # Important, if data has selection, for instance all elements of X are 1, then return None
        return None
    first_nodes = [ k for k in dag.find_first_nodes() 
            if len(dag.find_u_linked(k)) == 1 and k in complete_data ]
    # Need to check if data is complete
    for k in first_nodes:
        problem.unconf_first_nodes += [ (k + str(i), 
            datam.groupby(k).sum().loc[i]['prob'] )
                for i in datam.groupby(k).sum().index ]
    problem.set_p_to_zero([ x[0] for x in problem.unconf_first_nodes ])


def replace_first_nodes(first_nodes, constraint):
    """ 
    Gets an expr inside a constraint, for instance,
    (1, ['X00.Y00', 'Z0']) and if Z0 is in first nodes, 
    it replace Z0 by 1 and it multiplies 1 by the prob
    """
    coef, var = constraint[0], constraint[1]
    for i,v in enumerate(var):
        for n in first_nodes:
            if v == n[0]:
                var[i] = '1'
                coef *= n[1]
                break
    return ( coef, var )
            


def transform_constraint(constraint, zero_parameters = []):
    """ 
    To be used inside write_program method
    This functions gets a constraint in 
    causalProblem format and translate it to 
    program format
    """
    res =  [ ['' if k[0] == 1 else str(k[0]) ] + 
            [ i for i in k[1] if i != '1' ] 
        for k in constraint ] 
    res = [ [ j for j in i if j != '' ] for i in res  ]
    res = [ [ j for j in i ] for i in res 
            if not any([ j in zero_parameters for j in i ]) ]  # Check if there are zero parameters
    return res

def get_constraint_from_row(row_data, row_prob, program, cond = [ ], n = 0):
    """ 
    Function to be employed in load_data method in causalProgram
    One introduces the row data , row prob , Parser
    and function returns constraints 
    """
    row_cond = cond.iloc[n] if len(cond) > 0  else []
    query = [ f'{row_data.index[j]}={int(row_data.iloc[j])}'
                    for j,k in enumerate(list(row_data)) ]
    if len(row_cond) > 0:
        query_cond = [ f'{row_cond.index[j]}={int(row_cond.iloc[j])}'
                    for j,k in enumerate(list(row_cond)) ]
        return program.p('&'.join(query)) - Query(row_prob) * program.p('&'.join(query_cond))
    return   program.p('&'.join(query)) - Query(row_prob)

def get_query_data_do(row, cols, do, self):
    do_str = ','.join([
        f'{i}={int(row[i])}' 
        for i in do ])
    str_query = '&'.join([
        f'{i}({do_str})={int(row[i])}' 
        for i in cols ])
    return self.p(str_query)


class causalProblem:
    """
    Orchestrator for one or more Bounder objects.

    Backward compatibility:
    - Existing single-problem calls are proxied to the implicit default bounder.
    """

    _INTERNAL_ATTRS = {"_default_bounder", "_bounders", "_bounder_order", "_proxy_warned"}

    def __init__(self, dag, number_values = {}):
        from .Bounder import Bounder
        object.__setattr__(self, "_default_bounder", Bounder(dag, number_values))
        object.__setattr__(self, "_bounders", {})
        object.__setattr__(self, "_bounder_order", [])
        object.__setattr__(self, "_proxy_warned", False)

    def _warn_proxy(self):
        if not self._proxy_warned:
            warnings.warn(
                "Accessing single-problem APIs through causalProblem is kept for "
                "backward compatibility. Prefer explicit Bounder usage via "
                "get_bounder('default') or Bounder(...) for new code.",
                PendingDeprecationWarning,
                stacklevel=3,
            )
            object.__setattr__(self, "_proxy_warned", True)

    def __getattr__(self, name):
        default = self.__dict__.get("_default_bounder")
        if default is None:
            raise AttributeError(name)
        attr = getattr(default, name)
        if callable(attr):
            self._warn_proxy()
        return attr

    def __setattr__(self, name, value):
        if name in self._INTERNAL_ATTRS:
            object.__setattr__(self, name, value)
            return
        default = self.__dict__.get("_default_bounder")
        if default is not None and hasattr(default, name):
            setattr(default, name, value)
            return
        object.__setattr__(self, name, value)

    @property
    def bounders(self):
        return [self._default_bounder] + [self._bounders[k] for k in self._bounder_order]

    def list_bounders(self):
        return ["default"] + self._bounder_order.copy()

    def get_bounder(self, name = "default"):
        if name == "default":
            return self._default_bounder
        if name not in self._bounders:
            raise KeyError(f"Unknown bounder '{name}'. Available: {self.list_bounders()}")
        return self._bounders[name]

    def add_bounder(self, bounder, name = None, replace = False):
        from .Bounder import Bounder
        if name is None:
            name = f"bounder_{len(self._bounder_order) + 1}"
        if name == "default":
            raise ValueError("'default' is reserved for the implicit primary bounder.")
        if not isinstance(bounder, Bounder):
            raise TypeError("bounder must be an instance of Bounder.")
        if name in self._bounders and not replace:
            raise ValueError(f"Bounder '{name}' already exists. Use replace=True to overwrite.")
        if name not in self._bounders:
            self._bounder_order.append(name)
        self._bounders[name] = bounder
        return bounder

    def new_bounder(self, dag = None, number_values = None, name = None):
        from .Bounder import Bounder
        if dag is None:
            dag = deepcopy(self._default_bounder.dag)
        if number_values is None:
            number_values = deepcopy(self._default_bounder.number_values)
        bounder = Bounder(dag, number_values)
        self.add_bounder(bounder, name = name)
        return bounder

    def solve_bounders(self, *args, **kwargs):
        results = {}
        for name in self.list_bounders():
            results[name] = self.get_bounder(name).solve(*args, **kwargs)
        return results

    # Explicit backward-compatible wrappers for single-bounder operations.
    def p(self, *args, **kwargs):
        return self._default_bounder.p(*args, **kwargs)

    def E(self, *args, **kwargs):
        return self._default_bounder.E(*args, **kwargs)

    def set_estimand(self, *args, **kwargs):
        return self._default_bounder.set_estimand(*args, **kwargs)

    def set_ate(self, *args, **kwargs):
        return self._default_bounder.set_ate(*args, **kwargs)

    def add_assumption(self, *args, **kwargs):
        return self._default_bounder.add_assumption(*args, **kwargs)

    def add_constraint(self, *args, **kwargs):
        return self._default_bounder.add_constraint(*args, **kwargs)

    def set_p_to_zero(self, *args, **kwargs):
        return self._default_bounder.set_p_to_zero(*args, **kwargs)

    def load_data(self, *args, **kwargs):
        return self._default_bounder.load_data(*args, **kwargs)

    def load_data_do(self, *args, **kwargs):
        return self._default_bounder.load_data_do(*args, **kwargs)

    def load_data_kl(self, *args, **kwargs):
        return self._default_bounder.load_data_kl(*args, **kwargs)

    def load_data_gaussian(self, *args, **kwargs):
        return self._default_bounder.load_data_gaussian(*args, **kwargs)

    def read_data(self, *args, **kwargs):
        return self._default_bounder.read_data(*args, **kwargs)

    def write_program(self, *args, **kwargs):
        return self._default_bounder.write_program(*args, **kwargs)

    def solve(self, *args, **kwargs):
        # Simple path: one internal bounder, no orchestration requested.
        ci = kwargs.get("ci", False)
        if len(self._bounders) == 0:
            return self._default_bounder.solve(*args, **kwargs)

        # Orchestration path: solve all registered bounders.
        # For now, return per-bounder outputs; aggregation policies
        # (weights/composition rules) can be layered on top.
        return self.solve_bounders(*args, **kwargs)

    def is_active(self, *args, **kwargs):
        return self._default_bounder.is_active(*args, **kwargs)

    def generate_samples(self, *args, **kwargs):
        return self._default_bounder.generate_samples(*args, **kwargs)

    def calculate_ci(self, *args, **kwargs):
        return self._default_bounder.calculate_ci(*args, **kwargs)

    def check_constraints(self, *args, **kwargs):
        return self._default_bounder.check_constraints(*args, **kwargs)

    def add_prob_constraints(self, *args, **kwargs):
        return self._default_bounder.add_prob_constraints(*args, **kwargs)
