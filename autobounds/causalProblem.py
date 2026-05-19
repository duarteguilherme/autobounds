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
from concurrent.futures import ThreadPoolExecutor, as_completed


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
            "is_active",
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
            "is_active": self.problem.is_active,
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

    _INTERNAL_ATTRS = {
        "_default_bounder",
        "_bounders",
        "_bounder_order",
        "_proxy_warned",
        "_operation_log",
        "_is_replaying",
        "_has_covariates",
        "_read_data_state",
        "K",
        "covariates",
        "inference",
        "categorical",
        "main_model",
        "_used_discrete_covariate_path",
        "_covariate_support_size",
        "continuous_outcome",
        "continuous_bins",
        "continuous_method",
        "_continuous_outcome_name",
        "_continuous_estimand",
        "_continuous_bin_specs",
    }

    def __init__(
        self,
        dag,
        number_values = {},
        continuous_outcome = False,
        continuous_bins = 5,
        continuous_method = "midpoint",
    ):
        from .Bounder import Bounder
        if continuous_method not in {"ymax_ymin", "midpoint", "conservative"}:
            raise ValueError(
                "continuous_method must be one of {'ymax_ymin', 'midpoint', 'conservative'}."
            )
        if isinstance(continuous_outcome, str):
            continuous_outcome_name = continuous_outcome
            continuous_outcome_flag = True
        else:
            continuous_outcome_name = None
            continuous_outcome_flag = bool(continuous_outcome)
        object.__setattr__(self, "_default_bounder", Bounder(dag, number_values))
        object.__setattr__(self, "_bounders", {})
        object.__setattr__(self, "_bounder_order", [])
        object.__setattr__(self, "_proxy_warned", False)
        object.__setattr__(self, "_operation_log", [])
        object.__setattr__(self, "_is_replaying", False)
        object.__setattr__(self, "_has_covariates", False)
        object.__setattr__(self, "_read_data_state", None)
        object.__setattr__(self, "K", 20)
        object.__setattr__(self, "covariates", None)
        object.__setattr__(self, "inference", False)
        object.__setattr__(self, "categorical", True)
        object.__setattr__(self, "main_model", None)
        object.__setattr__(self, "_used_discrete_covariate_path", False)
        object.__setattr__(self, "_covariate_support_size", None)
        object.__setattr__(self, "continuous_outcome", continuous_outcome_flag)
        object.__setattr__(self, "continuous_bins", int(continuous_bins))
        object.__setattr__(self, "continuous_method", continuous_method)
        object.__setattr__(self, "_continuous_outcome_name", continuous_outcome_name)
        object.__setattr__(self, "_continuous_estimand", None)
        object.__setattr__(self, "_continuous_bin_specs", None)

    def _freeze_tabular(self, obj):
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            return obj.copy(deep=True)
        if isinstance(obj, str):
            return pd.read_csv(obj)
        if hasattr(obj, "read"):
            pos = None
            if hasattr(obj, "tell"):
                try:
                    pos = obj.tell()
                except Exception:
                    pos = None
            if hasattr(obj, "seek"):
                try:
                    obj.seek(0)
                except Exception:
                    pass
            df = pd.read_csv(obj)
            if pos is not None and hasattr(obj, "seek"):
                try:
                    obj.seek(pos)
                except Exception:
                    pass
            return df
        raise TypeError("Unsupported tabular input type for subsampling CI.")

    def _record_operation(self, method, args, kwargs):
        if self._is_replaying:
            return
        self._operation_log.append(
            {
                "method": method,
                "args": deepcopy(args),
                "kwargs": deepcopy(kwargs),
            }
        )

    def _continuous_outcome_matches(self, dep=None):
        if not self.continuous_outcome:
            return False
        if self._continuous_outcome_name is None:
            return dep is None
        return dep == self._continuous_outcome_name

    def _resolve_continuous_outcome_name(self):
        if not self.continuous_outcome:
            return None
        if self._continuous_outcome_name is None:
            raise ValueError(
                "continuous_outcome=True requires calling set_ate(..., dep=...) before loading data, "
                "or passing continuous_outcome='<outcome_name>' in the constructor."
            )
        return self._continuous_outcome_name

    def _digitize_continuous_outcome(self, datam, outcome):
        if outcome not in datam.columns:
            raise KeyError(f"Continuous outcome '{outcome}' not found in data columns.")
        if "prob" in datam.columns:
            raise NotImplementedError(
                "Continuous-outcome discretization currently requires raw row-level data, not summary data."
            )

        out = datam.copy(deep=True)
        values = pd.to_numeric(out[outcome], errors="raise")
        non_missing = values.dropna()
        if non_missing.shape[0] == 0:
            raise ValueError(f"Continuous outcome '{outcome}' has no observed values.")
        y_min = float(non_missing.min())
        y_max = float(non_missing.max())
        if np.isclose(y_min, y_max):
            normalized = pd.Series(np.zeros(out.shape[0], dtype=float), index=out.index)
        else:
            normalized = (values - y_min) / (y_max - y_min)

        n_unique = int(non_missing.nunique())
        n_bins_target = max(1, min(int(self.continuous_bins), n_unique))
        if n_bins_target == 1:
            codes = pd.Series(np.zeros(out.shape[0], dtype=int), index=out.index)
        else:
            try:
                codes = pd.qcut(normalized.rank(method="first"), q=n_bins_target, labels=False, duplicates="drop")
            except ValueError:
                codes = pd.Series(np.zeros(out.shape[0], dtype=int), index=out.index)
        codes = pd.Series(codes, index=out.index).astype("Int64")
        unique_codes = sorted(codes.dropna().unique().tolist())
        if len(unique_codes) == 0:
            raise ValueError(f"Failed to construct bins for continuous outcome '{outcome}'.")

        code_map = {int(raw_code): idx for idx, raw_code in enumerate(unique_codes)}
        remapped = codes.map(code_map).astype(int)
        out[outcome] = remapped

        bin_specs = []
        for raw_code in unique_codes:
            new_code = code_map[int(raw_code)]
            mask = remapped == new_code
            orig_vals = values.loc[mask]
            norm_vals = normalized.loc[mask]
            bin_specs.append(
                {
                    "bin": int(new_code),
                    "ymin": float(orig_vals.min()),
                    "ymax": float(orig_vals.max()),
                    "normalized_ymin": float(norm_vals.min()),
                    "normalized_ymax": float(norm_vals.max()),
                }
            )

        object.__setattr__(self, "_continuous_bin_specs", bin_specs)
        return out

    def _prepare_continuous_loaded_data(self, raw=None, summary=None):
        if raw is not None and summary is not None:
            raise ValueError("Provide only one of raw or summary data.")
        if summary is not None:
            raise NotImplementedError(
                "Continuous-outcome mode currently supports raw row-level data only."
            )
        if raw is None:
            return raw, summary
        datam = deepcopy(raw) if isinstance(raw, pd.DataFrame) else pd.read_csv(raw)
        transformed = self._digitize_continuous_outcome(datam, self._resolve_continuous_outcome_name())
        self._ensure_default_bounder_number_values(
            outcome=self._resolve_continuous_outcome_name(),
            n_bins=len(self._continuous_bin_specs),
        )
        return transformed, None

    def _ensure_default_bounder_number_values(self, outcome, n_bins):
        from .Bounder import Bounder

        if self._default_bounder.number_values.get(outcome, 2) == int(n_bins):
            return

        number_values = deepcopy(self._default_bounder.number_values)
        number_values[outcome] = int(n_bins)
        new_bounder = Bounder(
            deepcopy(self._default_bounder.dag),
            number_values,
        )
        for step in self._operation_log:
            method = step["method"]
            if method in {"load_data", "read_data", "set_ate", "set_estimand"}:
                continue
            getattr(new_bounder, method)(*deepcopy(step["args"]), **deepcopy(step["kwargs"]))
        object.__setattr__(self, "_default_bounder", new_bounder)

    def _threshold_bin_raw_data(self, datam, outcome, bin_id):
        if outcome not in datam.columns:
            raise KeyError(f"Outcome '{outcome}' not found in data columns.")
        out = datam.copy(deep=True)
        out[outcome] = (out[outcome] == bin_id).astype(int)
        return out

    def _threshold_bin_summary_data(self, datam, outcome, bin_id):
        if "prob" not in datam.columns:
            raise ValueError("Summary data must include a 'prob' column.")
        out = self._threshold_bin_raw_data(datam, outcome, bin_id)
        group_cols = [c for c in out.columns if c != "prob"]
        return out.groupby(group_cols, dropna=False, sort=False, as_index=False)["prob"].sum()

    def _build_category_problem(self, ind, dep, bin_id):
        number_values = deepcopy(self._default_bounder.number_values)
        number_values[dep] = 2
        category_problem = causalProblem(
            deepcopy(self._default_bounder.dag),
            number_values,
        )

        for step in self._operation_log:
            method = step["method"]
            if method in {"set_estimand", "set_ate", "load_data", "read_data"}:
                continue
            getattr(category_problem, method)(*deepcopy(step["args"]), **deepcopy(step["kwargs"]))

        category_problem.set_ate(ind, dep)

        for step in self._operation_log:
            method = step["method"]
            call_kwargs = deepcopy(step["kwargs"])
            if method == "load_data":
                if call_kwargs.get("raw") is not None:
                    call_kwargs["raw"] = self._threshold_bin_raw_data(call_kwargs["raw"], dep, bin_id)
                elif call_kwargs.get("summary") is not None:
                    call_kwargs["summary"] = self._threshold_bin_summary_data(
                        call_kwargs["summary"], dep, bin_id
                    )
                category_problem.load_data(**call_kwargs)
            elif method == "read_data":
                raw = call_kwargs.get("raw")
                if raw is None:
                    raise ValueError("Continuous-outcome binning requires read_data(raw=...).")
                call_kwargs["raw"] = self._threshold_bin_raw_data(raw, dep, bin_id)
                category_problem.read_data(**call_kwargs)

        return category_problem

    def _continuous_threshold_weights(self):
        if self._continuous_bin_specs is None or len(self._continuous_bin_specs) < 2:
            raise ValueError("Continuous-outcome threshold aggregation requires at least two bins.")
        if self.continuous_method in {"ymax_ymin", "conservative"}:
            lower_representatives = [float(spec["ymin"]) for spec in self._continuous_bin_specs]
            upper_representatives = [float(spec["ymax"]) for spec in self._continuous_bin_specs]
        elif self.continuous_method == "midpoint":
            midpoints = [
                0.5 * (float(spec["ymin"]) + float(spec["ymax"]))
                for spec in self._continuous_bin_specs
            ]
            lower_representatives = midpoints
            upper_representatives = midpoints
        else:
            raise ValueError(
                "continuous_method must be one of {'ymax_ymin', 'midpoint', 'conservative'}."
            )
        weights = []
        for idx in range(1, len(lower_representatives)):
            weights.append(
                {
                    "threshold": idx,
                    "lower_weight": float(lower_representatives[idx] - lower_representatives[idx - 1]),
                    "upper_weight": float(upper_representatives[idx] - upper_representatives[idx - 1]),
                    "lower_bin": idx - 1,
                    "upper_bin": idx,
                }
            )
        return lower_representatives, upper_representatives, weights

    def _build_continuous_weighted_problem(self, ind, dep, representatives, cond=None):
        number_values = deepcopy(self._default_bounder.number_values)
        weighted_problem = causalProblem(
            deepcopy(self._default_bounder.dag),
            number_values,
        )

        for step in self._operation_log:
            method = step["method"]
            if method in {"set_estimand", "set_ate", "load_data", "read_data"}:
                continue
            getattr(weighted_problem, method)(*deepcopy(step["args"]), **deepcopy(step["kwargs"]))

        query = None
        for level, weight in enumerate(representatives):
            term = (
                weighted_problem.p(f"{dep}({ind}=1)={level}", cond=cond) * float(weight)
                - weighted_problem.p(f"{dep}({ind}=0)={level}", cond=cond) * float(weight)
            )
            query = term if query is None else query + term
        weighted_problem.set_estimand(query)

        for step in self._operation_log:
            method = step["method"]
            call_kwargs = deepcopy(step["kwargs"])
            if method == "load_data":
                weighted_problem.load_data(**call_kwargs)
            elif method == "read_data":
                weighted_problem.read_data(**call_kwargs)

        return weighted_problem

    def _solve_continuous_outcome_ate(self, **solve_kwargs):
        if self._continuous_estimand is None or self._continuous_estimand.get("kind") != "ate":
            raise ValueError("Continuous-outcome solve currently supports set_ate(...) only.")
        if self.continuous_method == "conservative":
            return self._solve_continuous_outcome_ate_conservative(**solve_kwargs)
        want_ci = bool(solve_kwargs.get("ci", False))
        want_dgps = bool(solve_kwargs.get("return_dgps", False))
        if self._continuous_bin_specs is None:
            raise ValueError("Load data before solving a continuous outcome problem.")

        ind = self._continuous_estimand["ind"]
        dep = self._continuous_estimand["dep"]
        cond = self._continuous_estimand.get("cond")
        subsolve_kwargs = deepcopy(solve_kwargs)
        subsolve_kwargs["verbose_result"] = False
        lower_representatives, upper_representatives, _ = self._continuous_threshold_weights()

        if self.continuous_method == "midpoint":
            weighted_problem = self._build_continuous_weighted_problem(
                ind,
                dep,
                lower_representatives,
                cond=cond,
            )
            weighted_result = weighted_problem.solve(**deepcopy(subsolve_kwargs))
            out = {
                "point lb dual": float(weighted_result["point lb dual"]),
                "point ub dual": float(weighted_result["point ub dual"]),
                "point lb primal": float(weighted_result["point lb primal"]),
                "point ub primal": float(weighted_result["point ub primal"]),
                "continuous_outcome": True,
                "continuous_method": self.continuous_method,
                "continuous_bins": int(self.continuous_bins),
                "outcome": dep,
                "treatment": ind,
                "continuous_lower_representatives": lower_representatives,
                "continuous_upper_representatives": upper_representatives,
                "weighted_results": {"midpoint": weighted_result},
            }
            if want_ci:
                out["2.5% lb bounds"] = float(weighted_result["2.5% lb bounds"])
                out["1% lb bounds"] = float(weighted_result["1% lb bounds"])
                out["97.5% ub bounds"] = float(weighted_result["97.5% ub bounds"])
                out["99% ub bounds"] = float(weighted_result["99% ub bounds"])
                if "ci method" in weighted_result:
                    out["ci method"] = weighted_result["ci method"]
                if "ci workers" in weighted_result:
                    out["ci workers"] = weighted_result["ci workers"]
            if want_dgps:
                out["dgps"] = weighted_result["dgps"]
            return out

        if self.continuous_method != "ymax_ymin":
            raise ValueError(
                "continuous_method must be one of {'ymax_ymin', 'midpoint', 'conservative'}."
            )

        lower_problem = self._build_continuous_weighted_problem(
            ind,
            dep,
            lower_representatives,
            cond=cond,
        )
        upper_problem = self._build_continuous_weighted_problem(
            ind,
            dep,
            upper_representatives,
            cond=cond,
        )
        lower_result = lower_problem.solve(**deepcopy(subsolve_kwargs))
        upper_result = upper_problem.solve(**deepcopy(subsolve_kwargs))

        out = {
            "point lb dual": float(lower_result["point lb dual"]),
            "point ub dual": float(upper_result["point ub dual"]),
            "point lb primal": float(lower_result["point lb primal"]),
            "point ub primal": float(upper_result["point ub primal"]),
            "continuous_outcome": True,
            "continuous_method": self.continuous_method,
            "continuous_bins": int(self.continuous_bins),
            "outcome": dep,
            "treatment": ind,
            "continuous_lower_representatives": lower_representatives,
            "continuous_upper_representatives": upper_representatives,
            "weighted_results": {"lower": lower_result, "upper": upper_result},
        }
        if want_ci:
            out["2.5% lb bounds"] = float(lower_result["2.5% lb bounds"])
            out["1% lb bounds"] = float(lower_result["1% lb bounds"])
            out["97.5% ub bounds"] = float(upper_result["97.5% ub bounds"])
            out["99% ub bounds"] = float(upper_result["99% ub bounds"])
            if "ci method" in lower_result:
                out["ci method"] = lower_result["ci method"]
            elif "ci method" in upper_result:
                out["ci method"] = upper_result["ci method"]
            if "ci workers" in lower_result:
                out["ci workers"] = lower_result["ci workers"]
            elif "ci workers" in upper_result:
                out["ci workers"] = upper_result["ci workers"]
        if want_dgps:
            out["dgps"] = {
                "lower": lower_result["dgps"]["lower"],
                "upper": upper_result["dgps"]["upper"],
            }
        return out

    def _build_threshold_component_problem(self, ind, dep, cutoff, treatment_value, cond=None):
        number_values = deepcopy(self._default_bounder.number_values)
        number_values[dep] = 2
        threshold_problem = causalProblem(
            deepcopy(self._default_bounder.dag),
            number_values,
        )

        for step in self._operation_log:
            method = step["method"]
            if method in {"set_estimand", "set_ate", "load_data", "read_data"}:
                continue
            getattr(threshold_problem, method)(*deepcopy(step["args"]), **deepcopy(step["kwargs"]))

        threshold_problem.set_estimand(
            threshold_problem.p(f"{dep}({ind}={treatment_value})=1", cond=cond)
        )

        for step in self._operation_log:
            method = step["method"]
            call_kwargs = deepcopy(step["kwargs"])
            if method == "load_data":
                if call_kwargs.get("raw") is not None:
                    call_kwargs["raw"] = self._threshold_raw_data(call_kwargs["raw"], dep, cutoff, "geq")
                elif call_kwargs.get("summary") is not None:
                    call_kwargs["summary"] = self._threshold_summary_data(
                        call_kwargs["summary"], dep, cutoff, "geq"
                    )
                threshold_problem.load_data(**call_kwargs)
            elif method == "read_data":
                raw = call_kwargs.get("raw")
                if raw is None:
                    raise ValueError("Continuous-outcome thresholding requires read_data(raw=...).")
                call_kwargs["raw"] = self._threshold_raw_data(raw, dep, cutoff, "geq")
                threshold_problem.read_data(**call_kwargs)

        return threshold_problem

    def _solve_continuous_outcome_ate_conservative(self, **solve_kwargs):
        want_ci = bool(solve_kwargs.get("ci", False))
        want_dgps = bool(solve_kwargs.get("return_dgps", False))
        if self._continuous_bin_specs is None:
            raise ValueError("Load data before solving a continuous outcome problem.")

        ind = self._continuous_estimand["ind"]
        dep = self._continuous_estimand["dep"]
        cond = self._continuous_estimand.get("cond")
        subsolve_kwargs = deepcopy(solve_kwargs)
        subsolve_kwargs["verbose_result"] = False
        lower_representatives, upper_representatives, threshold_weights = self._continuous_threshold_weights()

        components = {}
        for treatment_value in [0, 1]:
            components[treatment_value] = {
                "point lb dual": float(lower_representatives[0]),
                "point ub dual": float(upper_representatives[0]),
                "point lb primal": float(lower_representatives[0]),
                "point ub primal": float(upper_representatives[0]),
                "threshold_results": {},
            }
            if want_ci:
                components[treatment_value]["2.5% lb bounds"] = float(lower_representatives[0])
                components[treatment_value]["1% lb bounds"] = float(lower_representatives[0])
                components[treatment_value]["97.5% ub bounds"] = float(upper_representatives[0])
                components[treatment_value]["99% ub bounds"] = float(upper_representatives[0])
            if want_dgps:
                components[treatment_value]["dgps"] = {
                    "lower": {"status": f"continuous_component_{treatment_value}_lower", "thresholds": []},
                    "upper": {"status": f"continuous_component_{treatment_value}_upper", "thresholds": []},
                }

        ci_method = None
        ci_workers = None
        for spec in threshold_weights:
            cutoff = int(spec["threshold"])
            lower_weight = float(spec["lower_weight"])
            upper_weight = float(spec["upper_weight"])
            for treatment_value in [0, 1]:
                threshold_problem = self._build_threshold_component_problem(
                    ind,
                    dep,
                    cutoff,
                    treatment_value,
                    cond=cond,
                )
                threshold_result = threshold_problem.solve(**deepcopy(subsolve_kwargs))
                components[treatment_value]["threshold_results"][cutoff] = {
                    "threshold": cutoff,
                    "lower_weight": lower_weight,
                    "upper_weight": upper_weight,
                    "result": threshold_result,
                }
                components[treatment_value]["point lb dual"] += lower_weight * float(threshold_result["point lb dual"])
                components[treatment_value]["point ub dual"] += upper_weight * float(threshold_result["point ub dual"])
                components[treatment_value]["point lb primal"] += lower_weight * float(threshold_result["point lb primal"])
                components[treatment_value]["point ub primal"] += upper_weight * float(threshold_result["point ub primal"])
                if want_ci:
                    components[treatment_value]["2.5% lb bounds"] += lower_weight * float(threshold_result["2.5% lb bounds"])
                    components[treatment_value]["1% lb bounds"] += lower_weight * float(threshold_result["1% lb bounds"])
                    components[treatment_value]["97.5% ub bounds"] += upper_weight * float(threshold_result["97.5% ub bounds"])
                    components[treatment_value]["99% ub bounds"] += upper_weight * float(threshold_result["99% ub bounds"])
                    if ci_method is None:
                        ci_method = threshold_result.get("ci method")
                    if ci_workers is None and "ci workers" in threshold_result:
                        ci_workers = threshold_result.get("ci workers")
                if want_dgps:
                    components[treatment_value]["dgps"]["lower"]["thresholds"].append(
                        {
                            "threshold": cutoff,
                            "weight": lower_weight,
                            "dgps": threshold_result["dgps"]["lower"],
                        }
                    )
                    components[treatment_value]["dgps"]["upper"]["thresholds"].append(
                        {
                            "threshold": cutoff,
                            "weight": upper_weight,
                            "dgps": threshold_result["dgps"]["upper"],
                        }
                    )

        y1 = components[1]
        y0 = components[0]
        out = {
            "point lb dual": float(y1["point lb dual"] - y0["point ub dual"]),
            "point ub dual": float(y1["point ub dual"] - y0["point lb dual"]),
            "point lb primal": float(y1["point lb primal"] - y0["point ub primal"]),
            "point ub primal": float(y1["point ub primal"] - y0["point lb primal"]),
            "continuous_outcome": True,
            "continuous_method": self.continuous_method,
            "continuous_bins": int(self.continuous_bins),
            "outcome": dep,
            "treatment": ind,
            "continuous_lower_representatives": lower_representatives,
            "continuous_upper_representatives": upper_representatives,
            "component_results": components,
        }
        if want_ci:
            out["2.5% lb bounds"] = float(y1["2.5% lb bounds"] - y0["97.5% ub bounds"])
            out["1% lb bounds"] = float(y1["1% lb bounds"] - y0["99% ub bounds"])
            out["97.5% ub bounds"] = float(y1["97.5% ub bounds"] - y0["2.5% lb bounds"])
            out["99% ub bounds"] = float(y1["99% ub bounds"] - y0["1% lb bounds"])
            if ci_method is not None:
                out["ci method"] = ci_method
            if ci_workers is not None:
                out["ci workers"] = ci_workers
        if want_dgps:
            out["dgps"] = {
                "lower": {
                    "status": "continuous_conservative_aggregate",
                    "components": {"treated": y1["dgps"]["lower"], "control": y0["dgps"]["upper"]},
                },
                "upper": {
                    "status": "continuous_conservative_aggregate",
                    "components": {"treated": y1["dgps"]["upper"], "control": y0["dgps"]["lower"]},
                },
            }
        return out

    def _record_load_data_operation(self, args, kwargs):
        if self._is_replaying:
            return
        sig = inspect.signature(self._default_bounder.load_data)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        data_kwargs = deepcopy(bound.arguments)
        data_kwargs["summary"] = self._freeze_tabular(data_kwargs.get("summary"))
        data_kwargs["raw"] = self._freeze_tabular(data_kwargs.get("raw"))
        self._operation_log.append(
            {
                "method": "load_data",
                "args": tuple(),
                "kwargs": data_kwargs,
            }
        )

    def _is_discrete_covariate(self, series):
        if (
            pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_integer_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or pd.api.types.is_object_dtype(series)
        ):
            return True
        if pd.api.types.is_float_dtype(series):
            vals = series.dropna().to_numpy()
            if vals.size == 0:
                return True
            return np.all(np.isclose(vals, np.round(vals)))
        return False

    def _use_empirical_covariate_path(self, datam, covariates, nk):
        cov_df = datam[covariates]
        is_discrete = all(self._is_discrete_covariate(cov_df[col]) for col in covariates)
        support_size = int(cov_df.drop_duplicates().shape[0])
        use_empirical = is_discrete and support_size <= int(nk)
        return use_empirical, support_size

    def _normalize_read_data_args(self, args, kwargs):
        if len(args) > 1:
            raise TypeError("read_data accepts at most one positional argument for raw data.")
        local_kwargs = dict(kwargs)
        if len(args) == 1:
            if "raw" in local_kwargs:
                raise TypeError("raw provided twice to read_data.")
            local_kwargs["raw"] = args[0]
        return {
            "raw": local_kwargs.get("raw", None),
            "covariates": local_kwargs.get("covariates", None),
            "inference": local_kwargs.get("inference", False),
            "cond": local_kwargs.get("cond", []),
            "categorical": local_kwargs.get("categorical", True),
            "model": local_kwargs.get("model", None),
            "nsamples": local_kwargs.get("nsamples", 1000),
            "nk": local_kwargs.get("nk", 200),
        }

    def _threshold_raw_data(self, datam, outcome, cutoff, direction):
        if outcome not in datam.columns:
            raise KeyError(f"Outcome '{outcome}' not found in data columns.")
        out = datam.copy(deep=True)
        if direction == "geq":
            out[outcome] = (out[outcome] >= cutoff).astype(int)
        elif direction == "leq":
            out[outcome] = (out[outcome] <= cutoff).astype(int)
        else:
            raise ValueError("direction must be either 'geq' or 'leq'.")
        return out

    def _threshold_summary_data(self, datam, outcome, cutoff, direction):
        if "prob" not in datam.columns:
            raise ValueError("Summary data must include a 'prob' column.")
        out = self._threshold_raw_data(datam, outcome, cutoff, direction)
        group_cols = [c for c in out.columns if c != "prob"]
        return out.groupby(group_cols, dropna=False, sort=False, as_index=False)["prob"].sum()

    def _get_threshold_source_values(self, outcome):
        for step in self._operation_log:
            if step["method"] == "read_data":
                raw = step["kwargs"].get("raw")
                if raw is not None and outcome in raw.columns:
                    return sorted(pd.Series(raw[outcome]).dropna().unique().tolist())
            if step["method"] == "load_data":
                raw = step["kwargs"].get("raw")
                summary = step["kwargs"].get("summary")
                if raw is not None and outcome in raw.columns:
                    return sorted(pd.Series(raw[outcome]).dropna().unique().tolist())
                if summary is not None and outcome in summary.columns:
                    return sorted(pd.Series(summary[outcome]).dropna().unique().tolist())
        raise ValueError(f"Could not infer thresholds because no data source contains outcome '{outcome}'.")

    def _build_threshold_problem(self, ind, dep, cutoff, direction):
        threshold_number_values = deepcopy(self._default_bounder.number_values)
        threshold_number_values[dep] = 2
        threshold_problem = causalProblem(
            deepcopy(self._default_bounder.dag),
            threshold_number_values,
        )

        for step in self._operation_log:
            method = step["method"]
            if method in {"set_estimand", "set_ate", "load_data", "read_data"}:
                continue
            getattr(threshold_problem, method)(*deepcopy(step["args"]), **deepcopy(step["kwargs"]))

        threshold_problem.set_ate(ind, dep)

        for step in self._operation_log:
            method = step["method"]
            call_kwargs = deepcopy(step["kwargs"])
            if method == "load_data":
                if call_kwargs.get("raw") is not None:
                    call_kwargs["raw"] = self._threshold_raw_data(call_kwargs["raw"], dep, cutoff, direction)
                elif call_kwargs.get("summary") is not None:
                    call_kwargs["summary"] = self._threshold_summary_data(
                        call_kwargs["summary"], dep, cutoff, direction
                    )
                threshold_problem.load_data(**call_kwargs)
            elif method == "read_data":
                raw = call_kwargs.get("raw")
                if raw is None:
                    raise ValueError("Threshold outcome simplification requires read_data(raw=...).")
                call_kwargs["raw"] = self._threshold_raw_data(raw, dep, cutoff, direction)
                threshold_problem.read_data(**call_kwargs)

        return threshold_problem

    def _record_read_data_operation(self, args, kwargs):
        if self._is_replaying:
            return
        data_kwargs = self._normalize_read_data_args(args, kwargs)
        data_kwargs["raw"] = self._freeze_tabular(data_kwargs.get("raw"))
        covariates = data_kwargs.get("covariates", None)
        object.__setattr__(self, "_has_covariates", covariates is not None and len(covariates) > 0)
        self._operation_log.append(
            {
                "method": "read_data",
                "args": tuple(),
                "kwargs": data_kwargs,
            }
        )

    def _solve_with_multinomial_binned_ci(self, *args, **kwargs):
        from .Bounder import Bounder

        if len(args) > 0:
            raise ValueError("Use keyword arguments when ci=True in causalProblem.solve().")
        if self._has_covariates:
            raise NotImplementedError("Covariate-aware CI path is not implemented yet.")

        maxtime = kwargs.get("maxtime", None)
        theta = kwargs.get("theta", 0.01)
        verbose_optimizer = kwargs.get("verbose_optimizer", False)
        verbose_result = kwargs.get("verbose_result", True)
        limits = kwargs.get("limits", [None, None])
        ci_method = kwargs.get("ci_method", "recentered_subsampling")
        if ci_method != "recentered_subsampling":
            raise ValueError(
                "Unsupported ci_method. Only 'recentered_subsampling' is available."
            )

        point_result = self._default_bounder.solve(
            ci=False,
            maxtime=maxtime,
            theta=theta,
            verbose_optimizer=verbose_optimizer,
            verbose_result=verbose_result,
            limits=limits,
        )

        read_steps = [step for step in self._operation_log if step["method"] == "read_data"]
        if len(read_steps) == 0:
            # Backward-compatible path for legacy load_data-driven workflows.
            return self._solve_with_subsampling_ci(*args, **kwargs)
        if len(read_steps) != 1:
            raise ValueError("Current multinomial-binned CI supports exactly one read_data(...) dataset.")
        raw_df = read_steps[0]["kwargs"].get("raw")
        if raw_df is None or not isinstance(raw_df, pd.DataFrame):
            raise ValueError("read_data(raw=...) is required for multinomial-binned CI.")
        if raw_df.shape[0] == 0:
            raise ValueError("Cannot compute CI with empty raw data.")

        # Build multinomial scores over observed outcomes to define K bins.
        observed_cols = [c for c in raw_df.columns if c != "prob"]
        y = raw_df[observed_cols].astype(str).agg("_".join, axis=1)
        y_codes, _ = pd.factorize(y)
        if len(np.unique(y_codes)) < 2:
            obs_scores = np.ones(raw_df.shape[0], dtype=float)
        else:
            exog = sm.add_constant(raw_df[observed_cols].astype(float), has_constant="add")
            try:
                model = sm.MNLogit(y_codes, exog)
                fit = model.fit(disp=False)
                pred = np.asarray(fit.predict(exog))
                obs_scores = pred[np.arange(pred.shape[0]), y_codes]
                if not np.isfinite(obs_scores).all():
                    raise ValueError("Non-finite multinomial predictions.")
            except Exception:
                freqs = pd.Series(y_codes).value_counts(normalize=True)
                obs_scores = pd.Series(y_codes).map(freqs).to_numpy()

        k_target = max(1, min(int(self.K), raw_df.shape[0]))
        try:
            bins = pd.qcut(
                pd.Series(obs_scores).rank(method="first"),
                q=k_target,
                labels=False,
                duplicates="drop",
            )
        except Exception:
            bins = pd.Series(np.zeros(raw_df.shape[0], dtype=int))
        if pd.Series(bins).dropna().nunique() == 0:
            bins = pd.Series(np.zeros(raw_df.shape[0], dtype=int))

        n = int(raw_df.shape[0])
        b = max(1, min(n, int(np.floor(n ** 0.7))))
        theta_n_lb = float(point_result["point lb dual"])
        theta_n_ub = float(point_result["point ub dual"])

        lb_sub, ub_sub = [], []
        for bin_id in sorted(pd.Series(bins).dropna().unique().tolist()):
            bin_df = raw_df.loc[pd.Series(bins) == bin_id]
            if bin_df.shape[0] == 0:
                continue
            replace = bin_df.shape[0] < b
            sub_df = bin_df.sample(n=(b if replace else min(b, bin_df.shape[0])), replace=replace).reset_index(drop=True)

            replay_bounder = Bounder(
                deepcopy(self._default_bounder.dag),
                deepcopy(self._default_bounder.number_values),
            )
            for step in self._operation_log:
                method = step["method"]
                call_args = deepcopy(step["args"])
                call_kwargs = deepcopy(step["kwargs"])
                if method == "read_data":
                    call_kwargs["raw"] = sub_df
                getattr(replay_bounder, method)(*call_args, **call_kwargs)
            out = replay_bounder.solve(
                ci=False,
                maxtime=maxtime,
                theta=theta,
                verbose_optimizer=False,
                verbose_result=False,
                limits=limits,
            )
            lb_sub.append(float(out["point lb dual"]))
            ub_sub.append(float(out["point ub dual"]))

        if len(lb_sub) == 0 or len(ub_sub) == 0:
            raise RuntimeError("No bin-level subsample solves were produced.")
        lb_arr = np.asarray(lb_sub, dtype=float)
        ub_arr = np.asarray(ub_sub, dtype=float)

        t_lb = np.sqrt(b) * (lb_arr - theta_n_lb)
        t_ub = np.sqrt(b) * (ub_arr - theta_n_ub)
        sqrt_n = np.sqrt(n)
        ci_out = {
            "2.5% lb bounds": float(theta_n_lb - np.quantile(t_lb, 0.975) / sqrt_n),
            "97.5% ub bounds": float(theta_n_ub - np.quantile(t_ub, 0.025) / sqrt_n),
            "1% lb bounds": float(theta_n_lb - np.quantile(t_lb, 0.99) / sqrt_n),
            "99% ub bounds": float(theta_n_ub - np.quantile(t_ub, 0.01) / sqrt_n),
        }
        ci_out["ci method"] = ci_method
        ci_out["K bins used"] = int(pd.Series(bins).nunique())
        ci_out["subsample size b"] = int(b)
        return {**point_result, **ci_out}

    def _subsample_rows(self, df, subsample_rate, subsample_size, random_state=None):
        n = int(df.shape[0])
        if n == 0:
            raise ValueError("Cannot subsample empty dataset.")
        if subsample_size is None:
            m = min(n, max(1, int(np.floor(n ** subsample_rate))))
        else:
            m = min(n, int(subsample_size))
        return df.sample(n=m, replace=False, random_state=random_state)

    def _resolve_subsample_size(self, n, subsample_rate, subsample_size):
        n = int(n)
        if n == 0:
            raise ValueError("Cannot subsample empty dataset.")
        if subsample_size is None:
            return min(n, max(1, int(np.floor(n ** subsample_rate))))
        return min(n, int(subsample_size))

    def _allocate_strata_counts(self, counts, m):
        counts = np.asarray(counts, dtype=int)
        n = int(counts.sum())
        if n == 0:
            raise ValueError("Cannot allocate subsamples over empty strata.")
        if m >= n:
            return counts.copy()

        weights = counts / n
        raw = m * weights
        alloc = np.floor(raw).astype(int)

        positive = counts > 0
        for idx in np.where((alloc == 0) & positive)[0]:
            alloc[idx] = 1

        alloc = np.minimum(alloc, counts)
        total = int(alloc.sum())

        if total > m:
            excess = total - m
            removable = np.argsort(-(alloc - 1))
            for idx in removable:
                if excess == 0:
                    break
                drop = min(excess, max(0, alloc[idx] - 1))
                alloc[idx] -= drop
                excess -= drop
        elif total < m:
            deficit = m - total
            room = counts - alloc
            add_order = np.argsort(-(raw - np.floor(raw)))
            while deficit > 0 and np.any(room > 0):
                progressed = False
                for idx in add_order:
                    if room[idx] <= 0 or deficit == 0:
                        continue
                    alloc[idx] += 1
                    room[idx] -= 1
                    deficit -= 1
                    progressed = True
                    if deficit == 0:
                        break
                if not progressed:
                    break
        return alloc

    def _subsample_rows_stratified(
        self,
        df,
        covariates,
        subsample_rate,
        subsample_size,
        random_state=None,
    ):
        if covariates is None or len(covariates) == 0:
            return self._subsample_rows(df, subsample_rate, subsample_size, random_state=random_state)

        n = int(df.shape[0])
        m = self._resolve_subsample_size(n, subsample_rate, subsample_size)
        if m >= n:
            return df.sample(n=n, replace=False, random_state=random_state)

        grouped = list(df.groupby(covariates, sort=False, dropna=False))
        counts = np.array([g.shape[0] for _, g in grouped], dtype=int)
        alloc = self._allocate_strata_counts(counts, m)

        rng = np.random.default_rng(random_state)
        chunks = []
        for (_, g), take in zip(grouped, alloc):
            if take <= 0:
                continue
            seed = int(rng.integers(0, 2**32 - 1))
            chunks.append(g.sample(n=int(take), replace=False, random_state=seed))
        if len(chunks) == 0:
            raise RuntimeError("Stratified subsampling produced no rows.")
        return pd.concat(chunks, axis=0).sample(frac=1, random_state=int(rng.integers(0, 2**32 - 1)))

    def _run_subsampling_replication(
        self,
        replay_context,
        rep_seed,
        maxtime,
        theta,
        verbose_optimizer,
        limits,
        subsample_rate,
        subsample_size,
        return_subsample_df=False,
    ):
        from .Bounder import Bounder

        res, rep_n_total, rep_m_total, rep_subsample_df = self._solve_from_operation_log(
            replay_context=replay_context,
            maxtime=maxtime,
            theta=theta,
            verbose_optimizer=verbose_optimizer,
            limits=limits,
            subsample=True,
            subsample_rate=subsample_rate,
            subsample_size=subsample_size,
            rep_seed=int(rep_seed),
            return_subsample_df=return_subsample_df,
        )
        out = (
            float(res["point lb dual"]),
            float(res["point ub dual"]),
            rep_n_total,
            rep_m_total,
        )
        if return_subsample_df:
            return out + (rep_subsample_df,)
        return out

    def _solve_covariate_read_data_by_strata(
        self,
        base_bounder,
        raw_df,
        covariates,
        cond,
        maxtime,
        theta,
        verbose_optimizer,
        limits,
        return_dgps=False,
    ):
        if raw_df is None or raw_df.shape[0] == 0:
            raise ValueError("Covariate read_data requires non-empty raw data.")
        if covariates is None or len(covariates) == 0:
            raise ValueError("covariates must be non-empty in covariate solve path.")

        point_lb_dual = 0.0
        point_ub_dual = 0.0
        point_lb_primal = 0.0
        point_ub_primal = 0.0
        dgps = None
        if return_dgps:
            dgps = {
                "lower": {"status": "aggregated", "strata": []},
                "upper": {"status": "aggregated", "strata": []},
            }

        total_n = float(raw_df.shape[0])
        for _, gdf in raw_df.groupby(covariates, sort=False, dropna=False):
            w = float(gdf.shape[0]) / total_n
            strata_bounder = deepcopy(base_bounder)
            strata_bounder.load_data(raw=gdf.drop(columns=covariates).reset_index(drop=True), cond=cond)
            out = strata_bounder.solve(
                ci=False,
                maxtime=maxtime,
                theta=theta,
                verbose_optimizer=verbose_optimizer,
                verbose_result=False,
                limits=limits,
                return_dgps=return_dgps,
            )
            point_lb_dual += float(out["point lb dual"]) * w
            point_ub_dual += float(out["point ub dual"]) * w
            point_lb_primal += float(out["point lb primal"]) * w
            point_ub_primal += float(out["point ub primal"]) * w
            if return_dgps:
                covariate_values = {
                    col: gdf.iloc[0][col]
                    for col in covariates
                }
                dgps["lower"]["strata"].append(
                    {
                        "covariates": covariate_values,
                        "weight": w,
                        "dgps": out["dgps"]["lower"],
                    }
                )
                dgps["upper"]["strata"].append(
                    {
                        "covariates": covariate_values,
                        "weight": w,
                        "dgps": out["dgps"]["upper"],
                    }
                )

        result = {
            "point lb dual": point_lb_dual,
            "point ub dual": point_ub_dual,
            "point lb primal": point_lb_primal,
            "point ub primal": point_ub_primal,
        }
        if return_dgps:
            result["dgps"] = dgps
        return result

    def _prepare_replay_context(self):
        from .Bounder import Bounder

        data_seen = False
        fast_path_ok = True
        for step in self._operation_log:
            method = step["method"]
            if method in {"load_data", "read_data"}:
                data_seen = True
                continue
            if data_seen:
                fast_path_ok = False
                break

        if not fast_path_ok:
            return {"fast_path": False}

        prototype = Bounder(
            deepcopy(self._default_bounder.dag),
            deepcopy(self._default_bounder.number_values),
        )
        data_steps = []
        for step in self._operation_log:
            method = step["method"]
            call_args = deepcopy(step["args"])
            call_kwargs = deepcopy(step["kwargs"])
            if method in {"load_data", "read_data"}:
                data_steps.append(
                    {
                        "method": method,
                        "args": call_args,
                        "kwargs": call_kwargs,
                    }
                )
                continue
            getattr(prototype, method)(*call_args, **call_kwargs)

        return {
            "fast_path": True,
            "prototype": prototype,
            "data_steps": data_steps,
        }

    def _solve_from_operation_log_fast(
        self,
        replay_context,
        maxtime,
        theta,
        verbose_optimizer,
        limits,
        subsample,
        subsample_rate,
        subsample_size,
        rep_seed,
        return_subsample_df=False,
        return_dgps=False,
    ):
        replay_bounder = deepcopy(replay_context["prototype"])
        rep_n_total = 0
        rep_m_total = 0
        rep_subsample_df = None
        rng = np.random.default_rng(int(rep_seed))
        covariate_read_payload = None

        for step in replay_context["data_steps"]:
            method = step["method"]
            call_kwargs = deepcopy(step["kwargs"])

            if method == "load_data":
                if call_kwargs.get("raw") is None and call_kwargs.get("summary") is None:
                    raise ValueError("Subsampling CI requires load_data with raw or summary data.")
                if call_kwargs.get("raw") is not None and subsample:
                    raw_df = call_kwargs["raw"]
                    sub_df = self._subsample_rows(
                        raw_df,
                        subsample_rate,
                        subsample_size,
                        random_state=int(rng.integers(0, 2**32 - 1)),
                    ).reset_index(drop=True)
                    rep_n_total += int(raw_df.shape[0])
                    rep_m_total += int(sub_df.shape[0])
                    if return_subsample_df and rep_subsample_df is None:
                        rep_subsample_df = sub_df.copy(deep=True)
                    call_kwargs["raw"] = sub_df
                elif call_kwargs.get("summary") is not None:
                    call_kwargs["summary"] = call_kwargs["summary"].copy(deep=True)
                replay_bounder.load_data(**call_kwargs)
                continue

            if method == "read_data":
                if call_kwargs.get("raw") is None:
                    raise ValueError("Subsampling CI with read_data requires raw data.")
                raw_df = call_kwargs["raw"]
                covariates = call_kwargs.get("covariates", None)
                cond = call_kwargs.get("cond", [])

                work_df = raw_df
                if subsample:
                    if covariates is not None and len(covariates) > 0:
                        work_df = self._subsample_rows_stratified(
                            raw_df,
                            covariates=covariates,
                            subsample_rate=subsample_rate,
                            subsample_size=subsample_size,
                            random_state=int(rng.integers(0, 2**32 - 1)),
                        ).reset_index(drop=True)
                    else:
                        work_df = self._subsample_rows(
                            raw_df,
                            subsample_rate,
                            subsample_size,
                            random_state=int(rng.integers(0, 2**32 - 1)),
                        ).reset_index(drop=True)
                    rep_n_total += int(raw_df.shape[0])
                    rep_m_total += int(work_df.shape[0])
                    if return_subsample_df and rep_subsample_df is None:
                        rep_subsample_df = work_df.copy(deep=True)
                elif rep_n_total == 0:
                    rep_n_total += int(raw_df.shape[0])
                    rep_m_total += int(work_df.shape[0])

                if covariates is not None and len(covariates) > 0:
                    covariate_read_payload = {
                        "raw_df": work_df,
                        "covariates": covariates,
                        "cond": cond,
                    }
                else:
                    replay_bounder.load_data(raw=work_df, cond=cond)
                continue

            raise ValueError(f"Unexpected method in fast replay context: {method}")

        if covariate_read_payload is not None:
            res = self._solve_covariate_read_data_by_strata(
                base_bounder=replay_bounder,
                raw_df=covariate_read_payload["raw_df"],
                covariates=covariate_read_payload["covariates"],
                cond=covariate_read_payload["cond"],
                maxtime=maxtime,
                theta=theta,
                verbose_optimizer=verbose_optimizer,
                limits=limits,
                return_dgps=return_dgps,
            )
        else:
            res = replay_bounder.solve(
                ci=False,
                maxtime=maxtime,
                theta=theta,
                verbose_optimizer=verbose_optimizer,
                verbose_result=False,
                limits=limits,
                return_dgps=return_dgps,
            )
        return res, rep_n_total, rep_m_total, rep_subsample_df

    def _solve_from_operation_log_slow(
        self,
        maxtime,
        theta,
        verbose_optimizer,
        limits,
        subsample,
        subsample_rate,
        subsample_size,
        rep_seed,
        return_subsample_df=False,
        return_dgps=False,
    ):
        from .Bounder import Bounder

        replay_bounder = Bounder(
            deepcopy(self._default_bounder.dag),
            deepcopy(self._default_bounder.number_values),
        )
        rep_n_total = 0
        rep_m_total = 0
        rep_subsample_df = None
        rng = np.random.default_rng(int(rep_seed))
        covariate_read_payload = None

        for step in self._operation_log:
            method = step["method"]
            call_args = deepcopy(step["args"])
            call_kwargs = deepcopy(step["kwargs"])

            if method == "load_data":
                if call_kwargs.get("raw") is None and call_kwargs.get("summary") is None:
                    raise ValueError("Subsampling CI requires load_data with raw or summary data.")
                if call_kwargs.get("raw") is not None and subsample:
                    raw_df = call_kwargs["raw"]
                    sub_df = self._subsample_rows(
                        raw_df,
                        subsample_rate,
                        subsample_size,
                        random_state=int(rng.integers(0, 2**32 - 1)),
                    ).reset_index(drop=True)
                    rep_n_total += int(raw_df.shape[0])
                    rep_m_total += int(sub_df.shape[0])
                    if return_subsample_df and rep_subsample_df is None:
                        rep_subsample_df = sub_df.copy(deep=True)
                    call_kwargs["raw"] = sub_df
                elif call_kwargs.get("summary") is not None:
                    call_kwargs["summary"] = call_kwargs["summary"].copy(deep=True)
                getattr(replay_bounder, method)(*call_args, **call_kwargs)
                continue

            if method == "read_data":
                if call_kwargs.get("raw") is None:
                    raise ValueError("Subsampling CI with read_data requires raw data.")
                raw_df = call_kwargs["raw"]
                covariates = call_kwargs.get("covariates", None)
                cond = call_kwargs.get("cond", [])

                work_df = raw_df
                if subsample:
                    if covariates is not None and len(covariates) > 0:
                        work_df = self._subsample_rows_stratified(
                            raw_df,
                            covariates=covariates,
                            subsample_rate=subsample_rate,
                            subsample_size=subsample_size,
                            random_state=int(rng.integers(0, 2**32 - 1)),
                        ).reset_index(drop=True)
                    else:
                        work_df = self._subsample_rows(
                            raw_df,
                            subsample_rate,
                            subsample_size,
                            random_state=int(rng.integers(0, 2**32 - 1)),
                        ).reset_index(drop=True)
                    rep_n_total += int(raw_df.shape[0])
                    rep_m_total += int(work_df.shape[0])
                    if return_subsample_df and rep_subsample_df is None:
                        rep_subsample_df = work_df.copy(deep=True)
                elif rep_n_total == 0:
                    rep_n_total += int(raw_df.shape[0])
                    rep_m_total += int(work_df.shape[0])

                if covariates is not None and len(covariates) > 0:
                    covariate_read_payload = {
                        "raw_df": work_df,
                        "covariates": covariates,
                        "cond": cond,
                    }
                else:
                    replay_bounder.load_data(raw=work_df, cond=cond)
                continue

            getattr(replay_bounder, method)(*call_args, **call_kwargs)

        if covariate_read_payload is not None:
            res = self._solve_covariate_read_data_by_strata(
                base_bounder=replay_bounder,
                raw_df=covariate_read_payload["raw_df"],
                covariates=covariate_read_payload["covariates"],
                cond=covariate_read_payload["cond"],
                maxtime=maxtime,
                theta=theta,
                verbose_optimizer=verbose_optimizer,
                limits=limits,
                return_dgps=return_dgps,
            )
        else:
            res = replay_bounder.solve(
                ci=False,
                maxtime=maxtime,
                theta=theta,
                verbose_optimizer=verbose_optimizer,
                verbose_result=False,
                limits=limits,
                return_dgps=return_dgps,
            )
        return res, rep_n_total, rep_m_total, rep_subsample_df

    def _solve_from_operation_log(
        self,
        replay_context,
        maxtime,
        theta,
        verbose_optimizer,
        limits,
        subsample,
        subsample_rate,
        subsample_size,
        rep_seed,
        return_subsample_df=False,
        return_dgps=False,
    ):
        if replay_context is not None and replay_context.get("fast_path", False):
            return self._solve_from_operation_log_fast(
                replay_context=replay_context,
                maxtime=maxtime,
                theta=theta,
                verbose_optimizer=verbose_optimizer,
                limits=limits,
                subsample=subsample,
                subsample_rate=subsample_rate,
                subsample_size=subsample_size,
                rep_seed=rep_seed,
                return_subsample_df=return_subsample_df,
                return_dgps=return_dgps,
            )
        return self._solve_from_operation_log_slow(
            maxtime=maxtime,
            theta=theta,
            verbose_optimizer=verbose_optimizer,
            limits=limits,
            subsample=subsample,
            subsample_rate=subsample_rate,
            subsample_size=subsample_size,
            rep_seed=rep_seed,
            return_subsample_df=return_subsample_df,
            return_dgps=return_dgps,
        )

    def _solve_with_subsampling_ci(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError("Use keyword arguments when ci=True in causalProblem.solve().")

        nsamples = int(kwargs.pop("nsamples", 1000))
        progress = bool(kwargs.pop("progress", False))
        return_rep_seeds = bool(kwargs.pop("return_rep_seeds", False))
        return_subsample_dfs = bool(kwargs.pop("return_subsample_dfs", False))
        rep_seeds = kwargs.pop("rep_seeds", None)
        maxtime = kwargs.get("maxtime", None)
        theta = kwargs.get("theta", 0.01)
        verbose_optimizer = kwargs.get("verbose_optimizer", False)
        verbose_result = kwargs.get("verbose_result", True)
        limits = kwargs.get("limits", [None, None])
        subsample_rate = float(kwargs.pop("subsample_rate", 0.7))
        subsample_size = kwargs.pop("subsample_size", None)
        ci_workers = int(kwargs.pop("ci_workers", kwargs.pop("executions", 1)))
        if ci_workers < 1:
            raise ValueError("ci_workers must be >= 1")
        ci_method = kwargs.pop("ci_method", "recentered_subsampling")
        if ci_method != "recentered_subsampling":
            raise ValueError(
                "Unsupported ci_method. Only 'recentered_subsampling' is available."
            )
        replay_context = self._prepare_replay_context()

        point_result, _, _, _ = self._solve_from_operation_log(
            replay_context=replay_context,
            maxtime=maxtime,
            theta=theta,
            verbose_optimizer=verbose_optimizer,
            limits=limits,
            subsample=False,
            subsample_rate=subsample_rate,
            subsample_size=subsample_size,
            rep_seed=0,
        )
        if verbose_result:
            print("Point estimates\n")
            print(f"Dual: [{point_result['point lb dual']}, {point_result['point ub dual']}]")
            print(f"Primal: [{point_result['point lb primal']}, {point_result['point ub primal']}]")

        if len(self._operation_log) == 0:
            raise ValueError("No recorded operations available for subsampling CI.")

        lb_samples, ub_samples = [], []
        subsample_dfs = []
        n_eff_total = 0
        m_eff_total = 0
        progress_step = max(1, nsamples // 10) if nsamples > 0 else 1

        def _maybe_report_progress(done):
            if not progress:
                return
            if done == nsamples or done % progress_step == 0:
                pct = int(round(100 * done / nsamples)) if nsamples > 0 else 100
                print(f"CI subsampling {done}/{nsamples} ({pct}%)", flush=True)

        if rep_seeds is None:
            rep_seeds = np.random.default_rng().integers(0, 2**32 - 1, size=nsamples)
        else:
            rep_seeds = np.asarray(rep_seeds, dtype=np.int64)
            if rep_seeds.ndim != 1:
                raise ValueError("rep_seeds must be a one-dimensional array-like.")
            if rep_seeds.size != nsamples:
                raise ValueError("rep_seeds length must match nsamples.")

        if ci_workers == 1:
            for idx, rep_seed in enumerate(rep_seeds, start=1):
                out = self._run_subsampling_replication(
                    replay_context=replay_context,
                    rep_seed=rep_seed,
                    maxtime=maxtime,
                    theta=theta,
                    verbose_optimizer=False,
                    limits=limits,
                    subsample_rate=subsample_rate,
                    subsample_size=subsample_size,
                    return_subsample_df=return_subsample_dfs,
                )
                if return_subsample_dfs:
                    lb, ub, rep_n_total, rep_m_total, rep_subsample_df = out
                else:
                    lb, ub, rep_n_total, rep_m_total = out
                    rep_subsample_df = None
                if n_eff_total == 0 and rep_n_total > 0:
                    n_eff_total = rep_n_total
                    m_eff_total = rep_m_total
                lb_samples.append(lb)
                ub_samples.append(ub)
                if return_subsample_dfs:
                    subsample_dfs.append(rep_subsample_df)
                _maybe_report_progress(idx)
        else:
            with ThreadPoolExecutor(max_workers=ci_workers) as ex:
                futures = [
                    ex.submit(
                        self._run_subsampling_replication,
                        replay_context,
                        rep_seed=rep_seed,
                        maxtime=maxtime,
                        theta=theta,
                        verbose_optimizer=False,
                        limits=limits,
                        subsample_rate=subsample_rate,
                        subsample_size=subsample_size,
                        return_subsample_df=return_subsample_dfs,
                    )
                    for rep_seed in rep_seeds
                ]
                completed = 0
                for fut in as_completed(futures):
                    out = fut.result()
                    if return_subsample_dfs:
                        lb, ub, rep_n_total, rep_m_total, rep_subsample_df = out
                    else:
                        lb, ub, rep_n_total, rep_m_total = out
                        rep_subsample_df = None
                    if n_eff_total == 0 and rep_n_total > 0:
                        n_eff_total = rep_n_total
                        m_eff_total = rep_m_total
                    lb_samples.append(lb)
                    ub_samples.append(ub)
                    if return_subsample_dfs:
                        subsample_dfs.append(rep_subsample_df)
                    completed += 1
                    _maybe_report_progress(completed)

        lb_arr = np.asarray(lb_samples, dtype=float)
        ub_arr = np.asarray(ub_samples, dtype=float)
        valid = (~np.isnan(lb_arr)) & (~np.isnan(ub_arr))
        if valid.sum() == 0:
            raise RuntimeError("All subsampling CI replications returned NaN.")
        lb_arr = lb_arr[valid]
        ub_arr = ub_arr[valid]

        if n_eff_total <= 0 or m_eff_total <= 0:
            raise ValueError(
                "recentered_subsampling requires raw row-level data loaded via "
                "read_data(raw=...) or load_data(raw=...)."
            )
        theta_n_lb = float(point_result["point lb dual"])
        theta_n_ub = float(point_result["point ub dual"])
        t_lb = np.sqrt(m_eff_total) * (lb_arr - theta_n_lb)
        t_ub = np.sqrt(m_eff_total) * (ub_arr - theta_n_ub)
        sqrt_n = np.sqrt(n_eff_total)
        q_lb_025, q_lb_975 = np.quantile(t_lb, [0.025, 0.975])
        q_lb_01, q_lb_99 = np.quantile(t_lb, [0.01, 0.99])
        q_ub_025, q_ub_975 = np.quantile(t_ub, [0.025, 0.975])
        q_ub_01, q_ub_99 = np.quantile(t_ub, [0.01, 0.99])
        ci_out = {
            # Lower endpoint of CI for lower bound (95% and 98% two-sided analogs).
            "2.5% lb bounds": float(theta_n_lb - q_lb_975 / sqrt_n),
            "1% lb bounds": float(theta_n_lb - q_lb_99 / sqrt_n),
            # Upper endpoint of CI for upper bound.
            "97.5% ub bounds": float(theta_n_ub - q_ub_025 / sqrt_n),
            "99% ub bounds": float(theta_n_ub - q_ub_01 / sqrt_n),
        }
        ci_out["ci method"] = ci_method
        ci_out["ci workers"] = ci_workers
        if return_rep_seeds:
            ci_out["subsample rep seeds"] = rep_seeds.astype(int).tolist()
        if return_subsample_dfs:
            ci_out["subsample raw dfs"] = subsample_dfs
        return {**point_result, **ci_out}

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
        out = self._default_bounder.set_estimand(*args, **kwargs)
        object.__setattr__(self, "_continuous_estimand", None)
        self._record_operation("set_estimand", args, kwargs)
        return out

    def set_ate(self, *args, **kwargs):
        if len(args) >= 2:
            ind, dep = args[0], args[1]
        else:
            ind = kwargs.get("ind")
            dep = kwargs.get("dep")
        cond = kwargs.get("cond", None)
        if len(args) >= 3:
            cond = args[2]

        if self.continuous_outcome:
            if self._continuous_outcome_name is None:
                object.__setattr__(self, "_continuous_outcome_name", dep)
            if self._continuous_outcome_matches(dep):
                object.__setattr__(
                    self,
                    "_continuous_estimand",
                    {"kind": "ate", "ind": ind, "dep": dep, "cond": cond},
                )
                object.__setattr__(self._default_bounder, "estimand", None)
                self._record_operation("set_ate", args, kwargs)
                return None

        out = self._default_bounder.set_ate(*args, **kwargs)
        self._record_operation("set_ate", args, kwargs)
        return out

    def add_assumption(self, *args, **kwargs):
        out = self._default_bounder.add_assumption(*args, **kwargs)
        self._record_operation("add_assumption", args, kwargs)
        return out

    def add_constraint(self, *args, **kwargs):
        out = self._default_bounder.add_constraint(*args, **kwargs)
        self._record_operation("add_constraint", args, kwargs)
        return out

    def set_p_to_zero(self, *args, **kwargs):
        out = self._default_bounder.set_p_to_zero(*args, **kwargs)
        self._record_operation("set_p_to_zero", args, kwargs)
        return out

    def load_data(self, *args, **kwargs):
        sig = inspect.signature(self._default_bounder.load_data)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        call_kwargs = deepcopy(bound.arguments)
        if self.continuous_outcome:
            call_kwargs["raw"], call_kwargs["summary"] = self._prepare_continuous_loaded_data(
                raw=call_kwargs.get("raw"),
                summary=call_kwargs.get("summary"),
            )
        self._record_load_data_operation(tuple(), call_kwargs)
        return self._default_bounder.load_data(**call_kwargs)

    def load_data_do(self, *args, **kwargs):
        return self._default_bounder.load_data_do(*args, **kwargs)

    def load_data_kl(self, *args, **kwargs):
        return self._default_bounder.load_data_kl(*args, **kwargs)

    def load_data_gaussian(self, *args, **kwargs):
        return self._default_bounder.load_data_gaussian(*args, **kwargs)

    def read_data(self, *args, **kwargs):
        data_kwargs = self._normalize_read_data_args(args, kwargs)
        raw = data_kwargs["raw"]
        if raw is None:
            raise Exception("Data was not introduced!")
        datam = deepcopy(raw) if isinstance(raw, pd.DataFrame) else pd.read_csv(raw)
        if self.continuous_outcome:
            datam, _ = self._prepare_continuous_loaded_data(raw=datam, summary=None)
            data_kwargs["raw"] = datam
        self._record_read_data_operation(tuple(), data_kwargs)
        covariates = data_kwargs["covariates"]
        cond = list(data_kwargs["cond"])
        categorical = bool(data_kwargs["categorical"])
        model = data_kwargs["model"]
        nk = int(data_kwargs["nk"])

        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "inference", bool(data_kwargs["inference"]))
        object.__setattr__(self, "categorical", categorical)
        object.__setattr__(self, "main_model", None)
        object.__setattr__(self, "_used_discrete_covariate_path", False)
        object.__setattr__(self, "_covariate_support_size", None)
        object.__setattr__(self, "_has_covariates", covariates is not None and len(covariates) > 0)

        if covariates is not None and len(covariates) > 0:
            if len(cond) > 0:
                raise Exception(
                    "Conditional data is not supported in read_data when covariates are introduced."
                )
            if not categorical:
                use_empirical, support_size = self._use_empirical_covariate_path(datam, covariates, nk)
                object.__setattr__(self, "_covariate_support_size", support_size)
                if use_empirical:
                    categorical = True
                    object.__setattr__(self, "categorical", True)
                    object.__setattr__(self, "_used_discrete_covariate_path", True)
            if not categorical:
                x = datam[covariates].to_numpy().reshape((-1, len(covariates)))
                x = sm.add_constant(x)
                y = datam.drop(columns=covariates).astype(str).agg("_".join, axis=1)
                y, _ = pd.factorize(y)
                if model is None:
                    model = sm.MNLogit(y, x)
                    object.__setattr__(self, "main_model", model.fit())
                else:
                    object.__setattr__(self, "main_model", model)

        object.__setattr__(
            self,
            "_read_data_state",
            {
                "raw": datam,
                "covariates": covariates,
                "inference": bool(data_kwargs["inference"]),
                "cond": cond,
                "categorical": categorical,
                "model": model,
                "nsamples": int(data_kwargs["nsamples"]),
                "nk": nk,
            },
        )
        return None

    def solve_discrete_outcome_thresholds(
        self,
        ind,
        dep,
        thresholds=None,
        direction="geq",
        **solve_kwargs,
    ):
        support = self._get_threshold_source_values(dep)
        if len(support) < 2:
            raise ValueError(f"Outcome '{dep}' must have at least two support points.")
        if thresholds is None:
            thresholds = support[1:] if direction == "geq" else support[:-1]
        thresholds = list(thresholds)
        if len(thresholds) == 0:
            raise ValueError("No thresholds available for the requested direction.")

        results = {}
        for cutoff in thresholds:
            threshold_problem = self._build_threshold_problem(ind, dep, cutoff, direction)
            results[cutoff] = threshold_problem.solve(**deepcopy(solve_kwargs))

        return {
            "treatment": ind,
            "outcome": dep,
            "direction": direction,
            "support": support,
            "thresholds": thresholds,
            "results": results,
        }

    def write_program(self, *args, **kwargs):
        return self._default_bounder.write_program(*args, **kwargs)

    def solve(self, *args, **kwargs):
        # Simple path: one internal bounder, no orchestration requested.
        ci = kwargs.get("ci", False)
        if len(self._bounders) == 0:
            if self.continuous_outcome:
                point_result = self._solve_continuous_outcome_ate(**kwargs)
                if kwargs.get("verbose_result", True):
                    print("Point estimates\n")
                    print(f"Dual: [{point_result['point lb dual']}, {point_result['point ub dual']}]")
                    print(f"Primal: [{point_result['point lb primal']}, {point_result['point ub primal']}]")
                return point_result
            if ci:
                return self._solve_with_subsampling_ci(*args, **kwargs)
            if any(step["method"] == "read_data" for step in self._operation_log):
                replay_context = self._prepare_replay_context()
                point_result, _, _, _ = self._solve_from_operation_log(
                    replay_context=replay_context,
                    maxtime=kwargs.get("maxtime", None),
                    theta=kwargs.get("theta", 0.01),
                    verbose_optimizer=kwargs.get("verbose_optimizer", False),
                    limits=kwargs.get("limits", [None, None]),
                    subsample=False,
                    subsample_rate=kwargs.get("subsample_rate", 0.7),
                    subsample_size=kwargs.get("subsample_size", None),
                    rep_seed=0,
                    return_dgps=kwargs.get("return_dgps", False),
                )
                if kwargs.get("verbose_result", True):
                    print("Point estimates\n")
                    print(f"Dual: [{point_result['point lb dual']}, {point_result['point ub dual']}]")
                    print(f"Primal: [{point_result['point lb primal']}, {point_result['point ub primal']}]")
                return point_result
            return self._default_bounder.solve(*args, **kwargs)

        # Orchestration path: solve all registered bounders.
        # For now, return per-bounder outputs; aggregation policies
        # (weights/composition rules) can be layered on top.
        return self.solve_bounders(*args, **kwargs)

    def is_active(self, *args, **kwargs):
        return self._default_bounder.is_active(*args, **kwargs)

    def check_constraints(self, *args, **kwargs):
        return self._default_bounder.check_constraints(*args, **kwargs)

    def add_prob_constraints(self, *args, **kwargs):
        out = self._default_bounder.add_prob_constraints(*args, **kwargs)
        self._record_operation("add_prob_constraints", args, kwargs)
        return out
