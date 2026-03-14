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
    }

    def __init__(self, dag, number_values = {}):
        from .Bounder import Bounder
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
    ):
        if raw_df is None or raw_df.shape[0] == 0:
            raise ValueError("Covariate read_data requires non-empty raw data.")
        if covariates is None or len(covariates) == 0:
            raise ValueError("covariates must be non-empty in covariate solve path.")

        point_lb_dual = 0.0
        point_ub_dual = 0.0
        point_lb_primal = 0.0
        point_ub_primal = 0.0

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
            )
            point_lb_dual += float(out["point lb dual"]) * w
            point_ub_dual += float(out["point ub dual"]) * w
            point_lb_primal += float(out["point lb primal"]) * w
            point_ub_primal += float(out["point ub primal"]) * w

        return {
            "point lb dual": point_lb_dual,
            "point ub dual": point_ub_dual,
            "point lb primal": point_lb_primal,
            "point ub primal": point_ub_primal,
        }

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
            )
        else:
            res = replay_bounder.solve(
                ci=False,
                maxtime=maxtime,
                theta=theta,
                verbose_optimizer=verbose_optimizer,
                verbose_result=False,
                limits=limits,
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
            )
        else:
            res = replay_bounder.solve(
                ci=False,
                maxtime=maxtime,
                theta=theta,
                verbose_optimizer=verbose_optimizer,
                verbose_result=False,
                limits=limits,
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
        )

    def _solve_with_subsampling_ci(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError("Use keyword arguments when ci=True in causalProblem.solve().")

        nsamples = int(kwargs.pop("nsamples", 200))
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
        if rep_seeds is None:
            rep_seeds = np.random.default_rng().integers(0, 2**32 - 1, size=nsamples)
        else:
            rep_seeds = np.asarray(rep_seeds, dtype=np.int64)
            if rep_seeds.ndim != 1:
                raise ValueError("rep_seeds must be a one-dimensional array-like.")
            if rep_seeds.size != nsamples:
                raise ValueError("rep_seeds length must match nsamples.")

        if ci_workers == 1:
            for rep_seed in rep_seeds:
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
        self._record_operation("set_estimand", args, kwargs)
        return out

    def set_ate(self, *args, **kwargs):
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
        self._record_load_data_operation(args, kwargs)
        return self._default_bounder.load_data(*args, **kwargs)

    def load_data_do(self, *args, **kwargs):
        return self._default_bounder.load_data_do(*args, **kwargs)

    def load_data_kl(self, *args, **kwargs):
        return self._default_bounder.load_data_kl(*args, **kwargs)

    def load_data_gaussian(self, *args, **kwargs):
        return self._default_bounder.load_data_gaussian(*args, **kwargs)

    def read_data(self, *args, **kwargs):
        self._record_read_data_operation(args, kwargs)
        data_kwargs = self._normalize_read_data_args(args, kwargs)
        raw = data_kwargs["raw"]
        if raw is None:
            raise Exception("Data was not introduced!")
        datam = deepcopy(raw) if isinstance(raw, pd.DataFrame) else pd.read_csv(raw)
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

    def write_program(self, *args, **kwargs):
        return self._default_bounder.write_program(*args, **kwargs)

    def solve(self, *args, **kwargs):
        # Simple path: one internal bounder, no orchestration requested.
        ci = kwargs.get("ci", False)
        if len(self._bounders) == 0:
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
