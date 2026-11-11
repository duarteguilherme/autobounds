from .workflows.continuous import ContinuousOutcomeWorkflow
from .workflows.inference import InferenceWorkflow
from .workflows.replay import ReplayWorkflow
from .data_utils import (
    generate_posterior_beta, generate_mn_sample, get_summary_from_raw, multiply_matrix_gaussian, solve_gaussian, solve_kl_p, get_dirichlet_sample, simplify_first_nodes, replace_first_nodes, transform_constraint, get_constraint_from_row, get_query_data_do
)
from .Bounder import Bounder
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
from numpy import log
import inspect
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed


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


# Simplifiers 
### 1) First nodes


class causalProblem:
    """
    Orchestrator for one or more Bounder objects.

    Backward compatibility:
    - Existing single-problem calls are proxied to the implicit default bounder.
    """

    _INTERNAL_ATTRS = {
        "_continuous_workflow", "_inference_workflow", "_replay_workflow",
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
        number_values = None,
        continuous_outcome = False,
        continuous_bins = 5,
        continuous_method = "midpoint",
    ):
        from .Bounder import Bounder
        if continuous_method == "conservative_binarized":
            continuous_method = "conservative_threshold"
        if continuous_method not in {"ymax_ymin", "midpoint", "conservative", "conservative_threshold"}:
            raise ValueError(
                "continuous_method must be one of {'ymax_ymin', 'midpoint', 'conservative', 'conservative_threshold'}."
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
        object.__setattr__(self, "_continuous_workflow", ContinuousOutcomeWorkflow(self))
        object.__setattr__(self, "_inference_workflow", InferenceWorkflow(self))
        object.__setattr__(self, "_replay_workflow", ReplayWorkflow(self))

    @staticmethod
    def _new_child_problem(dag, number_values):
        """Construct component problems without imposing subclass constructors."""
        return causalProblem(dag, number_values)

    def _freeze_tabular(self, obj):
        return self._replay_workflow._freeze_tabular(obj)

    def _record_operation(self, method, args, kwargs):
        return self._replay_workflow._record_operation(method, args, kwargs)

    def _continuous_outcome_matches(self, dep=None):
        return self._continuous_workflow._continuous_outcome_matches(dep)

    def _resolve_continuous_outcome_name(self):
        return self._continuous_workflow._resolve_continuous_outcome_name()

    def _digitize_continuous_outcome(self, datam, outcome):
        return self._continuous_workflow._digitize_continuous_outcome(datam, outcome)

    def _prepare_continuous_loaded_data(self, raw=None, summary=None):
        return self._continuous_workflow._prepare_continuous_loaded_data(raw, summary)

    def _ensure_default_bounder_number_values(self, outcome, n_bins):
        return self._continuous_workflow._ensure_default_bounder_number_values(outcome, n_bins)

    def _threshold_bin_raw_data(self, datam, outcome, bin_id):
        return self._continuous_workflow._threshold_bin_raw_data(datam, outcome, bin_id)

    def _threshold_bin_summary_data(self, datam, outcome, bin_id):
        return self._continuous_workflow._threshold_bin_summary_data(datam, outcome, bin_id)

    def _build_category_problem(self, ind, dep, bin_id):
        return self._continuous_workflow._build_category_problem(ind, dep, bin_id)

    def _continuous_threshold_weights(self):
        return self._continuous_workflow._continuous_threshold_weights()

    def _build_continuous_weighted_problem(self, ind, dep, representatives, cond=None):
        return self._continuous_workflow._build_continuous_weighted_problem(ind, dep, representatives, cond)

    def _build_continuous_component_problem(self, ind, dep, treatment_value, representatives, cond=None):
        return self._continuous_workflow._build_continuous_component_problem(ind, dep, treatment_value, representatives, cond)

    def _solve_continuous_outcome_ate(self, **solve_kwargs):
        return self._continuous_workflow._solve_continuous_outcome_ate(**solve_kwargs)

    def _solve_continuous_outcome_ate_conservative(self, **solve_kwargs):
        return self._continuous_workflow._solve_continuous_outcome_ate_conservative(**solve_kwargs)

    def _build_threshold_component_problem(self, ind, dep, cutoff, treatment_value, cond=None):
        return self._continuous_workflow._build_threshold_component_problem(ind, dep, cutoff, treatment_value, cond)

    def _solve_continuous_outcome_ate_conservative_threshold(self, **solve_kwargs):
        return self._continuous_workflow._solve_continuous_outcome_ate_conservative_threshold(**solve_kwargs)

    def _record_load_data_operation(self, args, kwargs):
        return self._replay_workflow._record_load_data_operation(args, kwargs)

    def _is_discrete_covariate(self, series):
        return self._inference_workflow._is_discrete_covariate(series)

    def _use_empirical_covariate_path(self, datam, covariates, nk):
        return self._inference_workflow._use_empirical_covariate_path(datam, covariates, nk)

    def _normalize_read_data_args(self, args, kwargs):
        return self._replay_workflow._normalize_read_data_args(args, kwargs)

    def _threshold_raw_data(self, datam, outcome, cutoff, direction):
        return self._continuous_workflow._threshold_raw_data(datam, outcome, cutoff, direction)

    def _threshold_summary_data(self, datam, outcome, cutoff, direction):
        return self._continuous_workflow._threshold_summary_data(datam, outcome, cutoff, direction)

    def _get_threshold_source_values(self, outcome):
        return self._continuous_workflow._get_threshold_source_values(outcome)

    def _build_threshold_problem(self, ind, dep, cutoff, direction):
        return self._continuous_workflow._build_threshold_problem(ind, dep, cutoff, direction)

    def _record_read_data_operation(self, args, kwargs):
        return self._replay_workflow._record_read_data_operation(args, kwargs)

    def _solve_with_multinomial_binned_ci(self, *args, **kwargs):
        return self._inference_workflow._solve_with_multinomial_binned_ci(*args, **kwargs)

    def _subsample_rows(self, df, subsample_rate, subsample_size, random_state=None):
        return self._inference_workflow._subsample_rows(df, subsample_rate, subsample_size, random_state)

    def _resolve_subsample_size(self, n, subsample_rate, subsample_size):
        return self._inference_workflow._resolve_subsample_size(n, subsample_rate, subsample_size)

    def _allocate_strata_counts(self, counts, m):
        return self._inference_workflow._allocate_strata_counts(counts, m)

    def _subsample_rows_stratified(
        self,
        df,
        covariates,
        subsample_rate,
        subsample_size,
        random_state=None,
    ):
        return self._inference_workflow._subsample_rows_stratified(df, covariates, subsample_rate, subsample_size, random_state)

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
        return self._inference_workflow._run_subsampling_replication(
            replay_context,
            rep_seed,
            maxtime,
            theta,
            verbose_optimizer,
            limits,
            subsample_rate,
            subsample_size,
            return_subsample_df,
        )

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
        return self._inference_workflow._solve_covariate_read_data_by_strata(
            base_bounder,
            raw_df,
            covariates,
            cond,
            maxtime,
            theta,
            verbose_optimizer,
            limits,
            return_dgps,
        )

    def _prepare_replay_context(self):
        """Cache the fixed prefix and replay later operations in their order.

        Operations after data loading may depend on sample-specific zero cells
        or root probabilities. Keep them after that data instead of moving them
        into the prototype merely to enable caching.
        """
        return self._replay_workflow._prepare_replay_context()

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
        return self._replay_workflow._solve_from_operation_log_fast(
            replay_context,
            maxtime,
            theta,
            verbose_optimizer,
            limits,
            subsample,
            subsample_rate,
            subsample_size,
            rep_seed,
            return_subsample_df,
            return_dgps,
        )

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
        return self._replay_workflow._solve_from_operation_log_slow(
            maxtime,
            theta,
            verbose_optimizer,
            limits,
            subsample,
            subsample_rate,
            subsample_size,
            rep_seed,
            return_subsample_df,
            return_dgps,
        )

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
        return self._replay_workflow._solve_from_operation_log(
            replay_context,
            maxtime,
            theta,
            verbose_optimizer,
            limits,
            subsample,
            subsample_rate,
            subsample_size,
            rep_seed,
            return_subsample_df,
            return_dgps,
        )

    def _solve_with_subsampling_ci(self, *args, **kwargs):
        return self._inference_workflow._solve_with_subsampling_ci(*args, **kwargs)

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
                import statsmodels.api as sm
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
        return self._continuous_workflow.solve_discrete_outcome_thresholds(ind, dep, thresholds, direction, **solve_kwargs)

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
