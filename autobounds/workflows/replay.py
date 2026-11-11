"""ReplayWorkflow: algorithms operating on an explicit owning problem."""
import inspect
from copy import deepcopy
import numpy as np
import pandas as pd
from ..Bounder import Bounder


class ReplayWorkflow:
    def __init__(self, problem):
        self.problem = problem

    def _freeze_tabular(self, obj):
        context = self.problem
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
        context = self.problem
        if context._is_replaying:
            return
        context._operation_log.append(
            {
                "method": method,
                "args": deepcopy(args),
                "kwargs": deepcopy(kwargs),
            }
        )

    def _record_load_data_operation(self, args, kwargs):
        context = self.problem
        if context._is_replaying:
            return
        sig = inspect.signature(context._default_bounder.load_data)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        data_kwargs = deepcopy(bound.arguments)
        data_kwargs["summary"] = context._freeze_tabular(data_kwargs.get("summary"))
        data_kwargs["raw"] = context._freeze_tabular(data_kwargs.get("raw"))
        context._operation_log.append(
            {
                "method": "load_data",
                "args": tuple(),
                "kwargs": data_kwargs,
            }
        )

    def _normalize_read_data_args(self, args, kwargs):
        context = self.problem
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
        context = self.problem
        if context._is_replaying:
            return
        data_kwargs = context._normalize_read_data_args(args, kwargs)
        data_kwargs["raw"] = context._freeze_tabular(data_kwargs.get("raw"))
        covariates = data_kwargs.get("covariates", None)
        object.__setattr__(context, "_has_covariates", covariates is not None and len(covariates) > 0)
        context._operation_log.append(
            {
                "method": "read_data",
                "args": tuple(),
                "kwargs": data_kwargs,
            }
        )

    def _prepare_replay_context(self):
        """Cache the fixed prefix and replay later operations in their order.

        Operations after data loading may depend on sample-specific zero cells
        or root probabilities. Keep them after that data instead of moving them
        into the prototype merely to enable caching.
        """
        context = self.problem
        from ..Bounder import Bounder

        prototype = Bounder(
            deepcopy(context._default_bounder.dag),
            deepcopy(context._default_bounder.number_values),
        )
        prototype.Parser._query_cache = {}
        replay_steps = []
        data_seen = False
        for step in context._operation_log:
            method = step["method"]
            call_args = deepcopy(step["args"])
            call_kwargs = deepcopy(step["kwargs"])
            if method in {"load_data", "read_data"}:
                data_seen = True
            if data_seen:
                replay_steps.append(
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
            "replay_steps": replay_steps,
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
        context = self.problem
        replay_bounder = replay_context["prototype"]._copy_for_replay()
        rep_n_total = 0
        rep_m_total = 0
        rep_subsample_df = None
        rng = np.random.default_rng(int(rep_seed))
        covariate_read_payload = None

        for step in replay_context["replay_steps"]:
            method = step["method"]
            call_args = deepcopy(step["args"])
            call_kwargs = deepcopy(step["kwargs"])

            if method == "load_data":
                if call_kwargs.get("raw") is None and call_kwargs.get("summary") is None:
                    raise ValueError("Subsampling CI requires load_data with raw or summary data.")
                if call_kwargs.get("raw") is not None and subsample:
                    raw_df = call_kwargs["raw"]
                    sub_df = context._subsample_rows(
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
                        work_df = context._subsample_rows_stratified(
                            raw_df,
                            covariates=covariates,
                            subsample_rate=subsample_rate,
                            subsample_size=subsample_size,
                            random_state=int(rng.integers(0, 2**32 - 1)),
                        ).reset_index(drop=True)
                    else:
                        work_df = context._subsample_rows(
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
            res = context._solve_covariate_read_data_by_strata(
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
        context = self.problem
        from ..Bounder import Bounder

        replay_bounder = Bounder(
            deepcopy(context._default_bounder.dag),
            deepcopy(context._default_bounder.number_values),
        )
        rep_n_total = 0
        rep_m_total = 0
        rep_subsample_df = None
        rng = np.random.default_rng(int(rep_seed))
        covariate_read_payload = None

        for step in context._operation_log:
            method = step["method"]
            call_args = deepcopy(step["args"])
            call_kwargs = deepcopy(step["kwargs"])

            if method == "load_data":
                if call_kwargs.get("raw") is None and call_kwargs.get("summary") is None:
                    raise ValueError("Subsampling CI requires load_data with raw or summary data.")
                if call_kwargs.get("raw") is not None and subsample:
                    raw_df = call_kwargs["raw"]
                    sub_df = context._subsample_rows(
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
                        work_df = context._subsample_rows_stratified(
                            raw_df,
                            covariates=covariates,
                            subsample_rate=subsample_rate,
                            subsample_size=subsample_size,
                            random_state=int(rng.integers(0, 2**32 - 1)),
                        ).reset_index(drop=True)
                    else:
                        work_df = context._subsample_rows(
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
            res = context._solve_covariate_read_data_by_strata(
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
        context = self.problem
        if replay_context is not None and replay_context.get("fast_path", False):
            return context._solve_from_operation_log_fast(
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
        return context._solve_from_operation_log_slow(
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
