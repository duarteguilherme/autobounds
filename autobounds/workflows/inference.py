"""InferenceWorkflow: algorithms operating on an explicit owning problem."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from ..Bounder import Bounder


class InferenceWorkflow:
    def __init__(self, problem):
        self.problem = problem

    def _is_discrete_covariate(self, series):
        context = self.problem
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
        context = self.problem
        cov_df = datam[covariates]
        is_discrete = all(context._is_discrete_covariate(cov_df[col]) for col in covariates)
        support_size = int(cov_df.drop_duplicates().shape[0])
        use_empirical = is_discrete and support_size <= int(nk)
        return use_empirical, support_size

    def _solve_with_multinomial_binned_ci(self, *args, **kwargs):
        context = self.problem
        from ..Bounder import Bounder

        if len(args) > 0:
            raise ValueError("Use keyword arguments when ci=True in causalProblem.solve().")
        if context._has_covariates:
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

        point_result = context._default_bounder.solve(
            ci=False,
            maxtime=maxtime,
            theta=theta,
            verbose_optimizer=verbose_optimizer,
            verbose_result=verbose_result,
            limits=limits,
        )

        read_steps = [step for step in context._operation_log if step["method"] == "read_data"]
        if len(read_steps) == 0:
            # Backward-compatible path for legacy load_data-driven workflows.
            return context._solve_with_subsampling_ci(*args, **kwargs)
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
            import statsmodels.api as sm
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

        k_target = max(1, min(int(context.K), raw_df.shape[0]))
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
                deepcopy(context._default_bounder.dag),
                deepcopy(context._default_bounder.number_values),
            )
            for step in context._operation_log:
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
        context = self.problem
        n = int(df.shape[0])
        if n == 0:
            raise ValueError("Cannot subsample empty dataset.")
        if subsample_size is None:
            m = min(n, max(1, int(np.floor(n ** subsample_rate))))
        else:
            m = min(n, int(subsample_size))
        return df.sample(n=m, replace=False, random_state=random_state)

    def _resolve_subsample_size(self, n, subsample_rate, subsample_size):
        context = self.problem
        n = int(n)
        if n == 0:
            raise ValueError("Cannot subsample empty dataset.")
        if subsample_size is None:
            return min(n, max(1, int(np.floor(n ** subsample_rate))))
        return min(n, int(subsample_size))

    def _allocate_strata_counts(self, counts, m):
        context = self.problem
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
        context = self.problem
        if covariates is None or len(covariates) == 0:
            return context._subsample_rows(df, subsample_rate, subsample_size, random_state=random_state)

        n = int(df.shape[0])
        m = context._resolve_subsample_size(n, subsample_rate, subsample_size)
        if m >= n:
            return df.sample(n=n, replace=False, random_state=random_state)

        grouped = list(df.groupby(covariates, sort=False, dropna=False))
        counts = np.array([g.shape[0] for _, g in grouped], dtype=int)
        alloc = context._allocate_strata_counts(counts, m)

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
        context = self.problem
        from ..Bounder import Bounder

        res, rep_n_total, rep_m_total, rep_subsample_df = context._solve_from_operation_log(
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
        context = self.problem
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
            strata_bounder = base_bounder._copy_for_replay()
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

    def _solve_with_subsampling_ci(self, *args, **kwargs):
        context = self.problem
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
        replay_context = context._prepare_replay_context()

        point_result, _, _, _ = context._solve_from_operation_log(
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

        if len(context._operation_log) == 0:
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
                out = context._run_subsampling_replication(
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
                        context._run_subsampling_replication,
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
