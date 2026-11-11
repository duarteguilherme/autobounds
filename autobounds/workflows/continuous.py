"""ContinuousOutcomeWorkflow: algorithms operating on an explicit owning problem."""
from copy import deepcopy
import numpy as np
import pandas as pd
from ..Bounder import Bounder


class ContinuousOutcomeWorkflow:
    def __init__(self, problem):
        self.problem = problem

    def _continuous_outcome_matches(self, dep=None):
        context = self.problem
        if not context.continuous_outcome:
            return False
        if context._continuous_outcome_name is None:
            return dep is None
        return dep == context._continuous_outcome_name

    def _resolve_continuous_outcome_name(self):
        context = self.problem
        if not context.continuous_outcome:
            return None
        if context._continuous_outcome_name is None:
            raise ValueError(
                "continuous_outcome=True requires calling set_ate(..., dep=...) before loading data, "
                "or passing continuous_outcome='<outcome_name>' in the constructor."
            )
        return context._continuous_outcome_name

    def _digitize_continuous_outcome(self, datam, outcome):
        context = self.problem
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
        n_bins_target = max(1, min(int(context.continuous_bins), n_unique))
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

        object.__setattr__(context, "_continuous_bin_specs", bin_specs)
        return out

    def _prepare_continuous_loaded_data(self, raw=None, summary=None):
        context = self.problem
        if raw is not None and summary is not None:
            raise ValueError("Provide only one of raw or summary data.")
        if summary is not None:
            raise NotImplementedError(
                "Continuous-outcome mode currently supports raw row-level data only."
            )
        if raw is None:
            return raw, summary
        datam = deepcopy(raw) if isinstance(raw, pd.DataFrame) else pd.read_csv(raw)
        transformed = context._digitize_continuous_outcome(datam, context._resolve_continuous_outcome_name())
        context._ensure_default_bounder_number_values(
            outcome=context._resolve_continuous_outcome_name(),
            n_bins=len(context._continuous_bin_specs),
        )
        return transformed, None

    def _ensure_default_bounder_number_values(self, outcome, n_bins):
        context = self.problem
        from ..Bounder import Bounder

        if context._default_bounder.number_values.get(outcome, 2) == int(n_bins):
            return

        number_values = deepcopy(context._default_bounder.number_values)
        number_values[outcome] = int(n_bins)
        new_bounder = Bounder(
            deepcopy(context._default_bounder.dag),
            number_values,
        )
        for step in context._operation_log:
            method = step["method"]
            if method in {"load_data", "read_data", "set_ate", "set_estimand"}:
                continue
            getattr(new_bounder, method)(*deepcopy(step["args"]), **deepcopy(step["kwargs"]))
        object.__setattr__(context, "_default_bounder", new_bounder)

    def _threshold_bin_raw_data(self, datam, outcome, bin_id):
        context = self.problem
        if outcome not in datam.columns:
            raise KeyError(f"Outcome '{outcome}' not found in data columns.")
        out = datam.copy(deep=True)
        out[outcome] = (out[outcome] == bin_id).astype(int)
        return out

    def _threshold_bin_summary_data(self, datam, outcome, bin_id):
        context = self.problem
        if "prob" not in datam.columns:
            raise ValueError("Summary data must include a 'prob' column.")
        out = context._threshold_bin_raw_data(datam, outcome, bin_id)
        group_cols = [c for c in out.columns if c != "prob"]
        return out.groupby(group_cols, dropna=False, sort=False, as_index=False)["prob"].sum()

    def _build_category_problem(self, ind, dep, bin_id):
        context = self.problem
        number_values = deepcopy(context._default_bounder.number_values)
        number_values[dep] = 2
        category_problem = context._new_child_problem(
            deepcopy(context._default_bounder.dag),
            number_values,
        )

        for step in context._operation_log:
            method = step["method"]
            if method in {"set_estimand", "set_ate", "load_data", "read_data"}:
                continue
            getattr(category_problem, method)(*deepcopy(step["args"]), **deepcopy(step["kwargs"]))

        category_problem.set_ate(ind, dep)

        for step in context._operation_log:
            method = step["method"]
            call_kwargs = deepcopy(step["kwargs"])
            if method == "load_data":
                if call_kwargs.get("raw") is not None:
                    call_kwargs["raw"] = context._threshold_bin_raw_data(call_kwargs["raw"], dep, bin_id)
                elif call_kwargs.get("summary") is not None:
                    call_kwargs["summary"] = context._threshold_bin_summary_data(
                        call_kwargs["summary"], dep, bin_id
                    )
                category_problem.load_data(**call_kwargs)
            elif method == "read_data":
                raw = call_kwargs.get("raw")
                if raw is None:
                    raise ValueError("Continuous-outcome binning requires read_data(raw=...).")
                call_kwargs["raw"] = context._threshold_bin_raw_data(raw, dep, bin_id)
                category_problem.read_data(**call_kwargs)

        return category_problem

    def _continuous_threshold_weights(self):
        context = self.problem
        if context._continuous_bin_specs is None or len(context._continuous_bin_specs) < 2:
            raise ValueError("Continuous-outcome threshold aggregation requires at least two bins.")
        if context.continuous_method in {"ymax_ymin", "conservative", "conservative_threshold"}:
            lower_representatives = [float(spec["ymin"]) for spec in context._continuous_bin_specs]
            upper_representatives = [float(spec["ymax"]) for spec in context._continuous_bin_specs]
        elif context.continuous_method == "midpoint":
            midpoints = [
                0.5 * (float(spec["ymin"]) + float(spec["ymax"]))
                for spec in context._continuous_bin_specs
            ]
            lower_representatives = midpoints
            upper_representatives = midpoints
        else:
            raise ValueError(
                "continuous_method must be one of {'ymax_ymin', 'midpoint', 'conservative', 'conservative_threshold'}."
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
        context = self.problem
        number_values = deepcopy(context._default_bounder.number_values)
        weighted_problem = context._new_child_problem(
            deepcopy(context._default_bounder.dag),
            number_values,
        )

        for step in context._operation_log:
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

        for step in context._operation_log:
            method = step["method"]
            call_kwargs = deepcopy(step["kwargs"])
            if method == "load_data":
                weighted_problem.load_data(**call_kwargs)
            elif method == "read_data":
                weighted_problem.read_data(**call_kwargs)

        return weighted_problem

    def _build_continuous_component_problem(self, ind, dep, treatment_value, representatives, cond=None):
        context = self.problem
        number_values = deepcopy(context._default_bounder.number_values)
        component_problem = context._new_child_problem(
            deepcopy(context._default_bounder.dag),
            number_values,
        )

        for step in context._operation_log:
            method = step["method"]
            if method in {"set_estimand", "set_ate", "load_data", "read_data"}:
                continue
            getattr(component_problem, method)(*deepcopy(step["args"]), **deepcopy(step["kwargs"]))

        query = None
        for level, weight in enumerate(representatives):
            term = component_problem.p(f"{dep}({ind}={treatment_value})={level}", cond=cond) * float(weight)
            query = term if query is None else query + term
        component_problem.set_estimand(query)

        for step in context._operation_log:
            method = step["method"]
            call_kwargs = deepcopy(step["kwargs"])
            if method == "load_data":
                component_problem.load_data(**call_kwargs)
            elif method == "read_data":
                component_problem.read_data(**call_kwargs)

        return component_problem

    def _solve_continuous_outcome_ate(self, **solve_kwargs):
        context = self.problem
        if context._continuous_estimand is None or context._continuous_estimand.get("kind") != "ate":
            raise ValueError("Continuous-outcome solve currently supports set_ate(...) only.")
        if context.continuous_method == "conservative":
            return context._solve_continuous_outcome_ate_conservative(**solve_kwargs)
        if context.continuous_method == "conservative_threshold":
            return context._solve_continuous_outcome_ate_conservative_threshold(**solve_kwargs)
        want_ci = bool(solve_kwargs.get("ci", False))
        want_dgps = bool(solve_kwargs.get("return_dgps", False))
        if context._continuous_bin_specs is None:
            raise ValueError("Load data before solving a continuous outcome problem.")

        ind = context._continuous_estimand["ind"]
        dep = context._continuous_estimand["dep"]
        cond = context._continuous_estimand.get("cond")
        subsolve_kwargs = deepcopy(solve_kwargs)
        subsolve_kwargs["verbose_result"] = False
        lower_representatives, upper_representatives, _ = context._continuous_threshold_weights()

        if context.continuous_method == "midpoint":
            weighted_problem = context._build_continuous_weighted_problem(
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
                "continuous_method": context.continuous_method,
                "continuous_bins": int(context.continuous_bins),
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

        if context.continuous_method not in {"ymax_ymin", "conservative"}:
            raise ValueError(
                "continuous_method must be one of {'ymax_ymin', 'midpoint', 'conservative', 'conservative_threshold'}."
            )

        lower_problem = context._build_continuous_weighted_problem(
            ind,
            dep,
            lower_representatives,
            cond=cond,
        )
        upper_problem = context._build_continuous_weighted_problem(
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
            "continuous_method": context.continuous_method,
            "continuous_bins": int(context.continuous_bins),
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

    def _solve_continuous_outcome_ate_conservative(self, **solve_kwargs):
        context = self.problem
        want_ci = bool(solve_kwargs.get("ci", False))
        want_dgps = bool(solve_kwargs.get("return_dgps", False))
        if context._continuous_bin_specs is None:
            raise ValueError("Load data before solving a continuous outcome problem.")

        ind = context._continuous_estimand["ind"]
        dep = context._continuous_estimand["dep"]
        cond = context._continuous_estimand.get("cond")
        subsolve_kwargs = deepcopy(solve_kwargs)
        subsolve_kwargs["verbose_result"] = False
        lower_representatives, upper_representatives, _ = context._continuous_threshold_weights()

        component_problems = {
            "ey1_lower": context._build_continuous_component_problem(ind, dep, 1, lower_representatives, cond=cond),
            "ey1_upper": context._build_continuous_component_problem(ind, dep, 1, upper_representatives, cond=cond),
            "ey0_lower": context._build_continuous_component_problem(ind, dep, 0, lower_representatives, cond=cond),
            "ey0_upper": context._build_continuous_component_problem(ind, dep, 0, upper_representatives, cond=cond),
        }
        component_results = {
            name: problem.solve(**deepcopy(subsolve_kwargs))
            for name, problem in component_problems.items()
        }

        out = {
            "point lb dual": float(component_results["ey1_lower"]["point lb dual"] - component_results["ey0_upper"]["point ub dual"]),
            "point ub dual": float(component_results["ey1_upper"]["point ub dual"] - component_results["ey0_lower"]["point lb dual"]),
            "point lb primal": float(component_results["ey1_lower"]["point lb primal"] - component_results["ey0_upper"]["point ub primal"]),
            "point ub primal": float(component_results["ey1_upper"]["point ub primal"] - component_results["ey0_lower"]["point lb primal"]),
            "continuous_outcome": True,
            "continuous_method": context.continuous_method,
            "continuous_bins": int(context.continuous_bins),
            "outcome": dep,
            "treatment": ind,
            "continuous_lower_representatives": lower_representatives,
            "continuous_upper_representatives": upper_representatives,
            "component_results": component_results,
        }
        if want_ci:
            out["2.5% lb bounds"] = float(component_results["ey1_lower"]["2.5% lb bounds"] - component_results["ey0_upper"]["97.5% ub bounds"])
            out["1% lb bounds"] = float(component_results["ey1_lower"]["1% lb bounds"] - component_results["ey0_upper"]["99% ub bounds"])
            out["97.5% ub bounds"] = float(component_results["ey1_upper"]["97.5% ub bounds"] - component_results["ey0_lower"]["2.5% lb bounds"])
            out["99% ub bounds"] = float(component_results["ey1_upper"]["99% ub bounds"] - component_results["ey0_lower"]["1% lb bounds"])
            if "ci method" in component_results["ey1_lower"]:
                out["ci method"] = component_results["ey1_lower"]["ci method"]
            elif "ci method" in component_results["ey1_upper"]:
                out["ci method"] = component_results["ey1_upper"]["ci method"]
            elif "ci method" in component_results["ey0_lower"]:
                out["ci method"] = component_results["ey0_lower"]["ci method"]
            elif "ci method" in component_results["ey0_upper"]:
                out["ci method"] = component_results["ey0_upper"]["ci method"]
            if "ci workers" in component_results["ey1_lower"]:
                out["ci workers"] = component_results["ey1_lower"]["ci workers"]
            elif "ci workers" in component_results["ey1_upper"]:
                out["ci workers"] = component_results["ey1_upper"]["ci workers"]
            elif "ci workers" in component_results["ey0_lower"]:
                out["ci workers"] = component_results["ey0_lower"]["ci workers"]
            elif "ci workers" in component_results["ey0_upper"]:
                out["ci workers"] = component_results["ey0_upper"]["ci workers"]
        if want_dgps:
            out["dgps"] = {
                "lower": {
                    "status": "continuous_conservative_aggregate",
                    "components": {
                        "treated": component_results["ey1_lower"]["dgps"]["lower"],
                        "control": component_results["ey0_upper"]["dgps"]["upper"],
                    },
                },
                "upper": {
                    "status": "continuous_conservative_aggregate",
                    "components": {
                        "treated": component_results["ey1_upper"]["dgps"]["upper"],
                        "control": component_results["ey0_lower"]["dgps"]["lower"],
                    },
                },
            }
        return out

    def _build_threshold_component_problem(self, ind, dep, cutoff, treatment_value, cond=None):
        context = self.problem
        number_values = deepcopy(context._default_bounder.number_values)
        number_values[dep] = 2
        threshold_problem = context._new_child_problem(
            deepcopy(context._default_bounder.dag),
            number_values,
        )

        for step in context._operation_log:
            method = step["method"]
            if method in {"set_estimand", "set_ate", "load_data", "read_data"}:
                continue
            getattr(threshold_problem, method)(*deepcopy(step["args"]), **deepcopy(step["kwargs"]))

        threshold_problem.set_estimand(
            threshold_problem.p(f"{dep}({ind}={treatment_value})=1", cond=cond)
        )

        for step in context._operation_log:
            method = step["method"]
            call_kwargs = deepcopy(step["kwargs"])
            if method == "load_data":
                if call_kwargs.get("raw") is not None:
                    call_kwargs["raw"] = context._threshold_raw_data(call_kwargs["raw"], dep, cutoff, "geq")
                elif call_kwargs.get("summary") is not None:
                    call_kwargs["summary"] = context._threshold_summary_data(
                        call_kwargs["summary"], dep, cutoff, "geq"
                    )
                threshold_problem.load_data(**call_kwargs)
            elif method == "read_data":
                raw = call_kwargs.get("raw")
                if raw is None:
                    raise ValueError("Continuous-outcome thresholding requires read_data(raw=...).")
                call_kwargs["raw"] = context._threshold_raw_data(raw, dep, cutoff, "geq")
                threshold_problem.read_data(**call_kwargs)

        return threshold_problem

    def _solve_continuous_outcome_ate_conservative_threshold(self, **solve_kwargs):
        context = self.problem
        want_ci = bool(solve_kwargs.get("ci", False))
        want_dgps = bool(solve_kwargs.get("return_dgps", False))
        if context._continuous_bin_specs is None:
            raise ValueError("Load data before solving a continuous outcome problem.")

        ind = context._continuous_estimand["ind"]
        dep = context._continuous_estimand["dep"]
        cond = context._continuous_estimand.get("cond")
        subsolve_kwargs = deepcopy(solve_kwargs)
        subsolve_kwargs["verbose_result"] = False
        lower_representatives, upper_representatives, threshold_weights = context._continuous_threshold_weights()

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
                threshold_problem = context._build_threshold_component_problem(
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
            "continuous_method": context.continuous_method,
            "continuous_bins": int(context.continuous_bins),
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
                    "status": "continuous_conservative_threshold_aggregate",
                    "components": {"treated": y1["dgps"]["lower"], "control": y0["dgps"]["upper"]},
                },
                "upper": {
                    "status": "continuous_conservative_threshold_aggregate",
                    "components": {"treated": y1["dgps"]["upper"], "control": y0["dgps"]["lower"]},
                },
            }
        return out

    def _threshold_raw_data(self, datam, outcome, cutoff, direction):
        context = self.problem
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
        context = self.problem
        if "prob" not in datam.columns:
            raise ValueError("Summary data must include a 'prob' column.")
        out = context._threshold_raw_data(datam, outcome, cutoff, direction)
        group_cols = [c for c in out.columns if c != "prob"]
        return out.groupby(group_cols, dropna=False, sort=False, as_index=False)["prob"].sum()

    def _get_threshold_source_values(self, outcome):
        context = self.problem
        for step in context._operation_log:
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
        context = self.problem
        threshold_number_values = deepcopy(context._default_bounder.number_values)
        threshold_number_values[dep] = 2
        threshold_problem = context._new_child_problem(
            deepcopy(context._default_bounder.dag),
            threshold_number_values,
        )

        for step in context._operation_log:
            method = step["method"]
            if method in {"set_estimand", "set_ate", "load_data", "read_data"}:
                continue
            getattr(threshold_problem, method)(*deepcopy(step["args"]), **deepcopy(step["kwargs"]))

        threshold_problem.set_ate(ind, dep)

        for step in context._operation_log:
            method = step["method"]
            call_kwargs = deepcopy(step["kwargs"])
            if method == "load_data":
                if call_kwargs.get("raw") is not None:
                    call_kwargs["raw"] = context._threshold_raw_data(call_kwargs["raw"], dep, cutoff, direction)
                elif call_kwargs.get("summary") is not None:
                    call_kwargs["summary"] = context._threshold_summary_data(
                        call_kwargs["summary"], dep, cutoff, direction
                    )
                threshold_problem.load_data(**call_kwargs)
            elif method == "read_data":
                raw = call_kwargs.get("raw")
                if raw is None:
                    raise ValueError("Threshold outcome simplification requires read_data(raw=...).")
                call_kwargs["raw"] = context._threshold_raw_data(raw, dep, cutoff, direction)
                threshold_problem.read_data(**call_kwargs)

        return threshold_problem

    def solve_discrete_outcome_thresholds(
        self,
        ind,
        dep,
        thresholds=None,
        direction="geq",
        **solve_kwargs,
    ):
        context = self.problem
        support = context._get_threshold_source_values(dep)
        if len(support) < 2:
            raise ValueError(f"Outcome '{dep}' must have at least two support points.")
        if thresholds is None:
            thresholds = support[1:] if direction == "geq" else support[:-1]
        thresholds = list(thresholds)
        if len(thresholds) == 0:
            raise ValueError("No thresholds available for the requested direction.")

        results = {}
        for cutoff in thresholds:
            threshold_problem = context._build_threshold_problem(ind, dep, cutoff, direction)
            results[cutoff] = threshold_problem.solve(**deepcopy(solve_kwargs))

        return {
            "treatment": ind,
            "outcome": dep,
            "direction": direction,
            "support": support,
            "thresholds": thresholds,
            "results": results,
        }
