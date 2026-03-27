#!/usr/bin/env python3
"""Five-method CI comparison for the adversarial binary-IV model."""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from matplotlib.ticker import MultipleLocator

NORMAL_975 = 1.959963984540054

# ATE lower-bound candidate order uses:
# [p00|z=0, p01|z=0, p10|z=0, p11|z=0, p00|z=1, p01|z=1, p10|z=1, p11|z=1]
LOWER_COEFFS = np.asarray(
    [
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [2, 0, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 2, 1, 1, 0, 0],
        [0, 0, 1, 1, 2, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 0, 2],
    ],
    dtype=float,
)
LOWER_CONSTS = np.asarray([-1, -1, -1, -1, -2, -2, -2, -2], dtype=float)
UPPER_COEFFS = np.asarray(
    [
        [0, 0, -1, 0, 0, -1, 0, 0],
        [0, -1, 0, 0, 0, 0, -1, 0],
        [0, -1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, -1, 0],
        [0, 0, -1, 0, 0, -2, -1, -1],
        [0, -1, -2, 0, -1, -1, 0, 0],
        [0, 0, -1, -1, 0, -2, -1, 0],
        [-1, -1, 0, 0, 0, -1, -2, 0],
    ],
    dtype=float,
)
UPPER_CONSTS = np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=float)


@dataclass(frozen=True)
class IVModel:
    name: str
    p_u1: float
    p_x1: np.ndarray
    p_y1: np.ndarray


MODEL_PRESETS = {
    "oracle_friendly": IVModel(
        name="oracle_friendly",
        p_u1=0.6994502696332044,
        p_x1=np.asarray(
            [
                [0.9751179333292523, 0.8973667628716856],
                [0.0320193004325558, 0.039495567667656564],
            ],
            dtype=float,
        ),
        p_y1=np.asarray(
            [
                [0.7207997050415031, 0.3660858867109061],
                [0.8317399245911306, 0.4216450673797748],
            ],
            dtype=float,
        ),
    ),
    "nearly_tied": IVModel(
        name="nearly_tied",
        p_u1=0.1943818562657974,
        p_x1=np.asarray(
            [
                [0.39799216004378635, 0.42568204435146356],
                [0.1871095123322955, 0.774155338244361],
            ],
            dtype=float,
        ),
        p_y1=np.asarray(
            [
                [0.26738822338835644, 0.3084806836370481],
                [0.2931557864130436, 0.7352640456719242],
            ],
            dtype=float,
        ),
    ),
}


def parse_n_grid(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def resolve_subsample_size(n: int, gamma: float, minimum: int = 1) -> int:
    return min(n, max(int(minimum), int(np.floor(n**gamma))))


def draw_iv_subsample(data: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    n = data.shape[0]
    if m >= n:
        return data.copy()

    z = data[:, 0]
    idx0 = np.flatnonzero(z == 0)
    idx1 = np.flatnonzero(z == 1)
    if idx0.size == 0 or idx1.size == 0:
        raise ValueError("Both instrument levels must be present.")
    if m < 2:
        raise ValueError("Subsample size must be at least 2 for binary-IV bounds.")

    take0 = int(np.floor(m * (idx0.size / n)))
    take0 = max(1, min(take0, idx0.size))
    take1 = m - take0
    if take1 < 1:
        take1 = 1
        take0 = m - 1
    if take1 > idx1.size:
        take1 = idx1.size
        take0 = m - take1
    if take0 < 1:
        take0 = 1
        take1 = m - 1
    if take0 > idx0.size or take1 > idx1.size:
        raise ValueError("Insufficient rows to preserve both instrument levels in subsample.")

    chosen0 = rng.choice(idx0, size=take0, replace=False)
    chosen1 = rng.choice(idx1, size=take1, replace=False)
    chosen = np.concatenate([chosen0, chosen1])
    rng.shuffle(chosen)
    return data[chosen]


def population_conditional_probs(model: IVModel) -> np.ndarray:
    probs = []
    p_u0 = 1.0 - model.p_u1
    for z in [0, 1]:
        for d in [0, 1]:
            p_d0 = (1.0 - model.p_x1[z, 0]) if d == 0 else model.p_x1[z, 0]
            p_d1 = (1.0 - model.p_x1[z, 1]) if d == 0 else model.p_x1[z, 1]
            for y in [0, 1]:
                p_y0 = (1.0 - model.p_y1[d, 0]) if y == 0 else model.p_y1[d, 0]
                p_y1 = (1.0 - model.p_y1[d, 1]) if y == 0 else model.p_y1[d, 1]
                probs.append(p_u0 * p_d0 * p_y0 + model.p_u1 * p_d1 * p_y1)
    # Reorder from loop order [(z,d,y)] to [p00_0, p01_0, p10_0, p11_0, ...].
    return np.asarray(
        [
            probs[0],
            probs[1],
            probs[2],
            probs[3],
            probs[4],
            probs[5],
            probs[6],
            probs[7],
        ],
        dtype=float,
    )[[0, 2, 1, 3, 4, 6, 5, 7]]


def lower_candidates_from_probs(probs: np.ndarray) -> np.ndarray:
    return LOWER_COEFFS @ probs + LOWER_CONSTS


def upper_candidates_from_probs(probs: np.ndarray) -> np.ndarray:
    return UPPER_COEFFS @ probs + UPPER_CONSTS


def empirical_conditional_probs(data: np.ndarray) -> np.ndarray:
    z = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    out = np.empty(8, dtype=float)
    pos = 0
    for z_level in [0, 1]:
        mask_z = z == z_level
        n_z = int(mask_z.sum())
        if n_z == 0:
            raise ValueError("Both instrument levels must be present.")
        for x_level, y_level in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            out[pos] = np.mean(mask_z & (x == x_level) & (y == y_level)) / np.mean(mask_z)
            pos += 1
    return out


def lower_bound_from_data(data: np.ndarray) -> tuple[float, int, np.ndarray]:
    probs = empirical_conditional_probs(data)
    candidates = lower_candidates_from_probs(probs)
    idx = int(np.argmax(candidates))
    return float(candidates[idx]), idx, candidates


def upper_bound_from_data(data: np.ndarray) -> tuple[float, int, np.ndarray]:
    probs = empirical_conditional_probs(data)
    candidates = upper_candidates_from_probs(probs)
    idx = int(np.argmin(candidates))
    return float(candidates[idx]), idx, candidates


def simulate_iv_data(model: IVModel, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.binomial(1, model.p_u1, size=n)
    z = rng.binomial(1, 0.5, size=n)
    p_x = model.p_x1[z, u]
    x = rng.binomial(1, p_x)
    p_y = model.p_y1[x, u]
    y = rng.binomial(1, p_y)
    return np.column_stack([z, x, y]).astype(np.int8, copy=False)


def candidate_standard_error(data: np.ndarray, coeffs: np.ndarray) -> float:
    z = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    probs = empirical_conditional_probs(data)
    pi0 = np.mean(z == 0)
    pi1 = 1.0 - pi0
    if pi0 == 0.0 or pi1 == 0.0:
        raise ValueError("Both instrument levels must be present.")
    phi = np.zeros(data.shape[0], dtype=float)
    cell_order = [
        (0, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (1, 1, 1),
    ]
    for cell_idx, (z_level, x_level, y_level) in enumerate(cell_order):
        weight = coeffs[cell_idx]
        if weight == 0.0:
            continue
        mask_z = z == z_level
        pi_z = pi0 if z_level == 0 else pi1
        event = (x == x_level) & (y == y_level)
        phi += weight * mask_z * (event.astype(float) - probs[cell_idx]) / pi_z
    return float(np.sqrt(np.mean(phi**2) / data.shape[0]))


def bootstrap_lower_endpoint(
    data: np.ndarray,
    b: int,
    alpha: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    estimates = np.empty(b, dtype=float)
    for i in range(b):
        boot = data[rng.integers(0, n, size=n)]
        estimates[i] = lower_bound_from_data(boot)[0]
    return float(np.quantile(estimates, alpha / 2.0))


def bootstrap_upper_endpoint(
    data: np.ndarray,
    b: int,
    alpha: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    estimates = np.empty(b, dtype=float)
    for i in range(b):
        boot = data[rng.integers(0, n, size=n)]
        estimates[i] = upper_bound_from_data(boot)[0]
    return float(np.quantile(estimates, 1.0 - (alpha / 2.0)))


def bayesian_dirichlet_lower_endpoint(
    data: np.ndarray,
    b: int,
    alpha: float,
    seed: int,
    prior: float = 1.0,
) -> float:
    z = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    rng = np.random.default_rng(seed)

    def cell_counts(z_level: int) -> np.ndarray:
        mask = z == z_level
        if not np.any(mask):
            raise ValueError("Both instrument levels must be present.")
        counts = np.zeros(4, dtype=float)
        order = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for idx, (x_level, y_level) in enumerate(order):
            counts[idx] = np.sum(mask & (x == x_level) & (y == y_level))
        return counts

    alpha0 = cell_counts(0) + float(prior)
    alpha1 = cell_counts(1) + float(prior)
    draws = np.empty(b, dtype=float)
    for i in range(b):
        q0 = rng.dirichlet(alpha0)
        q1 = rng.dirichlet(alpha1)
        probs = np.concatenate([q0, q1])
        draws[i] = float(np.max(lower_candidates_from_probs(probs)))
    return float(np.quantile(draws, alpha / 2.0))


def bayesian_dirichlet_upper_endpoint(
    data: np.ndarray,
    b: int,
    alpha: float,
    seed: int,
    prior: float = 1.0,
) -> float:
    z = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    rng = np.random.default_rng(seed)

    def cell_counts(z_level: int) -> np.ndarray:
        mask = z == z_level
        if not np.any(mask):
            raise ValueError("Both instrument levels must be present.")
        counts = np.zeros(4, dtype=float)
        order = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for idx, (x_level, y_level) in enumerate(order):
            counts[idx] = np.sum(mask & (x == x_level) & (y == y_level))
        return counts

    alpha0 = cell_counts(0) + float(prior)
    alpha1 = cell_counts(1) + float(prior)
    draws = np.empty(b, dtype=float)
    for i in range(b):
        q0 = rng.dirichlet(alpha0)
        q1 = rng.dirichlet(alpha1)
        probs = np.concatenate([q0, q1])
        draws[i] = float(np.min(upper_candidates_from_probs(probs)))
    return float(np.quantile(draws, 1.0 - (alpha / 2.0)))


def kl_divergence(p_hat: np.ndarray, q: np.ndarray) -> float:
    mask = p_hat > 0.0
    if np.any(q <= 0.0):
        return float("inf")
    return float(np.sum(p_hat[mask] * np.log(p_hat[mask] / q[mask])))


def mtn_kl_radius(n: int, k: int, alpha: float) -> float:
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1).")
    return float((2.0 * k * np.log(n + 1.0) - np.log(alpha)) / n)


def mtn_kl_lower_endpoint(
    data: np.ndarray,
    alpha: float,
    random_starts: int = 6,
    seed: int = 0,
) -> float:
    probs_hat = empirical_conditional_probs(data)
    counts_z0 = int(np.sum(data[:, 0] == 0))
    counts_z1 = int(np.sum(data[:, 0] == 1))
    if counts_z0 == 0 or counts_z1 == 0:
        raise ValueError("Both instrument levels must be present.")

    p0_hat = probs_hat[:4]
    p1_hat = probs_hat[4:]
    alpha_split = alpha / 2.0
    tau0 = mtn_kl_radius(counts_z0, 4, alpha_split)
    tau1 = mtn_kl_radius(counts_z1, 4, alpha_split)
    eps = 1e-12
    rng = np.random.default_rng(seed)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        q0 = np.clip(x[:4], eps, 1.0)
        q1 = np.clip(x[4:8], eps, 1.0)
        t = float(x[8])
        return q0, q1, t

    def objective(x: np.ndarray) -> float:
        return float(x[8])

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:4]) - 1.0},
        {"type": "eq", "fun": lambda x: np.sum(x[4:8]) - 1.0},
        {"type": "ineq", "fun": lambda x: tau0 - kl_divergence(p0_hat, np.clip(x[:4], eps, 1.0))},
        {"type": "ineq", "fun": lambda x: tau1 - kl_divergence(p1_hat, np.clip(x[4:8], eps, 1.0))},
    ]

    for idx in range(LOWER_COEFFS.shape[0]):
        coeff = LOWER_COEFFS[idx]
        const = LOWER_CONSTS[idx]
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, coeff=coeff, const=const: x[8] - (float(coeff @ x[:8]) + float(const)),
            }
        )

    bounds = [(eps, 1.0)] * 8 + [(-2.0, 2.0)]
    starts = [np.concatenate([p0_hat, p1_hat, [lower_bound_from_data(data)[0]]])]
    for _ in range(random_starts):
        q0 = rng.dirichlet(counts_z0 * p0_hat + 1.0)
        q1 = rng.dirichlet(counts_z1 * p1_hat + 1.0)
        t0 = float(np.max(LOWER_COEFFS @ np.concatenate([q0, q1]) + LOWER_CONSTS))
        starts.append(np.concatenate([q0, q1, [t0]]))

    best = None
    for start in starts:
        res = scipy.optimize.minimize(
            objective,
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
        )
        if not res.success:
            continue
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("MTN KL optimization failed to converge.")
    return float(best.x[8])


def mtn_kl_upper_endpoint(
    data: np.ndarray,
    alpha: float,
    random_starts: int = 6,
    seed: int = 0,
) -> float:
    probs_hat = empirical_conditional_probs(data)
    counts_z0 = int(np.sum(data[:, 0] == 0))
    counts_z1 = int(np.sum(data[:, 0] == 1))
    if counts_z0 == 0 or counts_z1 == 0:
        raise ValueError("Both instrument levels must be present.")

    p0_hat = probs_hat[:4]
    p1_hat = probs_hat[4:]
    alpha_split = alpha / 2.0
    tau0 = mtn_kl_radius(counts_z0, 4, alpha_split)
    tau1 = mtn_kl_radius(counts_z1, 4, alpha_split)
    eps = 1e-12
    rng = np.random.default_rng(seed)

    def objective(x: np.ndarray) -> float:
        return float(-x[8])

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:4]) - 1.0},
        {"type": "eq", "fun": lambda x: np.sum(x[4:8]) - 1.0},
        {"type": "ineq", "fun": lambda x: tau0 - kl_divergence(p0_hat, np.clip(x[:4], eps, 1.0))},
        {"type": "ineq", "fun": lambda x: tau1 - kl_divergence(p1_hat, np.clip(x[4:8], eps, 1.0))},
    ]

    for idx in range(UPPER_COEFFS.shape[0]):
        coeff = UPPER_COEFFS[idx]
        const = UPPER_CONSTS[idx]
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, coeff=coeff, const=const: (float(coeff @ x[:8]) + float(const)) - x[8],
            }
        )

    bounds = [(eps, 1.0)] * 8 + [(-2.0, 2.0)]
    starts = [np.concatenate([p0_hat, p1_hat, [upper_bound_from_data(data)[0]]])]
    for _ in range(random_starts):
        q0 = rng.dirichlet(counts_z0 * p0_hat + 1.0)
        q1 = rng.dirichlet(counts_z1 * p1_hat + 1.0)
        t0 = float(np.min(UPPER_COEFFS @ np.concatenate([q0, q1]) + UPPER_CONSTS))
        starts.append(np.concatenate([q0, q1, [t0]]))

    best = None
    for start in starts:
        res = scipy.optimize.minimize(
            objective,
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
        )
        if not res.success:
            continue
        if best is None or res.fun < best.fun:
            best = res

    if best is None:
        raise RuntimeError("MTN KL optimization failed to converge.")
    return float(best.x[8])


def recentered_subsampling_lower_endpoint(
    data: np.ndarray,
    b: int,
    alpha: float,
    gamma: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    m = resolve_subsample_size(n, gamma=gamma)
    theta_n = lower_bound_from_data(data)[0]
    draws = np.empty(b, dtype=float)
    for i in range(b):
        sub = draw_iv_subsample(data, m=m, rng=rng)
        draws[i] = lower_bound_from_data(sub)[0]
    t = np.sqrt(m) * (draws - theta_n)
    return float(theta_n - np.quantile(t, 1.0 - (alpha / 2.0)) / np.sqrt(n))


def recentered_subsampling_upper_endpoint(
    data: np.ndarray,
    b: int,
    alpha: float,
    gamma: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    m = resolve_subsample_size(n, gamma=gamma)
    theta_n = upper_bound_from_data(data)[0]
    draws = np.empty(b, dtype=float)
    for i in range(b):
        sub = draw_iv_subsample(data, m=m, rng=rng)
        draws[i] = upper_bound_from_data(sub)[0]
    t = np.sqrt(m) * (draws - theta_n)
    return float(theta_n - np.quantile(t, alpha / 2.0) / np.sqrt(n))


@dataclass
class RepResult:
    point_lb: float
    point_ub: float
    sample_active_idx: int
    sample_active_ub_idx: int
    oracle_lb: float
    oracle_ub: float
    bootstrap_lb: float
    bootstrap_ub: float
    subsample_lb: float
    subsample_ub: float
    mtn_kl_lb: float
    mtn_kl_ub: float
    bayes_dirichlet_lb: float
    bayes_dirichlet_ub: float


def run_one_rep(
    rep_id: int,
    model_name: str,
    n: int,
    b: int,
    alpha: float,
    gamma: float,
    seed: int,
) -> RepResult:
    model = MODEL_PRESETS[model_name]
    population_probs = population_conditional_probs(model)
    population_lb_candidates = lower_candidates_from_probs(population_probs)
    population_ub_candidates = upper_candidates_from_probs(population_probs)
    oracle_idx = int(np.argmax(population_lb_candidates))
    oracle_ub_idx = int(np.argmin(population_ub_candidates))

    data = simulate_iv_data(model, n=n, seed=seed)
    point_lb, sample_active_idx, sample_lb_candidates = lower_bound_from_data(data)
    point_ub, sample_active_ub_idx, sample_ub_candidates = upper_bound_from_data(data)

    oracle_lb_est = float(sample_lb_candidates[oracle_idx])
    oracle_lb_se = candidate_standard_error(data, coeffs=LOWER_COEFFS[oracle_idx])
    oracle_lb = float(oracle_lb_est - NORMAL_975 * oracle_lb_se)

    oracle_ub_est = float(sample_ub_candidates[oracle_ub_idx])
    oracle_ub_se = candidate_standard_error(data, coeffs=UPPER_COEFFS[oracle_ub_idx])
    oracle_ub = float(oracle_ub_est + NORMAL_975 * oracle_ub_se)

    bootstrap_lb = bootstrap_lower_endpoint(
        data=data,
        b=b,
        alpha=alpha,
        seed=seed + 10_000_000,
    )
    bootstrap_ub = bootstrap_upper_endpoint(
        data=data,
        b=b,
        alpha=alpha,
        seed=seed + 15_000_000,
    )
    subsample_lb = recentered_subsampling_lower_endpoint(
        data=data,
        b=b,
        alpha=alpha,
        gamma=gamma,
        seed=seed + 20_000_000,
    )
    subsample_ub = recentered_subsampling_upper_endpoint(
        data=data,
        b=b,
        alpha=alpha,
        gamma=gamma,
        seed=seed + 25_000_000,
    )
    mtn_kl_lb = mtn_kl_lower_endpoint(
        data=data,
        alpha=alpha,
        seed=seed + 30_000_000,
    )
    mtn_kl_ub = mtn_kl_upper_endpoint(
        data=data,
        alpha=alpha,
        seed=seed + 35_000_000,
    )
    bayes_dirichlet_lb = bayesian_dirichlet_lower_endpoint(
        data=data,
        b=b,
        alpha=alpha,
        seed=seed + 40_000_000,
    )
    bayes_dirichlet_ub = bayesian_dirichlet_upper_endpoint(
        data=data,
        b=b,
        alpha=alpha,
        seed=seed + 45_000_000,
    )
    return RepResult(
        point_lb=point_lb,
        point_ub=point_ub,
        sample_active_idx=sample_active_idx,
        sample_active_ub_idx=sample_active_ub_idx,
        oracle_lb=oracle_lb,
        oracle_ub=oracle_ub,
        bootstrap_lb=bootstrap_lb,
        bootstrap_ub=bootstrap_ub,
        subsample_lb=subsample_lb,
        subsample_ub=subsample_ub,
        mtn_kl_lb=mtn_kl_lb,
        mtn_kl_ub=mtn_kl_ub,
        bayes_dirichlet_lb=bayes_dirichlet_lb,
        bayes_dirichlet_ub=bayes_dirichlet_ub,
    )


def summarize_results(
    results: list[RepResult],
    true_lb: float,
    true_ub: float,
    oracle_lb_idx: int,
    oracle_ub_idx: int,
) -> dict[str, float]:
    point_lbs = np.asarray([res.point_lb for res in results], dtype=float)
    point_ubs = np.asarray([res.point_ub for res in results], dtype=float)
    oracle_lbs = np.asarray([res.oracle_lb for res in results], dtype=float)
    oracle_ubs = np.asarray([res.oracle_ub for res in results], dtype=float)
    bootstrap_lbs = np.asarray([res.bootstrap_lb for res in results], dtype=float)
    bootstrap_ubs = np.asarray([res.bootstrap_ub for res in results], dtype=float)
    subsample_lbs = np.asarray([res.subsample_lb for res in results], dtype=float)
    subsample_ubs = np.asarray([res.subsample_ub for res in results], dtype=float)
    mtn_kl_lbs = np.asarray([res.mtn_kl_lb for res in results], dtype=float)
    mtn_kl_ubs = np.asarray([res.mtn_kl_ub for res in results], dtype=float)
    bayes_dirichlet_lbs = np.asarray([res.bayes_dirichlet_lb for res in results], dtype=float)
    bayes_dirichlet_ubs = np.asarray([res.bayes_dirichlet_ub for res in results], dtype=float)
    active_lb_idx = np.asarray([res.sample_active_idx for res in results], dtype=int)
    active_ub_idx = np.asarray([res.sample_active_ub_idx for res in results], dtype=int)

    out = {
        "mean_point_lb": float(point_lbs.mean()),
        "mean_point_ub": float(point_ubs.mean()),
        "active_lb_match_rate": float(np.mean(active_lb_idx == oracle_lb_idx)),
        "active_ub_match_rate": float(np.mean(active_ub_idx == oracle_ub_idx)),
    }
    methods = {
        "oracle": (oracle_lbs, oracle_ubs),
        "bootstrap": (bootstrap_lbs, bootstrap_ubs),
        "subsample": (subsample_lbs, subsample_ubs),
        "mtn_kl": (mtn_kl_lbs, mtn_kl_ubs),
        "bayes_dirichlet": (bayes_dirichlet_lbs, bayes_dirichlet_ubs),
    }
    for name, (lbs, ubs) in methods.items():
        out[f"{name}_lb_coverage"] = float(np.mean(lbs <= true_lb))
        out[f"{name}_ub_coverage"] = float(np.mean(ubs >= true_ub))
        out[f"{name}_joint_coverage"] = float(np.mean((lbs <= true_lb) & (ubs >= true_ub)))
        out[f"{name}_lb_avg_width"] = float(np.mean(point_lbs - lbs))
        out[f"{name}_ub_avg_width"] = float(np.mean(ubs - point_ubs))
        out[f"{name}_joint_avg_width"] = float(np.mean(ubs - lbs))
    return out


def describe_model(
    model: IVModel,
    detailed: bool = False,
) -> tuple[float, int, np.ndarray] | tuple[float, float, int, int, np.ndarray, np.ndarray]:
    probs = population_conditional_probs(model)
    lb_candidates = lower_candidates_from_probs(probs)
    ub_candidates = upper_candidates_from_probs(probs)
    oracle_lb_idx = int(np.argmax(lb_candidates))
    oracle_ub_idx = int(np.argmin(ub_candidates))
    if not detailed:
        return (
            float(lb_candidates[oracle_lb_idx]),
            oracle_lb_idx,
            lb_candidates,
        )
    return (
        float(lb_candidates[oracle_lb_idx]),
        float(ub_candidates[oracle_ub_idx]),
        oracle_lb_idx,
        oracle_ub_idx,
        lb_candidates,
        ub_candidates,
    )


def format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "n/a"
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    return f"{minutes:d}m{sec:02d}s"


def save_summary_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    fieldnames = [
        "n",
        "mean_point_lb",
        "mean_point_ub",
        "active_lb_match_rate",
        "active_ub_match_rate",
        "oracle_lb_coverage",
        "oracle_lb_avg_width",
        "oracle_ub_coverage",
        "oracle_ub_avg_width",
        "oracle_joint_coverage",
        "oracle_joint_avg_width",
        "bootstrap_lb_coverage",
        "bootstrap_lb_avg_width",
        "bootstrap_ub_coverage",
        "bootstrap_ub_avg_width",
        "bootstrap_joint_coverage",
        "bootstrap_joint_avg_width",
        "subsample_lb_coverage",
        "subsample_lb_avg_width",
        "subsample_ub_coverage",
        "subsample_ub_avg_width",
        "subsample_joint_coverage",
        "subsample_joint_avg_width",
        "mtn_kl_lb_coverage",
        "mtn_kl_lb_avg_width",
        "mtn_kl_ub_coverage",
        "mtn_kl_ub_avg_width",
        "mtn_kl_joint_coverage",
        "mtn_kl_joint_avg_width",
        "bayes_dirichlet_lb_coverage",
        "bayes_dirichlet_lb_avg_width",
        "bayes_dirichlet_ub_coverage",
        "bayes_dirichlet_ub_avg_width",
        "bayes_dirichlet_joint_coverage",
        "bayes_dirichlet_joint_avg_width",
    ]
    with out_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_plots(rows: list[dict[str, float]], out_dir: Path, stem: str) -> list[Path]:
    n_vals = np.asarray([int(row["n"]) for row in rows], dtype=int)
    saved_paths: list[Path] = []

    methods = [
        ("oracle", "Oracle", "o"),
        ("bootstrap", "Bootstrap", "s"),
        ("subsample", "Subsampling", "^"),
        ("mtn_kl", "MTN KL", "d"),
        ("bayes_dirichlet", "Bayes Dirichlet", "x"),
    ]

    def _plot(
        metric_suffix: str,
        title: str,
        ylabel: str,
        filename: str,
        coverage_plot: bool,
        target_line: float | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, label, marker in methods:
            vals = np.asarray([float(row[f"{key}_{metric_suffix}"]) for row in rows], dtype=float)
            ax.plot(n_vals, vals, marker=marker, linewidth=2, label=label)
        if coverage_plot:
            if target_line is not None:
                ax.axhline(target_line, color="black", linestyle="--", linewidth=1, alpha=0.8)
            ax.set_ylim(0.6, 1.0)
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(MultipleLocator(0.025))
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        path = out_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved_paths.append(path)

    _plot("lb_coverage", "Balke-Pearl Lower-Bound One-Sided Coverage", "Coverage", f"{stem}_lower_coverage.png", True, 0.975)
    _plot("lb_avg_width", "Balke-Pearl Lower-Bound One-Sided Width", "Average Width", f"{stem}_lower_width.png", False)
    _plot("ub_coverage", "Balke-Pearl Upper-Bound One-Sided Coverage", "Coverage", f"{stem}_upper_coverage.png", True, 0.975)
    _plot("ub_avg_width", "Balke-Pearl Upper-Bound One-Sided Width", "Average Width", f"{stem}_upper_width.png", False)
    _plot("joint_coverage", "Balke-Pearl Joint Two-Sided Coverage", "Coverage", f"{stem}_joint_coverage.png", True, 0.95)
    _plot("joint_avg_width", "Balke-Pearl Joint Two-Sided Width", "Average Width", f"{stem}_joint_width.png", False)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["nearly_tied"],
        default="nearly_tied",
        help="Latent binary IV SCM preset.",
    )
    parser.add_argument("--r", type=int, default=1000, help="Monte Carlo replications per n.")
    parser.add_argument("--b", type=int, default=500, help="Bootstrap/subsampling draws per replication.")
    parser.add_argument(
        "--n-grid",
        type=str,
        default="100,200,300,500,1000,2000",
        help="Comma-separated sample sizes.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seed", type=int, default=20260314)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--subsample-gamma", type=float, default=(2.0 / 3.0))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N completed Monte Carlo replications within each n.",
    )
    args = parser.parse_args()

    n_grid = parse_n_grid(args.n_grid)
    model = MODEL_PRESETS[args.model]
    true_lb, true_ub, oracle_lb_idx, oracle_ub_idx, lb_candidates, ub_candidates = describe_model(
        model,
        detailed=True,
    )
    lb_order = np.sort(lb_candidates)[::-1]
    ub_order = np.sort(ub_candidates)
    lb_gap = float(lb_order[0] - lb_order[1])
    ub_gap = float(ub_order[1] - ub_order[0])
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_stem = f"adversarial_iv_{args.model}"

    print(f"model=adversarial::{model.name}")
    print(f"true_lower_bound={true_lb:.8f}")
    print(f"true_upper_bound={true_ub:.8f}")
    print(f"oracle_lower_candidate={oracle_lb_idx + 1}")
    print(f"oracle_upper_candidate={oracle_ub_idx + 1}")
    print(f"population_lower_gap_to_second={lb_gap:.8f}")
    print(f"population_upper_gap_to_second={ub_gap:.8f}")
    print("population_lower_candidates=" + ",".join(f"{x:.8f}" for x in lb_candidates))
    print("population_upper_candidates=" + ",".join(f"{x:.8f}" for x in ub_candidates))
    print(
        "methods=oracle_known_minimizer,standard_bootstrap_percentile,recentered_subsampling,mtn_kl_outer_region,bayesian_dirichlet",
        flush=True,
    )
    print(
        "subsampling_rule="
        f"m=min(n,max(1,floor(n**{args.subsample_gamma:.6f})))",
        flush=True,
    )
    print(
        "columns="
        "n,mean_point_lb,mean_point_ub,active_lb_match_rate,active_ub_match_rate,"
        "oracle_lb_coverage,oracle_lb_avg_width,oracle_ub_coverage,oracle_ub_avg_width,oracle_joint_coverage,oracle_joint_avg_width,"
        "bootstrap_lb_coverage,bootstrap_lb_avg_width,bootstrap_ub_coverage,bootstrap_ub_avg_width,bootstrap_joint_coverage,bootstrap_joint_avg_width,"
        "subsample_lb_coverage,subsample_lb_avg_width,subsample_ub_coverage,subsample_ub_avg_width,subsample_joint_coverage,subsample_joint_avg_width,"
        "mtn_kl_lb_coverage,mtn_kl_lb_avg_width,mtn_kl_ub_coverage,mtn_kl_ub_avg_width,mtn_kl_joint_coverage,mtn_kl_joint_avg_width,"
        "bayes_dirichlet_lb_coverage,bayes_dirichlet_lb_avg_width,bayes_dirichlet_ub_coverage,bayes_dirichlet_ub_avg_width,bayes_dirichlet_joint_coverage,bayes_dirichlet_joint_avg_width",
        flush=True,
    )
    print(f"workers={args.workers}", flush=True)
    print(f"output_dir={out_dir}", flush=True)

    spawned = np.random.SeedSequence(args.seed).spawn(len(n_grid))
    summary_rows: list[dict[str, float]] = []
    overall_t0 = time.monotonic()
    total_jobs = len(n_grid) * args.r
    completed_jobs = 0
    for n_idx, n in enumerate(n_grid):
        t0 = time.monotonic()
        rep_seeds = spawned[n_idx].generate_state(args.r, dtype=np.uint64)
        results: list[RepResult] = []
        print(
            f"starting_n={n} reps={args.r} resamples={args.b} "
            f"completed_jobs={completed_jobs}/{total_jobs}",
            flush=True,
        )
        if args.workers > 1:
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futures = [
                        ex.submit(
                            run_one_rep,
                            rep_id=rep_id,
                            model_name=args.model,
                            n=n,
                            b=args.b,
                            alpha=args.alpha,
                            gamma=args.subsample_gamma,
                            seed=int(rep_seeds[rep_id]),
                        )
                        for rep_id in range(args.r)
                    ]
                    done_in_n = 0
                    for fut in as_completed(futures):
                        results.append(fut.result())
                        done_in_n += 1
                        completed_jobs += 1
                        if done_in_n % args.progress_every == 0 or done_in_n == args.r:
                            elapsed = time.monotonic() - overall_t0
                            rate = completed_jobs / elapsed if elapsed > 0 else float("nan")
                            remaining = (total_jobs - completed_jobs) / rate if rate and np.isfinite(rate) else float("nan")
                            print(
                                f"progress n={n} rep={done_in_n}/{args.r} "
                                f"overall={completed_jobs}/{total_jobs} "
                                f"elapsed={format_eta(elapsed)} eta={format_eta(remaining)}",
                                flush=True,
                            )
            except PermissionError:
                results = []
                for rep_id in range(args.r):
                    results.append(
                        run_one_rep(
                            rep_id=rep_id,
                            model_name=args.model,
                            n=n,
                            b=args.b,
                            alpha=args.alpha,
                            gamma=args.subsample_gamma,
                            seed=int(rep_seeds[rep_id]),
                        )
                    )
                    completed_jobs += 1
                    done_in_n = rep_id + 1
                    if done_in_n % args.progress_every == 0 or done_in_n == args.r:
                        elapsed = time.monotonic() - overall_t0
                        rate = completed_jobs / elapsed if elapsed > 0 else float("nan")
                        remaining = (total_jobs - completed_jobs) / rate if rate and np.isfinite(rate) else float("nan")
                        print(
                            f"progress n={n} rep={done_in_n}/{args.r} "
                            f"overall={completed_jobs}/{total_jobs} "
                            f"elapsed={format_eta(elapsed)} eta={format_eta(remaining)}",
                            flush=True,
                        )
        else:
            results = []
            for rep_id in range(args.r):
                results.append(
                    run_one_rep(
                        rep_id=rep_id,
                        model_name=args.model,
                        n=n,
                        b=args.b,
                        alpha=args.alpha,
                        gamma=args.subsample_gamma,
                        seed=int(rep_seeds[rep_id]),
                    )
                )
                completed_jobs += 1
                done_in_n = rep_id + 1
                if done_in_n % args.progress_every == 0 or done_in_n == args.r:
                    elapsed = time.monotonic() - overall_t0
                    rate = completed_jobs / elapsed if elapsed > 0 else float("nan")
                    remaining = (total_jobs - completed_jobs) / rate if rate and np.isfinite(rate) else float("nan")
                    print(
                        f"progress n={n} rep={done_in_n}/{args.r} "
                        f"overall={completed_jobs}/{total_jobs} "
                        f"elapsed={format_eta(elapsed)} eta={format_eta(remaining)}",
                        flush=True,
                    )
        summary = summarize_results(
            results,
            true_lb=true_lb,
            true_ub=true_ub,
            oracle_lb_idx=oracle_lb_idx,
            oracle_ub_idx=oracle_ub_idx,
        )
        elapsed = time.monotonic() - t0
        row = {"n": n, **summary}
        summary_rows.append(row)
        print(
            f"{n},"
            f"{summary['mean_point_lb']:.6f},"
            f"{summary['mean_point_ub']:.6f},"
            f"{summary['active_lb_match_rate']:.4f},"
            f"{summary['active_ub_match_rate']:.4f},"
            f"{summary['oracle_lb_coverage']:.4f},"
            f"{summary['oracle_lb_avg_width']:.6f},"
            f"{summary['oracle_ub_coverage']:.4f},"
            f"{summary['oracle_ub_avg_width']:.6f},"
            f"{summary['oracle_joint_coverage']:.4f},"
            f"{summary['oracle_joint_avg_width']:.6f},"
            f"{summary['bootstrap_lb_coverage']:.4f},"
            f"{summary['bootstrap_lb_avg_width']:.6f},"
            f"{summary['bootstrap_ub_coverage']:.4f},"
            f"{summary['bootstrap_ub_avg_width']:.6f},"
            f"{summary['bootstrap_joint_coverage']:.4f},"
            f"{summary['bootstrap_joint_avg_width']:.6f},"
            f"{summary['subsample_lb_coverage']:.4f},"
            f"{summary['subsample_lb_avg_width']:.6f},"
            f"{summary['subsample_ub_coverage']:.4f},"
            f"{summary['subsample_ub_avg_width']:.6f},"
            f"{summary['subsample_joint_coverage']:.4f},"
            f"{summary['subsample_joint_avg_width']:.6f},"
            f"{summary['mtn_kl_lb_coverage']:.4f},"
            f"{summary['mtn_kl_lb_avg_width']:.6f},"
            f"{summary['mtn_kl_ub_coverage']:.4f},"
            f"{summary['mtn_kl_ub_avg_width']:.6f},"
            f"{summary['mtn_kl_joint_coverage']:.4f},"
            f"{summary['mtn_kl_joint_avg_width']:.6f},"
            f"{summary['bayes_dirichlet_lb_coverage']:.4f},"
            f"{summary['bayes_dirichlet_lb_avg_width']:.6f},"
            f"{summary['bayes_dirichlet_ub_coverage']:.4f},"
            f"{summary['bayes_dirichlet_ub_avg_width']:.6f},"
            f"{summary['bayes_dirichlet_joint_coverage']:.4f},"
            f"{summary['bayes_dirichlet_joint_avg_width']:.6f}",
            flush=True,
        )
        elapsed_total = time.monotonic() - overall_t0
        avg_n = elapsed_total / float(n_idx + 1)
        eta_n = avg_n * (len(n_grid) - n_idx - 1)
        print(
            f"elapsed_for_n={n}:{elapsed:.1f}s overall_elapsed={format_eta(elapsed_total)} "
            f"remaining_n_eta={format_eta(eta_n)}",
            flush=True,
        )

    csv_path = out_dir / f"{run_stem}.csv"
    save_summary_csv(summary_rows, csv_path)
    plot_paths = save_plots(
        summary_rows,
        out_dir=out_dir,
        stem=run_stem,
    )
    print(f"csv_path={csv_path}", flush=True)
    for path in plot_paths:
        print(f"plot_path={path}", flush=True)


if __name__ == "__main__":
    main()
