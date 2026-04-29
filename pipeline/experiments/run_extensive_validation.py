#!/usr/bin/env python3
"""
run_extensive_validation.py
===========================

Rigorous Stage-3 validation of the phase-margin certificate across multiple
LLM backends, prompts, and seeds.  Designed to leave no room for mis-claim:

  * Pre-registered hypotheses (declared at the top, evaluated mechanically).
  * Multiple probe seeds per (model, prompt) -> confidence interval on Phi.
  * Permutation test for the period-2 purity ratio -> significance p-value.
  * All raw artefacts saved per run:
      - probe trajectories (npy)
      - probe text traces (jsonl)
      - per-frequency phase fits (jsonl)
      - margin report (json)
      - long ground-truth rollout (npy + text jsonl)
  * Per-model and per-prompt aggregates emitted as CSV + JSON.
  * Diagnostic plots: margin vs. model, per-direction heatmap, purity dotplot.
  * Idempotent: a previously-completed run is detected and skipped unless
    --force is passed.

Usage
-----
    python experiments/run_extensive_validation.py
    python experiments/run_extensive_validation.py --models qwen2.5:7b qwen2.5:14b
    python experiments/run_extensive_validation.py --quick
    python experiments/run_extensive_validation.py --models llama3.1:70b-instruct-q4_K_M --probe-seeds 1
    python experiments/run_extensive_validation.py --force
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import platform
import socket
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

# Repo bootstrap so this script runs from anywhere
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))

from phase_margin import run_certification
from phase_margin.embedder import SentenceTransformersEmbedder
from phase_margin.ground_truth import detect_period_by_within_across, detect_regime
from phase_margin.llm.ollama_client import OllamaClient
from phase_margin.loops import ParaphraseLoop
from phase_margin.probe import text_basis_paraphrase
from phase_margin.types import ProbeConfig, Regime

EXPERIMENT_VERSION = "2026.04.28"

# ---------------------------------------------------------------------------
# Pre-registered configuration
# ---------------------------------------------------------------------------
DEFAULT_MODELS: list[str] = [
    "qwen2.5:7b",
    "qwen2.5:14b",
]

# Three semantically distinct prompts.
PROMPTS: dict[str, str] = {
    "factual":   "Deep learning models trained on large corpora exhibit emergent capabilities.",
    "emotional": (
        "I felt a wave of joy mixed with disbelief as the morning sun "
        "finally broke through after the long, sleepless night."
    ),
    "technical": (
        "The Adam optimiser maintains exponentially decaying running averages "
        "of gradients and squared gradients to compute adaptive per-parameter "
        "learning rates."
    ),
}

PRE_REGISTERED_HYPOTHESES = [
    "H1: when ground-truth period-2 purity > 1.15 the certificate predicts "
    "OSCILLATORY (or its minimum directional margin Phi_min < 0.10).",
    "H2: when ground-truth period-2 purity < 0.87 the certificate predicts "
    "CONTRACTIVE (or its minimum directional margin Phi_min > 1.00).",
    "H3 (speculative): on a fixed prompt, larger models exhibit larger Phi_min "
    "(less oscillatory paraphrase loops).",
    "H4 (speculative): the verbosity axis is more often the smallest-Phi_v axis "
    "than any other axis across (model, prompt) cells.",
]

DEFAULT_PROBE_CONFIG = dict(
    horizon=12,
    n_seeds=3,
    n_seeds_nominal=3,
    n_frequencies=4,
    epsilon=0.6,
    residual_cap=0.5,   # tightened from 0.9 -> 0.5: noise fits get rejected
)

GROUND_TRUTH_HORIZON = 40
PURITY_PERIODS = (2, 3)
N_PERMUTATIONS = 500
PURITY_OSC_THRESHOLD = 1.15
PURITY_CONT_THRESHOLD = 1.0 / PURITY_OSC_THRESHOLD


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ExperimentRun:
    model: str
    prompt_name: str
    prompt_text: str
    probe_seed: int
    # Certificate side
    phase_margin: float
    phi_min: float
    per_direction_margin: dict[str, float]
    predicted_regime: str
    elapsed_certify_s: float
    # Ground-truth side
    purity_period_2: float
    purity_period_3: float
    purity_p_value_2: float
    period_2_regime: str
    autocorr_regime: str
    autocorr_period_score: float
    autocorr_period_lag: int
    final_variance: float
    growth_rate: float
    elapsed_ground_truth_s: float
    # Bookkeeping
    artefact_dir: str
    error: str = ""

    @property
    def predicted_oscillatory(self) -> bool:
        return self.predicted_regime == "oscillatory" or self.phi_min < 0.10

    @property
    def predicted_contractive(self) -> bool:
        return self.predicted_regime == "contractive" and self.phi_min > 1.00

    @property
    def observed_oscillatory(self) -> bool:
        return self.purity_period_2 > PURITY_OSC_THRESHOLD

    @property
    def observed_contractive(self) -> bool:
        return self.purity_period_2 < PURITY_CONT_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_margin.extensive")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def host_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        import torch  # type: ignore
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = int(torch.cuda.device_count())
    except Exception:
        info["torch"] = "unavailable"
    return info


def safe_json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Regime):
        return o.value
    return str(o)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=safe_json_default))


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=safe_json_default) + "\n")


def permutation_pvalue_for_purity(
    trajectory: np.ndarray,
    period: int = 2,
    n_perms: int = N_PERMUTATIONS,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Return (observed_purity, p_value, null_distribution).

    Null hypothesis: the iteration index carries no period-`period` structure.
    We shuffle the trajectory rows uniformly at random and recompute the
    within/across-class purity ratio.  p-value is the fraction of shuffled
    purities >= observed.
    """
    Z = np.asarray(trajectory, dtype=float)
    rng = np.random.default_rng(seed)

    def _purity(traj: np.ndarray, p: int) -> float:
        classes = [traj[i::p] for i in range(p)]
        within: list[float] = []
        for C in classes:
            if len(C) < 2:
                continue
            D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
            within.append(D[np.triu_indices(len(C), k=1)].mean())
        cross: list[float] = []
        for i in range(p):
            for j in range(i + 1, p):
                D = np.linalg.norm(
                    classes[i][:, None, :] - classes[j][None, :, :], axis=-1
                )
                cross.append(D.mean())
        if not within or not cross:
            return float("nan")
        return float(np.mean(cross) / max(np.mean(within), 1e-12))

    observed = _purity(Z, period)
    null = np.zeros(n_perms, dtype=float)
    n = Z.shape[0]
    for k in range(n_perms):
        perm = rng.permutation(n)
        null[k] = _purity(Z[perm], period)
    p_value = float((null >= observed).sum() / n_perms)
    return observed, p_value, null


# ---------------------------------------------------------------------------
# Single (model, prompt, seed) experiment
# ---------------------------------------------------------------------------
def single_experiment(
    *,
    model: str,
    prompt_name: str,
    prompt_text: str,
    probe_seed: int,
    embedder: SentenceTransformersEmbedder,
    out_dir: Path,
    probe_cfg: dict,
    log: logging.Logger,
) -> ExperimentRun:
    artefact_dir = out_dir / "raw" / f"{model.replace(':','_')}__{prompt_name}__seed{probe_seed}"
    artefact_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"  >>> {model} / {prompt_name} / seed={probe_seed}  ->  {artefact_dir.name}")

    llm = OllamaClient(model=model)
    loop = ParaphraseLoop(
        llm=llm, embedder=embedder, initial_text=prompt_text,
    )
    basis = text_basis_paraphrase()
    cfg = ProbeConfig(**probe_cfg)
    save_json(asdict(cfg), artefact_dir / "probe_config.json")

    # ---- Certificate ---------------------------------------------------
    np.random.seed(probe_seed)  # affects probing-basis lazy embedding
    t0 = time.time()
    try:
        report = run_certification(
            loop=loop, basis=basis, config=cfg,
            use_text_perturbation=True, progress=False,
            probe_seed_base=probe_seed,
        )
        elapsed_certify = time.time() - t0
        log.info(f"      certificate: Phi = {report.phase_margin:+.3f} "
                 f"({report.regime.value}) in {elapsed_certify:.0f}s")
    except Exception as e:
        elapsed_certify = time.time() - t0
        err = f"certificate failed: {type(e).__name__}: {e}"
        log.error(err)
        log.error(traceback.format_exc())
        return ExperimentRun(
            model=model, prompt_name=prompt_name, prompt_text=prompt_text,
            probe_seed=probe_seed, phase_margin=float("nan"),
            phi_min=float("nan"), per_direction_margin={},
            predicted_regime="error", elapsed_certify_s=elapsed_certify,
            purity_period_2=float("nan"), purity_period_3=float("nan"),
            purity_p_value_2=float("nan"), period_2_regime="error",
            autocorr_regime="error", autocorr_period_score=float("nan"),
            autocorr_period_lag=0, final_variance=float("nan"),
            growth_rate=float("nan"), elapsed_ground_truth_s=0.0,
            artefact_dir=str(artefact_dir), error=err,
        )

    # Save the report and per-direction spectra
    save_json({
        "phase_margin": float(report.phase_margin),
        "regime": report.regime.value,
        "per_direction_margin": dict(report.per_direction_margin),
        "agent_spectra": {
            name: {
                "omegas":     spec.omegas.tolist(),
                "thetas":     spec.thetas.tolist(),
                "amplitudes": spec.amplitudes.tolist(),
                "residuals":  spec.residuals.tolist(),
            } for name, spec in report.agent_spectra.items()
        },
        "notes": report.notes,
    }, artefact_dir / "margin_report.json")

    phi_min = float(min(report.per_direction_margin.values())
                    if report.per_direction_margin else report.phase_margin)

    # ---- Ground-truth rollout -----------------------------------------
    log.info("      ground-truth rollout ...")
    t1 = time.time()
    loop.reset(seed=probe_seed)
    embeddings: list[np.ndarray] = []
    text_log: list[dict] = []
    for k in range(GROUND_TRUTH_HORIZON):
        try:
            z = loop.step(seed=probe_seed)
        except Exception as e:
            log.error(f"      step {k} crashed: {e}")
            break
        embeddings.append(z)
        text_log.append({"k": k, "text": loop.state})
    elapsed_ground_truth = time.time() - t1
    if not embeddings:
        log.error("      no ground-truth embeddings produced")
        return ExperimentRun(
            model=model, prompt_name=prompt_name, prompt_text=prompt_text,
            probe_seed=probe_seed, phase_margin=float(report.phase_margin),
            phi_min=phi_min, per_direction_margin=dict(report.per_direction_margin),
            predicted_regime=report.regime.value,
            elapsed_certify_s=elapsed_certify,
            purity_period_2=float("nan"), purity_period_3=float("nan"),
            purity_p_value_2=float("nan"), period_2_regime="error",
            autocorr_regime="error", autocorr_period_score=float("nan"),
            autocorr_period_lag=0, final_variance=float("nan"),
            growth_rate=float("nan"), elapsed_ground_truth_s=elapsed_ground_truth,
            artefact_dir=str(artefact_dir), error="no ground-truth steps",
        )
    traj = np.stack(embeddings)
    np.save(artefact_dir / "trajectory.npy", traj)
    save_jsonl(text_log, artefact_dir / "trajectory_text.jsonl")

    # ---- Purity-based regime detection + permutation p-value ---------
    purity2, p2, null2 = permutation_pvalue_for_purity(traj, period=2, seed=probe_seed)
    purity3, _,  _     = permutation_pvalue_for_purity(traj, period=3, seed=probe_seed)
    p2_regime = (
        "oscillatory" if purity2 > PURITY_OSC_THRESHOLD else
        "contractive" if purity2 < PURITY_CONT_THRESHOLD else
        "unknown"
    )

    # Autocorrelation-based detector for cross-comparison
    ac = detect_regime(traj, period_min=2, period_max=8)
    np.save(artefact_dir / "purity_null_period2.npy", null2)

    save_json({
        "purity_period_2": purity2,
        "purity_p_value_2": p2,
        "purity_period_3": purity3,
        "period_2_regime": p2_regime,
        "autocorr_regime": ac.regime.value,
        "autocorr_period_score": ac.period_score,
        "autocorr_period_lag": ac.period_lag,
        "final_variance": ac.final_variance,
        "growth_rate": ac.growth_rate,
    }, artefact_dir / "ground_truth.json")

    log.info(f"      purity(p=2) = {purity2:.3f}  (p-val {p2:.3g})  ->  {p2_regime}")
    log.info(f"      autocorr regime = {ac.regime.value}  "
             f"(score {ac.period_score:.3f} at lag {ac.period_lag})")
    log.info(f"      ground truth rollout in {elapsed_ground_truth:.0f}s")

    return ExperimentRun(
        model=model, prompt_name=prompt_name, prompt_text=prompt_text,
        probe_seed=probe_seed, phase_margin=float(report.phase_margin),
        phi_min=phi_min,
        per_direction_margin=dict(report.per_direction_margin),
        predicted_regime=report.regime.value,
        elapsed_certify_s=elapsed_certify,
        purity_period_2=purity2, purity_period_3=purity3,
        purity_p_value_2=p2, period_2_regime=p2_regime,
        autocorr_regime=ac.regime.value,
        autocorr_period_score=float(ac.period_score),
        autocorr_period_lag=int(ac.period_lag),
        final_variance=float(ac.final_variance),
        growth_rate=float(ac.growth_rate),
        elapsed_ground_truth_s=elapsed_ground_truth,
        artefact_dir=str(artefact_dir),
    )


# ---------------------------------------------------------------------------
# Aggregation, hypothesis evaluation, plots
# ---------------------------------------------------------------------------
def to_csv(runs: list[ExperimentRun], path: Path) -> None:
    import csv
    columns = [
        "model", "prompt_name", "probe_seed",
        "phase_margin", "phi_min", "predicted_regime",
        "purity_period_2", "purity_p_value_2", "period_2_regime",
        "purity_period_3",
        "autocorr_regime", "autocorr_period_score", "autocorr_period_lag",
        "final_variance", "growth_rate",
        "elapsed_certify_s", "elapsed_ground_truth_s",
        "predicted_oscillatory", "observed_oscillatory",
        "predicted_contractive", "observed_contractive",
        "error", "artefact_dir",
    ]
    # Per-direction margins as wide columns
    all_dirs: list[str] = []
    for r in runs:
        for d in r.per_direction_margin.keys():
            if d not in all_dirs:
                all_dirs.append(d)
    columns.extend([f"phi_v[{d}]" for d in all_dirs])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in runs:
            row = [
                r.model, r.prompt_name, r.probe_seed,
                f"{r.phase_margin:.6f}", f"{r.phi_min:.6f}", r.predicted_regime,
                f"{r.purity_period_2:.6f}", f"{r.purity_p_value_2:.6g}",
                r.period_2_regime,
                f"{r.purity_period_3:.6f}",
                r.autocorr_regime, f"{r.autocorr_period_score:.6f}",
                r.autocorr_period_lag,
                f"{r.final_variance:.6g}", f"{r.growth_rate:.6g}",
                f"{r.elapsed_certify_s:.1f}", f"{r.elapsed_ground_truth_s:.1f}",
                int(r.predicted_oscillatory), int(r.observed_oscillatory),
                int(r.predicted_contractive), int(r.observed_contractive),
                r.error, r.artefact_dir,
            ]
            for d in all_dirs:
                v = r.per_direction_margin.get(d, float("nan"))
                row.append(f"{float(v):.6f}")
            w.writerow(row)


def evaluate_hypotheses(runs: list[ExperimentRun]) -> dict[str, Any]:
    successful = [r for r in runs if not r.error]
    n = len(successful)
    if n == 0:
        return {"note": "no successful runs"}

    h1_pos = [r for r in successful if r.observed_oscillatory]
    h1_pred = sum(1 for r in h1_pos if r.predicted_oscillatory)
    h1_rate = h1_pred / max(len(h1_pos), 1)

    h2_pos = [r for r in successful if r.observed_contractive]
    h2_pred = sum(1 for r in h2_pos if r.predicted_contractive)
    h2_rate = h2_pred / max(len(h2_pos), 1)

    # H3: per-(prompt) sort by approximate model size proxy (param count parsed
    # from the model tag, e.g. ":7b" -> 7).
    def size(model_tag: str) -> float:
        part = model_tag.split(":")[-1].lower()
        for token in part.split("-"):
            tok = token.strip()
            if tok.endswith("b"):
                try:
                    return float(tok[:-1])
                except ValueError:
                    continue
        return float("nan")

    by_prompt: dict[str, list[tuple[float, float]]] = {}
    for r in successful:
        s = size(r.model)
        if np.isnan(s):
            continue
        by_prompt.setdefault(r.prompt_name, []).append((s, r.phi_min))
    h3_corr: dict[str, float] = {}
    for p, pairs in by_prompt.items():
        if len(pairs) < 2:
            continue
        arr = np.array(pairs)
        if np.std(arr[:, 0]) == 0:
            continue
        h3_corr[p] = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])

    # H4: how often is verbosity the smallest-margin direction?
    verbosity_smallest = 0
    total_with_dirs = 0
    for r in successful:
        if not r.per_direction_margin:
            continue
        total_with_dirs += 1
        smallest = min(r.per_direction_margin.items(), key=lambda kv: kv[1])[0]
        if smallest == "verbosity":
            verbosity_smallest += 1
    h4_rate = (verbosity_smallest / total_with_dirs) if total_with_dirs else float("nan")

    return {
        "n_successful_runs": n,
        "H1_oscillatory_recall": {
            "definition": (
                "P(predicted_oscillatory | observed_oscillatory)  "
                "where observed = period-2 purity > 1.15  and  "
                "predicted = report.regime == oscillatory OR phi_min < 0.10"
            ),
            "n_observed_oscillatory": len(h1_pos),
            "n_correctly_predicted":  h1_pred,
            "rate": h1_rate,
        },
        "H2_contractive_specificity": {
            "definition": (
                "P(predicted_contractive | observed_contractive)  "
                "where observed = purity < 0.87  and  "
                "predicted = regime == contractive AND phi_min > 1.00"
            ),
            "n_observed_contractive": len(h2_pos),
            "n_correctly_predicted":  h2_pred,
            "rate": h2_rate,
        },
        "H3_size_vs_phi_min_per_prompt_correlation": h3_corr,
        "H4_verbosity_is_smallest_margin": {
            "n_runs_with_directions": total_with_dirs,
            "n_verbosity_smallest":   verbosity_smallest,
            "rate": h4_rate,
        },
    }


def plot_summary(runs: list[ExperimentRun], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    successful = [r for r in runs if not r.error]
    if not successful:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Phi_min vs (model, prompt)
    models = sorted({r.model for r in successful})
    prompts = sorted({r.prompt_name for r in successful})
    fig, ax = plt.subplots(figsize=(max(6, 1.5 + 1.0 * len(models)), 4))
    width = 0.8 / max(len(prompts), 1)
    for j, p in enumerate(prompts):
        means = []
        stds = []
        for m in models:
            vals = [r.phi_min for r in successful
                    if r.model == m and r.prompt_name == p]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        x = np.arange(len(models)) + j * width - (len(prompts) - 1) * width / 2
        ax.bar(x, means, width=width, yerr=stds, label=p, capsize=3)
    ax.axhline(0.10, ls="--", color="0.4", lw=1)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel(r"min directional phase margin $\Phi_{\min}$ (rad)")
    ax.set_title("Phase-margin certificate vs. model and prompt")
    ax.legend(title="prompt", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "phi_min_vs_model.png", dpi=120)
    plt.close(fig)

    # 2. Period-2 purity vs (model, prompt)
    fig, ax = plt.subplots(figsize=(max(6, 1.5 + 1.0 * len(models)), 4))
    for j, p in enumerate(prompts):
        means = []
        stds = []
        for m in models:
            vals = [r.purity_period_2 for r in successful
                    if r.model == m and r.prompt_name == p]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        x = np.arange(len(models)) + j * width - (len(prompts) - 1) * width / 2
        ax.bar(x, means, width=width, yerr=stds, label=p, capsize=3)
    ax.axhline(PURITY_OSC_THRESHOLD, ls="--", color="C3", lw=1, label=f"osc th. ({PURITY_OSC_THRESHOLD})")
    ax.axhline(PURITY_CONT_THRESHOLD, ls="--", color="C2", lw=1, label=f"contr th. ({PURITY_CONT_THRESHOLD:.2f})")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("period-2 within/across purity")
    ax.set_title("Ground-truth purity vs. model and prompt")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "purity_vs_model.png", dpi=120)
    plt.close(fig)

    # 3. Per-direction heatmap (avg over seeds + prompts)
    all_dirs: list[str] = []
    for r in successful:
        for d in r.per_direction_margin.keys():
            if d not in all_dirs:
                all_dirs.append(d)
    if all_dirs:
        H = np.full((len(models), len(all_dirs)), np.nan)
        for i, m in enumerate(models):
            for j, d in enumerate(all_dirs):
                vals = [r.per_direction_margin[d] for r in successful
                        if r.model == m and d in r.per_direction_margin]
                if vals:
                    H[i, j] = float(np.mean(vals))
        fig, ax = plt.subplots(figsize=(2 + 1.2 * len(all_dirs),
                                        1 + 0.8 * len(models)))
        im = ax.imshow(H, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(all_dirs))); ax.set_xticklabels(all_dirs, rotation=20, ha="right")
        ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
        for i in range(len(models)):
            for j in range(len(all_dirs)):
                if not np.isnan(H[i, j]):
                    ax.text(j, i, f"{H[i, j]:+.2f}", ha="center", va="center",
                            color="white" if H[i, j] < np.nanmean(H) else "black",
                            fontsize=9)
        plt.colorbar(im, ax=ax, label=r"$\Phi_v$ (rad), avg over seeds & prompts")
        ax.set_title("Per-direction phase margin")
        fig.tight_layout()
        fig.savefig(out_dir / "per_direction_heatmap.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Run-level skip logic
# ---------------------------------------------------------------------------
def existing_run_record(artefact_dir: Path) -> ExperimentRun | None:
    margin_path = artefact_dir / "margin_report.json"
    gt_path = artefact_dir / "ground_truth.json"
    if not (margin_path.exists() and gt_path.exists()):
        return None
    try:
        margin = json.loads(margin_path.read_text())
        gt = json.loads(gt_path.read_text())
        per_dir = dict(margin.get("per_direction_margin", {}))
        phi_min = min(per_dir.values()) if per_dir else margin["phase_margin"]
        # Prompt and seed are encoded in the dir name
        parts = artefact_dir.name.split("__")
        model = parts[0].replace("_", ":", 1)
        prompt_name = parts[1]
        seed = int(parts[2].replace("seed", ""))
        return ExperimentRun(
            model=model, prompt_name=prompt_name, prompt_text="",
            probe_seed=seed,
            phase_margin=float(margin["phase_margin"]),
            phi_min=float(phi_min),
            per_direction_margin=per_dir,
            predicted_regime=str(margin["regime"]),
            elapsed_certify_s=0.0,
            purity_period_2=float(gt["purity_period_2"]),
            purity_period_3=float(gt["purity_period_3"]),
            purity_p_value_2=float(gt["purity_p_value_2"]),
            period_2_regime=str(gt["period_2_regime"]),
            autocorr_regime=str(gt["autocorr_regime"]),
            autocorr_period_score=float(gt["autocorr_period_score"]),
            autocorr_period_lag=int(gt["autocorr_period_lag"]),
            final_variance=float(gt["final_variance"]),
            growth_rate=float(gt["growth_rate"]),
            elapsed_ground_truth_s=0.0,
            artefact_dir=str(artefact_dir), error="",
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Ollama model tags to evaluate")
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS),
                        choices=list(PROMPTS),
                        help="Prompt categories to use")
    parser.add_argument("--probe-seeds", type=int, default=2,
                        help="Probe seeds per (model, prompt)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_PROBE_CONFIG["horizon"])
    parser.add_argument("--n-frequencies", type=int,
                        default=DEFAULT_PROBE_CONFIG["n_frequencies"])
    parser.add_argument("--epsilon", type=float, default=DEFAULT_PROBE_CONFIG["epsilon"])
    parser.add_argument("--ground-truth-horizon", type=int, default=GROUND_TRUTH_HORIZON)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--quick", action="store_true",
                        help="Single seed, shorter horizons (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run cells whose artefacts already exist")
    parser.add_argument("--results-root", default="results",
                        help="Where the experiment_<timestamp>/ folder is written")
    parser.add_argument("--tag", default="",
                        help="Optional tag appended to the experiment folder name")
    args = parser.parse_args(argv)

    if args.quick:
        args.probe_seeds = 1
        args.horizon = 6
        args.n_frequencies = 3
        args.ground_truth_horizon = 20
        args.n_permutations = 200

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_tag = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.results_root) / f"experiment_{timestamp}{folder_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "log.txt")

    log.info("=" * 76)
    log.info(f"phase-margin extensive validation  v{EXPERIMENT_VERSION}")
    log.info(f"output dir: {out_dir}")
    log.info("=" * 76)
    log.info("models   : " + ", ".join(args.models))
    log.info("prompts  : " + ", ".join(args.prompts))
    log.info(f"seeds    : {args.probe_seeds}")
    log.info(f"horizon  : probe {args.horizon}, ground-truth {args.ground_truth_horizon}")
    log.info(f"freqs    : {args.n_frequencies}")
    log.info(f"epsilon  : {args.epsilon}")
    log.info(f"perm test: {args.n_permutations} permutations")

    save_json({
        "experiment_version": EXPERIMENT_VERSION,
        "timestamp": timestamp,
        "argv": list(sys.argv),
        "args": vars(args),
        "host": host_environment(),
        "prompts": PROMPTS,
        "pre_registered_hypotheses": PRE_REGISTERED_HYPOTHESES,
    }, out_dir / "manifest.json")

    log.info("loading sentence-transformers embedder ...")
    embedder = SentenceTransformersEmbedder()
    log.info(f"  embedder: {embedder.model_name} (dim={embedder.dim})")

    probe_cfg = dict(DEFAULT_PROBE_CONFIG)
    probe_cfg["horizon"] = args.horizon
    probe_cfg["n_frequencies"] = args.n_frequencies
    probe_cfg["epsilon"] = args.epsilon
    save_json(probe_cfg, out_dir / "probe_config.default.json")

    runs: list[ExperimentRun] = []
    t_global = time.time()
    for model in args.models:
        for prompt_name in args.prompts:
            prompt_text = PROMPTS[prompt_name]
            for probe_seed in range(args.probe_seeds):
                tag = f"{model.replace(':','_')}__{prompt_name}__seed{probe_seed}"
                artefact_dir = out_dir / "raw" / tag
                if artefact_dir.exists() and not args.force:
                    rec = existing_run_record(artefact_dir)
                    if rec is not None:
                        log.info(f"  --- skip (already complete): {tag}")
                        runs.append(rec)
                        continue
                t0 = time.time()
                try:
                    run = single_experiment(
                        model=model, prompt_name=prompt_name, prompt_text=prompt_text,
                        probe_seed=probe_seed, embedder=embedder,
                        out_dir=out_dir, probe_cfg=probe_cfg, log=log,
                    )
                    runs.append(run)
                except Exception as e:  # pragma: no cover
                    log.error(f"single experiment crashed: {e}")
                    log.error(traceback.format_exc())
                log.info(f"      cell wall time: {time.time() - t0:.0f}s")

    elapsed = time.time() - t_global
    log.info("=" * 76)
    log.info(f"all cells finished in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Aggregate
    to_csv(runs, out_dir / "summary.csv")
    save_json([asdict(r) for r in runs], out_dir / "summary.json")

    h = evaluate_hypotheses(runs)
    save_json(h, out_dir / "hypotheses.json")
    log.info("hypothesis evaluation:")
    log.info(json.dumps(h, indent=2, default=safe_json_default))

    plot_summary(runs, out_dir / "plots")
    log.info(f"plots written to {out_dir / 'plots'}")

    log.info(