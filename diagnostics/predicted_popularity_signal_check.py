"""
diagnostics/predicted_popularity_signal_check.py

READ-ONLY diagnostic. Does NOT modify main.py, RL/agent.py,
RL/agent_double_dqn.py, RL/network.py, RL/environment.py, RL/trainer.py,
RL/replay_buffer.py, RL/config.py, RL/baseline.py, existing checkpoints,
or existing dashboards. Trains nothing. Loads no RL checkpoint.

QUESTION
--------
Does `predicted_popularity` (the LSTM's score at eviction time) carry
useful incremental ranking information beyond raw `frequency`, in the
*actual* cold-start evaluation regime (num_keys=5000, num_requests=5000,
seed=43) -- the same regime the 57.00% LFU / 56.5%-ish DQN numbers come
from?

METHOD
------
We drive the REAL CacheEvictionEnvironment (same construction as the
final benchmark block in main.py: fresh WorkloadGenerator(eval config),
fresh CacheState, fresh Predictor.from_checkpoint, fresh
StateNormalizer) using the existing, unmodified LFUPolicy from
RL/baseline.py as the eviction policy. LFU is used only as a realistic,
deterministic, already-validated driver to produce genuine cache
trajectories -- it is NOT what we are evaluating. We are evaluating the
*feature*, not the policy.

At every eviction decision we snapshot, for every resident slot, the raw
(pre-normalization) features already computed by the real system via
`CacheEvictionEnvironment.raw_state()`:

    frequency, recency, key_age, predicted_popularity

LABELING (no future leakage into any decision)
-----------------------------------------------
For a resident observed at decision t, we ask a purely retrospective
question, answered only from the single realized rollout that already
happened: "did this resident's frequency ever increase again before it
was evicted (or before the workload ended)?" Frequency only increases
while resident via a hit, so:

  - We track one contiguous "residency run" per cache slot: a run
    starts when a key is (re)inserted into a slot and ends the instant
    that slot's action index is chosen for eviction (or the workload
    ends).
  - Within a run, frequency is monotonically non-decreasing (it is
    cumulative hit count while resident).
  - label(entry) = 1 iff some LATER entry in the same run has strictly
    greater frequency than this entry (i.e. at least one more hit
    happened after this decision, before this residency ended).
  - The final entry of every run is always label=0 by construction
    (nothing observed after it in that run) -- this is intentional and
    not a bug: it means "no further hit was recorded before this
    residency ended."

This uses only information that (a) already happened by the end of the
single fixed rollout, and (b) is never fed back into any feature,
policy decision, or the predictor itself. It is a label computed after
the fact for analysis only.

We do NOT touch environment.py's public API surface beyond reading the
`_slot_keys` list (private, read-only introspection -- we do not set
it). This is required because raw_state() and current_state() report
per-slot *features* but not which physical key occupies which slot,
and we need slot identity to track residency runs across decisions.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from simulator.config import WorkloadConfig, DistributionType
from simulator.generator import WorkloadGenerator

from feature.state import CacheState

from lstm.config import ModelConfig
from lstm.predictor import Predictor

from RL.config import EnvironmentConfig
from RL.environment import CacheEvictionEnvironment
from RL.normalizer import StateNormalizer
from RL.baseline import LFUPolicy


# ---------------------------------------------------------------------
# Must match main.py's eval_workload_config / model_config exactly.
# This diagnostic reads that configuration, it does not redefine a new
# evaluation regime. If main.py's eval config ever changes, this must
# be updated to match -- it is deliberately hardcoded (not imported)
# because main.py builds several configs inline in main() and does not
# expose eval_workload_config as an importable constant.
# ---------------------------------------------------------------------
EVAL_WORKLOAD_CONFIG = WorkloadConfig(
    num_keys=5000,
    num_requests=5_000,
    distribution=DistributionType.ZIPFIAN,
    seed=43,
)
MODEL_CONFIG = ModelConfig(event_features=8, candidate_features=3)
LSTM_CHECKPOINT_PATH = "lstm_popularity_predictor.pt"


@dataclass
class ResidentEntry:
    decision_idx: int
    slot: int
    key: int
    frequency: float
    recency: float
    key_age: float
    predicted_popularity: float


def build_benchmark_style_environment() -> CacheEvictionEnvironment:
    """Same construction pattern as main.py's final benchmark_environment:
    fresh workload, fresh cache state, fresh predictor loaded from the
    already-trained LSTM checkpoint, fresh normalizer. No RL checkpoint
    is loaded -- this diagnostic never touches the DQN.
    """
    workload = WorkloadGenerator(EVAL_WORKLOAD_CONFIG)
    cache_state = CacheState()
    predictor = Predictor.from_checkpoint(
        checkpoint_path=LSTM_CHECKPOINT_PATH,
        model_config=MODEL_CONFIG,
        state=cache_state,
    )
    normalizer = StateNormalizer()
    environment_config = EnvironmentConfig()  # defaults match main.py's
    return CacheEvictionEnvironment(
        config=environment_config,
        workload_generator=workload,
        predictor=predictor,
        cache_state=cache_state,
        normalizer=normalizer,
    )


def collect_resident_entries() -> list[dict]:
    """Drive the real environment with the unmodified LFUPolicy and
    return one resolved row per (decision, resident slot) with a
    non-leaking future-hit label. Pure observation: no environment
    method is overridden or monkeypatched, no state is written.
    """
    env = build_benchmark_style_environment()
    policy = LFUPolicy()

    run_buffers: dict[int, list[ResidentEntry]] = {}
    resolved_rows: list[dict] = []

    def finalize_run(buf: list[ResidentEntry], resolution: str) -> None:
        if not buf:
            return
        final_freq = buf[-1].frequency
        for i, entry in enumerate(buf):
            is_last = i == len(buf) - 1
            label = 1 if (not is_last and final_freq > entry.frequency) else 0
            resolved_rows.append(
                {
                    "decision_idx": entry.decision_idx,
                    "slot": entry.slot,
                    "key": entry.key,
                    "frequency": entry.frequency,
                    "recency": entry.recency,
                    "key_age": entry.key_age,
                    "predicted_popularity": entry.predicted_popularity,
                    "label": label,
                    "resolution": resolution,
                }
            )

    def snapshot(decision_idx: int) -> None:
        raw = env.raw_state()
        slot_keys = list(env._slot_keys)  # read-only introspection
        for slot, key in enumerate(slot_keys):
            if key is None:
                continue
            entry = ResidentEntry(
                decision_idx=decision_idx,
                slot=slot,
                key=key,
                frequency=float(raw[slot, 0]),
                recency=float(raw[slot, 1]),
                key_age=float(raw[slot, 2]),
                predicted_popularity=float(raw[slot, 3]),
            )
            run_buffers.setdefault(slot, []).append(entry)

    env.reset()
    decision_idx = 0
    snapshot(decision_idx)

    while not env.done:
        action = policy.select_action(env.current_state())
        env.step(action)
        decision_idx += 1

        # The old resident of `action` was just evicted -- close its run.
        old_run = run_buffers.pop(action, [])
        finalize_run(old_run, resolution="evicted")

        # New state: new key in `action`, updated frequency for all
        # other still-resident slots.
        snapshot(decision_idx)

    # Workload exhausted: anything still resident never got evicted.
    for slot, buf in list(run_buffers.items()):
        finalize_run(buf, resolution="episode_end")

    return resolved_rows


# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------

def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    if _HAVE_SCIPY:
        rho, _ = spearmanr(x, y)
        return float(rho)
    # Fallback: manual Spearman via rank + Pearson, no scipy dependency.
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def eviction_side_quality(rows: list[dict], score_key: str, higher_is_kept: bool) -> dict:
    """Group rows by decision_idx. For each decision's full candidate
    set, simulate 'who would this score evict' (argmin of score if
    higher_is_kept else argmax) and check whether that candidate's
    label was 0 (good: didn't remove a future hit) or 1 (bad: removed
    a resident that would have been hit again).
    """
    by_decision: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_decision[r["decision_idx"]].append(r)

    good, bad, total = 0, 0, 0
    for decision_idx, candidates in by_decision.items():
        if len(candidates) < 2:
            continue
        scores = np.array([c[score_key] for c in candidates])
        idx = int(np.argmin(scores)) if higher_is_kept else int(np.argmax(scores))
        evicted_candidate = candidates[idx]
        total += 1
        if evicted_candidate["label"] == 0:
            good += 1
        else:
            bad += 1
    return {
        "policy": score_key,
        "decisions_evaluated": total,
        "good_evictions": good,
        "bad_evictions": bad,
        "good_eviction_rate": good / total if total else float("nan"),
    }


def topk_hit_rate(rows: list[dict], score_key: str) -> float:
    """Within each decision, does the single HIGHEST-scored candidate
    (by this score) end up with label==1 more often than chance? This
    measures whether the score identifies a resident worth keeping.
    """
    by_decision: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_decision[r["decision_idx"]].append(r)

    hits, total = 0, 0
    for decision_idx, candidates in by_decision.items():
        if len(candidates) < 2:
            continue
        scores = np.array([c[score_key] for c in candidates])
        top_idx = int(np.argmax(scores))
        total += 1
        hits += candidates[top_idx]["label"]
    return hits / total if total else float("nan")


def subset_report(rows: list[dict], predicate, label: str) -> None:
    subset = [r for r in rows if predicate(r)]
    n = len(subset)
    print(f"\n--- Subset: {label} (n={n}) ---")
    if n < 10:
        print("  Too few observations for a reliable correlation.")
        return
    freq = np.array([r["frequency"] for r in subset])
    pop = np.array([r["predicted_popularity"] for r in subset])
    lab = np.array([r["label"] for r in subset])
    base_rate = lab.mean()
    print(f"  base future-hit rate (label==1): {base_rate:.4f}")
    print(f"  Spearman(frequency, label):            {rank_correlation(freq, lab):.4f}")
    print(f"  Spearman(predicted_popularity, label):  {rank_correlation(pop, lab):.4f}")
    print(f"  top-1-by-frequency future-hit rate:            {topk_hit_rate(subset, 'frequency'):.4f}")
    print(f"  top-1-by-predicted_popularity future-hit rate: {topk_hit_rate(subset, 'predicted_popularity'):.4f}")


def main() -> None:
    print("Driving real CacheEvictionEnvironment with unmodified LFUPolicy")
    print(f"eval config: num_keys={EVAL_WORKLOAD_CONFIG.num_keys}, "
          f"num_requests={EVAL_WORKLOAD_CONFIG.num_requests}, "
          f"seed={EVAL_WORKLOAD_CONFIG.seed}")
    print("No RL checkpoint loaded. No training performed. No files written "
          "except this stdout report.\n")

    rows = collect_resident_entries()
    n = len(rows)
    print(f"Collected {n} resolved (decision, resident) observations across "
          f"{len(set(r['decision_idx'] for r in rows))} eviction decisions.\n")

    if n == 0:
        print("No observations collected -- cannot proceed with analysis.")
        sys.exit(1)

    freq_all = np.array([r["frequency"] for r in rows])
    pop_all = np.array([r["predicted_popularity"] for r in rows])
    label_all = np.array([r["label"] for r in rows])

    print("=" * 70)
    print("OVERALL (all frequency levels pooled)")
    print("=" * 70)
    print(f"n = {n}")
    print(f"base future-hit rate (label==1): {label_all.mean():.4f}")
    print(f"Spearman(frequency, label):            {rank_correlation(freq_all, label_all):.4f}")
    print(f"Spearman(predicted_popularity, label):  {rank_correlation(pop_all, label_all):.4f}")

    print("\n--- Eviction-side ranking quality (who would each score evict?) ---")
    for score_key in ("frequency", "predicted_popularity"):
        result = eviction_side_quality(rows, score_key, higher_is_kept=True)
        print(f"  evict-lowest-{score_key}: good_eviction_rate="
              f"{result['good_eviction_rate']:.4f} "
              f"(good={result['good_evictions']}, bad={result['bad_evictions']}, "
              f"n_decisions={result['decisions_evaluated']})")

    # Simple combined ranking: min-max normalize both scores per-decision
    # and rank-evict by their sum. Diagnostic only.
    by_decision: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_decision[r["decision_idx"]].append(r)
    combined_good, combined_bad, combined_total = 0, 0, 0
    for decision_idx, candidates in by_decision.items():
        if len(candidates) < 2:
            continue
        freq = np.array([c["frequency"] for c in candidates])
        pop = np.array([c["predicted_popularity"] for c in candidates])

        def norm(v: np.ndarray) -> np.ndarray:
            span = v.max() - v.min()
            return (v - v.min()) / span if span > 0 else np.zeros_like(v)

        combined_score = norm(freq) + norm(pop)
        idx = int(np.argmin(combined_score))
        combined_total += 1
        if candidates[idx]["label"] == 0:
            combined_good += 1
        else:
            combined_bad += 1
    print(f"  evict-lowest-combined(freq+predicted_popularity): "
          f"good_eviction_rate={combined_good / combined_total:.4f} "
          f"(good={combined_good}, bad={combined_bad}, n_decisions={combined_total})")

    print("\n--- Top-1-by-score future-hit rate (does the single highest-scored "
          "candidate actually get hit again?) ---")
    print(f"  top-1-by-frequency:            {topk_hit_rate(rows, 'frequency'):.4f}")
    print(f"  top-1-by-predicted_popularity: {topk_hit_rate(rows, 'predicted_popularity'):.4f}")

    print("\n" + "=" * 70)
    print("COLD-START SUBSETS (where frequency is least informative)")
    print("=" * 70)
    subset_report(rows, lambda r: r["frequency"] == 0, "frequency == 0")
    subset_report(rows, lambda r: r["frequency"] == 1, "frequency == 1")
    subset_report(rows, lambda r: r["frequency"] <= 2, "frequency <= 2")
    subset_report(rows, lambda r: r["frequency"] > 2, "frequency > 2 (for contrast)")

    print("\n" + "=" * 70)
    print("Observation counts by exact frequency value (cold-start detail)")
    print("=" * 70)
    freq_values, counts = np.unique(freq_all, return_counts=True)
    for v, c in list(zip(freq_values, counts))[:10]:
        cold_subset = [r for r in rows if r["frequency"] == v]
        cold_label_rate = np.mean([r["label"] for r in cold_subset])
        print(f"  frequency={v:.0f}: n={c}, future-hit rate={cold_label_rate:.4f}")

    print(
        "\nThis script only prints a report. It records no verdict itself -- "
        "the four-way verdict (A/B/C/D) should be assigned by reading the "
        "numbers above against the interpretation rule: incremental "
        "predicted_popularity signal in the cold-start subsets "
        "(frequency==0, frequency==1, frequency<=2) is the load-bearing "
        "evidence, not the pooled/overall correlation."
    )


if __name__ == "__main__":
    main()