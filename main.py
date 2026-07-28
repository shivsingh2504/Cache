
from __future__ import annotations

import argparse
from pathlib import Path

from simulator.config import DistributionType, WorkloadConfig
from simulator.generator import WorkloadGenerator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic cache access trace and save it to "
            "Parquet."
        )
    )
    parser.add_argument(
        "--num-keys",
        type=int,
        required=True,
        help="Total number of distinct cache keys.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        required=True,
        help="Total number of access events to generate.",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        choices=[member.value for member in DistributionType],
        default=DistributionType.ZIPFIAN.value,
        help="Key-popularity distribution to sample from.",
    )
    parser.add_argument(
        "--zipf-alpha",
        type=float,
        default=1.2,
        help="Zipf skew parameter (ignored for uniform).",
    )
    parser.add_argument(
        "--drift-interval",
        type=int,
        default=10_000,
        help="Requests between hotspot drift events.",
    )
    parser.add_argument(
        "--drift-fraction",
        type=float,
        default=0.1,
        help="Fraction of the key space the hotspot shifts by per drift.",
    )
    parser.add_argument(
        "--burst-prob",
        type=float,
        default=0.0,
        help="Per-request probability of triggering a burst.",
    )
    parser.add_argument(
        "--burst-length-min",
        type=int,
        default=10,
        help="Minimum burst length, in requests.",
    )
    parser.add_argument(
        "--burst-length-max",
        type=int,
        default=100,
        help="Maximum burst length, in requests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/trace.parquet"),
        help="Destination Parquet file path.",
    )
    return parser


def build_config(args: argparse.Namespace) -> WorkloadConfig:
    return WorkloadConfig(
        num_keys=args.num_keys,
        num_requests=args.num_requests,
        distribution=DistributionType(args.distribution),
        zipf_alpha=args.zipf_alpha,
        drift_interval=args.drift_interval,
        drift_fraction=args.drift_fraction,
        burst_prob=args.burst_prob,
        burst_length_range=(args.burst_length_min, args.burst_length_max),
        seed=args.seed,
    )


def print_summary(config: WorkloadConfig, output_path: Path) -> None:
    print("Cache Intelligence System — Workload Generator")
    print("-" * 50)
    print(f"Distribution:      {config.distribution.value}")
    print(f"Num keys:          {config.num_keys:,}")
    print(f"Num requests:      {config.num_requests:,}")
    if config.distribution is not DistributionType.UNIFORM:
        print(f"Zipf alpha:        {config.zipf_alpha}")
    if config.distribution is DistributionType.HOTSPOT_DRIFT:
        print(f"Drift interval:    {config.drift_interval:,}")
        print(f"Drift fraction:    {config.drift_fraction}")
    if config.burst_prob > 0:
        print(f"Burst probability: {config.burst_prob}")
        print(f"Burst length:      {config.burst_length_range}")
    print(f"Seed:              {config.seed}")
    print(f"Saved to:          {output_path}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = build_config(args)
    generator = WorkloadGenerator(config)
    generator.save(args.output)

    print_summary(config, args.output)


if __name__ == "__main__":
    main()