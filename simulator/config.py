
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DistributionType(Enum):
    UNIFORM = "uniform"
    ZIPFIAN = "zipfian"
    HOTSPOT_DRIFT = "hotspot_drift"


@dataclass(frozen=True)
class WorkloadConfig:
    num_keys: int
    num_requests: int
    distribution: DistributionType = DistributionType.ZIPFIAN
    zipf_alpha: float = 1.2
    drift_interval: int = 10_000
    drift_fraction: float = 0.1
    burst_probability: float = 0.0
    burst_length_range: tuple[int, int] = field(default=(10, 100))
    seed: int = 42

    def __post_init__(self) -> None:
        """Validates all configuration parameters.

        Raises:
            ValueError: If any parameter is out of its allowed range or
                otherwise malformed.
        """
        self._validate_num_keys()
        self._validate_num_requests()
        self._validate_zipf_alpha()
        self._validate_drift_interval()
        self._validate_drift_fraction()
        self._validate_burst_probability()
        self._validate_burst_length_range()

    def _validate_num_keys(self) -> None:
        if self.num_keys <= 0:
            raise ValueError(
                f"num_keys must be > 0, got {self.num_keys}"
            )

    def _validate_num_requests(self) -> None:
        if self.num_requests <= 0:
            raise ValueError(
                f"num_requests must be > 0, got {self.num_requests}"
            )

    def _validate_zipf_alpha(self) -> None:
        if self.zipf_alpha <= 0:
            raise ValueError(
                f"zipf_alpha must be > 0, got {self.zipf_alpha}"
            )

    def _validate_drift_interval(self) -> None:
        if self.drift_interval <= 0:
            raise ValueError(
                f"drift_interval must be > 0, got {self.drift_interval}"
            )

    def _validate_drift_fraction(self) -> None:
        if not (0 < self.drift_fraction <= 1):
            raise ValueError(
                "drift_fraction must be in (0, 1], got "
                f"{self.drift_fraction}"
            )

    def _validate_burst_probability(self) -> None:
        if not (0 <= self.burst_probability <= 1):
            raise ValueError(
                "burst_probability must be in [0, 1], got "
                f"{self.burst_probability}"
            )

    def _validate_burst_length_range(self) -> None:
        low, high = self.burst_length_range
        if low <= 0 or high <= 0:
            raise ValueError(
                "burst_length_range values must be > 0, got "
                f"{self.burst_length_range}"
            )
        if low > high:
            raise ValueError(
                "burst_length_range must satisfy min <= max, got "
                f"{self.burst_length_range}"
            )