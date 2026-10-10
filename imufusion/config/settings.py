from dataclasses import dataclass


@dataclass(frozen=True)
class GateConfig:
    use_gate: bool
    gate_ms2: float
    kalman_r_inflation: float


@dataclass(frozen=True)
class ComplementaryConfig:
    tau_s: float


@dataclass(frozen=True)
class KalmanConfig:
    q_gravity: float
    q_bias: float
    r_measurement: float
    p0_gravity: float
    p0_bias: float


@dataclass(frozen=True)
class FilterConfig:
    gate: GateConfig
    complementary: ComplementaryConfig
    kalman: KalmanConfig


# Values chosen by imufusion.eval.tune on the tuning trials; see results/tuning.json.
# Kalman p0 values and kalman_r_inflation were not tuned.
DEFAULT_CONFIG = FilterConfig(
    gate=GateConfig(use_gate=True, gate_ms2=3.0, kalman_r_inflation=1000.0),
    complementary=ComplementaryConfig(tau_s=1.0),
    kalman=KalmanConfig(q_gravity=1e-8, q_bias=1e-10, r_measurement=1e-2, p0_gravity=1e-2, p0_bias=1e-2),
)
