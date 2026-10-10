import numpy as np

from imufusion.eval.metrics import inclination_error_deg, summarize

Z = np.array([[0.0, 0.0, 1.0]])


def test_identical_is_zero():
    assert inclination_error_deg(Z, Z)[0] == 0


def test_opposite_is_180():
    assert np.isclose(inclination_error_deg(Z, -Z)[0], 180)


def test_perpendicular_is_90():
    assert np.isclose(inclination_error_deg(Z, np.array([[1.0, 0, 0]]))[0], 90)


def test_summarize_exact_values():
    out = summarize(np.array([3.0, 4.0]))
    assert np.isclose(out["rmse_deg"], np.sqrt(12.5)) and out["max_deg"] == 4.0 and out["mean_deg"] == 3.5


def test_rounding_above_one_gives_no_nan():
    assert not np.isnan(inclination_error_deg(Z * (1 + 1e-12), Z)[0])
