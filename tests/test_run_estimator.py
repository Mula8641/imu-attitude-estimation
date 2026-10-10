import numpy as np
import pytest

from imufusion.config.settings import DEFAULT_CONFIG
from imufusion.core.run_estimator import ESTIMATORS, run_estimator
from tests.conftest import make_trial


@pytest.mark.parametrize("name", ESTIMATORS)
def test_returns_unit_vectors(name):
    series, _ = make_trial(duration_s=5, accel_noise=0.1)
    out = run_estimator(name, series, DEFAULT_CONFIG)
    assert out.shape == (500, 3) and np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_unknown_name_raises():
    series, _ = make_trial(duration_s=1)
    with pytest.raises(ValueError):
        run_estimator("magic", series, DEFAULT_CONFIG)
