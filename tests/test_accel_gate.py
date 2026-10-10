import numpy as np

from imufusion.core.accel_gate import is_dynamic


def test_exact_gravity_is_static():
    assert not is_dynamic(np.array([0, 0, 9.81]), 1.5)


def test_just_above_gate_is_dynamic():
    assert is_dynamic(np.array([0, 0, 9.81 + 1.5 + 1e-6]), 1.5)


def test_uses_three_dimensional_norm():
    assert not is_dynamic(np.array([9.81, 0, 0]), 1.5)


def test_free_fall_is_dynamic():
    assert is_dynamic(np.zeros(3), 1.5)
