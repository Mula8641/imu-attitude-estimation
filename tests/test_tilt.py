import numpy as np
import pytest

from imufusion.core.tilt import accel_unit, gravity_from_roll_pitch, propagate_gravity, roll_pitch_from_accel


def test_accel_unit_of_flat_sensor():
    assert np.allclose(accel_unit(np.array([[0, 0, 9.81]])), [[0, 0, 1]])


def test_accel_unit_zero_raises():
    with pytest.raises(ValueError):
        accel_unit(np.zeros((2, 3)))


def test_roll_pitch_flat_is_zero():
    assert np.allclose(roll_pitch_from_accel(np.array([[0, 0, 9.81]])), 0)


def test_roll_ninety_degrees():
    assert np.isclose(roll_pitch_from_accel(np.array([[0, 9.81, 0]]))[0, 0], np.pi / 2)


def test_roll_pitch_round_trip():
    angles = np.random.default_rng(0).uniform(-np.radians(60), np.radians(60), (200, 2))
    assert np.allclose(roll_pitch_from_accel(gravity_from_roll_pitch(angles)), angles)


def test_propagate_matches_closed_form_rotation():
    omega, total_s, steps = 0.5, 2.0, 20000
    g = np.array([0.0, 0.0, 1.0])
    for _ in range(steps):
        g = propagate_gravity(g, np.array([omega, 0.0, 0.0]), total_s / steps)
    theta = omega * total_s
    # rotating the body by +theta about x moves a world-fixed vector by -theta
    assert np.allclose(g, [0.0, np.sin(theta), np.cos(theta)], atol=1e-3)


def test_propagate_returns_unit_vector():
    g = propagate_gravity(np.array([0.0, 0.6, 0.8]), np.array([3.0, -2.0, 1.0]), 0.05)
    assert np.isclose(np.linalg.norm(g), 1.0)
