import numpy as np
import pytest

from imufusion.data.quaternion import gravity_in_sensor_from_quaternion as up

IDENTITY = np.array([[1.0, 0, 0, 0]])
ROT_X_90 = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]])


@pytest.mark.parametrize("rotates", ["sensor_to_world", "world_to_sensor"])
def test_identity_gives_world_up(rotates):
    assert np.allclose(up(IDENTITY, "wxyz", rotates), [[0, 0, 1]])


def test_rot_x_90_sensor_to_world_gives_plus_y():
    assert np.allclose(up(ROT_X_90, "wxyz", "sensor_to_world"), [[0, 1, 0]])


def test_rot_x_90_world_to_sensor_gives_minus_y():
    assert np.allclose(up(ROT_X_90, "wxyz", "world_to_sensor"), [[0, -1, 0]])


def test_wxyz_and_xyzw_agree():
    q = np.array([[0.5, 0.5, -0.5, 0.5]])
    assert np.allclose(up(q, "wxyz", "sensor_to_world"), up(q[:, [1, 2, 3, 0]], "xyzw", "sensor_to_world"))


def test_unknown_arguments_raise():
    with pytest.raises(ValueError):
        up(IDENTITY, "abcd", "sensor_to_world")
    with pytest.raises(ValueError):
        up(IDENTITY, "wxyz", "sideways")
