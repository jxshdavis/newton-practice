import pytest
import numpy as np
import math
import newton


def test_basic_function():
    assert np.isclose(newton.newtons_method(np.cos, starting_value=2.95)["x"], math.pi)
    print("Basic function test passed")


def test_wrong_function_type():
    with pytest.raises(TypeError, match="Argument is not a function"):
        newton.newtons_method("1", starting_value=2.95)
    print("Wrong function type test passed.")


def test_wrong_starting_point_type():
    with pytest.raises(TypeError, match="Starting value must be a number"):
        newton.newtons_method(np.sin, starting_value="1")
    print("Wrong starting point type test passed.")


def test_diverging_gradient():
    with pytest.raises(
        RuntimeError,
        match="Second derivative is too close to zero, Newton's method may diverge.",
    ):
        newton.newtons_method(lambda x: x**4 / 4 - x**3 - x, starting_value=2)
    print("Divering gradient test passed.")


test_basic_function()
print()
test_wrong_function_type()
print()
test_diverging_gradient()
print()
test_wrong_starting_point_type()
