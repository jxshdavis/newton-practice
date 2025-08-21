import math
import numdifftools as nd
import numpy as np


def approximate_derivative(func, value, epsilon):
    """Approximates the derivative of a function at a given point using the difference quotient.

    Args:
        func (function): A differentiable real valued single variable function on R
        value (float): point at which to approximate the derivative of the function.
        epsilon (float): A small value to use in the difference quotient to approximate the derivative.

    Returns:
        flot: The approximate value of the derivative of the function at the given point.
    """
    return (func(value + epsilon) - func(value)) / epsilon


def multivar_newtons_method(
    function, starting_value, epsilon=10 ** (-5), tolerance=10 ** (-5)
):
    """An approximate implementation of Newton's method for univariate functions.

    Args:
        function (function): Function we are optimizing.
        starting_value (np.ndarray, optional): Starting point for the optimization procedure
        epsilon (float, optional): _description_. Step size for approximating the frist and second derivatives. Defaults to 10**(-3).
        tolerance (float, optional): _description_. How small the change in x_t-x_{t-1} must be in order to stop procedure. Defaults to 10**(-5).

    Returns:
        _type_: A string containing the interval last_guess +- tolerance to try and estimate the window in which the optimum occurs in.
    """
    if not callable(function):
        raise TypeError(f"Argument is not a function, it is of type {type(function)}")

    # handliing the case where starting value is not a real number, or numpy array
    if not isinstance(starting_value, (int, float, np.ndarray)):
        raise TypeError(
            f"Starting value must be a numpy ndarray it is of type {type(starting_value)}"
        )

    step = tolerance + 1
    current_val = starting_value

    while step > tolerance:
        grad = nd.Gradient(function, step=epsilon)(current_val)
        hess = nd.Hessian(function, step=epsilon)(current_val)
        # check that the hessian has large enough determinant
        if np.linalg.det(hess) < 1e-7:
            raise RuntimeError(
                "Determinant of Hessian is too close to zero, Newton's method may diverge."
            )

        next_val = current_val - np.linalg.inv(hess) @ grad
        step = np.sum((next_val - current_val) ** 2) ** (1 / 2)
        current_val = next_val

    print(f"A function critical value occurs near the interval ({current_val})!")
    return {"x": current_val, "f(x)": function(current_val)}


def newtons_method(f, starting_value, epsilon=10 ** (-5), tolerance=10 ** (-5)):
    if not callable(function):
        raise TypeError(f"Argument is not a function, it is of type {type(function)}")

    # handliing the case where starting value is not a real number
    if not isinstance(starting_value, (int, float)):
        raise TypeError(
            f"Starting value must be a number, it is of type {type(starting_value)}"
        )

    step = tolerance + 1
    current_val = starting_value

    while step > tolerance:
        first_derivative = approximate_derivative(function, current_val, epsilon)
        second_derivative = approximate_derivative(
            lambda x: approximate_derivative(function, x, epsilon), current_val, epsilon
        )

        if abs(second_derivative) < 1e-12:
            raise RuntimeError(
                "Second derivative is too close to zero, Newton's method may diverge."
            )

        next_val = current_val - first_derivative / second_derivative
        step = abs(next_val - current_val)
        current_val = next_val

    if second_derivative > 0:
        optima = "minimum"
    elif second_derivative < 0:
        optima = "maximum"

    print(
        f"A function {optima} occurs near the interval ({current_val - tolerance}, {current_val + tolerance})!"
    )
    return {"x": current_val, "f(x)": function(current_val)}


def test_multivar_newton():
    print("Testing on x^2+y^2 starting at (x,y)=(1,1)")
    print(
        multivar_newtons_method(
            lambda x: x[0] ** 2 + x[1] ** 2, starting_value=np.array([1, 1])
        )
    )


test_multivar_newton()


def test_newton():
    print("Testing on x^2 starting at x=2.")
    print(newtons_method(lambda x: x**2, starting_value=2))

    print()
    print("Testing on sin(x) starting at x=1.}")
    print(newtons_method(math.sin, starting_value=1))

    print()
    print("Testing on sin(sin(x)+cos(x)).")
    print(
        newtons_method(lambda x: math.sin(math.sin(x) + math.cos(x)), starting_value=1)
    )


# print("Testing on x**4/4-x**3-x starting at x=1.")
# print(newtons_method(lambda x: x**4/4-x**3-x, starting_value=1))
# print()
# print("Testing on x**4/4-x**3-x starting at x=100.")
# print(newtons_method(lambda x: x**4/4-x**3-x, starting_value=100))
# print()
# print("Testing on x**4/4-x**3-x starting at x=-100.")
# print(newtons_method(lambda x: x**4/4-x**3-x, starting_value=-100))
# print()
# print("Testing on x**4/4-x**3-x starting at x=1.9989999999999999.")
# print(newtons_method(lambda x: x**4/4-x**3-x, starting_value=2))


# print()
# print("Testing on x**4/4-x**3-x starting at x=3.")
# print(newtons_method(lambda x: x**4/4-x**3-x, starting_value=3))
# print()
# print("Testing on x**4/4-x**3-x starting at x=4.")
# print(newtons_method(lambda x: x**4/4-x**3-x, starting_value=4))
