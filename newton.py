

import math




def approximate_derivative(func, value, epsilon):
    return (func(value+epsilon)-func(value))/epsilon


def newtons_method(function, starting_value=0, epsilon=10**(-3), tolerance=10**(-5)):
    """An approximate implementation of Newton's method for univariate functions.

    Args:
        function (function): Function we are optimizing.
        starting_value (int, optional): Starting point for the optimization procedure. Defaults to 0.
        epsilon (_type_, optional): _description_. Step size for approximating the frist and second derivatives. Defaults to 10**(-3).
        tolerance (_type_, optional): _description_. How small the change in x_t-x_{t-1} must be in order to stop procedure. Defaults to 10**(-5).

    Returns:
        _type_: A string containing the interval last_guess +- tolerance to try and estimate the window in which the optimum occurs in.
    """
    
    step = tolerance+1
    current_val = starting_value
    
    while step > tolerance:
        first_derivative = approximate_derivative(function, current_val, epsilon)
        second_derivative = approximate_derivative(lambda x : approximate_derivative(function, x, epsilon), current_val, epsilon)

        next_val = current_val - first_derivative /  second_derivative
        step = abs(next_val-current_val)
        current_val = next_val
    
    

    if second_derivative > 0:
        optima = "minimum"
    else:
        optima = "maximum"
    
    return f"A function {optima} occurs near the interval ({current_val-tolerance}, {current_val+tolerance})!"


def test_newton():
    print("Testing on x^2 starting at x=2.")
    print(newtons_method(lambda x : x**2, starting_value=2))
    
    print()
    print("Testing on sin(x) starting at x=1.}")
    print(newtons_method(math.sin, starting_value=1))

    print()
    print("Testing on sin(sin(x)+cos(x)).")
    print(newtons_method(lambda x : math.sin(math.sin(x)+math.cos(x)), starting_value=1))
    

test_newton()



