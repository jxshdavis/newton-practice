

import math




def approximate_derivative(func, value, epsilon):
    return (func(value+epsilon)-func(value))/epsilon


def newtons_method(function, starting_value=0, epsilon=10**(-3), tolerance=10**(-5)):
    
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
    
    return print(f"A function {optima} occurs near the interval ({current_val-tolerance}, {current_val+tolerance})!")



 # testing the function on x^2
print("Testing on x^2")
print(newtons_method(lambda x : x**2, starting_value=2))

print("Sin(x)")
print(newtons_method(math.sin, starting_value=1))

print("Sin(sin+cos)")
print(newtons_method(lambda x : math.sin(math.sin(x)+math.cos(x)), starting_value=1))