import numpy as np
from random import randrange

gen_random = lambda: randrange(-5,5)

# Heading values (from data)
x = np.arange(0, 10, 0.1)

# Test data
y = np.sin(x)

# Regression parameters (initial estimates)
# Amplitude
a = gen_random()
# Offset
b = gen_random()
# Angular frequency
k = gen_random()
# Parameter vector
w = (a, b, k)

# Hypothesis function
h = lambda v, x_i: v[0]*np.sin(v[2]*x_i) + v[1]

# Cost function to minimize
def regression_loss(w):
    L = 0

    for x_i, y_i in zip(x, y):
        L += (h(w, x_i) - y_i)**2

    return L

# Estimate gradient using central difference
def gradient(f, a):
    # Input steps
    da = 1e-4
    # Gradient vector
    grad = []

    # Take partial derivative wrt each input
    for i in range(len(a)):
        # One-variable differential
        da_i = np.zeros_like(a) + 0.0
        da_i[i] = da

        # Central difference derivative approximation
        f_a = (f(a + da_i) - f(a - da_i))/(2*da)
        # Add this to gradient vector
        grad.append(f_a)

    return np.array(grad)

# Optimize parameters

# Gradient steps
steps = 10000
# Learning rate
lr = 2e-6

for i in range(steps):
    grad = gradient(regression_loss, w)
    w -= lr*grad

    if i % 1000 == 0:
        print('Loss:', regression_loss(w))

print('Final parameters:', w)
