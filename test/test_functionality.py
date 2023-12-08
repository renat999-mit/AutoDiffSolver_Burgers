"""
Unit test file to test all the functionality on a simple case,
specially the Jacobian computation.

In this case, we use the Lotka-Volterra Equations, which
simulates the dynamics of prey/predator population
"""

import sys
import os

# Path to the src directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')

# Add the src directory to the Python path
sys.path.append(src_path)

import numpy as np
import matplotlib.pyplot as plt
from auto_diff import Surreal
from numerical_scheme import eval_jacobian
from numerical_scheme import residual_order1
from numerical_scheme import time_integration

# Lotka-Volterra equations
def f(x_vec, params):
	if (isinstance(x_vec[0], Surreal)):
		result = np.zeros(2, dtype=object)
	else:
		result = np.zeros(2)
	alpha = params['alpha']
	beta = params['beta']
	delta = params['delta']
	gamma = params['gamma']

	result[0] = alpha * x_vec[0] - beta * x_vec[0] * x_vec[1]
	result[1] = delta * x_vec[0] * x_vec[1] - gamma * x_vec[1]

	return result

# Define the analytical Jacobian of the residual function
def jacobian_residual_order1(u_n1, params):
    x_n1, y_n1 = u_n1
    alpha = params['alpha']
    beta = params['beta']
    delta = params['delta']
    gamma = params['gamma']
    dt = params['dt']
    J11 = 1 - dt * (alpha - beta * y_n1)
    J12 = -dt * (-beta * x_n1)
    J21 = -dt * (delta * y_n1)
    J22 = 1 - dt * (delta * x_n1 - gamma)
    return np.array([[J11, J12], [J21, J22]])

alpha = 1.1
beta = 0.4
delta = 0.1
gamma = 0.4
params = {}
params['alpha'] = alpha
params['beta'] = beta
params['delta'] = delta
params['gamma'] = gamma
params['dt'] = 0.01

# Test jacobian of RHS function (f)
for _ in range(100):
	x = np.random.uniform(-100,100)
	y = np.random.uniform(-100,100)
	x_vec = np.array([x,y])
	jac = eval_jacobian(x_vec, f, params)
	true_jac = np.array([[alpha - beta * y, -beta * x],[delta * y, delta * x - gamma]])
	assert(jac[0,0] == true_jac[0,0]), f"jac[0,0] failed, (x,y) = ({x},{y}), jac[0,0] = {jac[0,0]}, true = {true_jac[0,0]}"
	assert(jac[0,1] == true_jac[0,1]), f"jac[0,1] failed, (x,y) = ({x},{y}), jac[0,1] = {jac[0,1]}, true = {true_jac[0,1]}"
	assert(jac[1,0] == true_jac[1,0]), f"jac[1,0] failed, (x,y) = ({x},{y}), jac[1,0] = {jac[1,0]}, true = {true_jac[1,0]}"
	assert(jac[1,1] == true_jac[1,1]), f"jac[1,1] failed, (x,y) = ({x},{y}), jac[1,1] = {jac[1,1]}, true = {true_jac[1,1]}"
     
# Test jacobian of residual function (order 1)
for _ in range(100):
	np.random.seed(0)
	x = np.random.uniform(-100,100)
	y = np.random.uniform(-100,100)
	x_vec = np.array([x,y])
	params['u_n'] =  x_vec.copy()
	params['f'] = f
	jac = eval_jacobian(x_vec, residual_order1, params)
	true_jac = jacobian_residual_order1(x_vec, params)
	assert(jac[0,0] == true_jac[0,0]), f"jac[0,0] failed, (x,y) = ({x},{y}), jac[0,0] = {jac[0,0]}, true = {true_jac[0,0]}"
	assert(jac[0,1] == true_jac[0,1]), f"jac[0,1] failed, (x,y) = ({x},{y}), jac[0,1] = {jac[0,1]}, true = {true_jac[0,1]}"
	assert(jac[1,0] == true_jac[1,0]), f"jac[1,0] failed, (x,y) = ({x},{y}), jac[1,0] = {jac[1,0]}, true = {true_jac[1,0]}"
	assert(jac[1,1] == true_jac[1,1]), f"jac[1,1] failed, (x,y) = ({x},{y}), jac[1,1] = {jac[1,1]}, true = {true_jac[1,1]}"

# Set parameters and initial conditions
params = {'alpha': 1.1, 'beta': 0.4, 'delta': 0.1, 'gamma': 0.4, 'f': f}
ic = np.array([40, 9])
t_0 = 0
t_f = 100
dt = 0.001
params['dt'] = dt

# Perform time integration
history = time_integration(t_0, t_f, dt, ic, params, 2, time_history = True)
T = np.linspace(t_0, t_f, int((t_f - t_0) / dt) + 1)

# Plotting
plt.plot(T, history[0, :], label='Prey (x)')
plt.plot(T, history[1, :], label='Predator (y)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Lotka-Volterra Predator-Prey Model')
plt.legend()
plt.show()
