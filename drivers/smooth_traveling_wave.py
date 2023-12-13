"""
Driver for solving Burgers Equation
du/dt = - u * du/dx + nu * d^2u/dx^2
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
from numerical_scheme import time_integration_implicit
from dynamics import burgers_rhs

# Define analytical solution
def tanh_traveling_wave(x, t, u_l, u_r, nu):
	return (u_r + u_l)/2. - (u_l - u_r)/2.*np.tanh((x - (u_r + u_l)/2.*t)*(u_l - u_r)/(4.*nu))

# Function to perform convergence study
def run_convergence(dts, x_points, t_0, t_f, params):

	error1 = np.zeros(len(dts))
	error2 = np.zeros(len(dts))
	dxs = np.zeros(len(dts))
	for i in range(len(dts)):
		# Spatial discretization parameters
		nodes = x_points[i]
		x = np.linspace(-10, 10, nodes)
		dx = x[1] - x[0]

		# Analytical solution parameters
		u_l = params['u_l']
		u_r = params['u_r']
		nu = params['nu']

		# Time parameters
		ic = tanh_traveling_wave(x, t_0, u_l, u_r, nu)
		dt = dts[i]

		# Create dictionary with all the information
		params['dx'] = dx
		params['nodes'] = nodes
		params['dt'] = dt
		params['f'] = burgers_rhs

		u_final_order1 = time_integration_implicit(t_0, t_f, dt, ic, params, order = 1, name = f"Burgers Equation, dx = {dx}, dt = {dt}")
		u_final_order2 = time_integration_implicit(t_0, t_f, dt, ic, params, order = 2, name = f"Burgers Equation, dx = {dx}, dt = {dt}")

		final_exact = tanh_traveling_wave(x, t_f, u_l, u_r, nu)
		error1[i] = np.sqrt(dx)*np.linalg.norm(final_exact - u_final_order1, 2)
		error2[i] = np.sqrt(dx)*np.linalg.norm(final_exact - u_final_order2, 2)
		dxs[i] = dx

	return error1, error2, dxs

def fit_line(dts, errors):
	slope = (np.log(errors[-1]) - np.log(errors[-2])) / (np.log(dts[-1]) - np.log(dts[-2]))
	intercept = np.log(errors[-1]) - slope * np.log(dts[-1])

	offset = 1e-1
	fitted_line = np.exp(slope * np.log(dts) + intercept + offset) 

	return fitted_line, slope


if __name__ == "__main__":

	single_run = False
	convergence_study = not single_run

	if single_run:
		# Spatial discretization parameters
		nodes = 100
		x = np.linspace(-10, 10, nodes)
		dx = x[1] - x[0]

		# Analytical solution parameters
		u_l = 1
		u_r = 0
		nu = 0.25

		# Time parameters
		t_0 = 0
		ic = tanh_traveling_wave(x, t_0, u_l, u_r, nu)
		t_f = 3
		dt = 0.1

		fig, ax = plt.subplots()

		ax.plot(x, ic, lw = 1, color = 'k', label = 'Initial condition')

		# Create dictionary with all the information
		params = {}
		params['dx'] = dx
		params['nodes'] = nodes
		params['nu'] = nu
		params['dt'] = dt
		params['f'] = burgers_rhs

		u_final_order1 = time_integration_implicit(t_0, t_f, dt, ic, params, order = 1, name = "Burgers Equation")
		u_final_order2 = time_integration_implicit(t_0, t_f, dt, ic, params, order = 2, name = "Burgers Equation")

		final_exact = tanh_traveling_wave(x, t_f, u_l, u_r, nu)
		ax.plot(x, u_final_order1, 'o-', ms = 2, lw = 1, color = 'b', label = f'Numerical BDF1, t = {t_f}')
		ax.plot(x, u_final_order2, 'o-', ms = 2, lw = 1, color = 'g', label = f'Numerical BDF2, t = {t_f}')
		ax.plot(x, final_exact, lw = 1, color = 'r', label = f'Exact, t = {t_f}')
		ax.legend()
		plt.show()

	if convergence_study:
		# Analytical solution parameters
		u_l = 1
		u_r = 0
		nu = 0.25
		params = {}
		params['u_l'] = u_l
		params['u_r'] = u_r
		params['nu'] = nu

		t_0 = 0
		t_f = 2
		dts = [0.2, 0.1, 0.05]
		x_points = [60, 120, 240]

		error1, error2, dxs = run_convergence(dts, x_points, t_0, t_f, params)

		fig, ax = plt.subplots()
		ax.loglog(dts, error1, 'o-', lw = 1, ms = 2, label= "BDF1", color = 'r')
		ax.loglog(dts, error2, 'o-', lw = 1, ms = 2, label= "BDF2", color = 'b')

		line1, slope1 = fit_line(dts, error1)
		line2, slope2 = fit_line(dts, error2)

		ax.loglog(dts[-2:], line1[-2:], '--', lw = 1, label = f"m = {slope1:.2f}", color = 'r')
		ax.loglog(dts[-2:], line2[-2:], '--', lw = 1, label = f"m = {slope2:.2f}", color = 'b')

		ax.set_xlabel('dt')
		ax.set_ylabel('Error')
		ax.legend()
		plt.show()
