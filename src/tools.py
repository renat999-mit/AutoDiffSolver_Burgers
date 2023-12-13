"""
File containing tools, such as analytic solution,
driver for convergence plots and plotting tools
"""

import numpy as np
import time
from dynamics import burgers_rhs
from numerical_scheme import time_integration_implicit
from numerical_scheme import time_integration_explicit

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

# Function to fit a line in loglog scale
def fit_line(dts, errors):
	slope = (np.log(errors[-1]) - np.log(errors[-2])) / (np.log(dts[-1]) - np.log(dts[-2]))
	intercept = np.log(errors[-1]) - slope * np.log(dts[-1])

	offset = 1e-1
	fitted_line = np.exp(slope * np.log(dts) + intercept + offset)

	return fitted_line, slope

# Function to perform convergence study
def run_convergence_modified(dts_implicit, dts_explicit, x_points, t_0, t_f, params):

	error1 = np.zeros(len(dts_implicit))
	error2 = np.zeros(len(dts_explicit))
	dxs = np.zeros(len(dts_implicit))
	for i in range(len(dts_implicit)):
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
		dt = dts_implicit[i]

		# Create dictionary with all the information
		params['dx'] = dx
		params['nodes'] = nodes
		params['dt'] = dt
		params['f'] = burgers_rhs

		time_0 = time.perf_counter()

		u_implicit_order2 = time_integration_implicit(t_0, t_f, dt, ic, params, order = 2, name = f"Burgers Equation, dx = {dx}, dt = {dt}", use_sparse_jac=True)
		
		time_1 = time.perf_counter()

		print(f'Implicit time = {time_1 - time_0} s')

		dt = dts_explicit[i]
		params['dt'] = dt
		
		time_0 = time.perf_counter()

		u_explicit_order2 = time_integration_explicit(t_0, t_f, dt, ic, params, order = 2, name = f"Burgers Equation, dx = {dx}, dt = {dt}")

		time_1 = time.perf_counter()

		print(f'Explicit time = {time_1 - time_0} s')

		final_exact = tanh_traveling_wave(x, t_f, u_l, u_r, nu)
		error1[i] = np.sqrt(dx)*np.linalg.norm(final_exact - u_implicit_order2, 2)
		error2[i] = np.sqrt(dx)*np.linalg.norm(final_exact - u_explicit_order2, 2)
		dxs[i] = dx

	return error1, error2, dxs