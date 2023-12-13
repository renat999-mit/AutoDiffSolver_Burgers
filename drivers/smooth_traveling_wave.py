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
from tools import run_convergence
from tools import fit_line
from tools import tanh_traveling_wave

if __name__ == "__main__":

	single_run = True
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
		t_0 = -7.5
		ic = tanh_traveling_wave(x, t_0, u_l, u_r, nu)
		t_f = 7.5
		dt = 0.5

		fig, ax = plt.subplots()

		ax.plot(x, ic, lw = 1, color = 'k', label = f'Initial condition, t_0 = {t_0}')

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
		ax.set_xlabel('x')
		ax.set_ylabel('u')
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
