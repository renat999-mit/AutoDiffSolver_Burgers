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
from numerical_scheme import time_integration_implicit, time_integration_explicit
from dynamics import burgers_rhs
from tools import fit_line
from tools import tanh_traveling_wave
from tools import run_convergence_modified

if __name__ == "__main__":

	single_run = False
	convergence_study = not single_run

	if single_run:
		# Spatial discretization parameters
		nodes = 60
		x = np.linspace(-10, 10, nodes)
		dx = x[1] - x[0]

		# Analytical solution parameters
		u_l = 1
		u_r = 0
		nu = 0.25

		# Time parameters
		t_0 = 0
		ic = tanh_traveling_wave(x, t_0, u_l, u_r, nu)
		t_f = 2
		dt_unstable = 0.2
		dt_stable = 0.1

		fig, ax = plt.subplots()

		ax.plot(x, ic, lw = 1, color = 'k', label = 'Initial condition')

		# Create dictionary with all the information
		params = {}
		params['dx'] = dx
		params['nodes'] = nodes
		params['nu'] = nu
		params['dt'] = dt_unstable
		params['f'] = burgers_rhs

		u_unstable_order2 = time_integration_explicit(t_0, t_f, dt_unstable, ic, params, order = 2, name = "Burgers Equation")
		
		params['dt'] = dt_stable
		u_stable_order2 = time_integration_explicit(t_0, t_f, dt_stable, ic, params, order = 2, name = "Burgers Equation")

		final_exact = tanh_traveling_wave(x, t_f, u_l, u_r, nu)

		ax.plot(x, u_unstable_order2, 'o-', ms = 2, lw = 1, color = 'b', label = f'Numerical RK2, t = {t_f}, dt = {dt_unstable}')
		ax.plot(x, u_stable_order2, 'o-', ms = 2, lw = 1, color = 'g', label = f'Numerical RK2, t = {t_f}, dt = {dt_stable}')
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
		dts_implicit = [0.2, 0.1, 0.05]
		dts_explicit = [4*(0.1/9), 2*(0.1/9), 0.1/9]
		x_points = [60, 120, 240]

		error1, error2, dxs = run_convergence_modified(dts_implicit, dts_explicit, x_points, t_0, t_f, params)

		fig, ax = plt.subplots()
		ax.loglog(dts_implicit, error1, 'o-', lw = 1, ms = 2, label= "BDF2", color = 'r')
		ax.loglog(dts_explicit, error2, 'o-', lw = 1, ms = 2, label= "RK2", color = 'b')

		line1, slope1 = fit_line(dts_implicit, error1)
		line2, slope2 = fit_line(dts_explicit, error2)

		ax.loglog(dts_implicit[-2:], line1[-2:], '--', lw = 1, label = f"m = {slope1:.2f}", color = 'r')
		ax.loglog(dts_explicit[-2:], line2[-2:], '--', lw = 1, label = f"m = {slope2:.2f}", color = 'b')

		ax.set_xlabel('dt')
		ax.set_ylabel('Error')
		ax.legend()
		plt.show()
