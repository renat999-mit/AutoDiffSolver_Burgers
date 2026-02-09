"""
Driver for minimal test of 1D Burgers Equation Solver
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
import argparse
from numerical_scheme import time_integration_implicit
from dynamics import burgers_rhs
from tools import tanh_traveling_wave

if __name__ == "__main__":

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

	# Create dictionary with all the information
	params = {}
	params['dx'] = dx
	params['nodes'] = nodes
	params['nu'] = nu
	params['dt'] = dt
	params['f'] = burgers_rhs

	u_final_order1 = time_integration_implicit(
		t_0, t_f, dt, ic, params,
		order = 1, name = f"Burgers Equation, dx = {dx:e}, dt = {dt:e}"
	)
	u_final_order2 = time_integration_implicit(
		t_0, t_f, dt, ic, params,
		order = 2, name = f"Burgers Equation, dx = {dx:e}, dt = {dt:e}"
	)

	final_exact = tanh_traveling_wave(x, t_f, u_l, u_r, nu)

	error_order1 = np.sqrt(dx)*np.linalg.norm(final_exact - u_final_order1, 2)
	error_order2 = np.sqrt(dx)*np.linalg.norm(final_exact - u_final_order2, 2)

	print('----------------------------------------')
	print(f'Scaled l^2 error for BDF1 = {error_order1:e}')
	print(f'Scaled l^2 error for BDF2 = {error_order2:e}')
	print('----------------------------------------')
