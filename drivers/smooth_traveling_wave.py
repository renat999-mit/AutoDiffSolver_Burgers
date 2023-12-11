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
from auto_diff import Surreal
from numerical_scheme import time_integration
from dynamics import burgers_rhs

def tanh_traveling_wave(x, t, u_l, u_r, nu):
	return (u_r + u_l)/2. - (u_l - u_r)/2.*np.tanh((x - (u_r + u_l)/2.*t)*(u_l - u_r)/(4.*nu))

if __name__ == "__main__":
	# Spatial discretization parameters
	nodes = 150
	x = np.linspace(-10, 10, nodes)
	dx = x[1] - x[0]

	# Analytical solution parameters
	u_l = 1
	u_r = 0
	nu = 0.25

	# Time parameters
	t_0 = -5
	ic = tanh_traveling_wave(x, t_0, u_l, u_r, nu)
	t_f = 5
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

	u_final_order1 = time_integration(t_0, t_f, dt, ic, params, order = 1, 
									name = "Burgers Equation")
	u_final_order2 = time_integration(t_0, t_f, dt, ic, params, order = 2, 
									name = "Burgers Equation")

	final_exact = tanh_traveling_wave(x, t_f, u_l, u_r, nu)
	ax.plot(x, u_final_order1, 'o-', ms = 2, lw = 1, 
		color = 'b', label = f'Numerical BDF1, t = {t_f}')
	ax.plot(x, u_final_order2, 'o-', ms = 2, lw = 1, 
		color = 'g', label = f'Numerical BDF2, t = {t_f}')
	ax.plot(x, final_exact, lw = 1, color = 'r', label = f'Exact, t = {t_f}')
	ax.legend()
	plt.show()