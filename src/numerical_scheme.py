"""
Functions related to the numerical time integration scheme
"""

import numpy as np
from auto_diff import Surreal

def residual_order1(u_n1, params):

	"""
	Compute the residual for a first-order backward difference formula (BDF1).

	Parameters:
	u_n1 (array): The current solution estimate at time step n+1.
	params (dict): A dictionary containing parameters like the previous solution 'u_n', 
				time step 'dt', and the function 'f' representing the RHS of the ODE.

	Returns:
	array: The residual of the BDF1 scheme for the given estimate u_n1.
	"""
	u_n = params['u_n']
	dt = params['dt']
	f = params['f']

	return u_n1 - u_n - dt * f(u_n1, params)

def residual_order2(u_n1, params):

	"""
	Compute the residual for a second-order backward difference formula (BDF2).

	Parameters:
	u_n1 (array): The current solution estimate at time step n+1.
	params (dict): A dictionary containing parameters like the previous solutions 'u_n' and 'u_nm1', 
					time step 'dt', and the function 'f' representing the RHS of the ODE.

	Returns:
	array: The residual of the BDF2 scheme for the given estimate u_n1.
	"""
	u_n = params['u_n']
	u_nm1 = params['u_nm1']
	dt = params['dt']
	f = params['f']

	return u_n1 - (4./3.)*u_n + (1./3.)*u_nm1 - (2./3.)*dt*f(u_n1, params)

def eval_jacobian(u,res_func,params):

	"""
	Compute the Jacobian matrix of a residual function at a given point.

	Parameters:
	u (array): The point at which to evaluate the Jacobian.
	res_func (function): The residual function for which the Jacobian is to be computed.
	params (dict): Additional parameters required by the residual function.

	Returns:
	ndarray: The Jacobian matrix of the residual function evaluated at 'u'.
	"""

	n = len(u)

	jacobian = np.zeros((n, n))

	# Iterate over each element of u
	for i in range(n):

		# Create a modified vector with the i-th element derivative part set to 1 (so effectively a Surreal)
		u_modified = [Surreal(u[j], (1 if j == i else 0)) for j in range(n)]

		# Evaluate the function
		func_evaluated = res_func(u_modified, params)

		# Extract the derivative parts for the Jacobian column
		for j in range(n):
			jacobian[j, i] = func_evaluated[j].derivative

	return jacobian

def newton_method(u_n, residual_func, params, tol = 1e-10, max_iter = 50, verbose = False):
	"""
    Perform the Newton-Raphson method to find a root of the given residual function.

    Parameters:
    u_n (array): Initial guess for the root.
    residual_func (function): The residual function whose root is to be found.
    params (dict): Parameters required by the residual function.
    tol (float, optional): Tolerance for convergence. Defaults to 1e-10.
    max_iter (int, optional): Maximum number of iterations. Defaults to 50.
    verbose (bool, optional): Flag to enable verbose output. Defaults to False.

    Returns:
    array: The solution (root) of the residual function.
    """
	it = 0
	while it <= max_iter:
		if verbose:
			print(f"it = {it}")
		
		# Evaluate residual
		res = residual_func(u_n, params)
		if np.linalg.norm(res) < tol:
			return u_n
		
		# Evaluate Jacobian
		jac = eval_jacobian(u_n, residual_func, params)

		# Compute delta_u and update guess
		delta_u = np.linalg.solve(jac, -res)
		u_n = u_n + delta_u

		it += 1

	assert(0), "Newton method failed"

def time_integration(t_0, t_f, dt, ic, params, order: int = 1, name: str = None, verbose = False, time_history = False):
	"""
    Perform time integration using either first or second order backward difference formula.

    Parameters:
    t_0 (float): Initial time.
    t_f (float): Final time.
    dt (float): Time step.
    ic (array): Initial condition.
    params (dict): Parameters required for the integration process.
    order (int, optional): Order of the numerical integration (1 for BDF1, 2 for BDF2). Defaults to 1.
    name (str, optional): Name of the case being solved. Used for printing purposes. Defaults to None.
    verbose (bool, optional): Flag to enable verbose output. Defaults to False.
    time_history (bool, optional): Flag to return the entire time history. Defaults to False.

    Returns:
    array: The solution at the final time if time_history is False, otherwise the entire time history.
    """
	if name:
		print(f"Solving case: {name}")
	print(f"Integrating in time with order = {order}")

	def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
		"""
		Small function to print a progress bar based on the time step
		"""
		percent = ("{0:.1f}").format(100 * (iteration / float(total)))
		filled_length = int(length * iteration // total)
		bar = fill * filled_length + '-' * (length - filled_length)
		print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    
		# Print New Line on Complete
		if iteration == total: 
			print()
	
	if verbose:
		print(f"t = {t_0}\n")
          
	time_steps = int((t_f - t_0) / dt)

	# Initialize time history if needed
	if time_history:
		history = np.zeros((len(ic), time_steps + 1))
		history[:, 0] = ic

	# Take first step using first order scheme
	params['u_n'] = ic.copy()
	u_1 = newton_method(ic, residual_order1, params, verbose = verbose)
	if time_history:
		history[:, 1] = u_1
		
	t = t_0 + dt
	if verbose:
		print(f"t = {t}\n")
	
	# Complete the time integration using first order scheme
	if order == 1:
		params['u_n'] = u_1.copy()
		for i in range(2, time_steps + 1):
			u_n1 = newton_method(params['u_n'], residual_order1, params, verbose = verbose)
			if time_history:
				history[:, i] = u_n1
			t += dt
			if verbose:
				print(f"t = {t}\n")
			params['u_n'] = u_n1.copy()

			# Print the progress bar
			print_progress_bar(i, time_steps, prefix='Progress:', suffix='Complete', length=50)

	# Complete the time integration using second order scheme	
	elif order == 2:       
		params['u_n'] = u_1.copy()
		params['u_nm1'] = ic.copy()

		for i in range(2, time_steps + 1):
			u_n1 = newton_method(params['u_n'], residual_order2, params, verbose = verbose)
			if time_history:
				history[:, i] = u_n1
			t += dt
			if verbose:
				print(f"t = {t}\n")
			params['u_nm1'] = params['u_n'].copy()
			params['u_n'] = u_n1.copy()

			# Print the progress bar
			print_progress_bar(i, time_steps, prefix='Progress:', suffix='Complete', length=50)
	else:
		assert(0), "Order selected not implemented"         
	if time_history:
		return history
	else:
		return params['u_n']
