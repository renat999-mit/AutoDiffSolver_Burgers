import numpy as np
from auto_diff import Surreal

def rhs(u_vector,
        params
       ):
  """
  Evaluates RHS vector du/dt = f

  f here comes from the spatial discretization
  of the Burgers Eq.  using finite difference
  """
  dx = params['dx']
  nodes = params['nodes']
  nu = params['nu']

  rhs = np.zeros(nodes)

  for i in range(1, nodes - 1):  # loop over interior nodes

    # Apply appropriate scheme based on advection velocity direction

    if i == 1:
      if u_vector[i] >= 0:
          firstorderup = u_vector[i] * (u_vector[i] - u_vector[i-1]) / dx
          rhs[i] = -firstorderup
      else:
          secondorderup = u_vector[i] * (3 * u_vector[i] - 4 * u_vector[i+1] + u_vector[i+2]) / (2 * dx)
          rhs[i] = -secondorderup

    elif i == nodes - 2:
      if u_vector[i] >= 0:
          secondorderup = u_vector[i] * (-3 * u_vector[i] + 4 * u_vector[i-1] - u_vector[i-2]) / (2 * dx)
          rhs[i] = -secondorderup
      else:
          firstorderup = u_vector[i] * (u_vector[i+1] - u_vector[i]) / dx
          rhs[i] = -firstorderup

    else:
      if u_vector[i] >= 0:
        secondorderup = u_vector[i] * (-3 * u_vector[i] + 4 * u_vector[i-1] - u_vector[i-2]) / (2 * dx)
      else:
        secondorderup = u_vector[i] * (3 * u_vector[i] - 4 * u_vector[i+1] + u_vector[i+2]) / (2 * dx)
      rhs[i] = -secondorderup

    # Diffusion Term
    diffusion = nu * (u_vector[i+1] - 2*u_vector[i] + u_vector[i-1]) / (dx**2)

    rhs[i] += diffusion

  return rhs

def residual_1order(u_n1,
                    params
                   ):

  """
  Evaluate residual for BDF1: u_n1 - u_n - dt * f
  """
  u_n = params['u_n']
  dt = params['dt']
  f = params['f']

  return u_n1 - u_n - dt * f(u_n1, params)

def residual_2order(u_n1,
                    params
                   ):

  """
  Evaluate residual for BDF2: u_n1 - 4/3 * u_n + 1/3 * u_nm1 - 2/3 * dt * f
  """
  u_n = params['u_n']
  u_nm1 = params['u_nm1']
  dt = params['dt']
  f = params['f']

  return u_n1 - (4./3.)*u_n + (1./3.)*u_nm1 - (2./3.)*dt*f(u_n1, params)

def eval_jacobian(u,
                  func,
                  params
                 ):

  """
  Computes Jacobian of func at u
  """

  n = len(u)

  jacobian = np.zeros((n, n))

  # Iterate over each element of u
  for i in range(n):

    # Create a modified vector with the i-th element derivative part set to 1 (so effectively a Surreal)
    u_modified = [Surreal(u[j].value, (1 if j == i else 0)) for j in range(n)]

    # Evaluate the function
    func_evaluated = func(u_modified, params)

    # Extract the derivative parts for the Jacobian column
    for j in range(n):
        jacobian[j, i] = func_evaluated[j].derivative

  return jacobian