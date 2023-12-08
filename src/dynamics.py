"""
In this file we have the RHSs for different
Eqs. we might want to solve (for now just Burgers Eq.).

This is, each function defines a different f
in the equation: du/dt = f
"""

import numpy as np
from auto_diff import Surreal

def burgers_rhs(u_vector,
                params
               ):
    """
    Evaluates RHS vector f of Burgers Eq: du/dt = f

    f here comes from the spatial discretization
    of the Burgers Eq. using second order finite difference
    with upwind scheme for the advective term
    """
    dx = params['dx']
    nodes = params['nodes']
    nu = params['nu']

    if (isinstance(u_vector[0], Surreal)):
        rhs = np.zeros(nodes, dtype = object)
    else:
        rhs = np.zeros(nodes)

    for i in range(1, nodes - 1):  # loop over interior nodes

      # Apply appropriate scheme based on advection velocity direction

      if i == 1:
          if u_vector[i] >= 0:
              firstorderup = u_vector[i] * (-u_vector[i] + u_vector[i-1]) / dx
              rhs[i] = firstorderup
          else:
              secondorderup = u_vector[i] * ( 3 * u_vector[i] - 4 * u_vector[i+1] + u_vector[i+2]) / (2 * dx)
              rhs[i] = secondorderup

      elif i == nodes - 2:
          if u_vector[i] >= 0:
              secondorderup = u_vector[i] * (-3 * u_vector[i] + 4 * u_vector[i-1] - u_vector[i-2]) / (2 * dx)
              rhs[i] = secondorderup
          else:
              firstorderup = u_vector[i] * (u_vector[i+1] - u_vector[i]) / dx
              rhs[i] = firstorderup

      else:
          if u_vector[i] >= 0:
              secondorderup = u_vector[i] * (-3 * u_vector[i] + 4 * u_vector[i-1] - u_vector[i-2]) / (2 * dx)
          else:
              secondorderup = u_vector[i] * (3 * u_vector[i] - 4 * u_vector[i+1] + u_vector[i+2]) / (2 * dx)
          rhs[i] = secondorderup

      # Diffusion Term
      diffusion = nu * (u_vector[i+1] - 2*u_vector[i] + u_vector[i-1]) / (dx**2)

      rhs[i] += diffusion

    return rhs

