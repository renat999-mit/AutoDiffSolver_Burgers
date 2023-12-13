# 1D Viscous Burgers Equation Solver with Automatic Differentation

Implementation of a finite difference solver for the 1D viscous Burgers equation in advective form. For implicit
time integration with BDF1 and BDF2, Jacobians for Newton solves are computed with in-house automatic differentation.

## Requirements

To run the solver, you will need a Python installation with access to `numpy`, `scipy`, and `matplotlib`.

## Running

There are two driver scripts in `AutoDiffSolver_Burgers/drivers` which you can use to run different cases.

![](https://github.com/renat999-mit/AutoDiffSolver_Burgers/blob/main/figures/bdf_solution.png)
