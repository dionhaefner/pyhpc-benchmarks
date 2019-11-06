# Turbulent kinetic energy benchmark

This is a parameterization for turbulence in large-scale ocean models.

When we model the whole global ocean, every grid cell is orders of magnitude
too large to resolve small-scale turbulence (even in our most costly simulations).
Therefore, we need a *parameterization* that lets us quantify the effect of turbulence
on the large-scale flow without resolving it explicitly.

This routine consists of some finite difference derivatives, but also some more challenging
operations like boolean mask operations and a tridiagonal matrix solver
*which cannot be fully vectorized*.
