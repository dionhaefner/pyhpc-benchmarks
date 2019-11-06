# Isoneutral mixing benchmark

This routine computes the mixing tensors that we need to simulate mixing in the ocean.

In the ocean, mixing takes place along surfaces of constant neutral density. At every
time step, we need to figure out where these surfaces lie in 3D space. This is usually
the single most costly operation in an ocean model.

Numerically, this routine consists of many finite difference derivatives and grid
interpolations. There are many horizontal data dependencies and some arrays have
up to 5 dimensions, but still only elementary math, and everything is vectorized.
