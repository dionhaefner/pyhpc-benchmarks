import jax
import jax.numpy as np


@jax.jit
def where(mask, a, b):
    return np.where(mask, a, b)


@jax.jit
def solve_implicit(ks, a, b, c, d, b_edge=None, d_edge=None):
    land_mask = (ks >= 0)[:, :, np.newaxis]
    edge_mask = land_mask & (
        np.arange(a.shape[2])[np.newaxis, np.newaxis, :] == ks[:, :, np.newaxis]
    )
    water_mask = land_mask & (
        np.arange(a.shape[2])[np.newaxis, np.newaxis, :] >= ks[:, :, np.newaxis]
    )

    a_tri = water_mask * a * np.logical_not(edge_mask)
    b_tri = where(water_mask, b, 1.0)
    if b_edge is not None:
        b_tri = where(edge_mask, b_edge, b_tri)
    c_tri = water_mask * c
    d_tri = water_mask * d
    if d_edge is not None:
        d_tri = where(edge_mask, d_edge, d_tri)

    return solve_tridiag(a_tri, b_tri, c_tri, d_tri), water_mask


@jax.jit
def solve_tridiag(a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape

    def compute_primes(last_primes, x):
        last_cp, last_dp = last_primes
        a, b, c, d = x
        cp = c / (b - a * last_cp)
        dp = (d - a * last_dp) / (b - a * last_cp)
        new_primes = np.stack((cp, dp))
        return new_primes, new_primes

    diags_stacked = np.stack([arr.transpose((2, 0, 1)) for arr in (a, b, c, d)], axis=1)
    _, primes = jax.lax.scan(
        compute_primes, np.zeros((2, *a.shape[:-1])), diags_stacked
    )

    def backsubstitution(last_x, x):
        cp, dp = x
        new_x = dp - cp * last_x
        return new_x, new_x

    _, sol = jax.lax.scan(backsubstitution, np.zeros(a.shape[:-1]), primes[::-1])
    return sol[::-1].transpose((1, 2, 0))


@jax.jit
def _calc_cr(rjp, rj, rjm, vel):
    """
    Calculates cr value used in superbee advection scheme
    """
    eps = 1e-20  # prevent division by 0
    return where(vel > 0.0, rjm, rjp) / where(np.abs(rj) < eps, eps, rj)


@jax.jit
def pad_z_edges(arr):
    arr_shape = list(arr.shape)
    arr_shape[2] += 2
    out = np.zeros(arr_shape, arr.dtype)
    out = jax.ops.index_update(out, jax.ops.index[:, :, 1:-1], arr)
    return out


@jax.jit
def limiter(cr):
    return np.maximum(0.0, np.maximum(np.minimum(1.0, 2 * cr), np.minimum(2.0, cr)))


def _adv_superbee(vel, var, mask, dx, axis, cost, cosu, dt_tracer):
    velfac = 1
    if axis == 0:
        sm1, s, sp1, sp2 = (
            (slice(1 + n, -2 + n or None), slice(2, -2), slice(None))
            for n in range(-1, 3)
        )
        dx = cost[np.newaxis, 2:-2, np.newaxis] * dx[1:-2, np.newaxis, np.newaxis]
    elif axis == 1:
        sm1, s, sp1, sp2 = (
            (slice(2, -2), slice(1 + n, -2 + n or None), slice(None))
            for n in range(-1, 3)
        )
        dx = (cost * dx)[np.newaxis, 1:-2, np.newaxis]
        velfac = cosu[np.newaxis, 1:-2, np.newaxis]
    elif axis == 2:
        vel, var, mask = (pad_z_edges(a) for a in (vel, var, mask))
        sm1, s, sp1, sp2 = (
            (slice(2, -2), slice(2, -2), slice(1 + n, -2 + n or None))
            for n in range(-1, 3)
        )
        dx = dx[np.newaxis, np.newaxis, :-1]
    else:
        raise ValueError("axis must be 0, 1, or 2")
    uCFL = np.abs(velfac * vel[s] * dt_tracer / dx)
    rjp = (var[sp2] - var[sp1]) * mask[sp1]
    rj = (var[sp1] - var[s]) * mask[s]
    rjm = (var[s] - var[sm1]) * mask[sm1]
    cr = limiter(_calc_cr(rjp, rj, rjm, vel[s]))
    return (
        velfac * vel[s] * (var[sp1] + var[s]) * 0.5
        - np.abs(velfac * vel[s]) * ((1.0 - cr) + uCFL * cr) * rj * 0.5
    )


_adv_superbee = jax.jit(_adv_superbee, static_argnums=(4,))


@jax.jit
def adv_flux_superbee_wgrid(
    var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer
):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    maskUtr = np.zeros_like(maskW)
    maskUtr = jax.ops.index_update(
        maskUtr, jax.ops.index[:-1, :, :], maskW[1:, :, :] * maskW[:-1, :, :]
    )

    adv_fe = np.zeros_like(maskW)
    adv_fe = jax.ops.index_update(
        adv_fe,
        jax.ops.index[1:-2, 2:-2, :],
        _adv_superbee(u_wgrid, var, maskUtr, dxt, 0, cost, cosu, dt_tracer),
    )

    maskVtr = np.zeros_like(maskW)
    maskVtr = jax.ops.index_update(
        maskVtr, jax.ops.index[:, :-1, :], maskW[:, 1:, :] * maskW[:, :-1, :]
    )
    adv_fn = np.zeros_like(maskW)
    adv_fn = jax.ops.index_update(
        adv_fn,
        jax.ops.index[2:-2, 1:-2, :],
        _adv_superbee(v_wgrid, var, maskVtr, dyt, 1, cost, cosu, dt_tracer),
    )

    maskWtr = np.zeros_like(maskW)
    maskWtr = jax.ops.index_update(
        maskWtr, jax.ops.index[:, :, :-1], maskW[:, :, 1:] * maskW[:, :, :-1]
    )
    adv_ft = np.zeros_like(maskW)
    adv_ft = jax.ops.index_update(
        adv_ft,
        jax.ops.index[2:-2, 2:-2, :-1],
        _adv_superbee(w_wgrid, var, maskWtr, dzw, 2, cost, cosu, dt_tracer),
    )

    return adv_fe, adv_fn, adv_ft


@jax.jit
def integrate_tke(
    u,
    v,
    w,
    maskU,
    maskV,
    maskW,
    dxt,
    dxu,
    dyt,
    dyu,
    dzt,
    dzw,
    cost,
    cosu,
    kbot,
    kappaM,
    mxl,
    forc,
    forc_tke_surface,
    tke,
    dtke,
):
    tau = 0
    taup1 = 1
    taum1 = 2

    dt_tracer = 1.0
    dt_mom = 1.0
    AB_eps = 0.1
    alpha_tke = 1.0
    c_eps = 0.7
    K_h_tke = 2000.0

    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskU)
    flux_top = np.zeros_like(maskU)

    sqrttke = np.sqrt(np.maximum(0.0, tke[:, :, :, tau]))

    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    dt_tke = dt_mom  # use momentum time step to prevent spurious oscillations

    """
    vertical mixing and dissipation of TKE
    """
    ks = kbot[2:-2, 2:-2] - 1

    a_tri = np.zeros_like(maskU[2:-2, 2:-2])
    b_tri = np.zeros_like(maskU[2:-2, 2:-2])
    c_tri = np.zeros_like(maskU[2:-2, 2:-2])
    d_tri = np.zeros_like(maskU[2:-2, 2:-2])
    delta = np.zeros_like(maskU[2:-2, 2:-2])

    delta = jax.ops.index_update(
        delta,
        jax.ops.index[:, :, :-1],
        dt_tke
        / dzt[np.newaxis, np.newaxis, 1:]
        * alpha_tke
        * 0.5
        * (kappaM[2:-2, 2:-2, :-1] + kappaM[2:-2, 2:-2, 1:]),
    )

    a_tri = jax.ops.index_update(
        a_tri,
        jax.ops.index[:, :, 1:-1],
        -delta[:, :, :-2] / dzw[np.newaxis, np.newaxis, 1:-1],
    )
    a_tri = jax.ops.index_update(
        a_tri, jax.ops.index[:, :, -1], -delta[:, :, -2] / (0.5 * dzw[-1])
    )

    b_tri = jax.ops.index_update(
        b_tri,
        jax.ops.index[:, :, 1:-1],
        1
        + (delta[:, :, 1:-1] + delta[:, :, :-2]) / dzw[np.newaxis, np.newaxis, 1:-1]
        + dt_tke * c_eps * sqrttke[2:-2, 2:-2, 1:-1] / mxl[2:-2, 2:-2, 1:-1],
    )
    b_tri = jax.ops.index_update(
        b_tri,
        jax.ops.index[:, :, -1],
        1
        + delta[:, :, -2] / (0.5 * dzw[-1])
        + dt_tke * c_eps / mxl[2:-2, 2:-2, -1] * sqrttke[2:-2, 2:-2, -1],
    )
    b_tri_edge = (
        1
        + delta / dzw[np.newaxis, np.newaxis, :]
        + dt_tke * c_eps / mxl[2:-2, 2:-2, :] * sqrttke[2:-2, 2:-2, :]
    )

    c_tri = jax.ops.index_update(
        c_tri,
        jax.ops.index[:, :, :-1],
        -delta[:, :, :-1] / dzw[np.newaxis, np.newaxis, :-1],
    )

    d_tri = tke[2:-2, 2:-2, :, tau] + dt_tke * forc[2:-2, 2:-2, :]
    d_tri = jax.ops.index_add(
        d_tri,
        jax.ops.index[:, :, -1],
        dt_tke * forc_tke_surface[2:-2, 2:-2] / (0.5 * dzw[-1]),
    )

    sol, water_mask = solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    tke = jax.ops.index_update(
        tke,
        jax.ops.index[2:-2, 2:-2, :, taup1],
        where(water_mask, sol, tke[2:-2, 2:-2, :, taup1]),
    )

    """
    Add TKE if surface density flux drains TKE in uppermost box
    """
    mask = tke[2:-2, 2:-2, -1, taup1] < 0.0
    tke_surf_corr = np.zeros_like(maskU[..., -1])
    tke_surf_corr = jax.ops.index_update(
        tke_surf_corr,
        jax.ops.index[2:-2, 2:-2],
        where(mask, -tke[2:-2, 2:-2, -1, taup1] * 0.5 * dzw[-1] / dt_tke, 0.0),
    )

    tke = jax.ops.index_update(
        tke,
        jax.ops.index[2:-2, 2:-2, -1, taup1],
        np.maximum(0.0, tke[2:-2, 2:-2, -1, taup1]),
    )

    """
    add tendency due to lateral diffusion
    """
    flux_east = jax.ops.index_update(
        flux_east,
        jax.ops.index[:-1, :, :],
        K_h_tke
        * (tke[1:, :, :, tau] - tke[:-1, :, :, tau])
        / (cost[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis])
        * maskU[:-1, :, :],
    )
    flux_north = jax.ops.index_update(
        flux_north,
        jax.ops.index[:, :-1, :],
        K_h_tke
        * (tke[:, 1:, :, tau] - tke[:, :-1, :, tau])
        / dyu[np.newaxis, :-1, np.newaxis]
        * maskV[:, :-1, :]
        * cosu[np.newaxis, :-1, np.newaxis],
    )
    tke = jax.ops.index_add(
        tke,
        jax.ops.index[2:-2, 2:-2, :, taup1],
        dt_tke
        * maskW[2:-2, 2:-2, :]
        * (
            (flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
            / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
            + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
            / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis])
        ),
    )

    """
    add tendency due to advection
    """
    flux_east, flux_north, flux_top = adv_flux_superbee_wgrid(
        tke[:, :, :, tau],
        u[..., tau],
        v[..., tau],
        w[..., tau],
        maskW,
        dxt,
        dyt,
        dzw,
        cost,
        cosu,
        dt_tracer,
    )

    dtke = jax.ops.index_update(
        dtke,
        jax.ops.index[2:-2, 2:-2, :, tau],
        maskW[2:-2, 2:-2, :]
        * (
            -(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
            / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
            - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
            / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis])
        ),
    )
    dtke = jax.ops.index_add(
        dtke, jax.ops.index[:, :, 0, tau], -flux_top[:, :, 0] / dzw[0]
    )
    dtke = jax.ops.index_add(
        dtke,
        jax.ops.index[:, :, 1:-1, tau],
        -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / dzw[1:-1],
    )
    dtke = jax.ops.index_add(
        dtke,
        jax.ops.index[:, :, -1, tau],
        -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * dzw[-1]),
    )

    """
    Adam Bashforth time stepping
    """
    tke = jax.ops.index_add(
        tke,
        jax.ops.index[:, :, :, taup1],
        dt_tracer
        * ((1.5 + AB_eps) * dtke[:, :, :, tau] - (0.5 + AB_eps) * dtke[:, :, :, taum1]),
    )

    return tke, dtke, tke_surf_corr


def prepare_inputs(*inputs, device):
    out = [np.array(k) for k in inputs]
    for o in out:
        o.block_until_ready()
    return out


def run(*inputs, device="cpu"):
    outputs = integrate_tke(*inputs)
    for o in outputs:
        o.block_until_ready()
    return outputs
