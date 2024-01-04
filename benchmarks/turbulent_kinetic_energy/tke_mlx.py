import mlx.core as mx


def where(mask, a, b):
    return mx.where(mask, a, b)


def solve_implicit(ks, a, b, c, d, b_edge=None, d_edge=None):
    land_mask = (ks >= 0)[:, :, None]
    edge_mask = land_mask * (
        mx.arange(a.shape[2])[None, None, :] == ks[:, :, None]
    )
    water_mask = land_mask * (
        mx.arange(a.shape[2])[None, None, :] >= ks[:, :, None]
    )

    a_tri = water_mask * a * mx.logical_not(edge_mask)
    b_tri = where(water_mask, b, 1.0)
    if b_edge is not None:
        b_tri = where(edge_mask, b_edge, b_tri)
    c_tri = water_mask * c
    d_tri = water_mask * d
    if d_edge is not None:
        d_tri = where(edge_mask, d_edge, d_tri)

    return solve_tridiag(a_tri, b_tri, c_tri, d_tri), water_mask


def solve_tridiag(a, b, c, d):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape

    n = a.shape[0]

    for i in range(1, n):
        w = a[i] / b[i - 1]
        b[i] += -w * c[i - 1]
        d[i] += -w * d[i - 1]

    out = mx.zeros_like(a)
    out[-1] = d[-1] / b[-1]

    for i in range(n - 2, -1, -1):
        out[i] = (d[i] - c[i] * out[i + 1]) / b[i]

    return out


def _calc_cr(rjp, rj, rjm, vel):
    """
    Calculates cr value used in superbee advection scheme
    """
    eps = 1e-20  # prevent division by 0
    return where(vel > 0.0, rjm, rjp) / where(mx.abs(rj) < eps, eps, rj)


def pad_z_edges(arr):
    arr_shape = list(arr.shape)
    arr_shape[2] += 2
    out = mx.zeros(arr_shape, arr.dtype)
    out[:, :, 1:-1] = arr
    return out


def limiter(cr):
    return mx.maximum(0.0, mx.maximum(mx.minimum(1.0, 2 * cr), mx.minimum(2.0, cr)))


def _adv_superbee(vel, var, mask, dx, axis, cost, cosu, dt_tracer):
    velfac = 1
    if axis == 0:
        sm1, s, sp1, sp2 = (
            (slice(1 + n, -2 + n or None), slice(2, -2), slice(None))
            for n in range(-1, 3)
        )
        dx = cost[None, 2:-2, None] * dx[1:-2, None, None]
    elif axis == 1:
        sm1, s, sp1, sp2 = (
            (slice(2, -2), slice(1 + n, -2 + n or None), slice(None))
            for n in range(-1, 3)
        )
        dx = (cost * dx)[None, 1:-2, None]
        velfac = cosu[None, 1:-2, None]
    elif axis == 2:
        vel, var, mask = (pad_z_edges(a) for a in (vel, var, mask))
        sm1, s, sp1, sp2 = (
            (slice(2, -2), slice(2, -2), slice(1 + n, -2 + n or None))
            for n in range(-1, 3)
        )
        dx = dx[None, None, :-1]
    else:
        raise ValueError("axis must be 0, 1, or 2")
    uCFL = mx.abs(velfac * vel[s] * dt_tracer / dx)
    rjp = (var[sp2] - var[sp1]) * mask[sp1]
    rj = (var[sp1] - var[s]) * mask[s]
    rjm = (var[s] - var[sm1]) * mask[sm1]
    cr = limiter(_calc_cr(rjp, rj, rjm, vel[s]))
    return (
        velfac * vel[s] * (var[sp1] + var[s]) * 0.5
        - mx.abs(velfac * vel[s]) * ((1.0 - cr) + uCFL * cr) * rj * 0.5
    )


def adv_flux_superbee_wgrid(
    adv_fe,
    adv_fn,
    adv_ft,
    var,
    u_wgrid,
    v_wgrid,
    w_wgrid,
    maskW,
    dxt,
    dyt,
    dzw,
    cost,
    cosu,
    dt_tracer,
):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    maskUtr = mx.zeros_like(maskW)
    maskUtr[:-1, :, :] = maskW[1:, :, :] * maskW[:-1, :, :]
    adv_fe[:] = 0.0
    adv_fe[1:-2, 2:-2, :] = _adv_superbee(
        u_wgrid, var, maskUtr, dxt, 0, cost, cosu, dt_tracer
    )

    maskVtr = mx.zeros_like(maskW)
    maskVtr[:, :-1, :] = maskW[:, 1:, :] * maskW[:, :-1, :]
    adv_fn[:] = 0.0
    adv_fn[2:-2, 1:-2, :] = _adv_superbee(
        v_wgrid, var, maskVtr, dyt, 1, cost, cosu, dt_tracer
    )

    maskWtr = mx.zeros_like(maskW)
    maskWtr[:, :, :-1] = maskW[:, :, 1:] * maskW[:, :, :-1]
    adv_ft[:] = 0.0
    adv_ft[2:-2, 2:-2, :-1] = _adv_superbee(
        w_wgrid, var, maskWtr, dzw, 2, cost, cosu, dt_tracer
    )


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

    dt_tracer = 1
    dt_mom = 1
    AB_eps = 0.1
    alpha_tke = 1.0
    c_eps = 0.7
    K_h_tke = 2000.0

    flux_east = mx.zeros_like(maskU)
    flux_north = mx.zeros_like(maskU)
    flux_top = mx.zeros_like(maskU)

    sqrttke = mx.sqrt(mx.maximum(0.0, tke[:, :, :, tau]))

    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    dt_tke = dt_mom  # use momentum time step to prevent spurious oscillations

    """
    vertical mixing and dissipation of TKE
    """
    ks = kbot[2:-2, 2:-2] - 1

    a_tri = mx.zeros_like(maskU[2:-2, 2:-2])
    b_tri = mx.zeros_like(maskU[2:-2, 2:-2])
    c_tri = mx.zeros_like(maskU[2:-2, 2:-2])
    d_tri = mx.zeros_like(maskU[2:-2, 2:-2])
    delta = mx.zeros_like(maskU[2:-2, 2:-2])

    delta[:, :, :-1] = (
        dt_tke
        / dzt[None, None, 1:]
        * alpha_tke
        * 0.5
        * (kappaM[2:-2, 2:-2, :-1] + kappaM[2:-2, 2:-2, 1:])
    )

    a_tri[:, :, 1:-1] = -delta[:, :, :-2] / dzw[None, None, 1:-1]
    a_tri[:, :, -1] = -delta[:, :, -2] / (0.5 * dzw[-1])

    b_tri[:, :, 1:-1] = (
        1
        + (delta[:, :, 1:-1] + delta[:, :, :-2]) / dzw[None, None, 1:-1]
        + dt_tke * c_eps * sqrttke[2:-2, 2:-2, 1:-1] / mxl[2:-2, 2:-2, 1:-1]
    )
    b_tri[:, :, -1] = (
        1
        + delta[:, :, -2] / (0.5 * dzw[-1])
        + dt_tke * c_eps / mxl[2:-2, 2:-2, -1] * sqrttke[2:-2, 2:-2, -1]
    )
    b_tri_edge = (
        1
        + delta / dzw[None, None, :]
        + dt_tke * c_eps / mxl[2:-2, 2:-2, :] * sqrttke[2:-2, 2:-2, :]
    )

    c_tri[:, :, :-1] = -delta[:, :, :-1] / dzw[None, None, :-1]

    d_tri[:] = tke[2:-2, 2:-2, :, tau] + dt_tke * forc[2:-2, 2:-2, :]
    d_tri[:, :, -1] += dt_tke * forc_tke_surface[2:-2, 2:-2] / (0.5 * dzw[-1])

    sol, water_mask = solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, b_edge=b_tri_edge)
    tke[2:-2, 2:-2, :, taup1] = where(water_mask, sol, tke[2:-2, 2:-2, :, taup1])

    """
    Add TKE if surface density flux drains TKE in uppermost box
    """
    tke_surf_corr = mx.zeros(maskU.shape[:2])
    mask = tke[2:-2, 2:-2, -1, taup1] < 0.0
    tke_surf_corr[2:-2, 2:-2] = where(
        mask, -tke[2:-2, 2:-2, -1, taup1] * 0.5 * dzw[-1] / dt_tke, 0.0
    )
    tke[2:-2, 2:-2, -1, taup1] = mx.maximum(0.0, tke[2:-2, 2:-2, -1, taup1])

    """
    add tendency due to lateral diffusion
    """
    flux_east[:-1, :, :] = (
        K_h_tke
        * (tke[1:, :, :, tau] - tke[:-1, :, :, tau])
        / (cost[None, :, None] * dxu[:-1, None, None])
        * maskU[:-1, :, :]
    )
    flux_east[-1, :, :] = 0.0
    flux_north[:, :-1, :] = (
        K_h_tke
        * (tke[:, 1:, :, tau] - tke[:, :-1, :, tau])
        / dyu[None, :-1, None]
        * maskV[:, :-1, :]
        * cosu[None, :-1, None]
    )
    flux_north[:, -1, :] = 0.0
    tke[2:-2, 2:-2, :, taup1] += (
        dt_tke
        * maskW[2:-2, 2:-2, :]
        * (
            (flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
            / (cost[None, 2:-2, None] * dxt[2:-2, None, None])
            + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
            / (cost[None, 2:-2, None] * dyt[None, 2:-2, None])
        )
    )

    """
    add tendency due to advection
    """
    adv_flux_superbee_wgrid(
        flux_east,
        flux_north,
        flux_top,
        tke[:, :, :, tau],
        u[:, :, :, tau],
        v[:, :, :, tau],
        w[:, :, :, tau],
        maskW,
        dxt,
        dyt,
        dzw,
        cost,
        cosu,
        dt_tracer,
    )

    dtke[2:-2, 2:-2, :, tau] = maskW[2:-2, 2:-2, :] * (
        -(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
        / (cost[None, 2:-2, None] * dxt[2:-2, None, None])
        - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
        / (cost[None, 2:-2, None] * dyt[None, 2:-2, None])
    )
    dtke[:, :, 0, tau] += -flux_top[:, :, 0] / dzw[0]
    dtke[:, :, 1:-1, tau] += -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / dzw[1:-1]
    dtke[:, :, -1, tau] += -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * dzw[-1])

    """
    Adam Bashforth time stepping
    """
    tke[:, :, :, taup1] += dt_tracer * (
        (1.5 + AB_eps) * dtke[:, :, :, tau] - (0.5 + AB_eps) * dtke[:, :, :, taum1]
    )

    return tke, dtke, tke_surf_corr


def prepare_inputs(*inputs, device):
    out = [mx.array(k) for k in inputs]
    for o in out:
        mx.eval(o)
    return out


def run(*inputs, device="cpu"):
    outputs = integrate_tke(*inputs)
    for o in outputs:
        mx.eval(o)
    return outputs
