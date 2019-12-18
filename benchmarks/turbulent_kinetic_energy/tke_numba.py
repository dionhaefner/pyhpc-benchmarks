import numpy as np
import numba as nb


@nb.jit(nopython=True, fastmath=True, cache=True)
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

    out = np.empty_like(a)
    out[-1] = d[-1] / b[-1]

    for i in range(n - 2, -1, -1):
        out[i] = (d[i] - c[i] * out[i + 1]) / b[i]

    return out


@nb.jit(nopython=True, fastmath=True, cache=True)
def _calc_cr(r_jp, r_j, r_jm, vel):
    """
    Calculates cr value used in superbee advection scheme
    """
    eps = 1e-20  # prevent division by 0
    if abs(r_j) < eps:
        fac = eps
    else:
        fac = r_j

    if vel > 0:
        return r_jm / fac
    else:
        return r_jp / fac


@nb.jit(nopython=True, fastmath=True, cache=True)
def limiter(cr):
    return max(0., max(min(1., 2. * cr), min(2., cr)))


@nb.jit(nopython=True, fastmath=True, cache=True)
def adv_flux_superbee_wgrid(adv_fe, adv_fn, adv_ft, var, u_wgrid, v_wgrid, w_wgrid, maskW, dxt, dyt, dzw, cost, cosu, dt_tracer):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    nx, ny, nz = var.shape

    maskUtr = np.zeros_like(maskW)
    maskUtr[:-1, :, :] = maskW[1:, :, :] * maskW[:-1, :, :]

    adv_fe[...] = 0.
    for i in range(1, nx-2):
        for j in range(2, ny-2):
            for k in range(nz):
                vel = u_wgrid[i, j, k]
                u_cfl = abs(vel * dt_tracer / (cost[j] * dxt[i]))
                r_jp = (var[i+2, j, k] - var[i+1, j, k]) * maskUtr[i+1, j, k]
                r_j = (var[i+1, j, k] - var[i, j, k]) * maskUtr[i, j ,k]
                r_jm = (var[i, j, k] - var[i-1, j, k]) * maskUtr[i-1, j, k]
                cr = limiter(_calc_cr(r_jp, r_j, r_jm, vel))
                adv_fe[i, j, k] = vel * (var[i+1, j, k] + var[i, j, k]) * 0.5 - abs(vel) * ((1. - cr) + u_cfl * cr) * r_j * 0.5

    maskVtr = np.zeros_like(maskW)
    maskVtr[:, :-1, :] = maskW[:, 1:, :] * maskW[:, :-1, :]

    adv_fn[...] = 0.
    for i in range(2, nx-2):
        for j in range(1, ny-2):
            for k in range(nz):
                vel = cosu[j] * v_wgrid[i, j, k]
                u_cfl = abs(vel * dt_tracer / (cost[j] * dyt[j]))
                r_jp = (var[i, j+2, k] - var[i, j+1, k]) * maskVtr[i, j+1, k]
                r_j = (var[i, j+1, k] - var[i, j, k]) * maskVtr[i, j, k]
                r_jm = (var[i, j, k] - var[i, j-1, k]) * maskVtr[i, j-1, k]
                cr = limiter(_calc_cr(r_jp, r_j, r_jm, v_wgrid[i, j, k]))
                adv_fn[i, j, k] = vel * (var[i, j+1, k] + var[i, j, k]) * 0.5 - abs(vel) * ((1. - cr) + u_cfl * cr) * r_j * 0.5

    maskWtr = np.zeros_like(maskW)
    maskWtr[:, :, :-1] = maskW[:, :, 1:] * maskW[:, :, :-1]

    adv_ft[...] = 0.
    for i in range(2, nx-2):
        for j in range(2, ny-2):
            for k in range(nz-1):
                kp1 = min(nz-2, k+1)
                kp2 = min(nz-1, k+2)
                km1 = max(0, k-1)

                vel = w_wgrid[i, j, k]
                u_cfl = abs(vel * dt_tracer / dzw[k])
                r_jp = (var[i, j, kp2] - var[i, j, k+1]) * maskWtr[i, j, kp1]
                r_j = (var[i, j, k+1] - var[i, j, k]) * maskWtr[i, j ,k]
                r_jm = (var[i, j, k] - var[i, j, km1]) * maskWtr[i, j, km1]
                cr = limiter(_calc_cr(r_jp, r_j, r_jm, vel))
                adv_ft[i, j, k] = vel * (var[i, j, k+1] + var[i, j, k]) * 0.5 - abs(vel) * ((1. - cr) + u_cfl * cr) * r_j * 0.5


@nb.jit(nopython=True, fastmath=True, cache=True)
def integrate_tke(u, v, w, maskU, maskV, maskW, dxt, dxu, dyt, dyu, dzt, dzw, cost, cosu, kbot, kappaM, mxl, forc, forc_tke_surface, tke, dtke):
    nx, ny, nz = maskU.shape

    tau = 0
    taup1 = 1
    taum1 = 2

    dt_tracer = 1
    dt_mom = 1
    AB_eps = 0.1
    alpha_tke = 1.
    c_eps = 0.7
    K_h_tke = 2000.

    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskU)
    flux_top = np.zeros_like(maskU)

    sqrttke = np.sqrt(np.maximum(0., tke[:, :, :, tau]))

    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    dt_tke = dt_mom  # use momentum time step to prevent spurious oscillations

    """
    vertical mixing and dissipation of TKE
    """
    a_tri = np.empty(nz)
    b_tri = np.empty(nz)
    c_tri = np.empty(nz)
    d_tri = np.empty(nz)
    delta = np.empty(nz)

    ke = nz - 1
    for i in range(2, nx-2):
        for j in range(2, ny-2):
            ks = kbot[i, j] - 1
            if ks < 0:
                continue

            for k in range(ks, ke):
                delta[k] = dt_tke / dzt[k+1] * alpha_tke * 0.5 * (kappaM[i, j, k] + kappaM[i, j, k+1])
            delta[ke] = 0.

            for k in range(ks+1, ke):
                a_tri[k] = -delta[k-1] / dzw[k]
            a_tri[ks] = 0.
            a_tri[ke] = -delta[ke-1] / (0.5 * dzw[ke])

            for k in range(ks+1, ke):
                b_tri[k] = 1 + delta[k] / dzw[k] + delta[k-1] / dzw[k] + dt_tke * c_eps * sqrttke[i, j, k] / mxl[i, j, k]
            b_tri[ke] = 1 + delta[ke - 1] / (0.5 * dzw[ke]) + dt_tke * c_eps * sqrttke[i, j, ke] / mxl[i, j, ke]
            b_tri[ks] = 1 + delta[ks] / dzw[ks] + dt_tke * c_eps * sqrttke[i, j, ks] / mxl[i, j, ks]

            for k in range(ks, ke):
                c_tri[k] = -delta[k] / dzw[k]
            c_tri[ke] = 0.

            d_tri[ks:] = tke[i, j, ks:, tau] + dt_tke * forc[i, j, ks:]
            d_tri[ke] += dt_tke * forc_tke_surface[i, j] / (0.5 * dzw[ke])

            tke[i, j, ks:, taup1] = solve_tridiag(a_tri[ks:], b_tri[ks:], c_tri[ks:], d_tri[ks:])

    """
    Add TKE if surface density flux drains TKE in uppermost box
    """
    tke_surf_corr = np.zeros((nx, ny))
    for i in range(2, nx-2):
        for j in range(2, ny-2):
            if tke[i, j, -1, taup1] >= 0.:
                continue
            tke_surf_corr[i, j] = -tke[i, j, -1, taup1] * (0.5 * dzw[-1]) / dt_tke
            tke[i, j, -1, taup1] = 0.

    """
    add tendency due to lateral diffusion
    """
    for i in range(nx-1):
        for j in range(ny):
            flux_east[i, j, :] = K_h_tke * (tke[i+1, j, :, tau] - tke[i, j, :, tau]) / (cost[j] * dxu[i]) * maskU[i, j, :]
    flux_east[-1, :, :] = 0.

    for j in range(ny-1):
        flux_north[:, j, :] = K_h_tke * (tke[:, j+1, :, tau] - tke[:, j, :, tau]) / dyu[j] * maskV[:, j, :] * cosu[j]
    flux_north[:, -1, :] = 0.

    for i in range(2, nx-2):
        for j in range(2, ny-2):
            tke[i, j, :, taup1] += dt_tke * maskW[i, j, :] * (
                (flux_east[i, j, :] - flux_east[i-1, j, :])/ (cost[j] * dxt[i])
                + (flux_north[i, j, :] - flux_north[i, j-1, :]) / (cost[j] * dyt[j])
            )

    """
    add tendency due to advection
    """
    adv_flux_superbee_wgrid(
        flux_east, flux_north, flux_top, tke[:, :, :, tau],
        u[..., tau], v[..., tau], w[..., tau], maskW, dxt, dyt, dzw,
        cost, cosu, dt_tracer
    )

    for i in range(2, nx-2):
        for j in range(2, ny-2):
            dtke[i, j, :, tau] = maskW[i, j, :] * (
                -(flux_east[i, j, :] - flux_east[i-1, j, :]) / (cost[j] * dxt[i])
                - (flux_north[i, j, :] - flux_north[i, j-1, :]) / (cost[j] * dyt[j])
            )
    dtke[:, :, 0, tau] += -flux_top[:, :, 0] / dzw[0]
    dtke[:, :, 1:-1, tau] += -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / dzw[1:-1]
    dtke[:, :, -1, tau] += -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * dzw[-1])

    """
    Adam Bashforth time stepping
    """
    tke[:, :, :, taup1] += dt_tracer * ((1.5 + AB_eps) * dtke[:, :, :, tau] - (0.5 + AB_eps) * dtke[:, :, :, taum1])

    return tke, dtke, tke_surf_corr


def run(*inputs, device='cpu'):
    outputs = integrate_tke(*inputs)
    return outputs
