import numpy as np
import numba as nb


@nb.jit(nopython=True)
def get_drhodT(salt, temp, p):
    rho0 = 1024.0
    z0 = 0.0
    theta0 = 283.0 - 273.15
    grav = 9.81
    betaT = 1.67e-4
    betaTs = 1e-5
    gammas = 1.1e-8

    zz = -p - z0
    thetas = temp - theta0
    return -(betaTs * thetas + betaT * (1 - gammas * grav * zz * rho0)) * rho0


@nb.jit(nopython=True)
def get_drhodS(salt, temp, p):
    betaS = 0.78e-3
    rho0 = 1024.
    return betaS * rho0 * np.ones_like(temp)


@nb.jit(nopython=True)
def dm_taper(sx):
    """
    tapering function for isopycnal slopes
    """
    iso_slopec = 1e-3
    iso_dslope = 1e-3
    return 0.5 * (1. + np.tanh((-np.abs(sx) + iso_slopec) / iso_dslope))


@nb.jit(nopython=True, cache=True)
def isoneutral_diffusion_pre(maskT, maskU, maskV, maskW, dxt, dxu, dyt, dyu, dzt, dzw, cost, cosu, salt, temp, zt, K_iso, K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    nx, ny, nz = maskT.shape

    epsln = 1e-20
    K_iso_steep = 50.
    tau = 0

    dTdx = np.zeros_like(K_11)
    dSdx = np.zeros_like(K_11)
    dTdy = np.zeros_like(K_11)
    dSdy = np.zeros_like(K_11)
    dTdz = np.zeros_like(K_11)
    dSdz = np.zeros_like(K_11)

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdT = maskT * get_drhodT(
        salt[:, :, :, tau], temp[:, :, :, tau], np.abs(zt)
    )
    drdS = maskT * get_drhodS(
        salt[:, :, :, tau], temp[:, :, :, tau], np.abs(zt)
    )

    """
    gradients at top face of T cells
    """
    dTdz[:, :, :-1] = maskW[:, :, :-1] * \
        (temp[:, :, 1:, tau] - temp[:, :, :-1, tau]) / \
        dzw[:-1].reshape(1, 1, -1)
    dSdz[:, :, :-1] = maskW[:, :, :-1] * \
        (salt[:, :, 1:, tau] - salt[:, :, :-1, tau]) / \
        dzw[:-1].reshape(1, 1, -1)

    """
    gradients at eastern face of T cells
    """
    dTdx[:-1, :, :] = maskU[:-1, :, :] * (temp[1:, :, :, tau] - temp[:-1, :, :, tau]) \
        / (dxu[:-1].reshape(-1, 1, 1) * cost.reshape(1, -1, 1))
    dSdx[:-1, :, :] = maskU[:-1, :, :] * (salt[1:, :, :, tau] - salt[:-1, :, :, tau]) \
        / (dxu[:-1].reshape(-1, 1, 1) * cost.reshape(1, -1, 1))

    """
    gradients at northern face of T cells
    """
    dTdy[:, :-1, :] = maskV[:, :-1, :] * \
        (temp[:, 1:, :, tau] - temp[:, :-1, :, tau]) \
        / dyu[:-1].reshape(1, -1, 1)
    dSdy[:, :-1, :] = maskV[:, :-1, :] * \
        (salt[:, 1:, :, tau] - salt[:, :-1, :, tau]) \
        / dyu[:-1].reshape(1, -1, 1)

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = np.zeros_like(K_11)
    diffloc[1:-2, 2:-2, 1:] = 0.25 * (K_iso[1:-2, 2:-2, 1:] + K_iso[1:-2, 2:-2, :-1]
                                      + K_iso[2:-1, 2:-2, 1:] + K_iso[2:-1, 2:-2, :-1])
    diffloc[1:-2, 2:-2, 0] = 0.5 * (K_iso[1:-2, 2:-2, 0] + K_iso[2:-1, 2:-2, 0])

    sumz = np.zeros_like(K_11)

    for i in range(1, nx-2):
        for j in range(1, ny-2):
            for ki in range(2):
                for k in range(0, nz - ki):
                    for ip in range(2):
                        drodxe = drdT[i + ip, j, k + ki] * dTdx[i, j, k + ki] \
                            + drdS[i + ip, j, k + ki] * dSdx[i, j, k + ki]
                        drodze = drdT[i + ip, j, k + ki] * dTdz[i + ip, j, k] \
                            + drdS[i + ip, j, k + ki] * \
                            dSdz[i + ip, j, k]
                        sxe = -drodxe / (min(0., drodze) - epsln)
                        taper_2 = dm_taper(sxe)
                        sumz[i, j, k + ki] += dzw[k] * maskU[i, j, k + ki] \
                            * np.maximum(K_iso_steep, diffloc[i, j, k + ki] * taper_2)
                        Ai_ez[i, j, k + ki, ip, 1 - ki] = taper_2 * \
                            sxe * maskU[i, j, k + ki]

    K_11[1:-2, 2:-2, :] = sumz[1:-2, 2:-2] / (4. * dzt.reshape(1, 1, -1))

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc[...] = 0
    diffloc[2:-2, 1:-2, 1:] = 0.25 * (K_iso[2:-2, 1:-2, 1:] + K_iso[2:-2, 1:-2, :-1]
                                      + K_iso[2:-2, 2:-1, 1:] + K_iso[2:-2, 2:-1, :-1])
    diffloc[2:-2, 1:-2, 0] = 0.5 * (K_iso[2:-2, 1:-2, 0] + K_iso[2:-2, 2:-1, 0])

    sumz = np.zeros_like(K_11)

    for i in range(2, nx-2):
        for j in range(1, ny-2):
            for ki in range(2):
                for k in range(0, nz - ki):
                    for jp in range(2):
                        drodyn = drdT[i, j + jp, k + ki] * dTdy[i, j, k + ki] + \
                            drdS[i, j + jp, k + ki] * dSdy[i, j, k + ki]
                        drodzn = drdT[i, j + jp, k + ki] * dTdz[i, j + jp, k] \
                            + drdS[i, j + jp, k + ki] * \
                            dSdz[i, j + jp, k]
                        syn = -drodyn / (min(0., drodzn) - epsln)
                        taper_2 = dm_taper(syn)
                        sumz[i, j, k + ki] += dzw[k] \
                            * maskV[i, j, k + ki] * np.maximum(K_iso_steep, diffloc[i, j, k + ki] * taper_2)
                        Ai_nz[i, j, k + ki, jp, 1 - ki] = taper_2 * \
                            syn * maskV[i, j, k + ki]

    K_22[2:-2, 1:-2, :] = sumz[2:-2, 1:-2] / (4. * dzt.reshape(1, 1, -1))

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    for i in range(2, nx-2):
        for j in range(2, ny-2):
            for k in range(nz-1):
                for kr in (0, 1):
                    drodzb = drdT[i, j, k + kr] * dTdz[i, j, k] \
                        + drdS[i, j, k + kr] * dSdz[i, j, k]

                    # eastward slopes at the top of T cells
                    sumx = 0.
                    for ip in (0, 1):
                        drodxb = drdT[i, j, k + kr] * dTdx[i - 1 + ip, j, k + kr] \
                            + drdS[i, j, k + kr] * dSdx[i - 1 + ip, j, k + kr]
                        sxb = -drodxb / (min(0., drodzb) - epsln)
                        taper = dm_taper(sxb)
                        sumx += dxu[i - 1 + ip] * K_iso[i, j, k] * taper * \
                            sxb**2 * maskW[i, j, k]
                        Ai_bx[i, j, k, ip, kr] = taper * sxb * maskW[i, j, k]

                    # northward slopes at the top of T cells
                    sumy = 0.
                    for jp in (0, 1):
                        facty = cosu[j - 1 + jp] * dyu[j - 1 + jp]
                        drodyb = drdT[i, j, k + kr] * dTdy[i, j + jp - 1, k + kr] \
                            + drdS[i, j, k + kr] * dSdy[i, j + jp - 1, k + kr]
                        syb = -drodyb / (min(0., drodzb) - epsln)
                        taper = dm_taper(syb)
                        sumy += facty * K_iso[i, j, k] * taper * syb**2 * maskW[i, j, k]
                        Ai_by[i, j, k, jp, kr] = taper * syb * maskW[i, j, k]

                K_33[i, j, k] = sumx / (4 * dxt[i]) + sumy / (4 * dyt[j] * cost[j])

    K_33[2:-2, 2:-2, -1] = 0.


def run(*inputs):
    isoneutral_diffusion_pre(*inputs)
    return inputs[-1]
