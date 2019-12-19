import numpy as np
import numba as nb


@nb.jit(nopython=True, fastmath=True)
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


@nb.jit(nopython=True, fastmath=True)
def get_drhodS(salt, temp, p):
    betaS = 0.78e-3
    rho0 = 1024.
    return betaS * rho0


@nb.jit(nopython=True, fastmath=True)
def dm_taper(sx):
    """
    tapering function for isopycnal slopes
    """
    iso_slopec = 1e-3
    iso_dslope = 1e-3
    return 0.5 * (1. + np.tanh((-np.abs(sx) + iso_slopec) / iso_dslope))


@nb.jit(nopython=True, fastmath=True, cache=True)
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

    drdT = np.empty_like(K_11)
    drdS = np.empty_like(K_11)
    dTdx = np.empty_like(K_11)
    dSdx = np.empty_like(K_11)
    dTdy = np.empty_like(K_11)
    dSdy = np.empty_like(K_11)
    dTdz = np.empty_like(K_11)
    dSdz = np.empty_like(K_11)

    """
    drho_dt and drho_ds at centers of T cells
    """
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                drdT[i, j, k] = maskT[i, j, k] * get_drhodT(
                    salt[i, j, k, tau], temp[i, j, k, tau], np.abs(zt[k])
                )
                drdS[i, j, k] = maskT[i, j, k] * get_drhodS(
                    salt[i, j, k, tau], temp[i, j, k, tau], np.abs(zt[k])
                )

    """
    gradients at top face of T cells
    """
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1):
                dTdz[i, j, k] = maskW[i, j, k] * \
                    (temp[i, j, k+1, tau] - temp[i, j, k, tau]) / \
                    dzw[k]
                dSdz[i, j, k] = maskW[i, j, k] * \
                    (salt[i, j, k+1, tau] - salt[i, j, k, tau]) / \
                    dzw[k]
            dTdz[i, j, -1] = 0.
            dSdz[i, j, -1] = 0.

    """
    gradients at eastern face of T cells
    """
    for i in range(nx-1):
        for j in range(ny):
            for k in range(nz):
                dTdx[i, j, k] = maskU[i, j, k] * (temp[i+1, j, k, tau] - temp[i, j, k, tau]) \
                    / (dxu[i] * cost[j])
                dSdx[i, j, k] = maskU[i, j, k] * (salt[i+1, j, k, tau] - salt[i, j, k, tau]) \
                    / (dxu[i] * cost[j])
    dTdx[-1, :, :] = 0.
    dSdx[-1, :, :] = 0.

    """
    gradients at northern face of T cells
    """
    for i in range(nx):
        for j in range(ny-1):
            for k in range(nz):
                dTdy[i, j, k] = maskV[i, j, k] * (temp[i, j+1, k, tau] - temp[i, j, k, tau]) \
                    / dyu[j]
                dSdy[i, j, k] = maskV[i, j, k] * (salt[i, j+1, k, tau] - salt[i, j, k, tau]) \
                    / dyu[j]
    dTdy[:, -1, :] = 0.
    dSdy[:, -1, :] = 0.

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    for i in range(1, nx-2):
        for j in range(2, ny-2):
            for k in range(0, nz):
                if k == 0:
                    diffloc = 0.5 * (K_iso[i, j, k] + K_iso[i+1, j, k])
                else:
                    diffloc = 0.25 * (K_iso[i, j, k] + K_iso[i, j, k-1] + K_iso[i+1, j, k] + K_iso[i+1, j, k-1])

                sumz = 0.

                for kr in (0, 1):
                    if k == 0 and kr == 0:
                        continue

                    for ip in (0, 1):
                        drodxe = drdT[i + ip, j, k] * dTdx[i, j, k] \
                            + drdS[i + ip, j, k] * dSdx[i, j, k]
                        drodze = drdT[i + ip, j, k] * dTdz[i + ip, j, k + kr - 1] \
                            + drdS[i + ip, j, k] * dSdz[i + ip, j, k + kr - 1]
                        sxe = -drodxe / (min(0., drodze) - epsln)
                        taper = dm_taper(sxe)
                        sumz += dzw[k + kr - 1] * maskU[i, j, k] * max(K_iso_steep, diffloc * taper)
                        Ai_ez[i, j, k, ip, kr] = taper * sxe * maskU[i, j, k]

                K_11[i, j, k] = sumz / (4. * dzt[k])

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    for i in range(2, nx-2):
        for j in range(1, ny-2):
            for k in range(0, nz):
                if k == 0:
                    diffloc = 0.5 * (K_iso[i, j, k] + K_iso[i, j+1, k])
                else:
                    diffloc = 0.25 * (K_iso[i, j, k] + K_iso[i, j, k-1] + K_iso[i, j+1, k] + K_iso[i, j+1, k-1])

                sumz = 0.

                for kr in (0, 1):
                    if k == 0 and kr == 0:
                        continue

                    for jp in (0, 1):
                        drodyn = drdT[i, j + jp, k] * dTdy[i, j, k] \
                            + drdS[i, j + jp, k] * dSdy[i, j, k]
                        drodzn = drdT[i, j + jp, k] * dTdz[i, j + jp, k + kr - 1] \
                            + drdS[i, j + jp, k] * dSdz[i, j + jp, k + kr - 1]
                        syn = -drodyn / (min(0., drodzn) - epsln)
                        taper = dm_taper(syn)
                        sumz += dzw[k + kr - 1] * maskV[i, j, k] * max(K_iso_steep, diffloc * taper)
                        Ai_nz[i, j, k, jp, kr] = taper * syn * maskV[i, j, k]

                K_22[i, j, k] = sumz / (4. * dzt[k])

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    for i in range(2, nx-2):
        for j in range(2, ny-2):
            for k in range(nz-1):
                sumx = 0.
                sumy = 0.

                for kr in (0, 1):
                    drodzb = drdT[i, j, k + kr] * dTdz[i, j, k] \
                        + drdS[i, j, k + kr] * dSdz[i, j, k]

                    # eastward slopes at the top of T cells
                    for ip in (0, 1):
                        drodxb = drdT[i, j, k + kr] * dTdx[i - 1 + ip, j, k + kr] \
                            + drdS[i, j, k + kr] * dSdx[i - 1 + ip, j, k + kr]
                        sxb = -drodxb / (min(0., drodzb) - epsln)
                        taper = dm_taper(sxb)
                        sumx += dxu[i - 1 + ip] * K_iso[i, j, k] * taper * sxb**2 * maskW[i, j, k]
                        Ai_bx[i, j, k, ip, kr] = taper * sxb * maskW[i, j, k]

                    # northward slopes at the top of T cells
                    for jp in (0, 1):
                        facty = cosu[j - 1 + jp] * dyu[j - 1 + jp]
                        drodyb = drdT[i, j, k + kr] * dTdy[i, j + jp - 1, k + kr] \
                            + drdS[i, j, k + kr] * dSdy[i, j + jp - 1, k + kr]
                        syb = -drodyb / (min(0., drodzb) - epsln)
                        taper = dm_taper(syb)
                        sumy += facty * K_iso[i, j, k] * taper * syb**2 * maskW[i, j, k]
                        Ai_by[i, j, k, jp, kr] = taper * syb * maskW[i, j, k]

                K_33[i, j, k] = sumx / (4 * dxt[i]) + sumy / (4 * dyt[j] * cost[j])

            K_33[i, j, -1] = 0.


def run(*inputs, device='cpu'):
    isoneutral_diffusion_pre(*inputs)
    return inputs[-7:]
