import numpy as np
import cupy as cp


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


def get_drhodS(salt, temp, p):
    betaS = 0.78e-3
    rho0 = 1024.
    return betaS * rho0 * cp.ones_like(temp)


def isoneutral_diffusion_pre(maskT, maskU, maskV, maskW, dxt, dxu, dyt, dyu, dzt, dzw, cost, cosu, salt, temp, zt, K_iso, K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    epsln = 1e-20
    iso_slopec = 1e-3
    iso_dslope = 1e-3
    K_iso_steep = 50.
    tau = 0

    dTdx = cp.zeros_like(K_11)
    dSdx = cp.zeros_like(K_11)
    dTdy = cp.zeros_like(K_11)
    dSdy = cp.zeros_like(K_11)
    dTdz = cp.zeros_like(K_11)
    dSdz = cp.zeros_like(K_11)

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdT = maskT * get_drhodT(
        salt[:, :, :, tau], temp[:, :, :, tau], cp.abs(zt)
    )
    drdS = maskT * get_drhodS(
        salt[:, :, :, tau], temp[:, :, :, tau], cp.abs(zt)
    )

    """
    gradients at top face of T cells
    """
    dTdz[:, :, :-1] = maskW[:, :, :-1] * \
        (temp[:, :, 1:, tau] - temp[:, :, :-1, tau]) / \
        dzw[np.newaxis, np.newaxis, :-1]
    dSdz[:, :, :-1] = maskW[:, :, :-1] * \
        (salt[:, :, 1:, tau] - salt[:, :, :-1, tau]) / \
        dzw[np.newaxis, np.newaxis, :-1]

    """
    gradients at eastern face of T cells
    """
    dTdx[:-1, :, :] = maskU[:-1, :, :] * (temp[1:, :, :, tau] - temp[:-1, :, :, tau]) \
        / (dxu[:-1, np.newaxis, np.newaxis] * cost[np.newaxis, :, np.newaxis])
    dSdx[:-1, :, :] = maskU[:-1, :, :] * (salt[1:, :, :, tau] - salt[:-1, :, :, tau]) \
        / (dxu[:-1, np.newaxis, np.newaxis] * cost[np.newaxis, :, np.newaxis])

    """
    gradients at northern face of T cells
    """
    dTdy[:, :-1, :] = maskV[:, :-1, :] * \
        (temp[:, 1:, :, tau] - temp[:, :-1, :, tau]) \
        / dyu[np.newaxis, :-1, np.newaxis]
    dSdy[:, :-1, :] = maskV[:, :-1, :] * \
        (salt[:, 1:, :, tau] - salt[:, :-1, :, tau]) \
        / dyu[np.newaxis, :-1, np.newaxis]

    def dm_taper(sx):
        """
        tapering function for isopycnal slopes
        """
        return 0.5 * (1. + cp.tanh((-cp.abs(sx) + iso_slopec) / iso_dslope))

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = cp.zeros_like(K_11)
    diffloc[1:-2, 2:-2, 1:] = 0.25 * (K_iso[1:-2, 2:-2, 1:] + K_iso[1:-2, 2:-2, :-1]
                                      + K_iso[2:-1, 2:-2, 1:] + K_iso[2:-1, 2:-2, :-1])
    diffloc[1:-2, 2:-2, 0] = 0.5 * \
        (K_iso[1:-2, 2:-2, 0] + K_iso[2:-1, 2:-2, 0])

    sumz = cp.zeros_like(K_11)[1:-2, 2:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for ip in range(2):
            drodxe = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
            drodze = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdz[1 + ip:-2 + ip, 2:-2, :-1 + kr or None] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * \
                dSdz[1 + ip:-2 + ip, 2:-2, :-1 + kr or None]
            sxe = -drodxe / (cp.minimum(0., drodze) - epsln)
            taper = dm_taper(sxe)
            sumz[:, :, ki:] += dzw[np.newaxis, np.newaxis, :-1 + kr or None] * maskU[1:-2, 2:-2, ki:] \
                * cp.maximum(K_iso_steep, diffloc[1:-2, 2:-2, ki:] * taper)
            Ai_ez[1:-2, 2:-2, ki:, ip, kr] = taper * \
                sxe * maskU[1:-2, 2:-2, ki:]
    K_11[1:-2, 2:-2, :] = sumz / (4. * dzt[np.newaxis, np.newaxis, :])

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc[...] = 0
    diffloc[2:-2, 1:-2, 1:] = 0.25 * (K_iso[2:-2, 1:-2, 1:] + K_iso[2:-2, 1:-2, :-1]
                                      + K_iso[2:-2, 2:-1, 1:] + K_iso[2:-2, 2:-1, :-1])
    diffloc[2:-2, 1:-2, 0] = 0.5 * \
        (K_iso[2:-2, 1:-2, 0] + K_iso[2:-2, 2:-1, 0])

    sumz = cp.zeros_like(K_11)[2:-2, 1:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for jp in range(2):
            drodyn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:] + \
                drdS[2:-2, 1 + jp:-2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
            drodzn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdz[2:-2, 1 + jp:-2 + jp, :-1 + kr or None] \
                + drdS[2:-2, 1 + jp:-2 + jp, ki:] * \
                dSdz[2:-2, 1 + jp:-2 + jp, :-1 + kr or None]
            syn = -drodyn / (cp.minimum(0., drodzn) - epsln)
            taper = dm_taper(syn)
            sumz[:, :, ki:] += dzw[np.newaxis, np.newaxis, :-1 + kr or None] \
                * maskV[2:-2, 1:-2, ki:] * cp.maximum(K_iso_steep, diffloc[2:-2, 1:-2, ki:] * taper)
            Ai_nz[2:-2, 1:-2, ki:, jp, kr] = taper * \
                syn * maskV[2:-2, 1:-2, ki:]
    K_22[2:-2, 1:-2, :] = sumz / (4. * dzt[np.newaxis, np.newaxis, :])

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    sumx = cp.zeros_like(K_11)[2:-2, 2:-2, :-1]
    sumy = cp.zeros_like(K_11)[2:-2, 2:-2, :-1]

    for kr in range(2):
        drodzb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdz[2:-2, 2:-2, :-1] \
            + drdS[2:-2, 2:-2, kr:-1 + kr or None] * dSdz[2:-2, 2:-2, :-1]

        # eastward slopes at the top of T cells
        for ip in range(2):
            drodxb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdx[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None] \
                + drdS[2:-2, 2:-2, kr:-1 + kr or None] * \
                dSdx[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None]
            sxb = -drodxb / (cp.minimum(0., drodzb) - epsln)
            taper = dm_taper(sxb)
            sumx += dxu[1 + ip:-3 + ip, np.newaxis, np.newaxis] * \
                K_iso[2:-2, 2:-2, :-1] * taper * \
                sxb**2 * maskW[2:-2, 2:-2, :-1]
            Ai_bx[2:-2, 2:-2, :-1, ip, kr] = taper * \
                sxb * maskW[2:-2, 2:-2, :-1]

        # northward slopes at the top of T cells
        for jp in range(2):
            facty = cosu[1 + jp:-3 + jp] * dyu[1 + jp:-3 + jp]
            drodyb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdy[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None] \
                + drdS[2:-2, 2:-2, kr:-1 + kr or None] * \
                dSdy[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None]
            syb = -drodyb / (cp.minimum(0., drodzb) - epsln)
            taper = dm_taper(syb)
            sumy += facty[np.newaxis, :, np.newaxis] * K_iso[2:-2, 2:-2, :-1] \
                * taper * syb**2 * maskW[2:-2, 2:-2, :-1]
            Ai_by[2:-2, 2:-2, :-1, jp, kr] = taper * \
                syb * maskW[2:-2, 2:-2, :-1]

    K_33[2:-2, 2:-2, :-1] = sumx / (4 * dxt[2:-2, np.newaxis, np.newaxis]) + \
        sumy / (4 * dyt[np.newaxis, 2:-2, np.newaxis]
                * cost[np.newaxis, 2:-2, np.newaxis])
    K_33[2:-2, 2:-2, -1] = 0.


def prepare_inputs(*inputs, gpu):
    return [cp.asarray(k) for k in inputs]


def run(*inputs, gpu=False):
    isoneutral_diffusion_pre(*inputs)
    cp.cuda.stream.get_current_stream().synchronize()
    return inputs[-7:]
