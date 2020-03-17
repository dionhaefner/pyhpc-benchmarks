import numpy
import jax
import jax.numpy as np


@jax.jit
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


@jax.jit
def get_drhodS(salt, temp, p):
    betaS = 0.78e-3
    rho0 = 1024.
    return betaS * rho0 * np.ones_like(temp)


@jax.jit
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
        salt[ tau, :, :,:], temp[ tau, :, :,:], np.abs(zt)[:, np.newaxis, np.newaxis]
    )
    drdS = maskT * get_drhodS(
        salt[ tau, :, :,:], temp[ tau, :, :,:], np.abs(zt)[:, np.newaxis, np.newaxis]
    )

    """
    gradients at top face of T cells
    """
    dTdz = jax.ops.index_update(
        dTdz, jax.ops.index[ :-1, :,:], maskW[ :-1, :,:] * \
        (temp[ tau, 1:, :,:] - temp[ tau, :-1, :,:]) / \
        dzw[:-1, np.newaxis, np.newaxis]
    )
    dSdz = jax.ops.index_update(
        dSdz, jax.ops.index[ :-1, :,:], maskW[ :-1, :,:] * \
        (salt[ tau, 1:, :,:] - salt[ tau, :-1, :,:]) / \
        dzw[:-1, np.newaxis, np.newaxis]
    )

    """
    gradients at eastern face of T cells
    """
    dTdx = jax.ops.index_update(
        dTdx, jax.ops.index[ :, :,:-1], maskU[ :, :,:-1] * (temp[tau, :, :, 1:] - temp[ tau, :, :,:-1]) \
        / (dxu[np.newaxis, np.newaxis, :-1] * cost[np.newaxis, :, np.newaxis])
    )
    dSdx = jax.ops.index_update(
        dSdx, jax.ops.index[ :, :,:-1], maskU[ :, :,:-1] *
        (salt[ tau, :, :,1:] - salt[ tau, :, :,:-1])
        / (dxu[np.newaxis, np.newaxis, :-1] * cost[np.newaxis, :, np.newaxis])
    )

    """
    gradients at northern face of T cells
    """
    dTdy = jax.ops.index_update(
        dTdy, jax.ops.index[ :, :-1,:], maskV[ :, :-1,:] * \
        (temp[ tau, :, 1:,:] - temp[ tau, :, :-1,:]) \
        / dyu[np.newaxis, :-1, np.newaxis]
    )
    dSdy = jax.ops.index_update(dSdy, jax.ops.index[ :, :-1,:], maskV[ :, :-1,:] * \
        (salt[ tau, :, 1:,:] - salt[ tau, :, :-1,:]) \
        / dyu[np.newaxis, :-1, np.newaxis]
    )

    def dm_taper(sx):
        """
        tapering function for isopycnal slopes
        """
        return 0.5 * (1. + np.tanh((-np.abs(sx) + iso_slopec) / iso_dslope))

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = np.zeros_like(K_11)
    diffloc = jax.ops.index_update(
        diffloc, jax.ops.index[ 1:, 2:-2,1:-2],
        0.25 * (K_iso[ 1:, 2:-2,1:-2] + K_iso[ :-1, 2:-2,1:-2]
               + K_iso[ 1:, 2:-2,2:-1] + K_iso[ :-1, 2:-2,2:-1])
    )
    diffloc = jax.ops.index_update(
        diffloc, jax.ops.index[ 0, 2:-2,1:-2],
        0.5 * (K_iso[ 0, 2:-2,1:-2] + K_iso[ 0, 2:-2,2:-1])
    )

    sumz = np.zeros_like(K_11)[:, 2:-2,1:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for ip in range(2):
            drodxe = drdT[ ki:, 2:-2,1 + ip:-2 + ip] * dTdx[ ki:, 2:-2,1:-2] \
                + drdS[ ki:, 2:-2,1 + ip:-2 + ip] * dSdx[ ki:, 2:-2,1:-2]
            drodze = drdT[ ki:, 2:-2,1 + ip:-2 + ip] * dTdz[ :-1 + kr or None, 2:-2,1 + ip:-2 + ip] \
                + drdS[ ki:, 2:-2,1 + ip:-2 + ip] * \
                dSdz[ :-1 + kr or None, 2:-2,1 + ip:-2 + ip]
            sxe = -drodxe / (np.minimum(0., drodze) - epsln)
            taper = dm_taper(sxe)
            sumz = jax.ops.index_update(
                sumz, jax.ops.index[ ki:, :,:], sumz[ki:, ...] + dzw[:-1 + kr or None, np.newaxis, np.newaxis] * maskU[ ki:, 2:-2,1:-2]
                * np.maximum(K_iso_steep, diffloc[ ki:, 2:-2,1:-2] * taper)
            )
            Ai_ez = jax.ops.index_update(
                Ai_ez, jax.ops.index[kr, ip, ki:, 2:-2, 1:-2],
                taper * sxe * maskU[ ki:, 2:-2,1:-2]
            )

    K_11 = jax.ops.index_update(
        K_11, jax.ops.index[ :, 2:-2,1:-2],
        sumz / (4. * dzt[:, np.newaxis, np.newaxis])
    )

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc = jax.ops.index_update(
        diffloc, jax.ops.index[...], 0
    )
    diffloc = jax.ops.index_update(
        diffloc, jax.ops.index[ 1:, 1:-2,2:-2],
        0.25 * (K_iso[ 1:, 1:-2,2:-2] + K_iso[ :-1, 1:-2,2:-2]
        + K_iso[ 1:, 2:-1,2:-2] + K_iso[ :-1, 2:-1,2:-2])
    )
    diffloc = jax.ops.index_update(
        diffloc, jax.ops.index[ 0, 1:-2,2:-2],
        0.5 *
        (K_iso[ 0, 1:-2,2:-2] + K_iso[ 0, 2:-1,2:-2])
    )

    sumz = np.zeros_like(K_11)[:, 1:-2,2:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for jp in range(2):
            drodyn = drdT[ ki:, 1 + jp:-2 + jp,2:-2] * dTdy[ ki:, 1:-2,2:-2] + \
                drdS[ ki:, 1 + jp:-2 + jp,2:-2] * dSdy[ ki:, 1:-2,2:-2]
            drodzn = drdT[ ki:, 1 + jp:-2 + jp,2:-2] * dTdz[ :-1 + kr or None, 1 + jp:-2 + jp,2:-2] \
                + drdS[ ki:, 1 + jp:-2 + jp,2:-2] * \
                dSdz[ :-1 + kr or None, 1 + jp:-2 + jp,2:-2]
            syn = -drodyn / (np.minimum(0., drodzn) - epsln)
            taper = dm_taper(syn)
            sumz = jax.ops.index_update(
                sumz, jax.ops.index[ ki:, :,:], sumz[ki:, ...] + dzw[:-1 + kr or None, np.newaxis, np.newaxis]
                * maskV[ ki:, 1:-2,2:-2] * np.maximum(K_iso_steep, diffloc[ ki:, 1:-2,2:-2] * taper)
            )
            Ai_nz = jax.ops.index_update(
                Ai_nz, jax.ops.index[kr, jp, ki:, 1:-2, 2:-2],
                taper * syn * maskV[ ki:, 1:-2,2:-2]
            )
    K_22 = jax.ops.index_update(
        K_22, jax.ops.index[ :, 1:-2,2:-2],
        sumz / (4. * dzt[:, np.newaxis, np.newaxis])
    )

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    sumx = np.zeros_like(K_11)[ :-1, 2:-2,2:-2]
    sumy = np.zeros_like(K_11)[ :-1, 2:-2,2:-2]

    for kr in range(2):
        drodzb = drdT[ kr:-1 + kr or None, 2:-2,2:-2] * dTdz[ :-1, 2:-2,2:-2] \
            + drdS[ kr:-1 + kr or None, 2:-2,2:-2] * dSdz[ :-1, 2:-2,2:-2]

        # eastward slopes at the top of T cells
        for ip in range(2):
            drodxb = drdT[ kr:-1 + kr or None, 2:-2,2:-2] * dTdx[ kr:-1 + kr or None, 2:-2,1 + ip:-3 + ip] \
                + drdS[ kr:-1 + kr or None, 2:-2,2:-2] * dSdx[ kr:-1 + kr or None, 2:-2,1 + ip:-3 + ip]
            sxb = -drodxb / (np.minimum(0., drodzb) - epsln)
            taper = dm_taper(sxb)
            sumx += dxu[np.newaxis, np.newaxis, 1 + ip:-3 + ip] * \
                K_iso[ :-1, 2:-2,2:-2] * taper * \
                sxb**2 * maskW[ :-1, 2:-2,2:-2]
            Ai_bx = jax.ops.index_update(
                Ai_bx, jax.ops.index[kr, ip, :-1, 2:-2, 2:-2],
                taper *
                sxb * maskW[ :-1, 2:-2,2:-2]
            )

        # northward slopes at the top of T cells
        for jp in range(2):
            facty = cosu[1 + jp:-3 + jp] * dyu[1 + jp:-3 + jp]
            drodyb = drdT[ kr:-1 + kr or None, 2:-2,2:-2] * dTdy[ kr:-1 + kr or None, 1 + jp:-3 + jp,2:-2] \
                + drdS[ kr:-1 + kr or None, 2:-2,2:-2] * dSdy[ kr:-1 + kr or None, 1 + jp:-3 + jp,2:-2]
            syb = -drodyb / (np.minimum(0., drodzb) - epsln)
            taper = dm_taper(syb)
            sumy += facty[np.newaxis, :, np.newaxis] * K_iso[ :-1, 2:-2,2:-2] \
                * taper * syb**2 * maskW[ :-1, 2:-2,2:-2]
            Ai_by = jax.ops.index_update(
                Ai_by, jax.ops.index[kr, jp, :-1, 2:-2, 2:-2],
                taper * syb * maskW[ :-1, 2:-2,2:-2]
            )

    K_33 = jax.ops.index_update(
        K_33, jax.ops.index[ :-1, 2:-2,2:-2],
        sumx / (4 * dxt[np.newaxis, np.newaxis, 2:-2]) + \
        sumy / (4 * dyt[np.newaxis, 2:-2, np.newaxis]
                * cost[np.newaxis, 2:-2, np.newaxis])
    )
    K_33 = jax.ops.index_update(
        K_33, jax.ops.index[ -1, 2:-2,2:-2], 0.
    )

    return K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by


def prepare_inputs(*inputs, device):
    out = [np.array(k.T) for k in inputs]
    for o in out:
        o.block_until_ready()
    return out


def run(*inputs, device='cpu'):
    outputs = isoneutral_diffusion_pre(*inputs)
    for o in outputs:
        o.block_until_ready()
    return tuple(o.T for o in outputs)
