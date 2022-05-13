import jax
import jax.numpy as jnp


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
    rho0 = 1024.0
    return betaS * rho0 * jnp.ones_like(temp)


@jax.jit
def isoneutral_diffusion_pre(
    maskT,
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
    salt,
    temp,
    zt,
    K_iso,
    K_11,
    K_22,
    K_33,
    Ai_ez,
    Ai_nz,
    Ai_bx,
    Ai_by,
):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    epsln = 1e-20
    iso_slopec = 1e-3
    iso_dslope = 1e-3
    K_iso_steep = 50.0
    tau = 0

    dTdx = jnp.zeros_like(K_11)
    dSdx = jnp.zeros_like(K_11)
    dTdy = jnp.zeros_like(K_11)
    dSdy = jnp.zeros_like(K_11)
    dTdz = jnp.zeros_like(K_11)
    dSdz = jnp.zeros_like(K_11)

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdT = maskT * get_drhodT(salt[:, :, :, tau], temp[:, :, :, tau], jnp.abs(zt))
    drdS = maskT * get_drhodS(salt[:, :, :, tau], temp[:, :, :, tau], jnp.abs(zt))

    """
    gradients at top face of T cells
    """
    dTdz = dTdz.at[:, :, :-1].set(
        maskW[:, :, :-1]
        * (temp[:, :, 1:, tau] - temp[:, :, :-1, tau])
        / dzw[jnp.newaxis, jnp.newaxis, :-1],
    )
    dSdz = dSdz.at[:, :, :-1].set(
        maskW[:, :, :-1]
        * (salt[:, :, 1:, tau] - salt[:, :, :-1, tau])
        / dzw[jnp.newaxis, jnp.newaxis, :-1],
    )

    """
    gradients at eastern face of T cells
    """
    dTdx = dTdx.at[:-1, :, :].set(
        maskU[:-1, :, :]
        * (temp[1:, :, :, tau] - temp[:-1, :, :, tau])
        / (dxu[:-1, jnp.newaxis, jnp.newaxis] * cost[jnp.newaxis, :, jnp.newaxis]),
    )
    dSdx = dSdx.at[:-1, :, :].set(
        maskU[:-1, :, :]
        * (salt[1:, :, :, tau] - salt[:-1, :, :, tau])
        / (dxu[:-1, jnp.newaxis, jnp.newaxis] * cost[jnp.newaxis, :, jnp.newaxis]),
    )

    """
    gradients at northern face of T cells
    """
    dTdy = dTdy.at[:, :-1, :].set(
        maskV[:, :-1, :]
        * (temp[:, 1:, :, tau] - temp[:, :-1, :, tau])
        / dyu[jnp.newaxis, :-1, jnp.newaxis],
    )
    dSdy = dSdy.at[:, :-1, :].set(
        maskV[:, :-1, :]
        * (salt[:, 1:, :, tau] - salt[:, :-1, :, tau])
        / dyu[jnp.newaxis, :-1, jnp.newaxis],
    )

    def dm_taper(sx):
        """
        tapering function for isopycnal slopes
        """
        return 0.5 * (1.0 + jnp.tanh((-jnp.abs(sx) + iso_slopec) / iso_dslope))

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = jnp.zeros_like(K_11)
    diffloc = diffloc.at[1:-2, 2:-2, 1:].set(
        0.25
        * (
            K_iso[1:-2, 2:-2, 1:]
            + K_iso[1:-2, 2:-2, :-1]
            + K_iso[2:-1, 2:-2, 1:]
            + K_iso[2:-1, 2:-2, :-1]
        ),
    )
    diffloc = diffloc.at[1:-2, 2:-2, 0].set(
        0.5 * (K_iso[1:-2, 2:-2, 0] + K_iso[2:-1, 2:-2, 0]),
    )

    sumz = jnp.zeros_like(K_11)[1:-2, 2:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for ip in range(2):
            drodxe = (
                drdT[1 + ip : -2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:]
                + drdS[1 + ip : -2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
            )
            drodze = (
                drdT[1 + ip : -2 + ip, 2:-2, ki:]
                * dTdz[1 + ip : -2 + ip, 2:-2, : -1 + kr or None]
                + drdS[1 + ip : -2 + ip, 2:-2, ki:]
                * dSdz[1 + ip : -2 + ip, 2:-2, : -1 + kr or None]
            )
            sxe = -drodxe / (jnp.minimum(0.0, drodze) - epsln)
            taper = dm_taper(sxe)
            sumz = sumz.at[:, :, ki:].set(
                sumz[..., ki:]
                + dzw[jnp.newaxis, jnp.newaxis, : -1 + kr or None]
                * maskU[1:-2, 2:-2, ki:]
                * jnp.maximum(K_iso_steep, diffloc[1:-2, 2:-2, ki:] * taper),
            )
            Ai_ez = Ai_ez.at[1:-2, 2:-2, ki:, ip, kr].set(
                taper * sxe * maskU[1:-2, 2:-2, ki:],
            )

    K_11 = K_11.at[1:-2, 2:-2, :].set(
        sumz / (4.0 * dzt[jnp.newaxis, jnp.newaxis, :]),
    )

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc = diffloc.at[...].set( 0)
    diffloc = diffloc.at[2:-2, 1:-2, 1:].set(
        0.25
        * (
            K_iso[2:-2, 1:-2, 1:]
            + K_iso[2:-2, 1:-2, :-1]
            + K_iso[2:-2, 2:-1, 1:]
            + K_iso[2:-2, 2:-1, :-1]
        ),
    )
    diffloc = diffloc.at[2:-2, 1:-2, 0].set(
        0.5 * (K_iso[2:-2, 1:-2, 0] + K_iso[2:-2, 2:-1, 0]),
    )

    sumz = jnp.zeros_like(K_11)[2:-2, 1:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for jp in range(2):
            drodyn = (
                drdT[2:-2, 1 + jp : -2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:]
                + drdS[2:-2, 1 + jp : -2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
            )
            drodzn = (
                drdT[2:-2, 1 + jp : -2 + jp, ki:]
                * dTdz[2:-2, 1 + jp : -2 + jp, : -1 + kr or None]
                + drdS[2:-2, 1 + jp : -2 + jp, ki:]
                * dSdz[2:-2, 1 + jp : -2 + jp, : -1 + kr or None]
            )
            syn = -drodyn / (jnp.minimum(0.0, drodzn) - epsln)
            taper = dm_taper(syn)
            sumz = sumz.at[:, :, ki:].set(
                sumz[..., ki:]
                + dzw[jnp.newaxis, jnp.newaxis, : -1 + kr or None]
                * maskV[2:-2, 1:-2, ki:]
                * jnp.maximum(K_iso_steep, diffloc[2:-2, 1:-2, ki:] * taper),
            )
            Ai_nz = Ai_nz.at[2:-2, 1:-2, ki:, jp, kr].set(
                taper * syn * maskV[2:-2, 1:-2, ki:],
            )
    K_22 = K_22.at[2:-2, 1:-2, :].set(
        sumz / (4.0 * dzt[jnp.newaxis, jnp.newaxis, :]),
    )

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    sumx = jnp.zeros_like(K_11)[2:-2, 2:-2, :-1]
    sumy = jnp.zeros_like(K_11)[2:-2, 2:-2, :-1]

    for kr in range(2):
        drodzb = (
            drdT[2:-2, 2:-2, kr : -1 + kr or None] * dTdz[2:-2, 2:-2, :-1]
            + drdS[2:-2, 2:-2, kr : -1 + kr or None] * dSdz[2:-2, 2:-2, :-1]
        )

        # eastward slopes at the top of T cells
        for ip in range(2):
            drodxb = (
                drdT[2:-2, 2:-2, kr : -1 + kr or None]
                * dTdx[1 + ip : -3 + ip, 2:-2, kr : -1 + kr or None]
                + drdS[2:-2, 2:-2, kr : -1 + kr or None]
                * dSdx[1 + ip : -3 + ip, 2:-2, kr : -1 + kr or None]
            )
            sxb = -drodxb / (jnp.minimum(0.0, drodzb) - epsln)
            taper = dm_taper(sxb)
            sumx += (
                dxu[1 + ip : -3 + ip, jnp.newaxis, jnp.newaxis]
                * K_iso[2:-2, 2:-2, :-1]
                * taper
                * sxb ** 2
                * maskW[2:-2, 2:-2, :-1]
            )
            Ai_bx = Ai_bx.at[2:-2, 2:-2, :-1, ip, kr].set(
                taper * sxb * maskW[2:-2, 2:-2, :-1],
            )

        # northward slopes at the top of T cells
        for jp in range(2):
            facty = cosu[1 + jp : -3 + jp] * dyu[1 + jp : -3 + jp]
            drodyb = (
                drdT[2:-2, 2:-2, kr : -1 + kr or None]
                * dTdy[2:-2, 1 + jp : -3 + jp, kr : -1 + kr or None]
                + drdS[2:-2, 2:-2, kr : -1 + kr or None]
                * dSdy[2:-2, 1 + jp : -3 + jp, kr : -1 + kr or None]
            )
            syb = -drodyb / (jnp.minimum(0.0, drodzb) - epsln)
            taper = dm_taper(syb)
            sumy += (
                facty[jnp.newaxis, :, jnp.newaxis]
                * K_iso[2:-2, 2:-2, :-1]
                * taper
                * syb ** 2
                * maskW[2:-2, 2:-2, :-1]
            )
            Ai_by = Ai_by.at[2:-2, 2:-2, :-1, jp, kr].set(
                taper * syb * maskW[2:-2, 2:-2, :-1],
            )

    K_33 = K_33.at[2:-2, 2:-2, :-1].set(
        sumx / (4 * dxt[2:-2, jnp.newaxis, jnp.newaxis])
        + sumy
        / (
            4
            * dyt[jnp.newaxis, 2:-2, jnp.newaxis]
            * cost[jnp.newaxis, 2:-2, jnp.newaxis]
        ),
    )
    K_33 = K_33.at[2:-2, 2:-2, -1].set( 0.0)

    return K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by


def prepare_inputs(*inputs, device):
    out = [jnp.array(k) for k in inputs]
    for o in out:
        o.block_until_ready()
    return out


def run(*inputs, device="cpu"):
    outputs = isoneutral_diffusion_pre(*inputs)
    for o in outputs:
        o.block_until_ready()
    return outputs
