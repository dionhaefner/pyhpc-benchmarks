import aesara
import aesara.tensor as aet


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
    return betaS * rho0 * aet.ones_like(temp)


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

    dTdx = aet.zeros_like(K_11)
    dSdx = aet.zeros_like(K_11)
    dTdy = aet.zeros_like(K_11)
    dSdy = aet.zeros_like(K_11)
    dTdz = aet.zeros_like(K_11)
    dSdz = aet.zeros_like(K_11)

    """
    drho_dt and drho_ds at centers of T cells
    """
    drdT = maskT * get_drhodT(
        salt[:, :, :, tau], temp[:, :, :, tau], abs(zt)
    )
    drdS = maskT * get_drhodS(
        salt[:, :, :, tau], temp[:, :, :, tau], abs(zt)
    )

    """
    gradients at top face of T cells
    """
    dTdz = aet.set_subtensor(dTdz[:, :, :-1], maskW[:, :, :-1] *
        (temp[:, :, 1:, tau] - temp[:, :, :-1, tau]) / \
        dzw[:, :, :-1]
    )
    dSdz = aet.set_subtensor(dSdz[:, :, :-1], maskW[:, :, :-1] *
        (salt[:, :, 1:, tau] - salt[:, :, :-1, tau]) / \
        dzw[:, :, :-1]
    )

    """
    gradients at eastern face of T cells
    """
    dTdx = aet.set_subtensor(dTdx[:-1, :, :], maskU[:-1, :, :] * (temp[1:, :, :, tau] - temp[:-1, :, :, tau])
        / (dxu[:-1, :, :] * cost[:, :, :])
    )
    dSdx = aet.set_subtensor(dSdx[:-1, :, :], maskU[:-1, :, :] * (salt[1:, :, :, tau] - salt[:-1, :, :, tau])
        / (dxu[:-1, :, :] * cost[:, :, :])
    )

    """
    gradients at northern face of T cells
    """
    dTdy = aet.set_subtensor(dTdy[:, :-1, :], maskV[:, :-1, :] *
        (temp[:, 1:, :, tau] - temp[:, :-1, :, tau]) \
        / dyu[:, :-1, :]
    )
    dSdy = aet.set_subtensor(dSdy[:, :-1, :], maskV[:, :-1, :] *
        (salt[:, 1:, :, tau] - salt[:, :-1, :, tau]) \
        / dyu[:, :-1, :]
    )

    def dm_taper(sx):
        """
        tapering function for isopycnal slopes
        """
        return 0.5 * (1. + aet.tanh((-abs(sx) + iso_slopec) / iso_dslope))

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = aet.zeros_like(K_11)
    diffloc = aet.set_subtensor(
        diffloc[1:-2, 2:-2, 1:],
        0.25 * (K_iso[1:-2, 2:-2, 1:] + K_iso[1:-2, 2:-2, :-1]
        + K_iso[2:-1, 2:-2, 1:] + K_iso[2:-1, 2:-2, :-1])
    )
    diffloc = aet.set_subtensor(
        diffloc[1:-2, 2:-2, 0],
        0.5 * (K_iso[1:-2, 2:-2, 0] + K_iso[2:-1, 2:-2, 0])
    )

    sumz = aet.zeros_like(K_11)[1:-2, 2:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for ip in range(2):
            drodxe = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
            drodze = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdz[1 + ip:-2 + ip, 2:-2, :-1 + kr or None] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * \
                dSdz[1 + ip:-2 + ip, 2:-2, :-1 + kr or None]
            sxe = -drodxe / (aet.minimum(0., drodze) - epsln)
            taper = dm_taper(sxe)
            sumz = aet.inc_subtensor(sumz[:, :, ki:], dzw[:, :, :-1 + kr or None] * maskU[1:-2, 2:-2, ki:]
                * aet.maximum(K_iso_steep, diffloc[1:-2, 2:-2, ki:] * taper)
            )
            Ai_ez = aet.set_subtensor(Ai_ez[1:-2, 2:-2, ki:, ip, kr], taper *
                sxe * maskU[1:-2, 2:-2, ki:]
            )
    K_11 = aet.set_subtensor(K_11[1:-2, 2:-2, :], sumz / (4. * dzt[:, :, :]))

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc = aet.set_subtensor(diffloc[...], 0)
    diffloc = aet.set_subtensor(
        diffloc[2:-2, 1:-2, 1:],
        0.25 * (K_iso[2:-2, 1:-2, 1:] + K_iso[2:-2, 1:-2, :-1]
        + K_iso[2:-2, 2:-1, 1:] + K_iso[2:-2, 2:-1, :-1])
    )
    diffloc = aet.set_subtensor(
        diffloc[2:-2, 1:-2, 0],
        0.5 * (K_iso[2:-2, 1:-2, 0] + K_iso[2:-2, 2:-1, 0])
    )

    sumz = aet.zeros_like(K_11)[2:-2, 1:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for jp in range(2):
            drodyn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:] + \
                drdS[2:-2, 1 + jp:-2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
            drodzn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdz[2:-2, 1 + jp:-2 + jp, :-1 + kr or None] \
                + drdS[2:-2, 1 + jp:-2 + jp, ki:] * \
                dSdz[2:-2, 1 + jp:-2 + jp, :-1 + kr or None]
            syn = -drodyn / (aet.minimum(0., drodzn) - epsln)
            taper = dm_taper(syn)
            sumz = aet.inc_subtensor(sumz[:, :, ki:], dzw[:, :, :-1 + kr or None]
                * maskV[2:-2, 1:-2, ki:] * aet.maximum(K_iso_steep, diffloc[2:-2, 1:-2, ki:] * taper))
            Ai_nz = aet.set_subtensor(Ai_nz[2:-2, 1:-2, ki:, jp, kr], taper *
                syn * maskV[2:-2, 1:-2, ki:]
            )
    K_22 = aet.set_subtensor(K_22[2:-2, 1:-2, :], sumz / (4. * dzt[:, :, :]))

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    sumx = aet.zeros_like(K_11)[2:-2, 2:-2, :-1]
    sumy = aet.zeros_like(K_11)[2:-2, 2:-2, :-1]

    for kr in range(2):
        drodzb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdz[2:-2, 2:-2, :-1] \
            + drdS[2:-2, 2:-2, kr:-1 + kr or None] * dSdz[2:-2, 2:-2, :-1]

        # eastward slopes at the top of T cells
        for ip in range(2):
            drodxb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdx[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None] \
                + drdS[2:-2, 2:-2, kr:-1 + kr or None] * \
                dSdx[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None]
            sxb = -drodxb / (aet.minimum(0., drodzb) - epsln)
            taper = dm_taper(sxb)
            sumx += dxu[1 + ip:-3 + ip, :, :] * \
                K_iso[2:-2, 2:-2, :-1] * taper * \
                sxb**2 * maskW[2:-2, 2:-2, :-1]
            Ai_bx = aet.set_subtensor(Ai_bx[2:-2, 2:-2, :-1, ip, kr], taper *
                sxb * maskW[2:-2, 2:-2, :-1])

        # northward slopes at the top of T cells
        for jp in range(2):
            facty = cosu[:, 1 + jp:-3 + jp] * dyu[:, 1 + jp:-3 + jp]
            drodyb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdy[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None] \
                + drdS[2:-2, 2:-2, kr:-1 + kr or None] * \
                dSdy[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None]
            syb = -drodyb / (aet.minimum(0., drodzb) - epsln)
            taper = dm_taper(syb)
            sumy += facty * K_iso[2:-2, 2:-2, :-1] \
                * taper * syb**2 * maskW[2:-2, 2:-2, :-1]
            Ai_by = aet.set_subtensor(Ai_by[2:-2, 2:-2, :-1, jp, kr], taper *
                syb * maskW[2:-2, 2:-2, :-1])

    K_33 = aet.set_subtensor(K_33[2:-2, 2:-2, :-1], sumx / (4 * dxt[2:-2, :, :]) +
        sumy / (4 * dyt[:, 2:-2, :]
                * cost[:, 2:-2, :]))
    K_33 = aet.set_subtensor(K_33[2:-2, 2:-2, -1], 0.)

    return K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by


t1d_x = aesara.tensor.TensorType(dtype='float64', broadcastable=(False, True, True))
t1d_y = aesara.tensor.TensorType(dtype='float64', broadcastable=(True, False, True))
t1d_z = aesara.tensor.TensorType(dtype='float64', broadcastable=(True, True, False))

symbolic_inputs = [
    aesara.tensor.dtensor3('maskT'),
    aesara.tensor.dtensor3('maskU'),
    aesara.tensor.dtensor3('maskV'),
    aesara.tensor.dtensor3('maskW'),
    t1d_x('dxt'),
    t1d_x('dxu'),
    t1d_y('dyt'),
    t1d_y('dyu'),
    t1d_z('dzt'),
    t1d_z('dzw'),
    t1d_y('cost'),
    t1d_y('cosu'),
    aesara.tensor.dtensor4('salt'),
    aesara.tensor.dtensor4('temp'),
    t1d_z('zt'),
    aesara.tensor.dtensor3('K_iso'),
    aesara.tensor.dtensor3('K_11'),
    aesara.tensor.dtensor3('K_22'),
    aesara.tensor.dtensor3('K_33'),
    aesara.tensor.dtensor5('Ai_ez'),
    aesara.tensor.dtensor5('Ai_nz'),
    aesara.tensor.dtensor5('Ai_bx'),
    aesara.tensor.dtensor5('Ai_by'),
]
isoneutral_aesara = aesara.function(symbolic_inputs, isoneutral_diffusion_pre(*symbolic_inputs))


def prepare_inputs(*inputs, device):
    inputs = list(inputs)

    for i in (4, 5):
        inputs[i] = inputs[i].reshape(-1, 1, 1)

    for i in (6, 7, 10, 11):
        inputs[i] = inputs[i].reshape(1, -1, 1)

    for i in (8, 9, 14):
        inputs[i] = inputs[i].reshape(1, 1, -1)

    return inputs


def run(*inputs, device='cpu'):
    outputs = isoneutral_aesara(*inputs)
    return outputs
