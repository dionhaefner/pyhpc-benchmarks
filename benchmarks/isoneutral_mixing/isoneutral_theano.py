import numpy as np
import theano
import theano.tensor as T


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
    return betaS * rho0 * T.ones_like(temp)


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

    dTdx = T.zeros_like(K_11)
    dSdx = T.zeros_like(K_11)
    dTdy = T.zeros_like(K_11)
    dSdy = T.zeros_like(K_11)
    dTdz = T.zeros_like(K_11)
    dSdz = T.zeros_like(K_11)

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
    dTdz = T.set_subtensor(dTdz[:, :, :-1], maskW[:, :, :-1] *
        (temp[:, :, 1:, tau] - temp[:, :, :-1, tau]) / \
        dzw[:, :, :-1]
    )
    dSdz = T.set_subtensor(dSdz[:, :, :-1], maskW[:, :, :-1] *
        (salt[:, :, 1:, tau] - salt[:, :, :-1, tau]) / \
        dzw[:, :, :-1]
    )

    """
    gradients at eastern face of T cells
    """
    dTdx = T.set_subtensor(dTdx[:-1, :, :], maskU[:-1, :, :] * (temp[1:, :, :, tau] - temp[:-1, :, :, tau])
        / (dxu[:-1, :, :] * cost[:, :, :])
    )
    dSdx = T.set_subtensor(dSdx[:-1, :, :], maskU[:-1, :, :] * (salt[1:, :, :, tau] - salt[:-1, :, :, tau])
        / (dxu[:-1, :, :] * cost[:, :, :])
    )

    """
    gradients at northern face of T cells
    """
    dTdy = T.set_subtensor(dTdy[:, :-1, :], maskV[:, :-1, :] *
        (temp[:, 1:, :, tau] - temp[:, :-1, :, tau]) \
        / dyu[:, :-1, :]
    )
    dSdy = T.set_subtensor(dSdy[:, :-1, :], maskV[:, :-1, :] *
        (salt[:, 1:, :, tau] - salt[:, :-1, :, tau]) \
        / dyu[:, :-1, :]
    )

    def dm_taper(sx):
        """
        tapering function for isopycnal slopes
        """
        return 0.5 * (1. + T.tanh((-abs(sx) + iso_slopec) / iso_dslope))

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    diffloc = T.zeros_like(K_11)
    diffloc = T.set_subtensor(
        diffloc[1:-2, 2:-2, 1:],
        0.25 * (K_iso[1:-2, 2:-2, 1:] + K_iso[1:-2, 2:-2, :-1]
        + K_iso[2:-1, 2:-2, 1:] + K_iso[2:-1, 2:-2, :-1])
    )
    diffloc = T.set_subtensor(
        diffloc[1:-2, 2:-2, 0],
        0.5 * (K_iso[1:-2, 2:-2, 0] + K_iso[2:-1, 2:-2, 0])
    )

    sumz = T.zeros_like(K_11)[1:-2, 2:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for ip in range(2):
            drodxe = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdx[1:-2, 2:-2, ki:] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * dSdx[1:-2, 2:-2, ki:]
            drodze = drdT[1 + ip:-2 + ip, 2:-2, ki:] * dTdz[1 + ip:-2 + ip, 2:-2, :-1 + kr or None] \
                + drdS[1 + ip:-2 + ip, 2:-2, ki:] * \
                dSdz[1 + ip:-2 + ip, 2:-2, :-1 + kr or None]
            sxe = -drodxe / (T.minimum(0., drodze) - epsln)
            taper = dm_taper(sxe)
            sumz = T.inc_subtensor(sumz[:, :, ki:], dzw[:, :, :-1 + kr or None] * maskU[1:-2, 2:-2, ki:]
                * T.maximum(K_iso_steep, diffloc[1:-2, 2:-2, ki:] * taper)
            )
            Ai_ez = T.set_subtensor(Ai_ez[1:-2, 2:-2, ki:, ip, kr], taper *
                sxe * maskU[1:-2, 2:-2, ki:]
            )
    K_11 = T.set_subtensor(K_11[1:-2, 2:-2, :], sumz / (4. * dzt[:, :, :]))

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    diffloc = T.set_subtensor(diffloc[...], 0)
    diffloc = T.set_subtensor(
        diffloc[2:-2, 1:-2, 1:],
        0.25 * (K_iso[2:-2, 1:-2, 1:] + K_iso[2:-2, 1:-2, :-1]
        + K_iso[2:-2, 2:-1, 1:] + K_iso[2:-2, 2:-1, :-1])
    )
    diffloc = T.set_subtensor(
        diffloc[2:-2, 1:-2, 0],
        0.5 * (K_iso[2:-2, 1:-2, 0] + K_iso[2:-2, 2:-1, 0])
    )

    sumz = T.zeros_like(K_11)[2:-2, 1:-2]
    for kr in range(2):
        ki = 0 if kr == 1 else 1
        for jp in range(2):
            drodyn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdy[2:-2, 1:-2, ki:] + \
                drdS[2:-2, 1 + jp:-2 + jp, ki:] * dSdy[2:-2, 1:-2, ki:]
            drodzn = drdT[2:-2, 1 + jp:-2 + jp, ki:] * dTdz[2:-2, 1 + jp:-2 + jp, :-1 + kr or None] \
                + drdS[2:-2, 1 + jp:-2 + jp, ki:] * \
                dSdz[2:-2, 1 + jp:-2 + jp, :-1 + kr or None]
            syn = -drodyn / (T.minimum(0., drodzn) - epsln)
            taper = dm_taper(syn)
            sumz = T.inc_subtensor(sumz[:, :, ki:], dzw[:, :, :-1 + kr or None]
                * maskV[2:-2, 1:-2, ki:] * T.maximum(K_iso_steep, diffloc[2:-2, 1:-2, ki:] * taper))
            Ai_nz = T.set_subtensor(Ai_nz[2:-2, 1:-2, ki:, jp, kr], taper *
                syn * maskV[2:-2, 1:-2, ki:]
            )
    K_22 = T.set_subtensor(K_22[2:-2, 1:-2, :], sumz / (4. * dzt[:, :, :]))

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    sumx = T.zeros_like(K_11)[2:-2, 2:-2, :-1]
    sumy = T.zeros_like(K_11)[2:-2, 2:-2, :-1]

    for kr in range(2):
        drodzb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdz[2:-2, 2:-2, :-1] \
            + drdS[2:-2, 2:-2, kr:-1 + kr or None] * dSdz[2:-2, 2:-2, :-1]

        # eastward slopes at the top of T cells
        for ip in range(2):
            drodxb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdx[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None] \
                + drdS[2:-2, 2:-2, kr:-1 + kr or None] * \
                dSdx[1 + ip:-3 + ip, 2:-2, kr:-1 + kr or None]
            sxb = -drodxb / (T.minimum(0., drodzb) - epsln)
            taper = dm_taper(sxb)
            sumx += dxu[1 + ip:-3 + ip, :, :] * \
                K_iso[2:-2, 2:-2, :-1] * taper * \
                sxb**2 * maskW[2:-2, 2:-2, :-1]
            Ai_bx = T.set_subtensor(Ai_bx[2:-2, 2:-2, :-1, ip, kr], taper *
                sxb * maskW[2:-2, 2:-2, :-1])

        # northward slopes at the top of T cells
        for jp in range(2):
            facty = cosu[:, 1 + jp:-3 + jp] * dyu[:, 1 + jp:-3 + jp]
            drodyb = drdT[2:-2, 2:-2, kr:-1 + kr or None] * dTdy[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None] \
                + drdS[2:-2, 2:-2, kr:-1 + kr or None] * \
                dSdy[2:-2, 1 + jp:-3 + jp, kr:-1 + kr or None]
            syb = -drodyb / (T.minimum(0., drodzb) - epsln)
            taper = dm_taper(syb)
            sumy += facty * K_iso[2:-2, 2:-2, :-1] \
                * taper * syb**2 * maskW[2:-2, 2:-2, :-1]
            Ai_by = T.set_subtensor(Ai_by[2:-2, 2:-2, :-1, jp, kr], taper *
                syb * maskW[2:-2, 2:-2, :-1])

    K_33 = T.set_subtensor(K_33[2:-2, 2:-2, :-1], sumx / (4 * dxt[2:-2, :, :]) +
        sumy / (4 * dyt[:, 2:-2, :]
                * cost[:, 2:-2, :]))
    K_33 = T.set_subtensor(K_33[2:-2, 2:-2, -1], 0.)

    return K_11, K_22, K_33, Ai_ez, Ai_nz, Ai_bx, Ai_by


t1d_x = theano.tensor.TensorType(dtype='float64', broadcastable=(False, True, True))
t1d_y = theano.tensor.TensorType(dtype='float64', broadcastable=(True, False, True))
t1d_z = theano.tensor.TensorType(dtype='float64', broadcastable=(True, True, False))

symbolic_inputs = [
    theano.tensor.dtensor3('maskT'),
    theano.tensor.dtensor3('maskU'),
    theano.tensor.dtensor3('maskV'),
    theano.tensor.dtensor3('maskW'),
    t1d_x('dxt'),
    t1d_x('dxu'),
    t1d_y('dyt'),
    t1d_y('dyu'),
    t1d_z('dzt'),
    t1d_z('dzw'),
    t1d_y('cost'),
    t1d_y('cosu'),
    theano.tensor.dtensor4('salt'),
    theano.tensor.dtensor4('temp'),
    t1d_z('zt'),
    theano.tensor.dtensor3('K_iso'),
    theano.tensor.dtensor3('K_11'),
    theano.tensor.dtensor3('K_22'),
    theano.tensor.dtensor3('K_33'),
    theano.tensor.dtensor5('Ai_ez'),
    theano.tensor.dtensor5('Ai_nz'),
    theano.tensor.dtensor5('Ai_bx'),
    theano.tensor.dtensor5('Ai_by'),
]
isoneutral_theano = theano.function(symbolic_inputs, isoneutral_diffusion_pre(*symbolic_inputs))


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
    outputs = isoneutral_theano(*inputs)
    return outputs
