import numpy as np


def const(t, S0):
    r"""Constant.

    .. math::

        S(t) = S0

    Args:
        t (array): array of time points
        S0 (float): constant value

    Returns:
        numpy.ndarray: Signal versus time, same size as t.
    """
    return np.full(t.shape, S0, dtype=t.dtype)


def lin(t, S0, R):
    r"""Linear function.

    .. math::

        S(t) = S0(1 + Rt)

    Args:
        t (array): array of time points
        S0 (float): signal scaling factor
        R (float): relaxation rate in units 1/[t]

    Returns:
        numpy.ndarray: Signal versus time, same size as t.
    """
    return S0 * (1 + R * t)

def lin_init(t, S, p0):
    r"""Estimate linear parameters.

    Args:
        t (array): array of time points
        S (array): signal at each value of t
        p0 (array): dimensionless initial values

    Returns:
        list: Estimates of S0 and T
    """
    S0 = S[0]
    return [S0*p0[0], p0[1]]


def quad(t, S0, R, A):
    r"""Quadratic function.

    .. math::

        S(t) = S0(1 + Rt + At^2)

    Args:
        t (array): array of time points
        S0 (float): signal scaling factor
        R (float): relaxation rate in units 1/[t]
        A (float): amplitude of quadratic term in units of 1/[t]^2

    Returns:
        numpy.ndarray: Signal versus time, same size as t.
    """
    return S0 * (1 + R * t + A * t**2)


def othree(t, S0, R, A, B):
    r"""Third order polynomial function.

    .. math::

        S(t) = S0(1 + Rt + At^2 + Bt^3)

    Args:
        t (array): array of time points
        S0 (float): signal scaling factor
        R (float): relaxation rate in units 1/[t]
        A (float): amplitude of quadratic term in units of [t]^2
        B (float): amplitude of third order term in units of [t]^3

    Returns:
        numpy.ndarray: Signal versus time, same size as t.
    """
    return S0 * (1 + R * t + A * t**2 + B * t**3)


def ofour(t, S0, R, A, B, C):
    r"""Foruth order polynomial function.

    .. math::

        S(t) = S0(1 + Rt + At^2 + Bt^3 + Ct^4)

    Args:
        t (array): array of time points
        S0 (float): signal scaling factor
        R (float): relaxation rate in units 1/[t]
        A (float): amplitude of quadratic term in units of [t]^2
        B (float): amplitude of third order term in units of [t]^3
        C (float): amplitude of fourth order term in units of [t]^4

    Returns:
        numpy.ndarray: Signal versus time, same size as t.
    """
    return S0 * (1 + R * t + A * t**2 + B * t**3 + C * t**4)


def exp_decay(t, S0, T):
    r"""Exponential decay.

    .. math::

        S = S_0 e^{-t/T}

    Args:
        t (array): array of time points
        S0 (float): signal scaling factor
        T (float): relaxation time in same units as t

    Returns:
        numpy.ndarray: Signal versus time, same size as t.
    """
    return S0*np.exp(-t/T)


def exp_decay_init(t, S, p0):
    r"""Estimate exponential decay parameters.

    Args:
        t (array): array of time points
        S (array): signal at each value of t
        p0 (array): dimensionless initial values

    Returns:
        list: Estimates of S0 and T
    """
    S0 = np.amax([np.amax(S),0])
    return [S0*p0[0], p0[1]]

def exp_recovery_2p(t, S0, T):
    r"""Exponential recovery with 2 parameters

    .. math::

        S = S_0 \left( 1 - 2 e^{-t/T} \right)

    Args:
        t (array): array of time points
        S0 (float): signal scaling factor
        T (float): relaxation time in same units as t

    Returns:
        numpy.ndarray: Signal versus time, same size as t.
    """
    return S0 * (1 - 2 * np.exp(-t/T))

def exp_recovery_2p_init(t, S, p0):
    r"""Estimate exponential recovery parameters.

    Args:
        t (array): array of time points
        S (array): signal at each value of t
        p0 (array): dimensionless initial values

    Returns:
        list: Estimates of S0 and T
    """
    S0 = np.amax(np.abs(S))
    return [S0*p0[0], p0[1]]

def abs_exp_recovery_2p(t, S0, T):
    r"""Absolute value of exponential recovery with 2 parameters

    .. math::

        S = \left| S_0 \left( 1 - 2 e^{-t/T} \right) \right|

    Args:
        t (array): array of time points
        S0 (float): signal scaling factor
        T (float): relaxation time in same units as t

    Returns:
        numpy.ndarray: Signal versus time, same size as t.
    """
    return np.abs(S0 * (1 - 2 * np.exp(-t/T)))

def abs_exp_recovery_2p_init(t, S, p0):
    r"""Estimate exponential recovery parameters.

    Args:
        t (array): array of time points
        S (array): signal at each value of t
        p0 (array): dimensionless initial values

    Returns:
        list: Estimates of S0 and T
    """
    S0 = np.amax(np.abs(S))
    return [S0*p0[0], p0[1]]

def exp_recovery_3p(t, S0, T, A):
    r"""Exponential recovery with 3 parameters

    .. math::

        S = S_0 \left( 1 - A e^{-t/T} \right) 

    Args:
        t (array): array of time points
        S0 (float): signal scaling factor
        T (float): relaxation time in same units as t
        A (float): Amplitude of exponential term

    Returns:
        numpy.ndarray: Signal versus time, same size as t.
    """
    return S0 * (1 - A * np.exp(-t/T))


def exp_recovery_3p_init(t, S, p0):
    r"""Estimate exponential recovery parameters.

    Args:
        t (array): array of time points
        S (array): signal at each value of t
        p0 (array): dimensionless initial values

    Returns:
        list: Estimates of S0 and T
    """
    S0 = np.amax(np.abs(S))
    return [S0*p0[0], p0[1], p0[2]]


def abs_exp_recovery_3p(t, S0, T, A):
    r"""Absolute value of exponential recovery with 3 parameters

    .. math::

        S = \left| S_0 \left( 1 - A e^{-t/T} \right) \right|

    Args:
        t (array): array of time points
        S0 (float): signal scaling factor
        T (float): relaxation time in same units as t
        A (float): Amplitude of exponential term

    Returns:
        numpy.ndarray: Signal versus time, same size as t.
    """
    return np.abs(S0 * (1 - A * np.exp(-t/T)))


def abs_exp_recovery_3p_init(t, signal, p0):
    r"""Estimate absolute exponential recovery parameters.

    Args:
        t (array): array of time points
        S (array): signal at each value of t
        p0 (array): dimensionless initial values

    Returns:
        list: Estimates of S0 and T
    """
    S0 = np.amax(np.abs(signal))
    return [S0*p0[0], p0[1], p0[2]]


def spgr_vfa(FA, S0, E):
    r"""
    Signal model for a Variable Flip Angle (VFA) scan.

    .. math::

        S(\alpha) = S_0 \sin(\alpha) \frac{1 - e^{-T_R/T_1}}{1 - \cos(\alpha) e^{-T_R/T_1}}

    This function models the MRI signal acquired using a spoiled gradient-echo 
    sequence in the steady-state with different flip angles.

    Args:
        FA (array): Flip angle :math:`\alpha` in degrees.
        S0 (float): Signal scaling factor :math:`S_0` in arbitrary units.
        E (float): Exponential fraction :math:`E = e^{-T_R / T_1}`, where 
            :math:`T_R` is the repetition time and :math:`T_1` is the 
            longitudinal relaxation time.

    Returns:
        array: Signal values at each flip angle.

    """
    FA = np.deg2rad(FA)
    sFA, cFA = np.sin(FA), np.cos(FA)
    if 0 in (1 - cFA * E):
        return S0 * sFA
    else:
        return S0 * sFA * (1-E) / (1 - cFA * E) 
    
    

def spgr_vfa_init(FA, signal, p0):
    """Data-driven initial values for VFA signal model fit.

    Args:
        FA (array): Flip angles in radians
        signal (array): signal values at each FA
        p0 (array): dimensionless initial values for S0 and E. 

    Returns:
        list: Initial values for S0 and E=exp(-TR/T1)
    """
    sFA = np.sin(FA)
    S0 = np.amax(np.abs(signal))/np.amax(sFA)
    return [S0*p0[0], p0[1]]