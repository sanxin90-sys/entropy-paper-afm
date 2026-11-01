"""Spectral radiation entropy (Planck 1913) for surface radiation fields.

Everything here works on a wavelength grid in metres. A radiation field is described
by its spectral radiance L_lambda (W m-2 sr-1 m-1) and by the solid angle it fills:

  * isotropic over the hemisphere (diffuse sky light, Lambertian reflection, emission):
        F = pi * int L dlambda,           J = pi * int S(L) dlambda
  * a beam of solid angle Omega at zenith angle theta on a horizontal surface:
        F = Omega cos(theta) int L dlambda, J = Omega cos(theta) int S(L) dlambda

with the Planck entropy radiance of unpolarised radiation (two polarisation states)

    S_lambda = (2 k c / lambda^4) [ (1+n) ln(1+n) - n ln n ],  n = lambda^5 L_lambda / (2 h c^2),

n being the photon occupation number per mode. For blackbody radiation at T the
hemispheric integral gives (4/3) sigma T^3, which is the first validation test.

Entropy is additive over modes: fields in disjoint solid angles (solar disk vs rest of
the sky) or disjoint wavelengths can be summed; fields that share direction and
wavelength (emitted + reflected longwave) must be added in radiance first.

Units: W m-2 (energy flux), W m-2 K-1 (entropy flux), K, m.
"""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy

H = 6.62607015e-34      # J s
KB = 1.380649e-23       # J K-1
CL = 2.99792458e8       # m s-1
SIGMA_SB = 5.670374419e-8
T_SUN = 5772.0          # K, IAU 2015 nominal (same as Article 1)
# Solid angle of the solar disk at 1 AU: angular radius 0.2666 deg (959.63 arcsec)
SUN_HALF_ANGLE_RAD = np.deg2rad(959.63 / 3600.0)
OMEGA_SUN = 2.0 * np.pi * (1.0 - np.cos(SUN_HALF_ANGLE_RAD))   # ~6.80e-5 sr


def trapz(y, x, axis=-1):
    return np.trapezoid(y, x, axis=axis)


# ----------------------------------------------------------------------------
# Planck functions
# ----------------------------------------------------------------------------
def planck_radiance(lam, t):
    """Blackbody spectral radiance B_lambda(T) (W m-2 sr-1 m-1); lam in m, broadcasts."""
    lam = np.asarray(lam, dtype=float)
    t = np.asarray(t, dtype=float)
    x = H * CL / (lam * KB * t)
    with np.errstate(over="ignore"):
        return 2.0 * H * CL ** 2 / lam ** 5 / np.expm1(x)


def occupation(lam, l_lam):
    """Photon occupation number n = lambda^5 L / (2 h c^2) (dimensionless)."""
    return np.asarray(l_lam, dtype=float) * np.asarray(lam, dtype=float) ** 5 / (2.0 * H * CL ** 2)


def entropy_radiance(lam, l_lam):
    """Spectral entropy radiance S_lambda (W m-2 sr-1 m-1 K-1) of unpolarised radiation."""
    lam = np.asarray(lam, dtype=float)
    n = np.clip(occupation(lam, l_lam), 0.0, None)
    bracket = (1.0 + n) * np.log1p(n) - xlogy(n, n)
    return 2.0 * KB * CL / lam ** 4 * bracket


def brightness_temperature(lam, l_lam):
    """Spectral brightness temperature (K): the T for which B_lambda(T) = L_lambda."""
    n = occupation(lam, l_lam)
    with np.errstate(divide="ignore", invalid="ignore"):
        return H * CL / (np.asarray(lam) * KB * np.log1p(1.0 / n))


# ----------------------------------------------------------------------------
# Fluxes of fields with a given geometry
# ----------------------------------------------------------------------------
def iso_flux(lam, l_lam, axis=-1):
    """(F, J) of an isotropic hemispheric field with spectral radiance l_lam."""
    return np.pi * trapz(l_lam, lam, axis), np.pi * trapz(entropy_radiance(lam, l_lam), lam, axis)


def beam_flux(lam, l_lam, omega=OMEGA_SUN, cos_theta=1.0, axis=-1):
    """(F, J) on a horizontal surface of a beam of solid angle omega at zenith angle theta
    (small-beam approximation: radiance uniform across the beam)."""
    f = omega * cos_theta * trapz(l_lam, lam, axis)
    j = omega * cos_theta * trapz(entropy_radiance(lam, l_lam), lam, axis)
    return f, j


def normalise_shape(lam, e_lam):
    """Spectral shape s_lambda = e_lambda / int e dlambda (1/m), so that int s = 1."""
    e = np.asarray(e_lam, dtype=float)
    return e / trapz(e, lam)


# ----------------------------------------------------------------------------
# Look-up tables: entropy flux as a function of energy flux for a fixed spectral shape
# ----------------------------------------------------------------------------
class IsoTable:
    """J(F) of an isotropic field with fixed spectral shape s (int s = 1): L = F s / pi.
    Tabulated in log-log and interpolated (error < 1e-5 relative, checked in tests)."""

    def __init__(self, lam, shape, fmin=1e-3, fmax=3000.0, n=400):
        self.lam = lam
        self.shape = normalise_shape(lam, shape)
        self.lf = np.linspace(np.log(fmin), np.log(fmax), n)
        f = np.exp(self.lf)
        l_lam = f[:, None] * self.shape[None, :] / np.pi
        _, j = iso_flux(lam, l_lam)
        self.lj = np.log(j)

    def __call__(self, f):
        f = np.asarray(f, dtype=float)
        out = np.zeros_like(f)
        pos = f > 0
        out[pos] = np.exp(np.interp(np.log(f[pos]), self.lf, self.lj))
        return out

    def exact(self, f):
        return iso_flux(self.lam, f * self.shape / np.pi)[1]


class BeamTable:
    """g(DNI) = Omega int S(DNI s / Omega) dlambda for a beam of fixed shape s; the entropy
    flux on a horizontal surface is J = cos(theta) g(DNI) (DNI = direct normal irradiance)."""

    def __init__(self, lam, shape, omega=OMEGA_SUN, fmin=1e-2, fmax=1500.0, n=400):
        self.lam = lam
        self.omega = omega
        self.shape = normalise_shape(lam, shape)
        self.lf = np.linspace(np.log(fmin), np.log(fmax), n)
        f = np.exp(self.lf)
        l_lam = f[:, None] * self.shape[None, :] / omega
        _, g = beam_flux(lam, l_lam, omega)
        self.lg = np.log(g)

    def __call__(self, dni):
        dni = np.asarray(dni, dtype=float)
        out = np.zeros_like(dni)
        pos = dni > 0
        out[pos] = np.exp(np.interp(np.log(dni[pos]), self.lf, self.lg))
        return out

    def exact(self, dni):
        return beam_flux(self.lam, dni * self.shape / self.omega, self.omega)[1]


# ----------------------------------------------------------------------------
# Diffuse fraction (Erbs, Klein & Duffie 1982)
# ----------------------------------------------------------------------------
def erbs_kd(kt):
    """Hourly diffuse fraction kd from the clearness index kt (Erbs et al. 1982)."""
    kt = np.asarray(kt, dtype=float)
    kd = np.where(kt <= 0.22, 1.0 - 0.09 * kt,
                  np.where(kt <= 0.80,
                           0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                           0.165))
    return np.clip(kd, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Longwave: emitted + reflected sky radiance of a surface with spectral emissivity
# ----------------------------------------------------------------------------
def lw_out_radiance(lam, eps_lam, ts, sky_lam):
    """Outgoing longwave radiance eps B(Ts) + (1 - eps) L_sky (isotropic sky, Lambertian surface).
    ts: (N,) ; sky_lam: (N, M) or (M,)."""
    ts = np.atleast_1d(np.asarray(ts, dtype=float))
    return eps_lam[None, :] * planck_radiance(lam[None, :], ts[:, None]) + (1.0 - eps_lam[None, :]) * sky_lam


def solve_ts(lam, eps_lam, lw_out, sky_lam, t0=None, n_iter=8):
    """Surface temperature Ts such that pi int [eps B(Ts) + (1-eps) L_sky] = LW_OUT (Newton)."""
    lw_out = np.atleast_1d(np.asarray(lw_out, dtype=float))
    refl = np.pi * trapz((1.0 - eps_lam)[None, :] * sky_lam, lam) if np.ndim(sky_lam) == 2 else \
        np.full(lw_out.shape, np.pi * trapz((1.0 - eps_lam) * sky_lam, lam))
    ts = (lw_out / SIGMA_SB) ** 0.25 if t0 is None else np.asarray(t0, dtype=float).copy()
    for _ in range(n_iter):
        b = planck_radiance(lam[None, :], ts[:, None])
        e = np.pi * trapz(eps_lam[None, :] * b, lam)
        # dB/dT = B * x e^x / (T (e^x - 1)),  x = hc / (lam k T)
        x = H * CL / (lam[None, :] * KB * ts[:, None])
        db = b * x / ts[:, None] / (-np.expm1(-x))
        de = np.pi * trapz(eps_lam[None, :] * db, lam)
        ts = ts - (e + refl - lw_out) / de
    return ts


def sky_radiance(lam, ta, lw_in, model="grey"):
    """Downwelling sky radiance (isotropic) consistent with the measured LW_IN.
    'grey'  : delta_a B(Ta), delta_a = LW_IN / (sigma Ta^4)  (Article 1 'diluted' variant)
    'window': B(Ta) outside 8-13 um, w B(Ta) inside, w chosen to reproduce LW_IN (clipped 0..1;
              when LW_IN > sigma Ta^4 the field is scaled grey instead).
    'bb_sky': blackbody at T_sky = (LW_IN/sigma)^1/4 (Article 1 core)."""
    ta = np.atleast_1d(np.asarray(ta, dtype=float))
    lw_in = np.atleast_1d(np.asarray(lw_in, dtype=float))
    b = planck_radiance(lam[None, :], ta[:, None])
    if model == "grey":
        return (lw_in / (SIGMA_SB * ta ** 4))[:, None] * b
    if model == "bb_sky":
        return planck_radiance(lam[None, :], (lw_in / SIGMA_SB)[:, None] ** 0.25)
    if model == "window":
        win = (lam >= 8e-6) & (lam <= 13e-6)
        full = np.pi * trapz(b, lam)
        inwin = np.pi * trapz(b * win[None, :], lam)
        w = 1.0 - (full - lw_in) / inwin
        out = b * np.where(win[None, :], np.clip(w, 0.0, 1.0)[:, None], 1.0)
        scale = lw_in / (np.pi * trapz(out, lam))     # exact match of LW_IN (also handles w < 0 or > 1)
        return out * scale[:, None]
    raise ValueError(model)
