"""Entropy flux terms of the surface entropy budget (METHODS_SPEC.md, section 2).

Pure functions on numpy arrays / pandas Series, plus `budget(df, opts)` which
assembles every term for the half-hours of a DataFrame produced by load.add_derived.

Units: energy fluxes W m-2, temperatures K, entropy fluxes W m-2 K-1.
Sign convention: fluxes keep the sign stored in the file (downward inputs positive,
upward outputs positive; nighttime negative H/LE kept). sigma = J_out - J_in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

# Physical constants for the spectral helper (CODATA 2018)
H_PLANCK = 6.62607015e-34   # J s
K_B = 1.380649e-23          # J K-1
C_LIGHT = 2.99792458e8      # m s-1


# ----------------------------------------------------------------------------
# Temperatures
# ----------------------------------------------------------------------------
def t_surface(lw_out, lw_in, eps=C.EPS_CORE, sigma=C.SIGMA_SB):
    """Radiometric surface temperature (K), eq. 9: LW_OUT = eps*sigma*Ts^4 + (1-eps)*LW_IN."""
    return ((lw_out - (1.0 - eps) * lw_in) / (eps * sigma)) ** 0.25


def lw_out_from_ts(ts, lw_in, eps=C.EPS_CORE, sigma=C.SIGMA_SB):
    """Forward model of eq. 9 (for the round-trip test)."""
    return eps * sigma * ts ** 4 + (1.0 - eps) * lw_in


def t_sky(lw_in, sigma=C.SIGMA_SB):
    """Blackbody-equivalent sky temperature (K), eq. 8."""
    return (lw_in / sigma) ** 0.25


# ----------------------------------------------------------------------------
# Dilution function (Landsberg & Tonge approximation; coefficients to confirm)
# ----------------------------------------------------------------------------
def x_dilution_approx(delta, coef=C.X_COEF):
    """Landsberg & Tonge approximation X(delta) = a + b ln(delta) + c delta (eq. 5).
    Checked numerically against x_dilution_exact: within 0.1 % for delta <= 0.1 but
    +1.6 % high at delta = 1 (gives 1.016 instead of 1). Kept for comparison only."""
    d = np.asarray(delta, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = coef[0] + coef[1] * np.log(d) + coef[2] * d
    return np.where(d > 0, x, np.nan)


_X_TABLE = None


def _x_table():
    """Exact X(delta) on a log grid, 1e-12 <= delta <= 1, built once from the spectral integral."""
    global _X_TABLE
    if _X_TABLE is None:
        lg = np.linspace(-12.0, 0.0, 241)
        _X_TABLE = (lg, np.array([x_dilution_exact(10.0 ** v) for v in lg]))
    return _X_TABLE


def x_dilution(delta):
    """Dilution function X(delta) (entropy flux of a diluted Planck spectrum relative to
    (4/3)F/T), evaluated from the exact spectral integral by interpolation in log10(delta).
    delta > 1 (unphysical for a diluted spectrum) is clipped to 1 (X = 1) *silently*: callers
    that may produce delta > 1 (j_lw_in 'diluted': delta_a = LW_IN/(sigma T_a^4) exceeds 1 in
    ~1 % of core half-hours, humid cloudy nights, max ~1.045) must count and log those cases
    (see count_delta_gt1 and run_all). delta <= 0 -> NaN.
    This replaces the approximate eq. 5 in all computations (see run log for the comparison)."""
    d = np.asarray(delta, dtype=float)
    lg, xt = _x_table()
    with np.errstate(divide="ignore", invalid="ignore"):
        l = np.log10(np.clip(d, 1e-12, 1.0))
    x = np.interp(l, lg, xt)
    return np.where(d > 0, x, np.nan)


def x_dilution_exact(delta, n_nu=20000):
    """Exact X(delta) by spectral integration of the Planck entropy of a diluted
    blackbody (occupation number n = delta/(exp(x)-1)); independent of T.

    X = L_s / ((4/3) F / T), with
    L_s = pi * int 2 k nu^2/c^2 [(1+n) ln(1+n) - n ln n] dnu,  F = delta * sigma T^4.
    Used in the tests to check the approximation coefficients numerically.
    """
    delta = float(delta)
    x = np.linspace(1e-6, 60.0, n_nu)              # x = h nu / k T
    n = delta / np.expm1(x)
    s_integrand = x ** 2 * ((1 + n) * np.log1p(n) - n * np.log(n))
    e_integrand = delta * x ** 3 / np.expm1(x)
    # both integrals carry the same prefactor up to k vs k x, so the ratio is dimensionless:
    # L_s / F = (k/ (k T)) * I_s / I_e  ->  L_s T / F = I_s / I_e ;  X = (3/4) I_s / I_e
    i_s = np.trapezoid(s_integrand, x)
    i_e = np.trapezoid(e_integrand, x)
    return 0.75 * i_s / i_e


def planck_entropy_flux(t, n_nu=20000):
    """Hemispheric entropy flux (W m-2 K-1) of undiluted blackbody radiation at T by
    numerical integration of the Planck spectral entropy; should equal (4/3) sigma T^3."""
    nu = np.linspace(1e9, 60 * K_B * t / H_PLANCK, n_nu)
    x = H_PLANCK * nu / (K_B * t)
    n = 1.0 / np.expm1(x)
    l_nu = (2 * K_B * nu ** 2 / C_LIGHT ** 2) * ((1 + n) * np.log1p(n) - n * np.log(n))
    return np.pi * np.trapezoid(l_nu, nu)


# ----------------------------------------------------------------------------
# Individual terms
# ----------------------------------------------------------------------------
def j_sw_in(sw_in, t_sun=C.T_SUN, variant="core"):
    """Incoming solar entropy flux. 'core': (4/3) SW_IN/T_sun (eq. 4, lower bound);
    'diluted': hemispheric-diluted upper bound with X(delta), delta = SW_IN/(sigma T_sun^4)."""
    sw = np.asarray(sw_in, dtype=float)
    base = (4.0 / 3.0) * sw / t_sun
    if variant == "core":
        return base
    if variant == "diluted":
        delta = sw / (C.SIGMA_SB * t_sun ** 4)
        x = x_dilution(delta)
        return np.where(sw > 0, base * x, 0.0)
    raise ValueError(variant)


def j_sw_out(sw_out, sw_in, is_day, t_sun=C.T_SUN, variant="a"):
    """Reflected solar entropy flux; zero when night or SW_IN <= 0 or SW_OUT <= 0
    (nighttime SW_OUT ~1-3 W m-2 is instrument offset; negative SW_OUT only at night).
    'a': (4/3) SW_OUT/T_sun (eq. 6); 'b': diluted blackbody (eq. 7)."""
    so = np.asarray(sw_out, dtype=float)
    valid = np.asarray(is_day, dtype=bool) & (np.asarray(sw_in, dtype=float) > 0) & (so > 0)
    base = (4.0 / 3.0) * so / t_sun
    if variant == "a":
        return np.where(valid, base, 0.0)
    if variant == "b":
        delta = so / (C.SIGMA_SB * t_sun ** 4)
        return np.where(valid, base * x_dilution(delta), 0.0)
    raise ValueError(variant)


def delta_air(lw_in, ta):
    """Apparent sky emissivity delta_a = LW_IN/(sigma T_a^4) (dilution factor of the sky
    radiation relative to a blackbody at air temperature)."""
    return np.asarray(lw_in, dtype=float) / (C.SIGMA_SB * np.asarray(ta, dtype=float) ** 4)


def count_delta_gt1(lw_in, ta):
    """Number of records with delta_a > 1 (clipped to X = 1 in the 'diluted' variants)."""
    d = delta_air(lw_in, ta)
    return int(np.nansum(d > 1.0)), float(np.nanmax(d)) if np.isfinite(d).any() else np.nan


def j_lw_in(lw_in, ta, variant="sky"):
    """Incoming longwave entropy flux. 'sky': (4/3) LW_IN/T_sky (eq. 8, core);
    'air': (4/3) LW_IN/T_a; 'diluted': (4/3)(LW_IN/T_a) X(delta_a), delta_a = LW_IN/(sigma T_a^4).
    In the 'diluted' variant delta_a > 1 (about 1 % of core half-hours, humid cloudy nights)
    is clipped to X = 1 by x_dilution; the count is logged by run_all."""
    lw = np.asarray(lw_in, dtype=float)
    if variant == "sky":
        return (4.0 / 3.0) * lw / t_sky(lw)
    if variant == "air":
        return (4.0 / 3.0) * lw / np.asarray(ta, dtype=float)
    if variant == "diluted":
        ta = np.asarray(ta, dtype=float)
        delta = delta_air(lw, ta)
        return (4.0 / 3.0) * lw / ta * x_dilution(delta)
    raise ValueError(variant)


def j_lw_out(lw_out, ts, jlw_in=None, eps=C.EPS_CORE, variant="core", lw_in=None, ta=None):
    """Outgoing longwave entropy flux.
    'core': (4/3) LW_OUT/T_s (eq. 10).
    'grey': (4/3) eps sigma T_s^3 X(eps) + (1-eps) J_LWin (eq. 11). The reflected part is
        carried at the entropy per unit energy of the *incoming* sky radiation. This is not
        fully consistent: reflection of hemispheric sky radiation with reflectance (1-eps)
        dilutes it further (delta' = (1-eps) delta_a ~ 0.02), so its entropy per unit energy is
        about twice X(delta_a)/T_a. The variant is therefore a sensitivity, not a corrected value.
    'grey_refl': eq. 11 with the reflected part treated as a diluted blackbody at T_a with
        delta' = (1-eps) delta_a: (1-eps) (4/3) (LW_IN/T_a) X((1-eps) delta_a). Reported as an
        upper bound of J_LWout (needs lw_in and ta)."""
    if variant == "core":
        return (4.0 / 3.0) * np.asarray(lw_out, dtype=float) / np.asarray(ts, dtype=float)
    ts = np.asarray(ts, dtype=float)
    emitted = (4.0 / 3.0) * eps * C.SIGMA_SB * ts ** 3 * x_dilution(eps)
    if variant == "grey":
        return emitted + (1.0 - eps) * np.asarray(jlw_in, dtype=float)
    if variant == "grey_refl":
        lw = np.asarray(lw_in, dtype=float); ta = np.asarray(ta, dtype=float)
        d_refl = (1.0 - eps) * delta_air(lw, ta)
        return emitted + (1.0 - eps) * (4.0 / 3.0) * lw / ta * x_dilution(d_refl)
    raise ValueError(variant)


def j_heat(flux, temp):
    """Entropy flux of a heat flux crossing a boundary at temperature T: Q/T (eqs. 12-13)."""
    return np.asarray(flux, dtype=float) / np.asarray(temp, dtype=float)


# ----------------------------------------------------------------------------
# Closure scenarios (section 3.2)
# ----------------------------------------------------------------------------
def closure_fluxes(h, le, rres, scenario="S0", cr=None):
    """Return (H*, LE*) for closure scenario S0..S3. rres = R' = Rn - H - LE.
    S3 divides both fluxes by the closure ratio `cr` (ratio of sums over the period)."""
    h = np.asarray(h, dtype=float)
    le = np.asarray(le, dtype=float)
    r = np.asarray(rres, dtype=float)
    if scenario == "S0":
        return h, le
    if scenario == "S1":
        return h, le + r
    if scenario == "S2":
        return h + r, le
    if scenario == "S3":
        if cr is None or not np.isfinite(cr) or cr <= 0:
            raise ValueError("S3 needs a positive closure ratio")
        return h / cr, le / cr
    raise ValueError(scenario)


def closure_ratio(h, le, rn):
    """CR_sum = sum(H+LE)/sum(Rn) over finite records."""
    h = np.asarray(h, dtype=float); le = np.asarray(le, dtype=float); rn = np.asarray(rn, dtype=float)
    ok = np.isfinite(h) & np.isfinite(le) & np.isfinite(rn)
    return float((h[ok] + le[ok]).sum() / rn[ok].sum())


# ----------------------------------------------------------------------------
# Full budget
# ----------------------------------------------------------------------------
def apply_bias(df: pd.DataFrame, bias) -> pd.DataFrame:
    """Perturb radiation components (one at a time in practice). bias: ((var, kind, value), ...)
    kind 'rel' multiplies by (1+value), 'abs' adds value (W m-2)."""
    if not bias:
        return df
    out = df.copy()
    for var, kind, value in bias:
        if kind == "rel":
            out[var] = out[var] * (1.0 + value)
        elif kind == "abs":
            out[var] = out[var] + value
        else:
            raise ValueError(kind)
    return out


J_COLS = ["J_SWin", "J_SWout", "J_LWin", "J_LWout", "J_H", "J_LE", "J_in", "J_out",
          "sigma", "sigma_rad", "sigma_turb"]


def sigma_analytic_terms(J: pd.DataFrame, t_sun=C.T_SUN) -> pd.DataFrame:
    """Analytic decomposition of the budget (spec 2.7): with eqs. 4, 6, 8, 10 and 12-13,
        sigma_S0 = absSW/T_s - (J_SWin - J_SWout) + LWterm - R'/T_s,
        LWterm   = (4/3) LW_OUT/T_s - (4/3) LW_IN/T_sky - (LW_OUT - LW_IN)/T_s
                 = sigma_SB [T_s^3/3 + T_sky^4/T_s - (4/3) T_sky^3]  (exact for eps = 1)
                 ~ 2 sigma_SB T_s (T_s - T_sky)^2  (second order in T_s - T_sky),
    so that sigma_S1 = sigma_S0 + R'/T_s ~ absSW (1/T_s - 4/(3 T_sun)) + LWterm.
    Returns the four terms per record (W m-2 K-1); the identity holds exactly for the core
    options (J_SWout zero at night makes 'swnet' differ from (4/3)(SW_IN - SW_OUT)/T_sun by < 1e-4).
    """
    out = pd.DataFrame(index=J.index)
    out["absSW_over_Ts"] = J["absSW"] / J["Ts"]
    out["minus_J_SWnet"] = -(J["J_SWin"] - J["J_SWout"])
    # Rn - absSW = LW_IN - LW_OUT, hence -(LW_OUT - LW_IN)/T_s = (Rn - absSW)/T_s
    out["LWterm"] = J["J_LWout"] - J["J_LWin"] + (J["Rn"] - J["absSW"]) / J["Ts"]
    out["minus_Rres_over_Ts"] = -J["Rres"] / J["Ts"]
    out["sigma_reconstructed"] = out[["absSW_over_Ts", "minus_J_SWnet", "LWterm", "minus_Rres_over_Ts"]].sum(axis=1)
    out["sigma_S1_analytic"] = J["absSW"] * (1.0 / J["Ts"] - 4.0 / (3.0 * t_sun))
    out["LWterm_second_order"] = 2.0 * C.SIGMA_SB * J["Ts"] * (J["Ts"] - J["Tsky"]) ** 2
    return out


def budget(df: pd.DataFrame, opts: C.Options = C.CORE, cr: float | None = None) -> pd.DataFrame:
    """Compute every entropy term for each row of df (any selection; NaNs propagate).

    Returns a DataFrame indexed like df with the J terms, sigma, sigma_rad, sigma_turb,
    the temperatures used, the (closure-adjusted) fluxes H_s, LE_s, Rn, Rres, Rn_abs
    and the random-uncertainty terms u_JH, u_JLE, u_sigma (W m-2 K-1).
    """
    d = apply_bias(df, opts.bias)
    ta = d["TA_F"].to_numpy(dtype=float) + C.T0
    lw_in = d["LW_IN_F"].to_numpy(dtype=float)
    lw_out = d["LW_OUT"].to_numpy(dtype=float)
    sw_in = d["SW_IN_F"].to_numpy(dtype=float)
    sw_out = d["SW_OUT"].to_numpy(dtype=float)
    is_day = (d["NIGHT"].to_numpy() == 0)
    ts = t_surface(lw_out, lw_in, opts.eps)
    tsky = t_sky(lw_in)
    rn = sw_in - sw_out + lw_in - lw_out
    h = d["H_F_MDS"].to_numpy(dtype=float)
    le = d["LE_F_MDS"].to_numpy(dtype=float)
    rres = rn - h - le
    if opts.closure == "S3" and cr is None:
        cr = closure_ratio(h, le, rn)
    h_s, le_s = closure_fluxes(h, le, rres, opts.closure, cr)
    t_turb = ts if opts.turb_temp == "ts" else ta

    out = pd.DataFrame(index=d.index)
    out["J_SWin"] = j_sw_in(sw_in, opts.t_sun, opts.swin_variant)
    out["J_SWout"] = j_sw_out(sw_out, sw_in, is_day, opts.t_sun, opts.swout_variant)
    out["J_LWin"] = j_lw_in(lw_in, ta, opts.lwin_temp)
    out["J_LWout"] = j_lw_out(lw_out, ts, out["J_LWin"].to_numpy(), opts.eps, opts.lwout_variant, lw_in=lw_in, ta=ta)
    out["J_H"] = j_heat(h_s, t_turb)
    out["J_LE"] = j_heat(le_s, t_turb)
    out["J_in"] = out["J_SWin"] + out["J_LWin"]
    out["J_out"] = out["J_SWout"] + out["J_LWout"] + out["J_H"] + out["J_LE"]
    out["sigma"] = out["J_out"] - out["J_in"]
    out["sigma_rad"] = out["J_SWout"] + out["J_LWout"] - out["J_SWin"] - out["J_LWin"]
    out["sigma_turb"] = out["J_H"] + out["J_LE"]
    out["Ts"] = ts
    out["Tsky"] = tsky
    out["Ta"] = ta
    out["H_s"] = h_s
    out["LE_s"] = le_s
    out["Rn"] = rn
    out["Rn_abs"] = sw_in - sw_out + lw_in
    out["absSW"] = sw_in - sw_out          # absorbed shortwave (the quantity that fixes the production, spec 2.7)
    out["Rres"] = rres
    out["u_JH"] = d["H_RANDUNC"].to_numpy(dtype=float) / t_turb
    out["u_JLE"] = d["LE_RANDUNC"].to_numpy(dtype=float) / t_turb
    out["u_sigma"] = np.sqrt(out["u_JH"] ** 2 + out["u_JLE"] ** 2)
    out.attrs["options"] = opts.to_dict()
    out.attrs["cr_used"] = cr
    return out
