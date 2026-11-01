"""Shared grids, reference spectra and paths for Article 2."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

import spectral as S

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
DERIVED = ROOT / "data_derived"
NPW_CODE = ROOT / "code" / "article1"         # companion-study pipeline (unchanged copies)
os.environ.setdefault("NPW_DATA_DIR", str(ROOT / "data"))   # AmeriFlux file location
os.environ.setdefault("NPW_SCRATCH_DIR", str(ROOT / "scratch"))
if str(NPW_CODE) not in sys.path:
    sys.path.append(str(NPW_CODE))

# Longwave grid: 2.5 um .. 1 mm, log-spaced (captures > 99.9 % of 250-340 K blackbody energy)
LW_UM = np.geomspace(2.5, 1000.0, 3000)
LW_M = LW_UM * 1e-6

# Reference conditions for the library analysis (typical BR-Npw midday values, Article 1 T1)
F_SWOUT_LEVELS = (25.0, 50.0, 75.0, 100.0, 150.0)   # W m-2 reflected shortwave
REF_TA = 301.6          # K, daytime mean air temperature (T1, day)
REF_LWIN = 419.3        # W m-2, daytime mean LW_IN (T1, day)
REF_TS = 303.0          # K, surface temperature for the LW library comparison
EPS_A1 = 0.98           # broadband emissivity of Article 1 core


def astm():
    """ASTM G173-03 reference spectra (pvlib copy): wavelength (m) and spectral irradiance
    (W m-2 m-1) for extraterrestrial, global tilt and direct + circumsolar."""
    from pvlib.spectrum import get_reference_spectra
    s = get_reference_spectra(standard="ASTM G173-03")
    lam = s.index.to_numpy(dtype=float) * 1e-9
    k = 1e9    # per nm -> per m
    et, gl, dr = (s[c].to_numpy(dtype=float) * k for c in ("extraterrestrial", "global", "direct"))
    # diffuse on the 37 deg tilted plane: global - direct * cos(incidence); AM1.5 zenith 48.19 deg,
    # tilt 37 deg facing the sun -> incidence 11.19 deg
    df = np.clip(gl - dr * np.cos(np.deg2rad(48.19 - 37.0)), 0.0, None)
    out = {"extraterrestrial": et, "global": gl, "direct": dr, "diffuse_tilt_raw": df}
    out["diffuse_clear"] = horizontal_sky_diffuse(lam, dr, df)
    return lam, out


def horizontal_sky_diffuse(lam, dni, dif_tilt, ground_rho=None):
    """Horizontal clear-sky diffuse spectrum D_h from the ASTM tilted 'diffuse' (global tilt minus
    direct x cos 11.19 deg), removing the ground-reflected light the 37 deg plane sees:
        dif_tilt = D_h (1 + cos 37)/2 + rho_g (DNI cos 48.19 + D_h) (1 - cos 37)/2   (isotropic sky).
    rho_g: ground spectral albedo; default = median ECOSTRESS soil reflectance (proxy for the
    light-soil albedo of the ASTM/SMARTS calculation)."""
    if ground_rho is None:
        ground_rho = _soil_median(lam)
    vs, vg = (1 + np.cos(np.deg2rad(37.0))) / 2, (1 - np.cos(np.deg2rad(37.0))) / 2
    gh_dir = dni * np.cos(np.deg2rad(48.19))
    d = (dif_tilt - ground_rho * gh_dir * vg) / (vs + ground_rho * vg)
    return np.clip(d, 0.0, None)


def _soil_median(lam):
    f = DERIVED / "soil_median_rho_astm.npy"
    if f.exists():
        return np.load(f)
    import ecostress as EC
    specs = [s for s in EC.join_samples(EC.read_library()) if s["cls"] == "soil" and EC.sw_usable(s)]
    r = np.median([EC.resample_sw(s, lam * 1e6) for s in specs], axis=0)
    DERIVED.mkdir(parents=True, exist_ok=True)
    np.save(f, r)
    return r


def wide_planck_shape():
    """Planck(T_sun) on a wide grid, for the 'diluted blackbody' reference (Article 1 variant b)."""
    lam = np.geomspace(5e-8, 2e-4, 6000)
    return lam, S.planck_radiance(lam, S.T_SUN)
