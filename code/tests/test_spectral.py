"""Validation of the spectral entropy code (run: python -m pytest code/tests -q from artigo2/)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spectral as S  # noqa: E402

LW = np.geomspace(1e-6, 2e-3, 6000)          # 1 um .. 2 mm


def test_blackbody_energy_and_entropy():
    for t in (250.0, 300.0, 330.0):
        f, j = S.iso_flux(LW, S.planck_radiance(LW, t))
        assert abs(f / (S.SIGMA_SB * t ** 4) - 1) < 1e-3
        assert abs(j / (4 / 3 * S.SIGMA_SB * t ** 3) - 1) < 1e-3


def test_sun_blackbody_beam():
    lam = np.geomspace(5e-8, 2e-4, 8000)
    f, j = S.beam_flux(lam, S.planck_radiance(lam, S.T_SUN))
    # irradiance at 1 AU and entropy flux (4/3) F / T_sun
    assert abs(f / (S.SIGMA_SB * S.T_SUN ** 4 * S.OMEGA_SUN / np.pi) - 1) < 1e-3
    assert abs(j / (4 / 3 * f / S.T_SUN) - 1) < 1e-3


def test_dilution_raises_entropy_per_energy():
    lam = np.geomspace(5e-8, 2e-4, 8000)
    shape = S.planck_radiance(lam, S.T_SUN)
    tab = S.IsoTable(lam, shape)
    for f in (10.0, 100.0, 1000.0):
        assert tab(np.array([f]))[0] / f > 4 / (3 * S.T_SUN)


def test_tables_match_exact():
    lam = np.geomspace(2.8e-7, 4e-6, 3000)
    shape = S.planck_radiance(lam, S.T_SUN)
    iso = S.IsoTable(lam, shape)
    beam = S.BeamTable(lam, shape)
    for f in (0.5, 37.0, 412.0, 999.0):
        assert abs(iso(np.array([f]))[0] / iso.exact(f) - 1) < 1e-4
        assert abs(beam(np.array([f]))[0] / beam.exact(f) - 1) < 1e-4


def test_solve_ts_roundtrip():
    lam = np.geomspace(2.5e-6, 1e-3, 3000)
    eps = np.where((lam > 8e-6) & (lam < 9.5e-6), 0.9, 0.98)
    ta = np.array([295.0, 300.0]); lw_in = np.array([400.0, 420.0])
    sky = S.sky_radiance(lam, ta, lw_in, "grey")
    ts_true = np.array([297.0, 305.0])
    lo = np.pi * np.trapezoid(S.lw_out_radiance(lam, eps, ts_true, sky), lam)
    ts = S.solve_ts(lam, eps, lo, sky)
    assert np.allclose(ts, ts_true, atol=1e-4)


def test_sky_models_reproduce_lw_in():
    lam = np.geomspace(2.5e-6, 1e-3, 3000)
    ta = np.array([290.0, 300.0, 300.0]); lw_in = np.array([350.0, 420.0, 470.0])
    for m in ("grey", "window", "bb_sky"):
        sky = S.sky_radiance(lam, ta, lw_in, m)
        f = np.pi * np.trapezoid(sky, lam)
        assert np.allclose(f / lw_in, 1, atol=2e-3), m


def test_erbs_limits():
    assert np.isclose(S.erbs_kd(0.0), 1.0)
    assert np.isclose(S.erbs_kd(0.9), 0.165)
    k = S.erbs_kd(np.linspace(0, 1, 101))
    assert np.all(np.diff(k) <= 1e-9 + 0.02)   # essentially non-increasing
