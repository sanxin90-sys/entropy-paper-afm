"""Part A: entropy of radiation reflected and emitted by ECOSTRESS surface classes,
spectral vs the broadband formulas of Article 1.

Writes results/tables/a1_library_spectra.csv (one row per spectrum), a2_class_summary.csv,
a3_validation.csv, a4_incoming_reference.csv and data_derived/library_arrays.npz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as K
import ecostress as EC
import spectral as S


def validation_table():
    rows = []
    for t in (250.0, 300.0, 330.0, S.T_SUN):
        lam = np.geomspace(5e-8, 2e-4, 8000) if t > 1000 else np.geomspace(1e-6, 2e-3, 8000)
        f, j = S.iso_flux(lam, S.planck_radiance(lam, t))
        rows.append({"check": f"blackbody {t:g} K, F/(sigma T^4)", "value": f / (S.SIGMA_SB * t ** 4), "target": 1.0})
        rows.append({"check": f"blackbody {t:g} K, J/((4/3) sigma T^3)", "value": j / (4 / 3 * S.SIGMA_SB * t ** 3), "target": 1.0})
    # also on the production LW grid
    for t in (280.0, 300.0, 320.0):
        f, j = S.iso_flux(K.LW_M, S.planck_radiance(K.LW_M, t))
        rows.append({"check": f"LW grid 2.5-1000 um, {t:g} K, F/(sigma T^4)", "value": f / (S.SIGMA_SB * t ** 4), "target": 1.0})
        rows.append({"check": f"LW grid 2.5-1000 um, {t:g} K, J/((4/3) sigma T^3)", "value": j / (4 / 3 * S.SIGMA_SB * t ** 3), "target": 1.0})
    return pd.DataFrame(rows)


def incoming_reference(lam, ref):
    """Entropy per unit energy of the reference solar fields (K-1), relative to 4/(3 T_sun)."""
    rows = []
    bb = 4.0 / (3.0 * S.T_SUN)
    f, j = S.beam_flux(lam, ref["extraterrestrial"] / S.OMEGA_SUN)
    rows.append({"field": "extraterrestrial ASTM E-490 shape, beam of the solar disk", "F": f, "J": j})
    f, j = S.beam_flux(lam, ref["direct"] / S.OMEGA_SUN)
    rows.append({"field": "direct normal AM1.5 (ASTM G173), beam of the solar disk", "F": f, "J": j})
    for name, key in (("diffuse clear-sky AM1.5 horizontal (ground light removed), isotropic", "diffuse_clear"),
                      ("diffuse from tilted ASTM, ground light not removed, isotropic", "diffuse_tilt_raw"),
                      ("global AM1.5 shape, isotropic", "global")):
        f, j = S.iso_flux(lam, ref[key] / np.pi)
        rows.append({"field": name, "F": f, "J": j})
    for fl in (50.0, 100.0, 200.0, 400.0):
        sh = S.normalise_shape(lam, ref["global"])
        f, j = S.iso_flux(lam, fl * sh / np.pi)
        rows.append({"field": f"global shape, isotropic, F = {fl:g} W m-2", "F": f, "J": j})
    out = pd.DataFrame(rows)
    out["J_over_F_Kinv"] = out["J"] / out["F"]
    out["ratio_to_4_over_3Tsun"] = out["J_over_F_Kinv"] / bb
    out["T_equiv_K"] = 4.0 / (3.0 * out["J_over_F_Kinv"])
    return out


def kubelka_munk_canopy(rho):
    """Crude canopy reflectance from a leaf spectrum: semi-infinite canopy of isotropic
    scatterers with single-scattering albedo w = rho + tau, tau = rho (two-stream limit
    R = (1 - sqrt(1 - w)) / (1 + sqrt(1 - w))). Used only as a spectral-shape sensitivity."""
    w = np.clip(2.0 * rho, 0.0, 0.999)
    s = np.sqrt(1.0 - w)
    return (1.0 - s) / (1.0 + s)


def main():
    K.TABLES.mkdir(parents=True, exist_ok=True)
    K.DERIVED.mkdir(parents=True, exist_ok=True)
    lam, ref = K.astm()
    lam_um = lam * 1e6
    e_gl = ref["global"]
    bb = 4.0 / (3.0 * S.T_SUN)

    val = validation_table()
    val.to_csv(K.TABLES / "a3_validation.csv", index=False)
    inc = incoming_reference(lam, ref)
    inc.to_csv(K.TABLES / "a4_incoming_reference.csv", index=False)

    # Article 1 variant (b): diluted blackbody at T_sun, isotropic
    lw_bb, pl = K.wide_planck_shape()
    tab_b = S.IsoTable(lw_bb, pl)

    specs = EC.join_samples(EC.read_library())
    # drop exact duplicates (same class and identical reflectance array), keep the first
    seen, uniq = set(), []
    for sp in specs:
        key = (sp["cls"], len(sp["wl_um"]), round(float(sp["refl"].sum()), 6), round(float(sp["wl_um"].sum()), 6))
        if key in seen:
            continue
        seen.add(key); uniq.append(sp)
    print(f"spectra after joining: {len(specs)}; duplicates dropped: {len(specs) - len(uniq)}")
    specs = uniq
    rows, sw_arr, sw_keys, lw_arr, lw_keys = [], [], [], [], []
    levels = np.array(K.F_SWOUT_LEVELS)

    # grey (spectrally flat) reflector: shape = incident global spectrum
    sh_grey = S.normalise_shape(lam, e_gl)

    # reference sky for LW
    ta, lwin = np.array([K.REF_TA]), np.array([K.REF_LWIN])
    sky_grey = S.sky_radiance(K.LW_M, ta, lwin, "grey")[0]
    sky_win = S.sky_radiance(K.LW_M, ta, lwin, "window")[0]

    for i, s in enumerate(specs):
        r = {k: s[k] for k in ("file", "cls", "subclass", "genus", "species", "sample", "owner", "joined")}
        if EC.sw_usable(s):
            rho = EC.resample_sw(s, lam_um)
            r["albedo_astm"] = float(np.trapezoid(rho * e_gl, lam) / np.trapezoid(e_gl, lam))
            r["vis_0.4_0.7"] = EC.band_mean(lam_um, rho, 0.4, 0.7)
            r["nir_0.7_1.3"] = EC.band_mean(lam_um, rho, 0.7, 1.3)
            r["swir_1.5_2.5"] = EC.band_mean(lam_um, rho, 1.5, 2.5)
            red, nir = EC.band_mean(lam_um, rho, 0.64, 0.67), EC.band_mean(lam_um, rho, 0.85, 0.88)
            r["ndvi"] = (nir - red) / (nir + red)
            outside = ((lam_um < s["wl_um"].min()) | (lam_um > s["wl_um"].max())).astype(float)
            r["frac_energy_extrapolated"] = float(np.trapezoid(rho * e_gl * outside, lam) / np.trapezoid(rho * e_gl, lam))
            sh = S.normalise_shape(lam, rho * e_gl)
            l_lam = levels[:, None] * sh[None, :] / np.pi
            _, j = S.iso_flux(lam, l_lam)
            jb = tab_b(levels)
            _, jg = S.iso_flux(lam, levels[:, None] * sh_grey[None, :] / np.pi)
            for f, jj, jjb, jjg in zip(levels, j, jb, jg):
                r[f"sw_ratio_core_F{f:g}"] = jj / (bb * f)          # vs Article 1 core (4/3) F / T_sun
                r[f"sw_ratio_b_F{f:g}"] = jj / jjb                  # vs Article 1 variant (b) diluted blackbody
                r[f"sw_ratio_grey_F{f:g}"] = jj / jjg               # vs spectrally flat reflector
                r[f"sw_JoverF_F{f:g}"] = jj / f
            # Kubelka-Munk canopy shape (green only, sensitivity)
            if s["cls"] == "green":
                shc = S.normalise_shape(lam, kubelka_munk_canopy(rho) * e_gl)
                _, jc = S.iso_flux(lam, 75.0 * shc / np.pi)
                r["sw_ratio_core_F75_KMcanopy"] = jc / (bb * 75.0)
            sw_arr.append(rho); sw_keys.append(i)
        if EC.lw_usable(s):
            eps = EC.resample_eps(s, K.LW_UM, "last_um")
            eps98 = EC.resample_eps(s, K.LW_UM, "0.98")
            b300 = S.planck_radiance(K.LW_M, 300.0)
            r["eps_8_14"] = EC.band_mean(K.LW_UM, eps, 8.0, 14.0)
            r["eps_bb_300K"] = float(np.trapezoid(eps * b300, K.LW_M) / np.trapezoid(b300, K.LW_M))
            r["eps_tail"] = float(eps[-1])
            r["lw_max_um"] = float(s["wl_um"].max())
            for tag, e_, sky in (("", eps, sky_grey), ("_tail098", eps98, sky_grey), ("_windowsky", eps, sky_win)):
                lout = S.lw_out_radiance(K.LW_M, e_, np.array([K.REF_TS]), sky)[0]
                f, j = S.iso_flux(K.LW_M, lout)
                ts_bb = ((f - (1 - K.EPS_A1) * K.REF_LWIN) / (K.EPS_A1 * S.SIGMA_SB)) ** 0.25
                j_core = 4.0 / 3.0 * f / ts_bb
                r[f"lw_F{tag}"] = f
                r[f"lw_ratio_core{tag}"] = j / j_core
                r[f"lw_Ts_bb_minus_true{tag}"] = ts_bb - K.REF_TS
            lw_arr.append(eps); lw_keys.append(i)
        rows.append(r)

    t = pd.DataFrame(rows)
    t.to_csv(K.TABLES / "a1_library_spectra.csv", index=False)
    np.savez_compressed(K.DERIVED / "library_arrays.npz", lam_sw=lam, rho=np.array(sw_arr), sw_idx=np.array(sw_keys),
                        lam_lw=K.LW_M, eps=np.array(lw_arr), lw_idx=np.array(lw_keys),
                        cls=np.array([s["cls"] for s in specs]), fname=np.array([s["file"] for s in specs]))

    # class summary
    cols = ["albedo_astm", "ndvi", "sw_ratio_core_F25", "sw_ratio_core_F75", "sw_ratio_core_F150", "sw_ratio_b_F75",
            "sw_ratio_grey_F75", "sw_ratio_core_F75_KMcanopy", "eps_8_14", "eps_bb_300K", "lw_ratio_core",
            "lw_ratio_core_tail098", "lw_ratio_core_windowsky", "lw_Ts_bb_minus_true"]
    summ = []
    for c, g in t.groupby("cls"):
        for col in cols:
            x = g[col].dropna()
            if len(x) == 0:
                continue
            summ.append({"cls": c, "variable": col, "n": len(x), "median": x.median(), "p05": x.quantile(0.05),
                         "p25": x.quantile(0.25), "p75": x.quantile(0.75), "p95": x.quantile(0.95), "min": x.min(), "max": x.max()})
    pd.DataFrame(summ).to_csv(K.TABLES / "a2_class_summary.csv", index=False)
    return t


if __name__ == "__main__":
    t = main()
    print(t.groupby("cls")[["albedo_astm", "sw_ratio_core_F75", "sw_ratio_b_F75", "sw_ratio_grey_F75", "eps_bb_300K", "lw_ratio_core"]].median())
