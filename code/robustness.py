"""Part C: robustness and transferability of the shortwave spectral correction (pre-submission
hardening against the usual desk-rejection points: one site, no benchmarks, measurement uncertainty).

C1  diffuse-fraction model ensemble (Erbs, Erbs-Driesse, Boland, Orgill-Hollands, DIRINT, DISC)
C2  radiometer-bias propagation (SW_IN, SW_OUT +-5 %) on the correction
C3  hierarchy of approximations for the shortwave terms (benchmarks)
C4  closed-form correction any tower can apply (power laws in flux), skill on BR-Npw
C5  transferability map: correction vs diffuse fraction and albedo, rescaling BR-Npw half-hours

Reuses the longwave terms of tower.py (data_derived/tower_terms.csv.gz: same rows, same order).
Writes results/tables/c1..c5 and results/run_log_robustness.txt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as K
import spectral as S
import config as C
import load as L
import entropy as E
import tower as TW

LOG = []
BB = 4.0 / (3.0 * S.T_SUN)


def log(m):
    print(m, flush=True); LOG.append(m)


def sw_terms(inc, out_tab, sw_in, sw_out, valid_out, pot, elev, **kw):
    jin, kd, dif = inc(sw_in, pot, elev, **kw)
    jout = out_tab(np.where(valid_out, np.clip(sw_out, 0, None), 0.0))
    return jin, jout, kd, dif


def main():
    df = L.add_derived(L.read_hh(), C.EPS_CORE)
    core = L.mask_core(df)
    d = df[core].copy()
    J0 = E.budget(d, C.CORE)
    tt = pd.read_csv(K.DERIVED / "tower_terms.csv.gz")
    assert len(tt) == len(d) and np.allclose(tt["sigma_A1"].to_numpy(), J0["sigma"].to_numpy())
    lam, ref = K.astm()
    sp = TW.class_spectra()
    out_tab = S.IsoTable(lam, sp["rho"]["green"] * ref["global"])
    inc = TW.IncomingModel(lam, ref)

    sw_in, sw_out = d["SW_IN_F"].to_numpy(float), d["SW_OUT"].to_numpy(float)
    is_day = d["NIGHT"].to_numpy() == 0
    valid_out = is_day & (sw_in > 0) & (sw_out > 0)
    full = L.read_hh()
    pot = full["SW_IN_POT"].shift(1).loc[d.index].to_numpy(float)
    t_mid = full["time"] + pd.Timedelta(minutes=15) - pd.Timedelta(minutes=30)
    elev = pd.Series(L.solar_elevation(t_mid), index=full.index).loc[d.index].to_numpy(float)
    times = pd.DatetimeIndex(t_mid.loc[d.index]).tz_localize("Etc/GMT+4")
    zen = 90.0 - elev
    wet = (d["season"] == "wet").to_numpy()

    # fixed (non-SW) part of the Article 2 sigma: everything except the two SW terms
    rest_A2 = tt["sigma_A2"].to_numpy() - tt["J_SWout_A2"].to_numpy() + tt["J_SWin_A2"].to_numpy()
    rest_A1 = J0["sigma"].to_numpy() - J0["J_SWout"].to_numpy() + J0["J_SWin"].to_numpy()
    sig_A1 = J0["sigma"].to_numpy()

    def summ(name, group, jin, jout, rest=rest_A2, extra=None):
        s = rest + jout - jin
        r = {"group": group, "configuration": name, "J_SWin": jin.mean(), "J_SWout": jout.mean(), "sigma": s.mean(),
             "d_sigma_vs_A1": s.mean() - sig_A1.mean(), "sigma_wet": s[wet].mean(), "sigma_dry": s[~wet].mean(),
             "d_wet_minus_dry": s[wet].mean() - s[~wet].mean()}
        if extra:
            r.update(extra)
        return r

    # ------------------------------------------------------------------ C1 diffuse models
    import pvlib.irradiance as I
    ghi = pd.Series(np.clip(sw_in, 0, None), index=times)
    z = pd.Series(zen, index=times)
    doy = times.dayofyear
    kd_models = {}
    with np.errstate(all="ignore"):
        kd_models["Erbs (reference)"] = None
        kd_models["Erbs-Driesse"] = I.erbs_driesse(ghi, z, times)["dhi"] / ghi
        kd_models["Boland"] = I.boland(ghi, z, times)["dhi"] / ghi
        kd_models["Orgill-Hollands"] = I.orgill_hollands(ghi, z, times)["dhi"] / ghi
        pa = d["PA_F"].to_numpy(float) * 1000.0
        dni_dirint = I.dirint(ghi, z, times, pressure=pd.Series(np.nan_to_num(pa, nan=99000.0), index=times))
        kd_models["DIRINT"] = 1.0 - dni_dirint.fillna(0) * np.cos(np.deg2rad(z)) / ghi
        dni_disc = I.disc(ghi, z, times)["dni"]
        kd_models["DISC"] = 1.0 - dni_disc.fillna(0) * np.cos(np.deg2rad(z)) / ghi
    rows = []
    for name, kd in kd_models.items():
        kdv = None if kd is None else np.asarray(kd, float)
        jin, jout, kdu, dif = sw_terms(inc, out_tab, sw_in, sw_out, valid_out, pot, elev, kd_given=kdv)
        share = dif[is_day].sum() / sw_in[is_day].sum()
        rows.append(summ(name, "C1 diffuse model", jin, jout, extra={"diffuse_share_day": share,
                         "diffuse_share_wet": dif[is_day & wet].sum() / sw_in[is_day & wet].sum(),
                         "diffuse_share_dry": dif[is_day & ~wet].sum() / sw_in[is_day & ~wet].sum()}))
        log(f"C1 {name}: daytime diffuse share {share:.3f}; sigma {rows[-1]['sigma']:.4f}; change vs A1 {rows[-1]['d_sigma_vs_A1']:+.4f}")
    c1 = pd.DataFrame(rows); c1.to_csv(K.TABLES / "c1_diffuse_models.csv", index=False)

    # ------------------------------------------------------------------ C2 radiometer bias
    rows = []
    for lab, fi, fo in (("none", 1, 1), ("SW_IN +5 %", 1.05, 1), ("SW_IN -5 %", 0.95, 1), ("SW_OUT +5 %", 1, 1.05), ("SW_OUT -5 %", 1, 0.95)):
        si, so = sw_in * fi, sw_out * fo
        jin, jout, _, _ = sw_terms(inc, out_tab, si, so, valid_out, pot, elev)
        a1_in, a1_out = BB * np.clip(si, 0, None), BB * np.where(valid_out, np.clip(so, 0, None), 0)
        # absorbed SW enters the turbulent/LW-independent part only through the measured budget; keep rest fixed
        corr = (jout - a1_out) - (jin - a1_in)
        rows.append({"bias": lab, "spectral_correction_mean": corr.mean(), "wet": corr[wet].mean(), "dry": corr[~wet].mean()})
        log(f"C2 {lab}: spectral correction of SW terms {corr.mean():+.4f} (wet {corr[wet].mean():+.4f}, dry {corr[~wet].mean():+.4f})")
    pd.DataFrame(rows).to_csv(K.TABLES / "c2_radiometer_bias.csv", index=False)

    # ------------------------------------------------------------------ C3 hierarchy of approximations
    jin_ref, jout_ref, kd, dif = sw_terms(inc, out_tab, sw_in, sw_out, valid_out, pot, elev)
    dirh = np.clip(sw_in, 0, None) - dif
    lb, pl = K.wide_planck_shape()
    dbb = S.IsoTable(lb, pl)                  # diluted blackbody at T_sun, isotropic
    so_valid = np.where(valid_out, np.clip(sw_out, 0, None), 0)
    rows = []
    rows.append(summ("(4/3)F/T_sun for every stream (Article 1 core)", "C3 benchmark", BB * np.clip(sw_in, 0, None), BB * so_valid, rest=rest_A1))
    rows.append(summ("F/T_sun for every stream (= absorbed SW / T_sun)", "C3 benchmark", np.clip(sw_in, 0, None) / S.T_SUN, so_valid / S.T_SUN, rest=rest_A1))
    rows.append(summ("Diluted blackbody for reflected only (Stephens & O'Brien form); incoming (4/3)F/T_sun",
                     "C3 benchmark", BB * np.clip(sw_in, 0, None), dbb(so_valid), rest=rest_A1))
    rows.append(summ("Diluted blackbody for reflected and diffuse, beam (4/3)F/T_sun, Erbs split",
                     "C3 benchmark", BB * dirh + dbb(dif), dbb(so_valid), rest=rest_A1))
    jin50, _, _ = inc(sw_in, pot, elev, kd_given=np.full(len(sw_in), 0.5))
    rows.append(summ("Spectral, fixed 50/50 direct/diffuse split (Holdaway-type split)", "C3 benchmark", jin50, jout_ref))
    rows.append(summ("Spectral, Erbs split (Article 2 reference)", "C3 benchmark", jin_ref, jout_ref))
    c3 = pd.DataFrame(rows); c3.to_csv(K.TABLES / "c3_benchmarks.csv", index=False)
    for _, r in c3.iterrows():
        log(f"C3 {r['configuration']}: sigma {r['sigma']:.4f} ({r['d_sigma_vs_A1']:+.4f}); wet-dry {r['d_wet_minus_dry']:.4f}")

    # ------------------------------------------------------------------ C4 closed-form correction
    # power laws J = c F^b fitted to the spectral tables over the observed flux range
    def fit_power(tab, fmin, fmax):
        f = np.geomspace(fmin, fmax, 200)
        b, lc = np.polyfit(np.log(f), np.log(tab(f)), 1)
        return float(np.exp(lc)), float(b)
    c_r, b_r = fit_power(out_tab, 5, 250)
    iso_blend_global = S.IsoTable(lam, ref["global"])
    iso_clear = S.IsoTable(lam, ref["diffuse_clear"])
    c_dg, b_dg = fit_power(iso_blend_global, 5, 600)
    c_dc, b_dc = fit_power(iso_clear, 5, 300)
    beam_tab = S.BeamTable(lam, ref["direct"], fmax=5000.0)
    c_b, b_b = fit_power(beam_tab, 20, 1200)
    # simple form (same inputs as the reference: SW_in, SW_out, kd, clearness index, solar elevation):
    #   reflected  c_r SWout^b_r
    #   diffuse    blend of clear-sky and global power laws with the same kt weight as the reference
    #   beam       cos(z) c_b DNI^b_b   (disk only; circumsolar neglected)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.clip(np.nan_to_num(np.where(pot > 0, np.clip(sw_in, 0, None) / pot, 0.0)), 0, 1)
    w = np.clip((TW.IncomingModel.KT_CLEAR - kt) / (TW.IncomingModel.KT_CLEAR - TW.IncomingModel.KT_CLOUD), 0, 1)
    cz = np.sin(np.deg2rad(np.clip(elev, 5, 90)))
    dni = np.where(dirh > 0, dirh / cz, 0.0)
    jdif_simple = (1 - w) * c_dc * np.power(dif, b_dc) + w * c_dg * np.power(dif, b_dg)
    jin_simple = jdif_simple + cz * c_b * np.power(dni, b_b)
    jout_simple = c_r * np.power(so_valid, b_r)
    beam_ratio = c_b
    s_ref = rest_A2 + jout_ref - jin_ref
    s_simple = rest_A2 + jout_simple - jin_simple
    day = is_day & (sw_in > 5)
    r2 = 1 - np.sum((s_simple - s_ref)[day] ** 2) / np.sum((s_ref[day] - s_ref[day].mean()) ** 2)
    corr_ref, corr_simple = s_ref - sig_A1, s_simple - sig_A1
    r2c = 1 - np.sum((corr_simple - corr_ref)[day] ** 2) / np.sum((corr_ref[day] - corr_ref[day].mean()) ** 2)
    c4 = pd.DataFrame([
        {"term": "reflected SW, J = c SW_out^b", "c": c_r, "b": b_r, "range_Wm2": "5-250"},
        {"term": "diffuse SW (global shape), J = c D^b", "c": c_dg, "b": b_dg, "range_Wm2": "5-600"},
        {"term": "diffuse SW (clear-sky shape), J = c D^b", "c": c_dc, "b": b_dc, "range_Wm2": "5-300"},
        {"term": "direct beam, J = cos(z) c DNI^b", "c": c_b, "b": b_b, "range_Wm2": "20-1200 DNI"},
        {"term": "skill: mean sigma simple vs spectral", "c": s_simple.mean(), "b": s_ref.mean(), "range_Wm2": ""},
        {"term": "skill: daytime R2 of half-hourly correction", "c": r2c, "b": np.nan, "range_Wm2": ""},
        {"term": "skill: daytime RMSE of half-hourly sigma (W m-2 K-1)", "c": float(np.sqrt(np.mean((s_simple - s_ref)[day] ** 2))), "b": np.nan, "range_Wm2": ""},
    ])
    c4.to_csv(K.TABLES / "c4_closed_form.csv", index=False)
    log(f"C4 reflected J = {c_r:.4e} SWout^{b_r:.4f}; diffuse(global) J = {c_dg:.4e} D^{b_dg:.4f}; diffuse(clear) {c_dc:.4e} D^{b_dc:.4f}; beam J = cos z {c_b:.4e} DNI^{b_b:.4f}")
    log(f"C4 skill: mean sigma simple {s_simple.mean():.4f} vs spectral {s_ref.mean():.4f}; daytime R2 of the half-hourly correction {r2c:.3f}; RMSE {np.sqrt(np.mean((s_simple - s_ref)[day] ** 2)):.4f}")

    # ------------------------------------------------------------------ C5 transferability map
    rows = []
    for ks in (0.5, 0.75, 1.0, 1.25, 1.5):
        kd_s = np.clip(kd * ks, 0, 1)
        for asc in (0.5, 0.75, 1.0, 1.5, 2.0):
            so = so_valid * asc
            jin, _, _ = inc(sw_in, pot, elev, kd_given=kd_s)
            jout = out_tab(so)
            corr = (jout - BB * so) - (jin - BB * np.clip(sw_in, 0, None))
            rows.append({"kd_scale": ks, "albedo_scale": asc, "diffuse_share_day": (kd_s * sw_in)[is_day].sum() / sw_in[is_day].sum(),
                         "albedo_day": so[is_day].sum() / sw_in[is_day].sum(), "correction_mean": corr.mean(),
                         "correction_rel_to_A1_sigma": corr.mean() / sig_A1.mean()})
    c5 = pd.DataFrame(rows); c5.to_csv(K.TABLES / "c5_transferability.csv", index=False)
    log("C5 correction (W m-2 K-1) by daytime diffuse share (rows) and albedo (cols):")
    piv = c5.pivot_table(index="diffuse_share_day", columns="albedo_day", values="correction_mean")
    log(piv.round(4).to_string())
    (K.ROOT / "results" / "run_log_robustness.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
