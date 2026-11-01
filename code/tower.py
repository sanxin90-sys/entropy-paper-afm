"""Part B: spectral radiation entropy terms applied to the BR-Npw half-hourly series
(Article 1 core sample, 2015-2016, n = 25,662), compared with the Article 1 broadband budget.

Reuses Article 1 code read-only (npw/code: config, load, entropy, stats). Writes
results/tables/b1_configurations.csv, b2_seasonal.csv, b3_bootstrap.csv, b4_swout_library_spread.csv,
b5_diurnal.csv, b6_monthly.csv and data_derived/tower_terms.parquet (or .csv.gz).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common as K
import spectral as S
import config as C          # Article 1
import load as L            # Article 1
import entropy as E         # Article 1
import stats as ST          # Article 1

LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


# ----------------------------------------------------------------------------
# spectra
# ----------------------------------------------------------------------------
def class_spectra():
    a = np.load(K.DERIVED / "library_arrays.npz", allow_pickle=False)
    cls = a["cls"]
    sw_cls = cls[a["sw_idx"]]
    lw_cls = cls[a["lw_idx"]]
    lib = pd.read_csv(K.TABLES / "a1_library_spectra.csv")
    out = {"lam_sw": a["lam_sw"], "lam_lw": a["lam_lw"], "rho_all": a["rho"], "rho_cls": sw_cls,
           "eps_all": a["eps"], "eps_cls": lw_cls, "rho": {}, "eps": {}}
    for c in ("green", "npv", "soil", "water"):
        out["rho"][c] = np.median(a["rho"][sw_cls == c], axis=0)
        out["eps"][c] = np.median(a["eps"][lw_cls == c], axis=0)
    # green spectra at the 5th / 95th percentile of the reflected-entropy ratio (library spread)
    g = lib[(lib["cls"] == "green") & lib["sw_ratio_core_F75"].notna()].reset_index(drop=True)
    gi = np.where(sw_cls == "green")[0]
    for q in (0.05, 0.95):
        k = (g["sw_ratio_core_F75"] - g["sw_ratio_core_F75"].quantile(q)).abs().idxmin()
        out["rho"][f"green_p{int(q * 100):02d}"] = a["rho"][gi[k]]
    # green emissivity extremes: lowest and highest broadband emissivity
    gl = lib[(lib["cls"] == "green") & lib["eps_bb_300K"].notna()].reset_index(drop=True)
    li = np.where(lw_cls == "green")[0]
    out["eps"]["green_min"] = a["eps"][li[gl["eps_bb_300K"].idxmin()]]
    out["eps"]["green_max"] = a["eps"][li[gl["eps_bb_300K"].idxmax()]]
    return out


# ----------------------------------------------------------------------------
# term calculators
# ----------------------------------------------------------------------------
CS_HALF_ANGLE_DEG = 2.5     # outer edge of the circumsolar region seen by a pyrheliometer (5 deg field)
OMEGA_CS = 2.0 * np.pi * (np.cos(S.SUN_HALF_ANGLE_RAD) - np.cos(np.deg2rad(CS_HALF_ANGLE_DEG)))


class IncomingModel:
    """Spectral entropy of incoming shortwave on a horizontal surface.

    SW_IN = diffuse + direct, kd from Erbs on kt = SW_IN / SW_IN_POT (optionally scaled or shifted).
    * diffuse: isotropic field whose spectral shape blends from the clear-sky horizontal diffuse
      (kt >= KT_CLEAR) to the global spectrum (overcast, kt <= KT_CLOUD), linear in kt between;
    * direct: a fraction (1 - f_cs) in the solar-disk solid angle and f_cs spread uniformly over
      the circumsolar annulus 0.27-2.5 deg (same spectrum: ASTM direct normal).
    Half-hours with solar elevation < 5 deg or SW_IN_POT <= 20 W m-2 are fully diffuse."""
    KT_CLEAR, KT_CLOUD = 0.65, 0.30

    def __init__(self, lam, ref, diffuse_clear="diffuse_clear", nw=11):
        self.w = np.linspace(0.0, 1.0, nw)
        sc = S.normalise_shape(lam, ref[diffuse_clear]); sg = S.normalise_shape(lam, ref["global"])
        self.iso = [S.IsoTable(lam, (1 - w) * sc + w * sg) for w in self.w]
        self.disk = S.BeamTable(lam, ref["direct"], fmax=5000.0)
        self.ann = S.BeamTable(lam, ref["direct"], omega=OMEGA_CS, fmin=1e-4, fmax=5000.0)

    def j_dif(self, dif, w):
        """Entropy of isotropic diffuse with blend weight w (0 = clear shape, 1 = global shape),
        linear interpolation between the tabulated weights."""
        w = np.clip(np.asarray(w, float), 0, 1)
        pos = w * (len(self.w) - 1)
        i0 = np.clip(np.floor(pos).astype(int), 0, len(self.w) - 2)
        t = pos - i0
        out = np.zeros(len(dif))
        for k in range(len(self.w) - 1):
            sel = i0 == k
            if sel.any():
                out[sel] = (1 - t[sel]) * self.iso[k](dif[sel]) + t[sel] * self.iso[k + 1](dif[sel])
        return out

    def __call__(self, sw_in, sw_pot, elev_deg, kd_mode="erbs", kd_shift=0.0, kd_scale=1.0,
                 shape="blend", f_cs=0.02, kd_given=None):
        sw = np.clip(np.asarray(sw_in, float), 0.0, None)
        pot = np.asarray(sw_pot, float)
        el = np.nan_to_num(np.asarray(elev_deg, float), nan=-90.0)
        cosz = np.sin(np.deg2rad(el))
        with np.errstate(divide="ignore", invalid="ignore"):
            kt = np.where(pot > 0, sw / pot, 0.0)
        kt = np.clip(np.nan_to_num(kt), 0.0, 1.0)
        kd = S.erbs_kd(kt) if kd_given is None else np.clip(np.nan_to_num(np.asarray(kd_given, float), nan=1.0), 0.0, 1.0)
        if kd_mode == "all_diffuse":
            kd = np.ones_like(kd)
        elif kd_mode == "all_beam":
            kd = np.zeros_like(kd)
        kd = np.clip(kd * kd_scale + kd_shift, 0.0, 1.0)
        low = (el < 5.0) | ~(pot > 20.0)
        if kd_mode != "all_beam":
            kd = np.where(low, 1.0, kd)
        dif = kd * sw
        dirh = sw - dif
        dni = np.where(cosz > 0.0, dirh / np.maximum(cosz, 0.087), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cz = np.where(dni > 0, dirh / dni, 0.0)    # equals cos z except where DNI was capped
        if shape == "blend":
            w = np.clip((self.KT_CLEAR - kt) / (self.KT_CLEAR - self.KT_CLOUD), 0.0, 1.0)
        elif shape == "clear":
            w = np.zeros_like(kt)
        elif shape == "global":
            w = np.ones_like(kt)
        else:
            raise ValueError(shape)
        j = self.j_dif(dif, w) + cz * (self.disk((1 - f_cs) * dni) + self.ann(f_cs * dni))
        return j, kd, dif


def j_sw_out_spectral(sw_out, valid, lam, rho, e_inc):
    """Lambertian reflection with spectral shape rho * E_inc; zero where Article 1 sets it to zero."""
    tab = S.IsoTable(lam, rho * e_inc)
    so = np.where(valid, np.clip(np.asarray(sw_out, float), 0.0, None), 0.0)
    return tab(so)


def j_sw_out_mixture(sw_out, valid, months, lam, rho_a, rho_b, frac_b_by_month, e_inc):
    """Reflection by a linear mixture (1 - f) rho_a + f rho_b, f fixed per calendar month."""
    out = np.zeros(len(sw_out))
    for m, f in frac_b_by_month.items():
        sel = np.asarray(months) == m
        if sel.any():
            out[sel] = j_sw_out_spectral(np.asarray(sw_out)[sel], np.asarray(valid)[sel], lam, (1 - f) * rho_a + f * rho_b, e_inc)
    return out


def j_lw_out_spectral(lw_out, lw_in, ta, lam, eps, sky_model="grey", chunk=3000):
    """Outgoing LW entropy from eps(lambda) B(Ts) + (1 - eps) L_sky, with Ts solved to reproduce
    LW_OUT. Returns (J_LWout, Ts, J_LWin from the same sky field)."""
    n = len(lw_out)
    j = np.full(n, np.nan); ts = np.full(n, np.nan); jin = np.full(n, np.nan)
    lo, li, t = (np.asarray(x, float) for x in (lw_out, lw_in, ta))
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        sky = S.sky_radiance(lam, t[a:b], li[a:b], sky_model)
        tsk = S.solve_ts(lam, eps, lo[a:b], sky)
        rad = S.lw_out_radiance(lam, eps, tsk, sky)
        _, jj = S.iso_flux(lam, rad)
        _, ji = S.iso_flux(lam, sky)
        j[a:b], ts[a:b], jin[a:b] = jj, tsk, ji
    return j, ts, jin


# ----------------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------------
def assemble(J0, sw_in=None, sw_out=None, lw=None, lw_in=None, closure="S0"):
    """Budget with any subset of the four radiation terms replaced. lw = (J_LWout, Ts) changes
    the surface temperature used for J_H and J_LE as well."""
    d = pd.DataFrame(index=J0.index)
    d["J_SWin"] = J0["J_SWin"] if sw_in is None else sw_in
    d["J_SWout"] = J0["J_SWout"] if sw_out is None else sw_out
    d["J_LWin"] = J0["J_LWin"] if lw_in is None else lw_in
    if lw is None:
        d["J_LWout"], ts = J0["J_LWout"], J0["Ts"]
    else:
        d["J_LWout"], ts = lw[0], lw[1]
    h, le = J0["H_s"], J0["LE_s"]
    if closure == "S1":
        h, le = J0["H_s"], J0["LE_s"] + J0["Rres"]
    d["J_H"] = h / ts
    d["J_LE"] = le / ts
    d["J_in"] = d["J_SWin"] + d["J_LWin"]
    d["J_out"] = d["J_SWout"] + d["J_LWout"] + d["J_H"] + d["J_LE"]
    d["sigma"] = d["J_out"] - d["J_in"]
    d["sigma_rad"] = d["J_SWout"] + d["J_LWout"] - d["J_SWin"] - d["J_LWin"]
    d["sigma_turb"] = d["J_H"] + d["J_LE"]
    d["Ts"] = ts
    return d


def summarise(name, d, masks, group):
    r = {"group": group, "configuration": name}
    for mname, m in masks.items():
        x = d[m]
        r[f"sigma_{mname}"] = x["sigma"].mean()
        r[f"SR_{mname}"] = x["J_out"].mean() / x["J_in"].mean()
    for c in ("J_SWin", "J_SWout", "J_LWin", "J_LWout", "J_H", "J_LE", "sigma_rad", "sigma_turb"):
        r[c] = d.loc[masks["all"], c].mean()
    r["d_wet_minus_dry"] = r["sigma_wet"] - r["sigma_dry"]
    return r


def block_boot_mean(x, dates, mask, block_days, n_boot=C.BOOT_N, seed=C.BOOT_SEED):
    """Day-block bootstrap of the half-hourly mean of x over mask (Article 1 resampling)."""
    f = pd.DataFrame({"x": np.asarray(x, float)})
    sums, cnt, days = ST._daily_sums(f, np.asarray(dates), np.asarray(mask), ["x"])
    rng = np.random.default_rng(seed)
    idx = ST._block_indices(len(days), n_boot, block_days, rng)
    return sums[idx, 0].sum(axis=1) / cnt[idx].sum(axis=1)


def main():
    t0 = time.time()
    df = L.add_derived(L.read_hh(), C.EPS_CORE)
    core = L.mask_core(df)
    assert int(core.sum()) == C.CORE_N_REGRESSION
    d = df[core].copy()
    J0 = E.budget(d, C.CORE)
    log(f"core sample n = {len(d)}; Article 1 core sigma = {J0['sigma'].mean():.5f} (expected 0.49536)")
    assert abs(J0["sigma"].mean() - 0.49536) < 5e-5

    lam, ref = K.astm()
    sp = class_spectra()
    lam_lw = sp["lam_lw"]
    sw_in, sw_out = d["SW_IN_F"].to_numpy(float), d["SW_OUT"].to_numpy(float)
    lw_in, lw_out = d["LW_IN_F"].to_numpy(float), d["LW_OUT"].to_numpy(float)
    ta = d["TA_F"].to_numpy(float) + C.T0
    is_day = (d["NIGHT"].to_numpy() == 0)
    valid_out = is_day & (sw_in > 0) & (sw_out > 0)
    elev = d["sun_elev"].to_numpy(float)
    pot = d["SW_IN_POT"].to_numpy(float)

    masks = {"all": np.ones(len(d), bool), "wet": (d["season"] == "wet").to_numpy(), "dry": (d["season"] == "dry").to_numpy(),
             "day": is_day, "night": ~is_day}

    # -------------------------------------------------------------- SW in
    # Timing: measured SW_IN_F lags the solar geometry by about one half-hour at BR-Npw (independent
    # review, docs/REVIEW_physics_code.md finding 4). The reference therefore takes kt and the solar
    # elevation from the previous half-hour (geometry shifted by -30 min); the unshifted geometry is
    # a sensitivity.
    full = L.read_hh()
    pot_prev = full["SW_IN_POT"].shift(1)
    elev_prev = pd.Series(L.solar_elevation(full["time"] + pd.Timedelta(minutes=15) - pd.Timedelta(minutes=30)), index=full.index)
    pot_s = pot_prev.loc[d.index].to_numpy(float)
    elev_s = elev_prev.loc[d.index].to_numpy(float)
    kt0 = np.where(pot > 20, sw_in / np.where(pot > 0, pot, np.nan), np.nan)
    kts = np.where(pot_s > 20, sw_in / np.where(pot_s > 0, pot_s, np.nan), np.nan)
    log(f"kt > 1 in {(kt0 > 1).sum()} half-hours with the stamped geometry, {(kts > 1).sum()} with geometry shifted -30 min")

    inc = IncomingModel(lam, ref)
    inc_raw = IncomingModel(lam, ref, diffuse_clear="diffuse_tilt_raw")
    jin = {}
    jin["ref"], kd, dif = inc(sw_in, pot_s, elev_s)
    d["kd"] = kd; d["SW_dif"] = dif
    log(f"clearness kt (day, shifted geometry): mean {np.nanmean(kts):.3f}; diffuse share of daytime SW_IN (ratio of sums) "
        f"{dif[is_day].sum() / sw_in[is_day].sum():.3f}; wet {dif[is_day & masks['wet']].sum() / sw_in[is_day & masks['wet']].sum():.3f}, "
        f"dry {dif[is_day & masks['dry']].sum() / sw_in[is_day & masks['dry']].sum():.3f}")
    jin["unshifted geometry"], _, _ = inc(sw_in, pot, elev)
    jin["diffuse shape clear-sky only"], _, _ = inc(sw_in, pot_s, elev_s, shape="clear")
    jin["diffuse shape global only"], _, _ = inc(sw_in, pot_s, elev_s, shape="global")
    jin["diffuse shape with ground light (tilted ASTM)"], _, _ = inc_raw(sw_in, pot_s, elev_s)
    jin["circumsolar 0 %"], _, _ = inc(sw_in, pot_s, elev_s, f_cs=0.0)
    jin["circumsolar 5 %"], _, _ = inc(sw_in, pot_s, elev_s, f_cs=0.05)
    jin["circumsolar 10 %"], _, _ = inc(sw_in, pot_s, elev_s, f_cs=0.10)
    jin["kd x 0.75"], _, _ = inc(sw_in, pot_s, elev_s, kd_scale=0.75)
    jin["kd x 1.25"], _, _ = inc(sw_in, pot_s, elev_s, kd_scale=1.25)
    jin["kd / 1.079 (Erbs bias, Abal et al. 2017)"], _, _ = inc(sw_in, pot_s, elev_s, kd_scale=1 / 1.079)
    jin["all SW_IN diffuse (upper bound)"], _, _ = inc(sw_in, pot_s, elev_s, kd_mode="all_diffuse")
    jin["all SW_IN beam (lower bound)"], _, _ = inc(sw_in, pot_s, elev_s, kd_mode="all_beam")
    for k, v in jin.items():
        log(f"J_SWin [{k}] mean {v.mean():.5f}")
    log(f"Article 1 core J_SWin {J0['J_SWin'].mean():.5f}")

    # -------------------------------------------------------------- SW out
    jout = {}
    e_gl = ref["global"]
    for c, rho in sp["rho"].items():
        jout[c] = j_sw_out_spectral(sw_out, valid_out, lam, rho, e_gl)
    jout["grey"] = j_sw_out_spectral(sw_out, valid_out, lam, np.ones_like(lam), e_gl)
    import library as LB
    # albedo-constrained green + water mixture: water fraction per month that brings the mixture's
    # ASTM-weighted albedo to the tower's monthly midday albedo (Article 1, t8 pooled)
    t8 = pd.read_csv(K.ROOT / "data" / "article1_tables" / "t8_midday_albedo_by_month.csv")
    t8 = t8[t8["year"] == "pooled"].set_index("month")["albedo_midday"]
    alb = lambda r: float(np.trapezoid(r * e_gl, lam) / np.trapezoid(e_gl, lam))
    ag, aw = alb(sp["rho"]["green"]), alb(sp["rho"]["water"])
    fw = {int(m): float(np.clip((ag - a) / (ag - aw), 0, 1)) for m, a in t8.items()}
    log(f"albedo-constrained mixture: green albedo {ag:.3f}, water {aw:.3f}; water fraction by month " + ", ".join(f"{m}:{f:.2f}" for m, f in fw.items()))
    jout["green+water (albedo-matched)"] = j_sw_out_mixture(sw_out, valid_out, d["month"].to_numpy(), lam, sp["rho"]["green"], sp["rho"]["water"], fw, e_gl)
    jout["green_KMcanopy"] = j_sw_out_spectral(sw_out, valid_out, lam, LB.kubelka_munk_canopy(sp["rho"]["green"]), e_gl)
    log(f"J_SWout spectral (green median) mean {jout['green'].mean():.5f} vs Article 1 core {J0['J_SWout'].mean():.5f}")

    # library spread: every SW-usable spectrum (period mean J_SWout)
    rows = []
    for i in range(len(sp["rho_all"])):
        j = j_sw_out_spectral(sw_out, valid_out, lam, sp["rho_all"][i], e_gl)
        rows.append({"cls": sp["rho_cls"][i], "J_SWout_mean": j.mean(), "J_SWout_wet": j[masks["wet"]].mean(), "J_SWout_dry": j[masks["dry"]].mean()})
    spread = pd.DataFrame(rows)
    spread.to_csv(K.TABLES / "b4_swout_library_spread.csv", index=False)
    log(f"library spread of mean J_SWout (all {len(spread)} spectra): {spread['J_SWout_mean'].min():.5f}-{spread['J_SWout_mean'].max():.5f}")

    # -------------------------------------------------------------- LW
    lwo = {}
    for c in ("green", "npv", "soil", "water", "green_min", "green_max"):
        tl = time.time()
        j, ts, jli = j_lw_out_spectral(lw_out, lw_in, ta, lam_lw, sp["eps"][c], "grey")
        lwo[c] = (j, ts, jli)
        log(f"LW {c}: J_LWout {j.mean():.5f} (A1 {J0['J_LWout'].mean():.5f}); Ts mean {ts.mean():.3f} K (A1 {J0['Ts'].mean():.3f}); {time.time() - tl:.0f} s")
    for tail in ("last_um+0.02", "last_um-0.02"):
        e_t = sp["eps"]["green"].copy()
        beyond = sp["lam_lw"] * 1e6 > 15.3
        e_t[beyond] = np.clip(e_t[beyond] + (0.02 if "+" in tail else -0.02), 0, 1)
        j, ts, jli = j_lw_out_spectral(lw_out, lw_in, ta, lam_lw, e_t, "grey")
        lwo[f"green tail {tail[7:]}"] = (j, ts, jli)
        log(f"LW green, emissivity beyond 15.3 um {tail[7:]}: J_LWout {j.mean():.5f}; Ts mean {ts.mean():.3f} K")
    j, ts, jli_w = j_lw_out_spectral(lw_out, lw_in, ta, lam_lw, sp["eps"]["green"], "window")
    lwo["green_windowsky"] = (j, ts, jli_w)
    log(f"LW green window sky: J_LWout {j.mean():.5f}; J_LWin(window) {jli_w.mean():.5f} vs A1 core {J0['J_LWin'].mean():.5f}; grey-sky J_LWin {lwo['green'][2].mean():.5f}")

    # -------------------------------------------------------------- configurations
    G = lwo["green"]
    cfg = [
        ("A1 core (broadband)", "headline", dict()),
        ("+ J_SWout spectral (green)", "stepwise", dict(sw_out=jout["green"])),
        ("+ J_SWin spectral (beam + diffuse)", "stepwise", dict(sw_in=jin["ref"])),
        ("+ J_LWout spectral (green eps, Ts)", "stepwise", dict(lw=G[:2])),
        ("A2 spectral (SWin + SWout + LWout)", "headline", dict(sw_in=jin["ref"], sw_out=jout["green"], lw=G[:2])),
        ("A2 spectral + J_LWin grey sky (diluted, T_a)", "sky", dict(sw_in=jin["ref"], sw_out=jout["green"], lw=G[:2], lw_in=G[2])),
        ("A2 spectral + J_LWin window sky", "sky", dict(sw_in=jin["ref"], sw_out=jout["green"], lw=lwo["green_windowsky"][:2], lw_in=lwo["green_windowsky"][2])),
    ]
    base = dict(sw_in=jin["ref"], sw_out=jout["green"], lw=G[:2])
    for c in ("npv", "soil", "water", "grey", "green_p05", "green_p95", "green_KMcanopy", "green+water (albedo-matched)"):
        cfg.append((f"A2, reflected shape {c}", "sens_swout", {**base, "sw_out": jout[c]}))
    for k in jin:
        if k != "ref":
            cfg.append((f"A2, {k}", "sens_swin", {**base, "sw_in": jin[k]}))
    for c in ("npv", "soil", "water", "green_min", "green_max", "green tail +0.02", "green tail -0.02"):
        cfg.append((f"A2, emissivity {c}", "sens_lw", {**base, "lw": lwo[c][:2]}))
    ft_in = sw_in / S.T_SUN
    ft_out = np.where(valid_out, sw_out, 0.0) / S.T_SUN
    cfg.append(("Broadband F/T for solar streams (no 4/3; = absorbed SW / T_sun)", "convention", dict(sw_in=ft_in, sw_out=ft_out)))
    cfg.append(("A1 core, closure S1", "closure", dict(closure="S1")))
    cfg.append(("A2 spectral, closure S1", "closure", {**base, "closure": "S1"}))

    rows, store = [], {}
    for name, grp, kw in cfg:
        dd = assemble(J0, **kw)
        store[name] = dd
        rows.append(summarise(name, dd, masks, grp))
    t1 = pd.DataFrame(rows)
    ref_sig = t1.loc[0, "sigma_all"]
    t1["d_sigma_vs_A1"] = t1["sigma_all"] - ref_sig
    t1.to_csv(K.TABLES / "b1_configurations.csv", index=False)

    # -------------------------------------------------------------- bootstrap
    A1, A2 = store["A1 core (broadband)"], store["A2 spectral (SWin + SWout + LWout)"]
    dates = d["date"].to_numpy()
    brow = []
    for bd in (C.BOOT_BLOCK_DAYS_HEADLINE, C.BOOT_BLOCK_DAYS_OPTIMISTIC):
        for lab, x in (("sigma A1", A1["sigma"]), ("sigma A2", A2["sigma"]), ("A2 - A1 (paired)", A2["sigma"] - A1["sigma"])):
            for per, m in (("all", masks["all"]), ("wet", masks["wet"]), ("dry", masks["dry"])):
                b = block_boot_mean(x.to_numpy(), dates, m, bd)
                brow.append({"block_days": bd, "quantity": lab, "period": per, "mean": float(np.mean(np.asarray(x)[m])),
                             "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5))})
        for lab, x in (("wet - dry, A1", A1["sigma"]), ("wet - dry, A2", A2["sigma"])):
            ba = block_boot_mean(x.to_numpy(), dates, masks["wet"], bd, seed=C.BOOT_SEED)
            bb = block_boot_mean(x.to_numpy(), dates, masks["dry"], bd, seed=C.BOOT_SEED + 1)
            diff = ba - bb
            brow.append({"block_days": bd, "quantity": lab, "period": "wet-dry", "mean": float(x[masks["wet"]].mean() - x[masks["dry"]].mean()),
                         "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5)), "n_le0": int((diff <= 0).sum())})
    pd.DataFrame(brow).to_csv(K.TABLES / "b3_bootstrap.csv", index=False)

    # -------------------------------------------------------------- seasonal / diurnal / monthly
    terms = pd.DataFrame({"date": d["date"].to_numpy(), "month": d["month"].to_numpy(), "slot": d["slot"].to_numpy(),
                          "season": d["season"].to_numpy(), "is_day": is_day, "kd": kd, "SW_IN": sw_in, "SW_OUT": sw_out,
                          "SW_dif": dif, "absSW": J0["absSW"].to_numpy()})
    for c in ("J_SWin", "J_SWout", "J_LWin", "J_LWout", "J_H", "J_LE", "sigma", "Ts"):
        terms[f"{c}_A1"] = A1[c].to_numpy()
        terms[f"{c}_A2"] = A2[c].to_numpy()
    terms.to_csv(K.DERIVED / "tower_terms.csv.gz", index=False)
    seas = []
    for per, m in masks.items():
        x = terms[m]
        r = {"period": per, "n": int(m.sum()), "SW_IN": x["SW_IN"].mean(), "SW_dif": x["SW_dif"].mean(), "SW_OUT": x["SW_OUT"].mean(),
             "absSW": x["absSW"].mean(), "kd_day_ratio_of_sums": x.loc[x["is_day"], "SW_dif"].sum() / max(x.loc[x["is_day"], "SW_IN"].sum(), 1e-9)}
        for c in ("J_SWin", "J_SWout", "J_LWin", "J_LWout", "J_H", "J_LE", "sigma"):
            r[f"{c}_A1"] = x[f"{c}_A1"].mean(); r[f"{c}_A2"] = x[f"{c}_A2"].mean()
            r[f"d_{c}"] = r[f"{c}_A2"] - r[f"{c}_A1"]
        r["sigma_per_absSW_A1"] = r["sigma_A1"] / r["absSW"] if r["absSW"] > 0 else np.nan
        r["sigma_per_absSW_A2"] = r["sigma_A2"] / r["absSW"] if r["absSW"] > 0 else np.nan
        seas.append(r)
    pd.DataFrame(seas).to_csv(K.TABLES / "b2_seasonal.csv", index=False)
    cols = ["J_SWin", "J_SWout", "J_LWout", "sigma"]
    di = terms.groupby(["season", "slot"]).agg(**{f"{c}_{v}": (f"{c}_{v}", "mean") for c in cols for v in ("A1", "A2")},
                                                 kd=("kd", "mean"), SW_IN=("SW_IN", "mean"), n=("sigma_A1", "size")).reset_index()
    di.to_csv(K.TABLES / "b5_diurnal.csv", index=False)
    mo = terms.groupby("month").agg(**{f"{c}_{v}": (f"{c}_{v}", "mean") for c in cols for v in ("A1", "A2")},
                                    kd_day=("kd", lambda s: s[terms.loc[s.index, "is_day"]].mean()), SW_IN=("SW_IN", "mean"),
                                    n=("sigma_A1", "size")).reset_index()
    mo["d_sigma"] = mo["sigma_A2"] - mo["sigma_A1"]
    mo.to_csv(K.TABLES / "b6_monthly.csv", index=False)

    log(f"done in {time.time() - t0:.0f} s")
    (K.ROOT / "results" / "run_log_tower.txt").write_text("\n".join(LOG) + "\n")
    return t1


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    t = main()
    print(t[["configuration", "sigma_all", "d_sigma_vs_A1", "sigma_wet", "sigma_dry", "d_wet_minus_dry", "SR_all", "J_SWin", "J_SWout", "J_LWin", "J_LWout"]].round(4).to_string())
