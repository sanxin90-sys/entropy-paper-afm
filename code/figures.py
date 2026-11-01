"""Figures G1-G5 of Article 2 (same style as Article 1: matplotlib, Okabe-Ito, PDF + PNG 300 dpi)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402

import common as K                 # noqa: E402
import spectral as S               # noqa: E402

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "figure.dpi": 100, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
})
CLS = {"green": ("#009E73", "Green vegetation"), "npv": ("#D55E00", "Non-photosynthetic veg."),
       "soil": ("#CC79A7", "Soil"), "water": ("#0072B2", "Water")}
WET, DRY = "#0072B2", "#D55E00"
BB = 4.0 / (3.0 * S.T_SUN)


def save(fig, stem):
    K.FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(K.FIGS / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def g1_spectra():
    lam, ref = K.astm()
    a = np.load(K.DERIVED / "library_arrays.npz")
    cls = a["cls"]; sw_c = cls[a["sw_idx"]]; lw_c = cls[a["lw_idx"]]
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 8.5))
    ax = axs[0]
    for key, lab, col, ls in (("extraterrestrial", "Extraterrestrial", "#999999", "-"), ("global", "Global AM1.5", "#000000", "-"),
                              ("direct", "Direct normal AM1.5", "#E69F00", "--"), ("diffuse_clear", "Clear-sky diffuse (derived)", "#56B4E9", "-")):
        ax.plot(lam * 1e6, ref[key] * 1e-9, color=col, ls=ls, lw=1.0, label=lab)
    ax.set_xlim(0.28, 2.6); ax.set_ylabel(r"Irradiance (W m$^{-2}$ nm$^{-1}$)"); ax.legend(frameon=False)
    ax.set_title("(a) ASTM G173-03 reference solar spectra")
    ax = axs[1]
    lu = a["lam_sw"] * 1e6
    for c, (col, lab) in CLS.items():
        r = a["rho"][sw_c == c]
        if len(r) == 0:
            continue
        ax.plot(lu, np.median(r, axis=0), color=col, label=f"{lab} (n = {len(r)})")
        if len(r) > 4:
            ax.fill_between(lu, np.percentile(r, 25, axis=0), np.percentile(r, 75, axis=0), color=col, alpha=0.2, lw=0)
    ax.set_xlim(0.35, 2.5); ax.set_ylim(0, 0.7); ax.set_ylabel("Reflectance"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("(b) ECOSTRESS reflectance, class median and interquartile range")
    ax = axs[2]
    ll = a["lam_lw"] * 1e6
    for c, (col, lab) in CLS.items():
        e = a["eps"][lw_c == c]
        if len(e) == 0:
            continue
        ax.plot(ll, np.median(e, axis=0), color=col, label=f"{lab} (n = {len(e)})")
        if len(e) > 4:
            ax.fill_between(ll, np.percentile(e, 25, axis=0), np.percentile(e, 75, axis=0), color=col, alpha=0.2, lw=0)
    ax.axvspan(15.4, 16, color="#eeeeee", lw=0)
    ax.set_xlim(3, 16); ax.set_ylim(0.75, 1.0); ax.set_xlabel(r"Wavelength ($\mu$m)"); ax.set_ylabel(r"Emissivity $1-R$")
    ax.legend(frameon=False, fontsize=7, loc="lower right"); ax.set_title("(c) Thermal-infrared emissivity (Kirchhoff)")
    fig.tight_layout(); save(fig, "g1_spectra")


def g2_entropy_per_energy():
    """J/F relative to 4/(3 T_sun) as a function of flux, for each field geometry."""
    lam, ref = K.astm()
    a = np.load(K.DERIVED / "library_arrays.npz")
    sw_c = a["cls"][a["sw_idx"]]
    f = np.geomspace(5, 1000, 60)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    lb, pl = K.wide_planck_shape()
    tb = S.IsoTable(lb, pl)
    ax.plot(f, tb(f) / f / BB, color="#000000", ls=":", label=r"Diluted blackbody at $T_{sun}$, isotropic")
    td = S.IsoTable(lam, ref["diffuse_clear"])
    ax.plot(f, td(f) / f / BB, color="#56B4E9", lw=2, label="Diffuse sky light, isotropic (clear-sky shape)")
    tg = S.IsoTable(lam, ref["global"])
    ax.plot(f, tg(f) / f / BB, color="#56B4E9", ls="--", label="Diffuse, global-spectrum shape")
    beam = S.BeamTable(lam, ref["direct"], fmax=5000)
    ax.plot(f, beam(f) / f / BB, color="#E69F00", lw=2, label="Direct beam, solar-disk solid angle")
    for c in ("green", "npv", "soil"):
        rho = np.median(a["rho"][sw_c == c], axis=0)
        t = S.IsoTable(lam, rho * ref["global"])
        ax.plot(f[f <= 300], t(f[f <= 300]) / f[f <= 300] / BB, color=CLS[c][0], label=f"Reflected, {CLS[c][1].lower()} (median)")
    ax.axhline(1.0, color="#999999", lw=1)
    ax.text(6, 1.12, r"Broadband $(4/3)F/T_{sun}$ (broadband core)", color="#666666", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel(r"Energy flux $F$ (W m$^{-2}$)")
    ax.set_ylabel(r"$(J/F)\,/\,(4/3T_{sun})$")
    ax.set_ylim(0, 6.5); ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout(); save(fig, "g2_entropy_per_energy")


def g3_decomposition(b1: pd.DataFrame, b3: pd.DataFrame):
    """Change in mean sigma by term, with plausible sensitivity ranges, against closure and season."""
    s = b1.set_index("configuration")["sigma_all"]
    a1 = s["A1 core (broadband)"]
    swout = s["+ J_SWout spectral (green)"] - a1
    swin = s["+ J_SWin spectral (beam + diffuse)"] - a1
    lw = s["+ J_LWout spectral (green eps, Ts)"] - a1
    net = s["A2 spectral (SWin + SWout + LWout)"] - a1
    sens = b1[b1["group"].isin(["sens_swout", "sens_swin", "sens_lw"])]
    rng = {g: (sens[sens["group"] == g]["sigma_all"].min() - a1, sens[sens["group"] == g]["sigma_all"].max() - a1) for g in ("sens_swout", "sens_swin", "sens_lw")}
    plaus = sens[~sens["configuration"].str.contains("all SW_IN")]
    net_lo, net_hi = plaus["sigma_all"].min() - a1, plaus["sigma_all"].max() - a1
    closure = s["A1 core, closure S1"] - a1
    seas = b1.set_index("configuration").loc["A1 core (broadband)", "d_wet_minus_dry"]
    rows = [(r"$J_{SW,out}$ spectral", swout, "#F0E442", None), (r"$J_{SW,in}$ beam + diffuse", swin, "#56B4E9", None),
            (r"$J_{LW,out}$ spectral", lw, "#009E73", None), ("Net spectral correction", net, "#000000", (net_lo, net_hi)),
            ("Closure residual (S1 - S0)", closure, "#999999", None), ("Wet - dry contrast (broadband)", seas, "#999999", None)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, col, er) in zip(y, rows):
        ax.barh(yi, v, color=col, edgecolor="#333333", lw=0.6, height=0.6, alpha=0.9 if col != "#999999" else 0.5)
        if er is not None:
            ax.plot([er[0], er[1]], [yi, yi], color="#D55E00", lw=2.5, solid_capstyle="butt")
        xt = v + 0.003 if v >= 0 else min(v, er[0] if er is not None else v) - 0.003
        ax.text(xt, yi, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel(r"Change in mean $\sigma$ (W m$^{-2}$ K$^{-1}$)")
    ax.set_xlim(-0.09, 0.11)
    # title moved to the caption (Elsevier: no titles on figures)
    fig.tight_layout(); save(fig, "g3_decomposition")


def g4_diurnal_monthly(b5: pd.DataFrame, b6: pd.DataFrame):
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 5.6))
    for ax, season, col in ((axs[0, 0], "wet", WET), (axs[0, 1], "dry", DRY)):
        d = b5[b5["season"] == season]
        h = d["slot"] / 2.0 + 0.25
        ax.plot(h, d["J_SWin_A2"] - d["J_SWin_A1"], color="#56B4E9", label=r"$\Delta J_{SW,in}$")
        ax.plot(h, d["J_SWout_A2"] - d["J_SWout_A1"], color="#E69F00", label=r"$\Delta J_{SW,out}$")
        ax.plot(h, d["sigma_A2"] - d["sigma_A1"], color="#000000", lw=2, label=r"$\Delta\sigma$")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6)); ax.set_xlabel("Local standard time (h)")
        ax.set_title(f"({'a' if season == 'wet' else 'b'}) {season} season, diurnal composite")
        ax.set_ylim(-0.22, 0.22)
    axs[0, 0].set_ylabel(r"Spectral - broadband (W m$^{-2}$ K$^{-1}$)")
    axs[0, 0].legend(frameon=False, fontsize=7)
    ax = axs[1, 0]
    m = b6["month"]
    for mm in (11, 12, 1, 2, 3, 4):
        ax.axvspan(mm - 0.5, mm + 0.5, color=WET, alpha=0.07, lw=0)
    ax.bar(m, b6["d_sigma"], color="#000000", width=0.6)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_ylabel(r"$\Delta\sigma$ (W m$^{-2}$ K$^{-1}$)"); ax.set_title("(c) Monthly net correction")
    ax = axs[1, 1]
    for mm in (11, 12, 1, 2, 3, 4):
        ax.axvspan(mm - 0.5, mm + 0.5, color=WET, alpha=0.07, lw=0)
    tt = pd.read_csv(K.DERIVED / "tower_terms.csv.gz")
    tt = tt[tt["is_day"]]
    kd_m = tt.groupby("month")["SW_dif"].sum() / tt.groupby("month")["SW_IN"].sum()
    ax.plot(kd_m.index, kd_m.to_numpy(), color="#56B4E9", marker="o", ms=4)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_ylabel(r"Diffuse share of daytime $SW_{in}$ (Erbs)"); ax.set_title("(d) Modelled diffuse fraction")
    fig.tight_layout(); save(fig, "g4_diurnal_monthly")


def g5_sensitivity(b1: pd.DataFrame):
    s = b1[b1["group"].str.startswith("sens") | (b1["configuration"].str.contains("sky"))].copy()
    a2 = b1.set_index("configuration").loc["A2 spectral (SWin + SWout + LWout)", "sigma_all"]
    s["d"] = s["sigma_all"] - a2
    s = s.reindex(s["d"].abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(s) + 1.2))
    y = np.arange(len(s))
    cols = {"sens_swout": "#E69F00", "sens_swin": "#56B4E9", "sens_lw": "#009E73", "sky": "#CC79A7"}
    ax.barh(y, s["d"], color=[cols.get(g, "#999999") for g in s["group"]], height=0.6)
    ax.set_yticks(y); ax.set_yticklabels([c.replace("A2, ", "").replace("A2 spectral + ", "") for c in s["configuration"]], fontsize=7)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel(r"$\sigma$ minus spectral reference (W m$^{-2}$ K$^{-1}$); reference = " + f"{a2:.3f}")
    fig.tight_layout(); save(fig, "g5_sensitivity")


def main():
    b1 = pd.read_csv(K.TABLES / "b1_configurations.csv")
    b3 = pd.read_csv(K.TABLES / "b3_bootstrap.csv")
    b5 = pd.read_csv(K.TABLES / "b5_diurnal.csv")
    b6 = pd.read_csv(K.TABLES / "b6_monthly.csv")
    g1_spectra(); g2_entropy_per_energy(); g3_decomposition(b1, b3); g4_diurnal_monthly(b5, b6); g5_sensitivity(b1)


if __name__ == "__main__":
    main()


def g6_benchmarks_transfer():
    c3 = pd.read_csv(K.TABLES / "c3_benchmarks.csv")
    c1 = pd.read_csv(K.TABLES / "c1_diffuse_models.csv")
    c5 = pd.read_csv(K.TABLES / "c5_transferability.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    labs = ["(4/3)F/T$_{sun}$ (broadband core)", "F/T$_{sun}$", "Diluted BB, reflected only", "Diluted BB, reflected + diffuse",
            "Spectral, fixed 50/50 split", "Spectral, Erbs split (ref.)"]
    v = c3["d_sigma_vs_A1"].to_numpy()
    y = np.arange(len(v))[::-1]
    ax.barh(y, v, color=["#999999", "#999999", "#E69F00", "#56B4E9", "#CC79A7", "#000000"], height=0.6)
    lo, hi = c1["d_sigma_vs_A1"].min(), c1["d_sigma_vs_A1"].max()
    ax.plot([lo, hi], [y[-1], y[-1]], color="#D55E00", lw=2.5, solid_capstyle="butt")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel(r"Change in mean $\sigma$ vs broadband core (W m$^{-2}$ K$^{-1}$)", fontsize=8)
    ax.set_title("(a) Benchmarks (orange: 6 diffuse models)", fontsize=9)
    ax = axs[1]
    piv = c5.pivot_table(index="diffuse_share_day", columns="albedo_day", values="correction_mean")
    X, Y = np.meshgrid(piv.columns.to_numpy(), piv.index.to_numpy())
    cs = ax.contourf(X, Y, piv.to_numpy(), levels=np.arange(-0.06, 0.035, 0.01), cmap="RdBu_r", vmin=-0.06, vmax=0.06)
    ax.contour(X, Y, piv.to_numpy(), levels=[0], colors="k", linewidths=1.2)
    ax.plot([0.167], [0.396], marker="*", color="k", ms=10)
    ax.text(0.175, 0.405, "BR-Npw", fontsize=7)
    fig.colorbar(cs, ax=ax, label=r"Correction (W m$^{-2}$ K$^{-1}$)")
    ax.set_xlabel("Daytime albedo"); ax.set_ylabel("Daytime diffuse share of SW$_{in}$")
    ax.set_title("(b) Correction vs cloudiness and albedo", fontsize=9)
    ax.grid(False)
    fig.tight_layout(); save(fig, "g6_benchmarks_transfer")


if __name__ == "__main__":
    g6_benchmarks_transfer()
