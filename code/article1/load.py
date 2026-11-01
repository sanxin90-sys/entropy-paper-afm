"""Load the ONEFlux half-hourly file, derive time variables, solar position,
surface/sky temperatures, net radiation and residual, and build the selection masks.

Nothing here modifies the raw file. -9999 is read as missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------
def read_hh(path=C.HH_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=C.USECOLS, na_values=C.NA_VALUES)
    ts = pd.to_datetime(df["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M")
    df = df.assign(
        time=ts,
        year=ts.dt.year.astype(int),
        month=ts.dt.month.astype(int),
        date=ts.dt.normalize(),
        hour=ts.dt.hour + ts.dt.minute / 60.0,   # start of the half-hour, decimal hours LST
        doy=ts.dt.dayofyear.astype(int),
    )
    df["slot"] = (df["hour"] * 2).round().astype(int)      # 0..47
    df["season"] = np.where(df["month"].isin(C.WET_MONTHS), "wet", "dry")
    # wet season labelled by the year in which it ends (Nov-Dec belong to next year's wet season)
    df["season_year"] = np.where((df["season"] == "wet") & (df["month"] >= 11), df["year"] + 1, df["year"])
    df["season3"] = "dry"
    for name, months in C.SEASON3.items():
        df.loc[df["month"].isin(months), "season3"] = name
    # in-house solar position at the half-hour midpoint
    el = solar_elevation(ts + pd.Timedelta(minutes=15))
    df["sun_elev"] = el
    df["day_elev"] = (el > 0).astype(int)        # 1 = sun above horizon
    df["is_day"] = df["NIGHT"] == 0               # primary day flag (ONEFlux NIGHT, from SW_IN_POT)
    return df


def solar_elevation(t: pd.Series, lat=C.LAT, lon=C.LON, tz_h=C.UTC_OFFSET_H) -> np.ndarray:
    """Solar elevation angle (deg) from the NOAA/Spencer low-precision algorithm.

    t: local standard time (UTC + tz_h). Accuracy ~0.1-0.2 deg, adequate for a day flag.
    """
    doy = t.dt.dayofyear.to_numpy()
    hr = (t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0).to_numpy()
    gamma = 2 * np.pi / 365.0 * (doy - 1 + (hr - 12) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    time_offset = eqtime + 4.0 * lon - 60.0 * tz_h        # minutes
    tst = hr * 60.0 + time_offset                           # true solar time, minutes
    ha = np.deg2rad(tst / 4.0 - 180.0)
    phi = np.deg2rad(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1, 1)
    return 90.0 - np.rad2deg(np.arccos(cosz))


# ----------------------------------------------------------------------------
# Derived physical variables (independent of the entropy options except eps)
# ----------------------------------------------------------------------------
def t_surface(lw_out, lw_in, eps=C.EPS_CORE, sigma=C.SIGMA_SB):
    """Radiometric surface temperature (K) from LW_OUT = eps*sigma*Ts^4 + (1-eps)*LW_IN (eq. 9)."""
    return ((lw_out - (1.0 - eps) * lw_in) / (eps * sigma)) ** 0.25


def t_sky(lw_in, sigma=C.SIGMA_SB):
    """Blackbody-equivalent sky temperature (K): T_sky = (LW_IN/sigma)^(1/4) (eq. 8)."""
    return (lw_in / sigma) ** 0.25


def add_derived(df: pd.DataFrame, eps=C.EPS_CORE) -> pd.DataFrame:
    """Add T_a, T_s(eps), T_sky, Rn (4 components), residual R', albedo."""
    out = df.copy()
    out["Ta"] = out["TA_F"] + C.T0
    out["Ts"] = t_surface(out["LW_OUT"], out["LW_IN_F"], eps)
    out["Tsky"] = t_sky(out["LW_IN_F"])
    out["Rn"] = out["SW_IN_F"] - out["SW_OUT"] + out["LW_IN_F"] - out["LW_OUT"]
    out["Rn_abs"] = out["SW_IN_F"] - out["SW_OUT"] + out["LW_IN_F"]
    out["H"] = out["H_F_MDS"]
    out["LE"] = out["LE_F_MDS"]
    out["Rres"] = out["Rn"] - out["H"] - out["LE"]          # R' = Rn - H - LE (storage + imbalance)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["albedo"] = np.where(out["is_day"] & (out["SW_IN_F"] > 0), out["SW_OUT"] / out["SW_IN_F"], np.nan)
    return out


# ----------------------------------------------------------------------------
# Selection masks (section 4.2)
# ----------------------------------------------------------------------------
def mask_rad_out(df):
    """Measured radiation outputs present (SW_OUT and LW_OUT)."""
    return df["SW_OUT"].notna() & df["LW_OUT"].notna()


def mask_rad_in_measured(df):
    return (df["SW_IN_F_QC"] == 0) & (df["LW_IN_F_QC"] == 0)


def mask_turb_measured(df):
    return (df["H_F_MDS_QC"] == 0) & (df["LE_F_MDS_QC"] == 0)


def mask_core(df, years=C.CORE_YEARS):
    """Core filter: measured H, LE, SW_IN, LW_IN, SW_OUT, LW_OUT, TA present, given years."""
    return (df["year"].isin(years) & mask_turb_measured(df) & mask_rad_in_measured(df)
            & mask_rad_out(df) & df["TA_F"].notna())


def mask_core_dawn_relaxed(df, years=C.CORE_YEARS, swpot_max=C.DAWN_RELAX_SWPOT_MAX):
    """Sensitivity sample: as mask_core but SW_IN_F_QC <= 1 is accepted when SW_IN_POT < swpot_max
    (re-admits the low-sun half-hours whose measured SW_IN ONEFlux replaced by a filled value;
    their J_SWin contribution is ~0.004 W m-2 K-1). Not the core: SW_IN_F is not measured there."""
    sw_ok = (df["SW_IN_F_QC"] == 0) | ((df["SW_IN_F_QC"] <= 1) & (df["SW_IN_POT"] < swpot_max))
    return (df["year"].isin(years) & mask_turb_measured(df) & sw_ok & (df["LW_IN_F_QC"] == 0)
            & mask_rad_out(df) & df["TA_F"].notna())


def mask_radiation_only(df, years=C.CORE_YEARS):
    """Radiation-only sample: 4 measured components, regardless of H/LE flags (J_rad terms only)."""
    return df["year"].isin(years) & mask_rad_in_measured(df) & mask_rad_out(df) & df["TA_F"].notna()


def mask_gapfilled(df, years=C.CORE_YEARS):
    """Gap-filled sample: SW_OUT & LW_OUT present, _F fluxes regardless of QC. Not measurements."""
    return (df["year"].isin(years) & mask_rad_out(df) & df["TA_F"].notna()
            & df["H_F_MDS"].notna() & df["LE_F_MDS"].notna())


def filter_log(df, years=C.CORE_YEARS):
    """Count of records surviving each successive core condition (for the run log)."""
    steps = []
    m = df["year"].isin(years)
    steps.append((f"year in {years}", int(m.sum())))
    m &= mask_turb_measured(df); steps.append(("H_F_MDS_QC==0 and LE_F_MDS_QC==0", int(m.sum())))
    m &= mask_rad_in_measured(df); steps.append(("SW_IN_F_QC==0 and LW_IN_F_QC==0", int(m.sum())))
    m &= mask_rad_out(df); steps.append(("SW_OUT and LW_OUT present", int(m.sum())))
    m &= df["TA_F"].notna(); steps.append(("TA_F present", int(m.sum())))
    return steps


def data_flags(df, mask):
    """Data-quality flags that shaped the rules (section 2.2). Returns dict of counts."""
    d = df[mask]
    return {
        "n": int(len(d)),
        "SW_OUT_negative": int((d["SW_OUT"] < 0).sum()),
        "SW_OUT_negative_day": int(((d["SW_OUT"] < 0) & d["is_day"]).sum()),
        "SW_OUT_gt_SW_IN_when_SW_IN_gt_50": int(((d["SW_OUT"] > d["SW_IN_F"]) & (d["SW_IN_F"] > C.SWOUT_GT_SWIN_THRESHOLD)).sum()),
        "SW_OUT_gt_SW_IN_any": int((d["SW_OUT"] > d["SW_IN_F"]).sum()),
        "SW_OUT_gt_SW_IN_night": int(((d["SW_OUT"] > d["SW_IN_F"]) & ~d["is_day"]).sum()),
        "NIGHT_vs_elevation_disagreements": int((d["is_day"].astype(int) != d["day_elev"]).sum()),
        "NIGHT_vs_SW_IN_POT_disagreements": int(((d["NIGHT"] == 0) != (d["SW_IN_POT"] > 0)).sum()),
        "H_RANDUNC_missing": int(d["H_RANDUNC"].isna().sum()),
        "LE_RANDUNC_missing": int(d["LE_RANDUNC"].isna().sum()),
        "G_F_MDS_present": int(d["G_F_MDS"].notna().sum()),
        "TS_1_present": int(d["TS_F_MDS_1"].notna().sum()),
        "Rn4_minus_NETRAD_mean": float((d["Rn"] - d["NETRAD"]).mean()),
        "Rn4_minus_NETRAD_sd": float((d["Rn"] - d["NETRAD"]).std()),
        "Ts_minus_Ta_mean": float((d["Ts"] - d["Ta"]).mean()),
        "Ts_minus_Ta_day": float((d["Ts"] - d["Ta"])[d["is_day"]].mean()),
        "Ts_minus_Ta_night": float((d["Ts"] - d["Ta"])[~d["is_day"]].mean()),
        "Tsky_minus_Ta_mean": float((d["Tsky"] - d["Ta"]).mean()),
    }


# ----------------------------------------------------------------------------
# Selection diagnostics (T0b: fraction selected by 30-min slot; T0c: core vs excluded)
# ----------------------------------------------------------------------------
def selection_by_slot(df, core, years=C.CORE_YEARS) -> pd.DataFrame:
    """Fraction of the half-hours of `years` that pass each filter, by 30-min slot
    (exposes the ONEFlux dawn hole at 05:00-06:30 LST and the evening turbulent-QC dip)."""
    y = df["year"].isin(years)
    t = pd.DataFrame({
        "slot": df.loc[y, "slot"].to_numpy(),
        "core": core[y].to_numpy().astype(int),
        "rad_out_present": mask_rad_out(df)[y].to_numpy().astype(int),
        "rad_in_qc0": mask_rad_in_measured(df)[y].to_numpy().astype(int),
        "turb_qc0": mask_turb_measured(df)[y].to_numpy().astype(int),
        "sw_in_qc0": (df.loc[y, "SW_IN_F_QC"] == 0).to_numpy().astype(int),
    })
    g = t.groupby("slot")
    out = g.mean().rename(columns=lambda c: f"frac_{c}")
    out.insert(0, "hour", out.index / 2.0)
    out["n_total"] = g.size()
    out["n_core"] = g["core"].sum()
    return out.reset_index()


def selection_bias_table(df, J_sigma, years=C.CORE_YEARS) -> pd.DataFrame:
    """Rain, radiation and clearness in the core sample vs the half-hours excluded although the
    four radiation components are present (quantifies the selection bias, spec 4.2)."""
    y = df["year"].isin(years)
    core = mask_core(df, years)
    excl = y & mask_rad_out(df) & ~core
    rows = []
    for lab, m in ((f"all {years[0]}-{years[-1]}", y), ("core", core), ("excluded, radiation present", excl)):
        d = df[m]
        day = d["is_day"]
        rows.append({
            "sample": lab, "n": int(len(d)),
            "frac_P_F_gt0": float((d["P_F"] > 0).mean()),
            "P_F_mm_per_hh": float(d["P_F"].mean()),
            "SW_IN_day_Wm2": float(d.loc[day, "SW_IN_F"].mean()),
            "clearness_day": float(d.loc[day, "SW_IN_F"].sum() / d.loc[day, "SW_IN_POT"].sum()),
            "LW_IN_Wm2": float(d["LW_IN_F"].mean()), "TA_C": float(d["TA_F"].mean()),
            "VPD_hPa": float(d["VPD_F"].mean()), "WS_ms": float(d["WS_F"].mean()),
            "frac_day": float(day.mean()),
            "sigma_with_F_fluxes": float(J_sigma[m].mean()) if J_sigma is not None else np.nan,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Coverage table T0
# ----------------------------------------------------------------------------
def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    core_all_years = (mask_turb_measured(df) & mask_rad_in_measured(df) & mask_rad_out(df) & df["TA_F"].notna())
    tmp = df.assign(
        has_rad_out=mask_rad_out(df).astype(int),
        h_qc0=(df["H_F_MDS_QC"] == 0).astype(int),
        le_qc0=(df["LE_F_MDS_QC"] == 0).astype(int),
        core=core_all_years.astype(int),
        rad_in_qc0=mask_rad_in_measured(df).astype(int),
        ts1=df["TS_F_MDS_1"].notna().astype(int),
    )
    g = tmp.groupby(["year", "month"])
    t0 = g.agg(
        n_total=("time", "size"),
        n_SW_OUT_LW_OUT=("has_rad_out", "sum"),
        n_SW_IN_LW_IN_QC0=("rad_in_qc0", "sum"),
        n_H_QC0=("h_qc0", "sum"),
        n_LE_QC0=("le_qc0", "sum"),
        n_core=("core", "sum"),
        n_TS1=("ts1", "sum"),
        P_F_sum_mm=("P_F", "sum"),
        TA_mean_C=("TA_F", "mean"),
    ).reset_index()
    # complete days in the core sample
    cd = tmp[tmp["core"] == 1].groupby(["year", "month", "date"]).size()
    cd = (cd >= C.MIN_HH_PER_DAY).groupby(level=[0, 1]).sum().rename("n_complete_days").reset_index()
    t0 = t0.merge(cd, on=["year", "month"], how="left")
    t0["n_complete_days"] = t0["n_complete_days"].fillna(0).astype(int)
    t0["core_fraction"] = t0["n_core"] / t0["n_total"]
    return t0
