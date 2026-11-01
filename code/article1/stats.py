"""Aggregation and sampling uncertainty (METHODS_SPEC.md, sections 4.5 and 5.4).

Period statistics are time means of half-hourly values; SR is a ratio of means.
Two estimators of a period mean are reported:
  * raw      = mean of the selected half-hours (conditional on the selection pattern);
  * stratified = climatological month x 30-min-slot mean: cell means (cells with
    n >= MIN_HH_PER_CELL), equal slot weight within a month, months weighted by their
    calendar length. It removes the uneven selection across hours (dawn hole) and months.
Bootstrap unit = calendar day, resampled with replacement within the stratum, as moving
blocks of `block_days` consecutive available days (7 days = headline, 1 day = optimistic).
"""
from __future__ import annotations

import calendar
import warnings

import numpy as np
import pandas as pd

import config as C

SUM_COLS = ["J_SWin", "J_SWout", "J_LWin", "J_LWout", "J_H", "J_LE", "J_in", "J_out",
            "sigma", "sigma_rad", "sigma_turb", "Rn", "Rn_abs", "absSW", "Rres", "H_s", "LE_s"]


def period_stats(J: pd.DataFrame, mask=None) -> dict:
    """Means of every term over the rows selected by mask, SR = <J_out>/<J_in>,
    sigma per absorbed shortwave (K-1, the physically meaningful normalisation, spec 2.7),
    its analytic reference 1/<T_s> - 4/(3 T_sun), eta_sigma = <sigma>/<Rn_abs> (own
    normalisation, dominated by LW_IN), random-uncertainty of the mean sigma."""
    d = J if mask is None else J[mask]
    out = {"n": int(len(d))}
    if len(d) == 0:
        return out
    for c in SUM_COLS:
        out[c] = float(d[c].mean())
    out["Ts_K"] = float(d["Ts"].mean())
    out["SR"] = out["J_out"] / out["J_in"]
    out["sigma_per_absSW_Kinv"] = out["sigma"] / out["absSW"] if out["absSW"] else np.nan
    out["sigma_S1_per_absSW_Kinv"] = (out["sigma"] + float((d["Rres"] / d["Ts"]).mean())) / out["absSW"] if out["absSW"] else np.nan
    out["ref_1_over_Ts_minus_4_3Tsun_Kinv"] = 1.0 / out["Ts_K"] - 4.0 / (3.0 * C.T_SUN)
    out["eta_sigma_Rnabs_Kinv"] = out["sigma"] / out["Rn_abs"]
    u = d["u_sigma"].to_numpy(dtype=float)
    u = u[np.isfinite(u)]
    out["u_sigma_rand_mean"] = float(np.sqrt((u ** 2).sum()) / len(d)) if len(u) else np.nan
    out["frac_randunc_present"] = float(len(u) / len(d))
    return out


# ----------------------------------------------------------------------------
# Stratified (month x slot) estimator
# ----------------------------------------------------------------------------
def _month_weights(months_present, years):
    """Calendar-day weight of each month (mean length over `years`), normalised over the months present."""
    w = {m: np.mean([calendar.monthrange(y, m)[1] for y in years]) for m in months_present}
    tot = sum(w.values())
    return {m: v / tot for m, v in w.items()}


def stratified_mean(J: pd.DataFrame, months, slots, mask, cols=("sigma", "J_out", "J_in"),
                    min_n=C.MIN_HH_PER_CELL, years=C.CORE_YEARS) -> dict:
    """Climatological month x slot mean of each col over the rows in mask.
    Returns the means, SR (ratio of the stratified means), number of populated cells and
    the number of cells dropped for n < min_n."""
    cols = list(cols)
    d = J.loc[mask, cols]
    key = pd.DataFrame({"month": np.asarray(months)[np.asarray(mask)], "slot": np.asarray(slots)[np.asarray(mask)]}, index=d.index)
    g = d.groupby([key["month"], key["slot"]])
    n = g.size()
    cell = g.mean()
    keep = n >= min_n
    cell = cell[keep]
    mon = cell.groupby(level=0).mean()                 # equal slot weight within month
    w = _month_weights(mon.index.tolist(), years)
    wv = np.array([w[m] for m in mon.index])
    out = {f"{c}_strat": float((mon[c].to_numpy() * wv).sum()) for c in cols}
    if "J_out" in cols and "J_in" in cols:
        out["SR_strat"] = out["J_out_strat"] / out["J_in_strat"]
    out["n_cells_strat"] = int(keep.sum())
    out["n_cells_dropped_strat"] = int((~keep).sum())
    out["n_months_strat"] = int(len(mon))
    return out


# ----------------------------------------------------------------------------
# Daily block bootstrap
# ----------------------------------------------------------------------------
def daily_means(J: pd.DataFrame, dates: pd.Series, mask=None, min_hh=C.MIN_HH_PER_DAY) -> pd.DataFrame:
    """Daily means of the J terms over complete days (>= min_hh selected half-hours)."""
    d = J if mask is None else J[mask]
    dt = dates if mask is None else dates[mask]
    g = d.groupby(dt.to_numpy())
    n = g.size()
    daily = g[SUM_COLS].mean()
    daily["n_hh"] = n
    return daily[n >= min_hh]


def sr_daily(daily: pd.DataFrame) -> float:
    """Lin (2015) style: SR from daily means, ratio of means over complete days."""
    if len(daily) == 0:
        return np.nan
    return float(daily["J_out"].mean() / daily["J_in"].mean())


def sr_window(J: pd.DataFrame, hours: pd.Series, mask, window) -> tuple[float, int]:
    """SR (ratio of means) in a clock window [h0, h1) LST; wraps midnight if h1 <= h0."""
    h0, h1 = window
    hr = hours
    if h1 > h0:
        w = (hr >= h0) & (hr < h1)
    else:
        w = (hr >= h0) | (hr < h1)
    d = J[mask & w]
    if len(d) == 0:
        return np.nan, 0
    return float(d["J_out"].mean() / d["J_in"].mean()), int(len(d))


def _daily_sums(J: pd.DataFrame, dates, mask, cols):
    d = J.loc[mask, cols]
    dt = pd.Series(np.asarray(dates)[np.asarray(mask)], index=d.index)
    g = d.groupby(dt.to_numpy())
    sums = g.sum(min_count=1)
    cnt = g.size().to_numpy(dtype=float)
    return sums.to_numpy(dtype=float), cnt, sums.index.to_numpy()


def _block_indices(nd, n_boot, block_days, rng):
    """(n_boot, nd) indices of resampled days: iid days (block_days = 1) or moving blocks of
    consecutive available days (circular), ceil(nd/block_days) blocks per replicate."""
    if block_days > 1:
        nb = int(np.ceil(nd / block_days))
        starts = rng.integers(0, nd, size=(n_boot, nb))
        idx = (starts[:, :, None] + np.arange(block_days)[None, None, :]) % nd
        return idx.reshape(n_boot, -1)[:, :nd]
    return rng.integers(0, nd, size=(n_boot, nd))


def bootstrap_means(J: pd.DataFrame, dates, mask, cols=("J_out", "J_in", "sigma", "sigma_rad", "sigma_turb"),
                    n_boot=C.BOOT_N, seed=C.BOOT_SEED, block_days=C.BOOT_BLOCK_DAYS_HEADLINE):
    """Resample calendar days (or consecutive blocks of `block_days` days) with
    replacement; return array (n_boot, len(cols)+1) of period means with SR last.

    Means are sum of daily sums / sum of daily counts, i.e. identical to the
    half-hourly mean of the resampled days (days with few half-hours get less weight).
    """
    cols = list(cols)
    sums, cnt, days = _daily_sums(J, dates, mask, cols)
    nd = len(days)
    rng = np.random.default_rng(seed)
    if nd == 0:
        return np.full((n_boot, len(cols) + 1), np.nan), cols + ["SR"]
    idx = _block_indices(nd, n_boot, block_days, rng)
    s = sums[idx].sum(axis=1)                   # (n_boot, ncols)
    c = cnt[idx].sum(axis=1)[:, None]
    means = s / c
    sr = means[:, cols.index("J_out")] / means[:, cols.index("J_in")]
    return np.column_stack([means, sr]), cols + ["SR"]


def bootstrap_stratified(J: pd.DataFrame, dates, months, slots, mask, cols=("sigma", "J_out", "J_in"),
                         n_boot=C.BOOT_N, seed=C.BOOT_SEED, block_days=C.BOOT_BLOCK_DAYS_HEADLINE,
                         min_n=C.MIN_HH_PER_CELL, years=C.CORE_YEARS):
    """Block bootstrap (days) of the stratified month x slot estimator: in each replicate the
    cell means are recomputed from the resampled days. Returns (n_boot, len(cols)+1) with SR last."""
    cols = list(cols)
    m = np.asarray(mask)
    d = J.loc[m, cols].to_numpy(dtype=float)
    dt = np.asarray(dates)[m]; mo = np.asarray(months)[m].astype(int); sl = np.asarray(slots)[m].astype(int)
    udays, di = np.unique(dt, return_inverse=True)
    nd = len(udays)
    if nd == 0:
        return np.full((n_boot, len(cols) + 1), np.nan), cols + ["SR"]
    # per-day x (month, slot) sums and counts; a day belongs to one month, so 48 cells per day
    day_month = np.zeros(nd, dtype=int)
    day_month[di] = mo
    sums = np.zeros((nd, 48, len(cols))); cnt = np.zeros((nd, 48))
    np.add.at(sums, (di, sl), d)
    np.add.at(cnt, (di, sl), 1.0)
    months_present = np.unique(day_month)
    w = _month_weights(months_present.tolist(), years)
    rng = np.random.default_rng(seed)
    idx = _block_indices(nd, n_boot, block_days, rng)
    out = np.full((n_boot, len(cols)), np.nan)
    for k in range(n_boot):
        sel = idx[k]
        # aggregate resampled days by month: (12, 48)
        s_m = np.zeros((13, 48, len(cols))); c_m = np.zeros((13, 48))
        np.add.at(s_m, day_month[sel], sums[sel])
        np.add.at(c_m, day_month[sel], cnt[sel])
        acc = np.zeros(len(cols)); wtot = 0.0
        for mth in months_present:
            ok = c_m[mth] >= min_n
            if not ok.any():
                continue
            cell = s_m[mth][ok] / c_m[mth][ok][:, None]
            acc += w[mth] * cell.mean(axis=0); wtot += w[mth]
        if wtot > 0:
            out[k] = acc / wtot
    sr = out[:, cols.index("J_out")] / out[:, cols.index("J_in")]
    return np.column_stack([out, sr]), cols + ["SR"]


def ci(arr, axis=0, q=(2.5, 97.5)):
    return np.nanpercentile(arr, q, axis=axis)


def bootstrap_summary(J, dates, mask, block_days=C.BOOT_BLOCK_DAYS_HEADLINE, suffix="", **kw) -> dict:
    """2.5/97.5 percentiles of the period mean sigma (and sigma_rad, sigma_turb) and of SR."""
    b, names = bootstrap_means(J, dates, mask, block_days=block_days, **kw)
    out = {}
    for name in ("sigma", "SR", "sigma_rad", "sigma_turb"):
        j = names.index(name)
        lo, hi = ci(b[:, j])
        out[f"{name}_boot_lo{suffix}"] = float(lo)
        out[f"{name}_boot_hi{suffix}"] = float(hi)
    out["n_boot_days"] = int(len(np.unique(np.asarray(dates)[np.asarray(mask)]))) if np.asarray(mask).sum() else 0
    return out


def bootstrap_difference(J, dates, mask_a, mask_b, block_days=C.BOOT_BLOCK_DAYS_HEADLINE, **kw) -> dict:
    """Bootstrap of <sigma>_a - <sigma>_b and SR_a - SR_b, strata resampled independently.
    Also the fraction of replicates with difference <= 0 (descriptive one-sided p), reported
    as a bound '< 1/n_boot' when no replicate crosses zero."""
    seed = kw.pop("seed", C.BOOT_SEED)
    ba, names = bootstrap_means(J, dates, mask_a, block_days=block_days, seed=seed, **kw)
    bb, _ = bootstrap_means(J, dates, mask_b, block_days=block_days, seed=seed + 1, **kw)
    n_boot = ba.shape[0]
    out = {}
    for name in ("sigma", "SR"):
        j = names.index(name)
        diff = ba[:, j] - bb[:, j]
        lo, hi = ci(diff)
        out[f"d_{name}"] = float(np.nanmean(ba[:, j]) - np.nanmean(bb[:, j]))
        out[f"d_{name}_lo"] = float(lo)
        out[f"d_{name}_hi"] = float(hi)
        k = int(np.sum(diff <= 0))
        out[f"d_{name}_n_le0"] = k
        out[f"d_{name}_frac_le0"] = f"< {1 / n_boot:g} (0 of {n_boot})" if k == 0 else f"{k / n_boot:.3f} ({k} of {n_boot})"
    return out


# ----------------------------------------------------------------------------
# Composites
# ----------------------------------------------------------------------------
def diurnal_composite(values: pd.Series, slots: pd.Series, dates: pd.Series, mask,
                      min_n=C.MIN_HH_PER_SLOT, n_boot=C.BOOT_N, seed=C.BOOT_SEED) -> pd.DataFrame:
    """Mean by 30-min slot (TIMESTAMP_START labelling: slot 23 = 11:30-12:00) with a
    day-resampling bootstrap band (2.5-97.5 %). lo/hi are NaN when n_boot < 100 (no band);
    the column n_boot records the replicates actually used."""
    v = values[mask].to_numpy(dtype=float)
    s = slots[mask].to_numpy()
    d = np.asarray(dates)[np.asarray(mask)]
    udays, di = np.unique(d, return_inverse=True)
    mat = np.full((len(udays), 48), np.nan)
    mat[di, s] = v
    finite = np.isfinite(mat)
    n = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.nansum(mat, axis=0) / np.maximum(n, 1), np.nan)
    mean = np.where(n >= min_n, mean, np.nan)
    lo = np.full(48, np.nan); hi = np.full(48, np.nan)
    nb_used = 0
    if len(udays) > 1 and n_boot >= 100:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(udays), size=(n_boot, len(udays)))
        vals0 = np.where(finite, mat, 0.0)
        reps = np.empty((n_boot, 48))
        for k in range(n_boot):
            sel = idx[k]
            cnt = finite[sel].sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                reps[k] = vals0[sel].sum(axis=0) / cnt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slots (no data) -> NaN
            lo, hi = np.nanpercentile(reps, [2.5, 97.5], axis=0)
        lo = np.where(n >= min_n, lo, np.nan); hi = np.where(n >= min_n, hi, np.nan)
        nb_used = n_boot
    return pd.DataFrame({"slot": np.arange(48), "hour": np.arange(48) / 2.0, "mean": mean, "lo": lo, "hi": hi, "n": n,
                         "n_boot": nb_used})


def monthly_table(J: pd.DataFrame, df: pd.DataFrame, mask, by_year=False) -> pd.DataFrame:
    """Monthly means of all terms (pooled over years or per year), SR as ratio of means,
    albedo as ratio of sums (daytime), Ts - Ta, sigma per absorbed SW, P_F monthly sum
    (from every record of the month, restricted to the years present in mask)."""
    d = J[mask].copy()
    d["month"] = df.loc[mask, "month"].to_numpy()
    d["year"] = df.loc[mask, "year"].to_numpy()
    d["SW_IN"] = df.loc[mask, "SW_IN_F"].to_numpy()
    d["SW_OUT_day"] = np.where(df.loc[mask, "is_day"], df.loc[mask, "SW_OUT"], np.nan)
    d["SW_IN_day"] = np.where(df.loc[mask, "is_day"], df.loc[mask, "SW_IN_F"], np.nan)
    d["dT"] = d["Ts"] - d["Ta"]
    keys = ["year", "month"] if by_year else ["month"]
    g = d.groupby(keys)
    t = g[SUM_COLS + ["dT"]].mean()
    t["SR"] = t["J_out"] / t["J_in"]
    t["sigma_per_absSW_Kinv"] = t["sigma"] / t["absSW"]
    t["albedo_day"] = g["SW_OUT_day"].sum() / g["SW_IN_day"].sum()
    t["n"] = g.size()
    years = sorted(d["year"].unique())
    p = df[df["year"].isin(years)].groupby(["year", "month"])["P_F"].sum().rename("P_F_mm")
    if by_year:
        t = t.join(p)
    else:
        t = t.join(p.groupby("month").mean().rename(f"P_F_mm_mean_{years[0]}_{years[-1]}"))
    return t.reset_index()
