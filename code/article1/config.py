"""Single source of truth for paths, constants, selection rules and options.

Surface entropy budget of BR-Npw (Northern Pantanal Wetland), ONEFlux v1.3 r1.
See docs/METHODS_SPEC.md for the derivation of every rule fixed here.
All energy fluxes W m-2; entropy fluxes W m-2 K-1; temperatures K inside formulas.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (portable: everything relative to the project root = parent of code/)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NPW_DATA_DIR", ROOT / "data"))
HH_FILE = DATA_DIR / "AMF_BR-Npw_FLUXNET_FLUXMET_HH_2013-2017_v1.3_r1.csv"
# sha256 of the AmeriFlux FLUXNET (ONEFlux) v1.3 r1 half-hourly file used for every result
HH_FILE_SHA256 = "48fbd354fc9c402e04881eedf018ddda011d9fb4981f4714caacd2e231c896f1"
SCRATCH_DIR = Path(os.environ.get("NPW_SCRATCH_DIR", ROOT / "scratch"))
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# ----------------------------------------------------------------------------
# Site (BIF, Verified)
# ----------------------------------------------------------------------------
SITE_ID = "BR-Npw"
LAT = -16.498          # deg
LON = -56.412          # deg (east positive)
UTC_OFFSET_H = -4.0    # local standard time, no DST
ELEV_M = 120.0

# ----------------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------------
SIGMA_SB = 5.670374419e-8   # W m-2 K-4, CODATA 2018
T_SUN = 5772.0              # K, IAU 2015 nominal solar effective temperature
T_SUN_ALT = 5778.0          # K, sensitivity (value used in the preliminary script)
T0 = 273.15                 # K
RHO_C_WATER = 4.18e6        # J m-3 K-1
C_SOIL_RANGE = (2.5e6, 3.5e6)   # J m-3 K-1, wet soil volumetric heat capacity (scenario)
Z_SOIL_RANGE = (0.05, 0.10)     # m, effective depth represented by TS_1 (unknown; scenario)
H_WATER_RANGE = (0.2, 0.5, 1.0) # m, water column depth scenarios (no water-level data)
DT_HH_S = 1800.0            # s, half-hour

EPS_CORE = 0.98
EPS_SENSITIVITY = (0.95, 0.97, 0.99)
EPS_FIGURE = tuple(sorted(set(EPS_SENSITIVITY) | {EPS_CORE}))   # eps values drawn in F5

# Landsberg & Tonge (1980) approximation of the dilution function X(delta).
# Approximation, diagnostic only: the pipeline uses the exact spectral integral
# (entropy.x_dilution); these coefficients are logged for comparison (1.6 % high at delta = 1).
X_COEF = (0.9652, -0.2777, 0.0511)

# ----------------------------------------------------------------------------
# Data selection
# ----------------------------------------------------------------------------
CORE_YEARS = (2015, 2016)
SENS_YEARS = (2014, 2017)
ALL_YEARS = (2013, 2014, 2015, 2016, 2017)
WET_MONTHS = (11, 12, 1, 2, 3, 4)
DRY_MONTHS = (5, 6, 7, 8, 9, 10)
# optional 3-season split (Preliminary, pending flood information)
SEASON3 = {"dry": (5, 6, 7, 8, 9), "transition": (10, 11, 12), "flooded": (1, 2, 3, 4)}
FLOODED_MONTHS_SCENARIO = (1, 2, 3, 4)   # months to which the water-column storage scenario applies

CORE_N_REGRESSION = 25662  # Verified core sample size 2015-2016 (regression value)
# Dawn hole: ONEFlux replaces measured SW_IN by a filled value (QC = 1) in low-sun half-hours
# (05:00-06:30 LST), which the SW_IN_F_QC == 0 rule removes from the core sample. The relaxed
# sensitivity sample re-admits SW_IN_F_QC <= 1 when SW_IN_POT is below this threshold (W m-2).
DAWN_RELAX_SWPOT_MAX = 50.0
MATCHED_MONTH_MIN_HH = 500   # months with >= this many core half-hours in BOTH years (2016 vs 2015)

MIN_HH_PER_DAY = 40        # complete-day criterion (of 48)
MIN_HH_PER_SLOT = 20       # diurnal composite: minimum half-hours per 30-min slot per season
MIN_HH_PER_CELL = 5        # stratified (month x slot) estimator: minimum half-hours per cell
MIDDAY_HOURS = (11, 13)    # [11:00, 13:00) local standard time (TIMESTAMP_START labelling)
MIDNIGHT_HOURS = (23, 1)   # 23:00-01:00 (wraps midnight)
MIDDAY_ELEV_DEG = 60.0     # alternative midday window
NOON_DIAG_HOURS = (10, 14) # window for the H vs (T_s - T_a) diagnostic (F7)
SWOUT_GT_SWIN_THRESHOLD = 50.0   # W m-2: flag SW_OUT > SW_IN_F when SW_IN_F > this

USECOLS = [
    "TIMESTAMP_START", "TIMESTAMP_END",
    "TA_F", "TA_F_QC", "SW_IN_POT", "SW_IN_F", "SW_IN_F_QC", "LW_IN_F", "LW_IN_F_QC",
    "VPD_F", "PA_F", "P_F", "WS_F", "RH", "NETRAD", "SW_OUT", "LW_OUT",
    "TS_F_MDS_1", "TS_F_MDS_2", "TS_F_MDS_1_QC", "TS_F_MDS_2_QC", "G_F_MDS",
    "LE_F_MDS", "LE_F_MDS_QC", "LE_CORR", "LE_RANDUNC",
    "H_F_MDS", "H_F_MDS_QC", "H_CORR", "H_RANDUNC", "NIGHT",
]
NA_VALUES = [-9999, -9999.0, "-9999"]

# ----------------------------------------------------------------------------
# Uncertainty
# ----------------------------------------------------------------------------
BOOT_N = 1000
BOOT_SEED = 20261030
BOOT_BLOCK_DAYS_HEADLINE = 7   # moving-block length for the headline intervals (anomaly ACF lag-1 0.34)
BOOT_BLOCK_DAYS_OPTIMISTIC = 1 # independent-days bootstrap, reported as the optimistic bound
RAD_BIAS = {                   # assumed instrument bounds (no metadata in BIF)
    "SW_IN_F": ("rel", 0.05),
    "SW_OUT": ("rel", 0.05),
    "LW_IN_F": ("abs", 10.0),
    "LW_OUT": ("abs", 10.0),
}

# ----------------------------------------------------------------------------
# Figure style
# ----------------------------------------------------------------------------
COLORS = {               # Okabe-Ito, colour-blind safe; wet/dry fixed across figures
    "wet": "#0072B2", "dry": "#D55E00", "all": "#000000",
    "sw_in": "#E69F00", "sw_out": "#F0E442", "lw_in": "#56B4E9", "lw_out": "#009E73",
    "h": "#CC79A7", "le": "#56B4E9", "sigma": "#000000", "grey": "#999999",
    "water": "#009E73",
    "2015": "#009E73", "2016": "#CC79A7", "2014": "#999999", "2017": "#999999",
}


# ----------------------------------------------------------------------------
# Methodological options (one-at-a-time deviations from CORE)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Options:
    """Formulation choices for the entropy budget (section 2 and 5.3 of the spec)."""
    eps: float = EPS_CORE
    t_sun: float = T_SUN
    lwin_temp: str = "sky"      # 'sky' (T_sky, core) | 'air' (T_a) | 'diluted' (T_a with X(delta_a))
    turb_temp: str = "ts"       # 'ts' (T_s, core) | 'ta' (T_a)
    swout_variant: str = "a"    # 'a' (T_sun, core) | 'b' (diluted blackbody X(delta_r))
    swin_variant: str = "core"  # 'core' (4/3 SW/T_sun) | 'diluted' (hemispheric upper bound)
    lwout_variant: str = "core" # 'core' (4/3 LW_OUT/T_s) | 'grey' (eq. 11) | 'grey_refl' (eq. 11 + diluted reflection)
    closure: str = "S0"         # 'S0' | 'S1' (all to LE) | 'S2' (all to H) | 'S3' (Bowen preserving, period CR)
    bias: tuple = field(default_factory=tuple)  # ((variable, factor_or_offset, kind), ...)

    def to_dict(self):
        return asdict(self)


CORE = Options()


def config_hash() -> str:
    """Short hash of every constant above (for the run log and INDEX.md)."""
    g = {k: v for k, v in globals().items()
         if k.isupper() and not k.startswith("_") and isinstance(v, (int, float, str, tuple, list, dict))}
    g["CORE"] = CORE.to_dict()
    s = json.dumps(g, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def code_hash() -> str:
    """Short sha256 of the concatenated source of every code/*.py file (code provenance)."""
    h = hashlib.sha256()
    for p in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:12]


def file_sha256(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
