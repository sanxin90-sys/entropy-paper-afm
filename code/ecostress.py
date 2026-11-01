"""Read the ECOSTRESS spectral library (v1.0, ASCII) straight from the zip and build the
surface-class spectra used in Article 2.

Classes (functional types, not Pantanal species; see docs/METHODS_A2.md):
  green : vegetation (tree, shrub, grass), leaf samples
  npv   : non-photosynthetic vegetation (bark, branches, dry leaves, needles, flowers, lichen)
  soil  : terrestrial soils (lunar soil excluded)
  water : liquid water (tap, distilled, sea); snow, ice and frost excluded

A spectrum is usable for the shortwave (SW) analysis if it covers 0.40-2.45 um, and for the
longwave (LW) analysis if it covers 3-13.5 um (emissivity eps = 1 - hemispherical reflectance,
Kirchhoff). VSWIR (ASD) and TIR (Nicolet) files of the same sample are joined at 2.5 um.
"""
from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP = Path(os.environ.get("ECOSTRESS_ZIP", ROOT / "data" / "ecospeclib.zip"))

CLASS_OF_TYPE = {"vegetation": "green", "nonphotosyntheticvegetation": "npv", "soil": "soil", "water": "water"}
WATER_KEEP = ("tapwater", "distilledwater", "seawater")


def _parse(text: str):
    head, data = {}, []
    lines = text.splitlines()
    i = 0
    for i, ln in enumerate(lines):
        if not ln.strip():
            break
        if ":" in ln:
            k, v = ln.split(":", 1)
            head[k.strip()] = v.strip()
    for ln in lines[i + 1:]:
        p = ln.split()
        if len(p) >= 2:
            try:
                data.append((float(p[0]), float(p[1])))
            except ValueError:
                continue
    a = np.array(data, dtype=float)
    a = a[np.argsort(a[:, 0])]
    # drop duplicated wavelengths
    _, idx = np.unique(a[:, 0], return_index=True)
    return head, a[idx]


def read_library(zip_path=ZIP) -> list[dict]:
    """All spectra of the four classes, one dict per file (header fields + wl_um + refl)."""
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            base = name.split("/")[-1]
            if "__MACOSX" in name or not base.endswith(".spectrum.txt"):
                continue
            typ = base.split(".")[0]
            if typ not in CLASS_OF_TYPE:
                continue
            if typ == "soil" and base.startswith("soil.lunar"):
                continue
            if typ == "water" and base.split(".")[1] not in WATER_KEEP:
                continue
            head, a = _parse(z.read(name).decode("latin-1"))
            if len(a) < 10:
                continue
            refl = np.clip(a[:, 1] / 100.0, 0.0, 1.0)
            out.append({
                "file": base, "type": typ, "cls": CLASS_OF_TYPE[typ],
                "subclass": base.split(".")[1], "genus": head.get("Genus", ""), "species": head.get("Species", ""),
                "sample": head.get("Sample No.", "").lower(), "owner": head.get("Owner", ""),
                "range": head.get("Wavelength Range", ""), "measurement": head.get("Measurement", ""),
                "wl_um": a[:, 0], "refl": refl,
            })
    return out


def join_samples(specs: list[dict]) -> list[dict]:
    """Join VSWIR and TIR files of the same sample (same class, genus, species, sample number,
    owner) into one spectrum: VSWIR below 2.5 um, TIR above. Unpaired files are kept as they are."""
    key = lambda s: (s["cls"], s["subclass"], s["genus"], s["species"], s["sample"], s["owner"])
    groups: dict = {}
    for s in specs:
        groups.setdefault(key(s), []).append(s)
    out = []
    for k, g in groups.items():
        vs = [s for s in g if s["wl_um"].min() < 1.0]
        tr = [s for s in g if s["wl_um"].min() >= 1.0]
        if vs and tr and len(vs) == 1 and len(tr) == 1 and vs[0]["wl_um"].max() < 3.0:
            v, t = vs[0], tr[0]
            mv = v["wl_um"] <= 2.5
            mt = t["wl_um"] > 2.5
            s = dict(v)
            s["wl_um"] = np.concatenate([v["wl_um"][mv], t["wl_um"][mt]])
            s["refl"] = np.concatenate([v["refl"][mv], t["refl"][mt]])
            s["file"] = v["file"] + " + " + t["file"]
            s["joined"] = True
            out.append(s)
        else:
            for s in g:
                s = dict(s); s["joined"] = False
                out.append(s)
    return out


def sw_usable(s):
    return s["wl_um"].min() <= 0.40 + 1e-6 and s["wl_um"].max() >= 2.45


def lw_usable(s):
    return s["wl_um"].min() <= 3.0 and s["wl_um"].max() >= 13.5


def resample_sw(s, lam_um):
    """Reflectance on the SW grid (ASTM G173 wavelengths). Constant extension below the first
    and above the last measured wavelength (energy outside 0.35/0.40-2.5 um without TIR data:
    ~3 % of the global spectrum)."""
    return np.interp(lam_um, s["wl_um"], s["refl"], left=s["refl"][0], right=s["refl"][-1])


def resample_eps(s, lam_um, tail="last_um"):
    """Emissivity eps = 1 - R on the LW grid. Beyond the last measured wavelength (14 or
    15.4 um; ~40 % of the 300 K blackbody energy lies beyond 15 um) eps is held at the mean
    of the last measured micrometre ('last_um') or set to 0.98 ('0.98')."""
    w, r = s["wl_um"], s["refl"]
    eps = 1.0 - np.interp(lam_um, w, r, left=r[0], right=r[-1])
    beyond = lam_um > w.max()
    if tail.startswith("last_um"):
        # optional offset, e.g. 'last_um+0.02' or 'last_um-0.02' (tail sensitivity)
        off = float(tail[len("last_um"):] or 0.0)
        val = 1.0 - r[w >= w.max() - 1.0].mean() + off
    elif tail == "0.98":
        val = 0.98
    else:
        raise ValueError(tail)
    eps[beyond] = val
    return np.clip(eps, 0.0, 1.0)


def band_mean(lam_um, y, lo, hi, w=None):
    m = (lam_um >= lo) & (lam_um <= hi)
    if w is None:
        return float(np.trapezoid(y[m], lam_um[m]) / (lam_um[m][-1] - lam_um[m][0]))
    return float(np.trapezoid(y[m] * w[m], lam_um[m]) / np.trapezoid(w[m], lam_um[m]))
