# Spectral versus broadband radiation entropy at BR-Npw: code and derived tables

This repository contains the code, run logs and derived tables for the article "Diffuse and reflected light control broadband radiation-entropy errors in flux-tower budgets: spectral benchmarks and closed-form corrections from a Pantanal wetland" (submitted to *Agricultural and Forest Meteorology*).

The code computes the Planck spectral entropy of surface radiation streams from the ECOSTRESS spectral library and the ASTM G173-03 solar spectra. It then places these spectral terms in the surface entropy budget of the AmeriFlux site BR-Npw (northern Pantanal, Brazil) for 2015–2016. It also runs the robustness tests, broadband benchmarks, closed-form power laws and transfer grid reported in the article.

The broadband budget comes from a companion study of the same tower. Its code lives at https://github.com/sanxin90-sys/entropy-paper (Zenodo: https://doi.org/10.5281/zenodo.23853238). The four modules this analysis needs are copied unchanged into `code/article1/`.

The input data are not included (see "Input data").

## Contents

```
code/
  spectral.py      Planck spectral entropy, isotropic and beam integrals, lookup tables, k_d models, T_s solution
  ecostress.py     reads the ECOSTRESS library zip and builds the surface-class spectra
  common.py        paths, grids, ASTM G173-03 spectra (via pvlib), horizontal clear-sky diffuse spectrum
  library.py       Part A: spectral library analysis   -> results/tables/a1-a4, data_derived/library_arrays.npz
  tower.py         Part B: tower budget, 37 configurations, bootstrap -> results/tables/b1-b6, data_derived/tower_terms.csv.gz
  robustness.py    Part C: diffuse-fraction models, radiometer bias, benchmarks, power laws, transfer grid -> results/tables/c1-c5
  figures.py       Figs. 1-6 -> results/figures
  tests/           unit tests of the spectral functions (blackbody limits, T_s inversion, sky models)
  article1/        config.py, load.py, entropy.py, stats.py from the companion study (unchanged)
data/
  article1_tables/t8_midday_albedo_by_month.csv   derived table from the companion study, used by tower.py
data_derived/      intermediate arrays written by the pipeline
results/           tables, figures and run logs used for every number in the article
docs/code_review.md  code review made with an AI assistant before the final run (see below)
```

## Software

The results were produced with Python 3.13.17, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, matplotlib 3.11.2 and pvlib 0.16.1. pytest is used only to run the tests. Other recent versions will probably work.

```
pip install -r requirements.txt
```

## Input data

1. **AmeriFlux BR-Npw.** Download the FLUXNET product v1.3_r1 (2013–2017) from https://ameriflux.lbl.gov (DOI https://doi.org/10.17190/AMF/1881563, CC-BY-4.0, free account required). Put `AMF_BR-Npw_FLUXNET_FLUXMET_HH_2013-2017_v1.3_r1.csv` in `data/`, or set the environment variable `NPW_DATA_DIR` to its folder.
2. **ECOSTRESS spectral library v1.0.** Download the full ASCII library from https://speclib.jpl.nasa.gov (Meerdink et al., 2019). Save the zip as `data/ecospeclib.zip`, or set `ECOSTRESS_ZIP` to its path. The code reads the files straight from the zip.
3. **ASTM G173-03.** The reference spectra are taken from pvlib, so no download is needed.

## Running

From `code/`:

```
python3 -m pytest -q tests     # unit tests
python3 library.py             # Part A, a few minutes
python3 tower.py               # Part B, about 10 minutes
python3 robustness.py          # Part C, uses data_derived/tower_terms.csv.gz from Part B
python3 -c "import figures as F; F.main(); F.g6_benchmarks_transfer()"
```

Each part writes a run log in `results/`.

## Code review and use of AI

An AI assistant (Claude, Anthropic) wrote and tested this code and carried out the code review in `docs/code_review.md`, under the authors' supervision. The authors checked the code and the results. In that review, "Article 1" means the companion study and "Article 2" means this article.

## Licence

Code: MIT (see `LICENSE`). Derived tables and figures: CC-BY-4.0. The derived tables include quantities computed from AmeriFlux data (CC-BY-4.0) and the ECOSTRESS library; cite those sources when you reuse them.

## Citation

Please cite the article (Agricultural and Forest Meteorology, submitted) and the Zenodo archive of this repository. Please also cite the data sources:

- Vourlitis, G., Dalmagro, H., Nogueira, J. de S., Johnson, M., Arruda, P., 2026. AmeriFlux FLUXNET-1F BR-Npw Northern Pantanal Wetland, Ver. v1.3_r1 (2013-2017). AmeriFlux AMP. https://doi.org/10.17190/AMF/1881563
- Meerdink, S.K., Hook, S.J., Roberts, D.A., Abbott, E.A., 2019. The ECOSTRESS spectral library version 1.0. Remote Sens. Environ. 230, 111196. https://doi.org/10.1016/j.rse.2019.05.015
