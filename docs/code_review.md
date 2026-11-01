> Note added for the public repository: this review was made with an AI assistant (Claude, Anthropic) before the final run, and the authors checked its findings. "Article 1" is the companion study of the same tower; "Article 2" is the article this repository supports. File paths refer to the authors' working folders.

# Independent review: physics, numerics and code of Article 2 (spectral radiation entropy, BR-Npw)

Reviewer: Claude, adversarial review, 2026-10-30. Scratch scripts: `/tmp/claude-0/review_a2/t1.py` to `t8.py`. No file under `analises/` was modified except this report.

## Summary

The core physics is correct. Eq. (1), the factor 2 for polarisation, the per-metre units, the beam and isotropic geometries, energy conservation of the beam/diffuse split, the Newton Ts solve and the bootstrap all check out. Every headline number reproduces from the tables (claims 1, 3 and 4 hold as stated; claim 2 needs a wording fix).

The weak point is the **incoming SW term**. It carries most of the net change (-0.055 of the -0.022 net) and rests on three choices that each move J_SWin by 0.002 to 0.015 W m-2 K-1, which is of the same order as the net headline change:

* the diffuse spectral shape: it is contaminated by ground-reflected light, and the clear-sky shape is applied to cloudy diffuse as well;
* the circumsolar light, which is forced into the solar disk;
* Erbs kd.

Separately, there is a half-hour offset between measured SW_IN and the solar geometry. It does not move the means much, but it produces the 17 h artefact and the asymmetric diurnal kd.

| # | Finding | Severity | Effect on mean sigma (W m-2 K-1) |
|---|---|---|---|
| 1 | ASTM "diffuse" includes about 15 % ground-reflected light (37 deg tilt) | changes a headline number | +0.002 to +0.005 (J_SWin -0.002 to -0.005) |
| 2 | Circumsolar (2.5 deg) light treated as filling only the solar disk | changes a headline number | -0.0016 (2 %), -0.0035 (5 %), -0.0064 (10 %) |
| 3 | Clear-sky diffuse shape used for all (incl. cloudy) diffuse | changes interpretation (uncertainty) | up to -0.015 (existing "global shape" run) |
| 4 | SW_IN_F is about 30 min out of phase with SW_IN_POT / sun_elev | changes interpretation (diurnal figure g4, kd) | -0.0004 all; dry -0.0008, wet +0.0001 |
| 5 | Isotropic/Lambertian assumption = upper bound; size quantified | minor (confirmed small) | diffuse -0.5 to -1.8 % of J_dif (about +0.0004 to +0.0013); reflected < 0.1 % |
| 6 | Claim 2 wording: diffuse is 3.6x the (4/3)/T_sun reference, 3.3-3.5x the beam | minor (text) | none |
| 7 | `frac_energy_extrapolated` bug (trapezoid bridges the gap) | minor (diagnostic only) | none; Methods "about 3 %" is wrong |
| 8 | Duplicate spectra in library (22 SW, 21 LW) | minor | negligible on medians |
| 9 | `all_beam` limit drops SW at elev <= 0; tables clamp below fmin | minor | < 0.25 % of SW energy in the bound only |
| 10 | Tail-perturbation inconsistency for spectra ending at 14 um | minor | negligible (tail sens. +-0.0001) |

Read together, findings 1 to 5 put the incoming-SW contribution at roughly -0.045 to -0.07, not a point value of -0.055. The net change in mean sigma becomes roughly -0.037 to -0.015, against -0.022 reported. Its sign stays negative in every case except the existing kd-0.1 run, where it is still -0.010. The wet-dry contrast reduction (0.075 to 0.064) does not depend on these choices to first order. Finding 4 adds +0.0008 to dry only, which shrinks the contrast by about 0.001.

---

## Verified items (no action needed)

**V1. Eq. (1), polarisation, units.** S_lambda = (2kc/lambda^4)[(1+n)ln(1+n) - n ln n] with n = lambda^5 L/(2hc^2). The derivation: 2 polarisations x c/lambda^4 modes per (m2 s sr m), and n is the occupation per mode, so the 2 in front and the 2 inside n are consistent. Units are per metre of wavelength throughout, and `common.astm()` converts per nm to per m (x1e9). Table a3 reproduces sigma T^4 and (4/3) sigma T^3 to 3e-6 on wide grids and to 3e-5 on the 2.5-1000 um LW grid. The 7 unit tests pass.

**V2. Beam geometry.** J = Omega cos(theta) int S(DNI s/Omega) dlambda is the small-cone limit of the projected solid angle. It is correct for Omega_sun = 6.80e-5 sr.

**V3. Energy conservation of the split.** I recomputed the split for all 25,662 half-hours: max |F_dif + cz_eff * DNI - SW_IN| = 1.1e-13 W m-2. The DNI cap (cos z >= 0.087) never binds in the reference run, because sin 5 deg = 0.0872 > 0.087 and every elev < 5 deg half-hour is forced to kd = 1. It binds only in the `all_beam` limit (see finding 9). cz_eff is handled consistently.

**V4. Additivity of beam and diffuse.** The two fields occupy disjoint solid angles up to the overlap of the isotropic diffuse with the disk, which is Omega_sun/pi = 2e-5 of the hemisphere. The resulting error is about 2e-5 x J_dif and negligible. Summing them is legitimate.

**V5. Table interpolation.** In range, IsoTable max relative error is 7e-7 and BeamTable 8e-6. Max DNI in the data is 1654 W m-2; BeamTable's fmax is 5000 and 9 half-hours have DNI > 1500. Values below fmin are clamped, not extrapolated: 24 half-hours have 0 < DNI < 1e-2, and each gets g(1e-2) = 9.4e-6 W m-2 K-1. That is negligible. No SW_OUT value falls in (0, 1e-3), and no diffuse value is below 1e-3 except 0.

**V6. Time convention of SW_IN_POT vs sun_elev.** Both refer to the half-hour midpoint. RMS of SW_IN_POT - 1361 E0 sin(elev_mid) is 10 W m-2, against 57 W m-2 for a +-15 min shift. kd is therefore computed with consistent geometry; the problem in finding 4 is in SW_IN_F itself.

**V7. Ts solve.** Newton converges quadratically. Max |F - LW_OUT| is 2e-2 after 1 iteration, 4e-7 after 2 and 3e-13 after 3, for green, and 2e-11 after 3 for green_min. Eight iterations is ample. Replacing Ts in J_H and J_LE is consistent with Article 1, which uses the same radiometric Ts. The change is small: J_H + J_LE goes down by 0.00025, and by 0.0020 with green_min (Ts +1.12 K).

**V8. Claim 3.** On a 1/10 tower subsample:
* J_LWout(spectral) / [(4/3) LW_OUT / Ts_A1] - 1 has mean +0.042 % and range -0.004 % to +0.137 %; the library value at 303 K is +0.059 %.
* Against the solved spectral Ts, the ratio is +0.09 % (green) and +0.42 % (green_min).

State which Ts the comparison uses. J_LWout is practically independent of eps: all classes give 2.01032. The emissivity only matters through Ts in J_H and J_LE. That is a result worth stating in the paper.

**V9. Bootstrap (`block_boot_mean`).** It reuses Article 1's `_daily_sums` / `_block_indices` and is correct as a ratio-of-sums estimator. The paired A2-A1 difference is correctly bootstrapped on the difference series. Wet and dry are resampled independently with different seeds, which is acceptable because their days are disjoint. One caveat, inherited from Article 1: circular moving blocks run over "available days", so a block can join non-consecutive dates, for example the end of April with November. This has a negligible effect.

**V10. Numbers vs claims.**
* **Claim 1:** green median ratio is 5.09 (F25), 4.77 (F75) and 4.57 (F150), so "4.6-5.1x" holds for green. Ratio to the diluted blackbody: green 1.004, npv 1.047, soil 1.079, water 0.905. "Within 0-8 %" holds for green, npv and soil; water is -9.5 %, so say so.
* **Claim 4:** b1 gives -0.02221; reflected +0.03261, incoming -0.05543, LW +0.00061; wet-dry 0.07485 to 0.06362. Paired 7-day bootstrap CI for A2-A1 is [-0.0242, -0.0201]. All match.

---

## Findings

### 1. The "clear-sky diffuse" shape contains ground-reflected light — changes a headline number

**What.** `common.astm()` computes `global_tilt - direct*cos(11.19 deg)`. ASTM G173 global is on a 37 deg tilted plane that sees the ground with view factor (1 - cos 37)/2 = 0.101, and G173/SMARTS uses a light-soil spectral albedo. The derived "diffuse" is therefore:
* sky diffuse on the tilt, plus
* about 0.1 x rho_soil(lambda) x G_h(lambda), which is red-shifted and raises entropy per energy.

**Evidence** (`t7.py`). I modelled tilted diffuse = 0.899 D_h + 0.101 rho (DNI cos 48.19 + D_h) and solved for D_h(lambda):

| ground rho | ground-reflected share of ASTM "diffuse" | J_dif(corrected)/J_dif(used) at 50-100 W m-2 | tower mean J_SWin |
|---|---|---|---|
| ECOSTRESS soil median | 14.6 % | 0.923 | 0.10261 (-0.0054) |
| grey 0.3 | 18.2 % | 0.955 | 0.10482 (-0.0032) |
| grey 0.2 | 12.3 % | 0.971 | 0.10596 (-0.0021) |

The energy-weighted mean wavelength of the used diffuse shape is 0.666 um. Removing the ground term makes it bluer.

**Effect.** J_SWin falls by 0.002-0.005, so mean sigma rises by 0.002-0.005. The incoming-term change goes from -0.055 to about -0.050, and the net change from -0.022 to about -0.017.

**Fix.** Pick one of these:
* Use a horizontal sky-diffuse spectrum computed directly: SMARTS2 with the G173 atmosphere and tilt 0, or ASTM G173 recomputed with zero albedo.
* At minimum, remove the ground term as above with the G173 light-soil albedo and the isotropic-sky transposition (1 + cos 37)/2.

Report the old shape as a sensitivity. Correct Sec. 2.3, which says "clear-sky diffuse".

### 2. Circumsolar light forced into the solar disk — changes a headline number (direction: underestimates J_SWin)

**What.** ASTM G173 "direct" is direct + circumsolar within a 2.5 deg half-angle (5.8 deg pyrheliometer field). Erbs' "beam" is defined by the same type of instrument. Treating all of it as a disk beam (Omega = 6.8e-5 sr) understates its entropy, because the circumsolar part is spread over about 88 times the solid angle.

**Evidence** (`t6.py`, `t7.py`). Moving a fraction f of the beam energy to the 0.27-2.5 deg annulus (Omega = 5.9e-3 sr) with the same spectrum:

| f | J_beam change at DNI 800 | J_beam change at DNI 300 | tower J_SWin change |
|---|---|---|---|
| 2 % | +4.5 % | +3.7 % | +0.0016 |
| 5 % | +10 % | +8.3 % | +0.0035 |
| 10 % | +19 % | +15 % | +0.0064 |

Mean J_beam on the tower is 0.0360 and mean J_dif is 0.0720.

Circumsolar is 1-3 % of DNI for G173-type clear skies. It can exceed 10 % under the dry-season smoke at the Pantanal. That would also add to the dry-season J_SWin and reduce the wet-dry contrast.

Limb darkening (linear law, u = 0.6) changes J_beam by -0.36 %, which is negligible.

**Fix.** Split the beam into a disk part and a circumsolar annulus with a fraction f, for example from the Buie et al. (2003) circumsolar ratio or from SMARTS for the G173 atmosphere. Report f = 0.02 as the reference and 0.05/0.10 as sensitivities, with a season-dependent f in the dry season. State in Sec. 2.5 that the disk-only beam is a lower bound on beam entropy.

### 3. The clear-sky diffuse shape is applied to all diffuse, including overcast — changes interpretation

**What.** About 40 % of daytime SW_IN is diffuse in Erbs, and much of it is cloud-scattered. Cloud-scattered light is spectrally close to the global spectrum. The existing "diffuse shape = global" run gives J_SWin 0.1230 against 0.1080, which shifts mean sigma by -0.015. That is larger than findings 1 and 2 combined. The reference choice sits at the low-entropy end of the plausible range.

**Fix.** Use a kt-dependent shape blend, for example clear-sky shape for kt > 0.65, global shape for kt < 0.3, and linear in between. Alternatively, give the incoming term as a range [clear-shape, global-shape] in the abstract instead of the point value -0.055.

### 4. SW_IN_F is about one half-hour out of phase with SW_IN_POT and sun_elev — changes interpretation (diurnal kd, figure g4)

**Evidence** (`t2.py`, `t3.py`, `t4.py`).
* At equal solar elevation the clearness index is strongly asymmetric. In the dry season, median kt is 0.27 at 07:00-07:30 (elev 17 deg) but 0.78 at 16:00-16:30 (elev 16.5 deg).
* At 17:00-17:30 in the dry season, median kt is 1.33 and the 95th percentile is 3.1. Measured SW_IN averages 89.6 W m-2 against SW_IN_POT 78 W m-2, which is physically impossible.
* Shifting the geometry used for kt by -30 min:
  * maximises the correlation of SW_IN_F with the TOA curve (r = 0.9035 against 0.8930 at 0);
  * makes morning and afternoon kt symmetric (median kt at 10-20 deg elevation: 0.435 morning, 0.448 afternoon, against 0.17 and 0.76).
* SW_IN_POT itself agrees with the midpoint geometry (V6).

Either the radiometer data are stamped one interval late, or the series is offset in the FLUXNET product.

**Consequences.**
* With the current alignment, Erbs kd is too high in the morning (0.94 against 0.70 at slot 14, dry season).
* It is too low in the afternoon (0.34 against 0.68 at slot 33), because kt > 0.8 forces kd = 0.165.
* The jump to kd = 1 at 17:00 is caused by elev < 5 deg on 73 % of those half-hours. This is the artefact visible in g4.

With the -30 min shift the diurnal kd is smooth and symmetric (dry season, slots 32-35: 0.59, 0.68, 0.80, 0.95). The period means barely move: J_SWin +0.0004 overall, +0.0008 dry, -0.0001 wet, because the morning and afternoon errors cancel.

**Fix.**
* Diagnose the offset against TIMESTAMP_END and the site metadata, and check other BR-Npw years.
* Compute kt and sun_elev with the matched timing, or shift SW_IN_F by one slot for the kd model.
* Rerun g4 and b5.
* Mention in Article 1 (and the qualification text) that day flags and absSW timing inherit the offset, although the Article 1 broadband J terms are unaffected.

### 5. Isotropic diffuse and Lambertian reflection are upper bounds; the bias is small — minor

At fixed spectral irradiance, Jensen's inequality on the concave S(L) with the measure cos(theta) dOmega shows that isotropic radiance maximises J. So both assumptions are upper bounds.

**Size** (`t6.py`), with the angular radiance pattern held independent of wavelength:

| field | change in J |
|---|---|
| CIE clear-sky diffuse, sun zenith 20 deg | -0.53 % |
| CIE clear-sky diffuse, sun zenith 50 deg | -1.34 % |
| CIE clear-sky diffuse, sun zenith 70 deg | -1.83 % |
| overcast (1 + 2 cos)/3 | -0.13 % |
| reflected, Minnaert k = 0.7 | -0.08 % |
| reflected, Minnaert k = 1.3 | -0.05 % |
| reflected, hot spot + limb brightening | -0.03 % |

The effect is small because, for strongly diluted light (n ~ 1e-7 to 1e-8), the entropy per energy is dominated by ln(1/n). Anisotropy only perturbs this by <x ln x>/(1 + ln 1/n), around 1 %.

On the tower, J_dif goes down by about 0.0004-0.0013, so mean sigma goes up by the same amount. The reflected term is not affected at the stated precision.

**Fix.** Add one sentence and these numbers to Sec. 2.5 or the discussion. "Upper bound" is the correct wording for both. Combined with finding 2, the beam/diffuse treatment has biases of opposite sign.

### 6. Claim 2 wording — minor

Table a4 and `t6.py` give:
* clear-sky diffuse (117 W m-2): J/F = 3.57 x (4/3)/T_sun;
* direct beam (900 W m-2): 1.052x;
* ratio of the two: 3.40.

At 80/900 W m-2 the ratio is 3.47, and at 120/800 W m-2 it is 3.30. "3.6x" is the diffuse-to-reference ratio, not the diffuse-to-beam ratio.

**Fix.** Write: "the diffuse carries about 3.6 times the entropy per unit energy of the (4/3)/T_sun reference and 3.3-3.5 times that of the beam". Note that finding 1 lowers 3.6 to about 3.3-3.5.

### 7. `frac_energy_extrapolated` is wrong (library.py) — minor, diagnostic only

The boolean mask selects points below the measured minimum and above the measured maximum, then calls `np.trapezoid` on the concatenated arrays. The trapezoid therefore adds one spurious panel spanning the whole measured range (0.35 to 2.5 um).

**Evidence** (`t5.py`, medians):

| subset | reported | correct |
|---|---|---|
| unjoined green | 13.3 % | 0.59 % (max 21 % against 1.0 %) |
| joined green | ~0.3 % | ~0.3 % |
| npv | ~0.3 % | ~0.3 % |
| soil | ~0.2 % | ~0.2 % |
| water | 4.9 % | 4.9 % |

The Methods statement "about 3 % of the reflected energy" matches neither column. It is about 0.2-1 %, and 5 % for water.

**Fix.** Use `1 - trapz(y[inside], lam[inside]) / trapz(y, lam)`, or integrate below and above separately. Correct the Methods number.

### 8. Duplicate spectra — minor

Of 660 SW-usable spectra, 638 are unique, with 22 duplicate groups. Of 605 LW-usable spectra, 584 are unique. Also, 72 green VSWIR spectra stay unjoined because their group holds more than one VSWIR or TIR file. All soils (0 joined) and water stay unjoined too, so soil SW (n = 41) and LW (n = 52) come from different sample sets.

The effect on class medians and on b4 is negligible, but the counts in Sec. 2.4 include duplicates.

**Fix.** Deduplicate by array hash before taking class medians and counts. State that the soil SW and LW sets are not paired.

### 9. Edge handling in the bounds — minor

* In `kd_mode="all_beam"`, the elev < 5 rule is skipped. Half-hours with SW_IN > 0 and elev <= 0 (n = 1,239, mean 10.7 W m-2, 0.23 % of SW energy) get DNI = 0, so their energy and entropy are dropped. The "lower bound" is therefore slightly low.
* Values below a table's fmin are clamped to J(fmin) (V5).

**Fix.** In all_beam, send elev <= 0 energy to the diffuse field, or floor cos z as in the main run. Optionally return 0 or the exact value below fmin.

### 10. Tail perturbation inconsistency — minor

The tower `green tail +-0.02` runs perturb eps only beyond 15.3 um. For spectra ending at 14.0 um (JHU soils and water), the held tail starts at 14 um. The Methods text says "beyond the last measured wavelength". The result is insensitive anyway (sigma changes by +-0.0001).

**Fix.** Perturb beyond each spectrum's own last wavelength, or state the 15.3 um cutoff.

---

## Other notes (no change requested)

* **Erbs kd.** Using kt = SW_IN_F/SW_IN_POT on half-hours is a standard application, and SW_IN_POT is TOA radiation on the horizontal. kt > 1 (cloud enhancement, and the offset in finding 4) is clipped to 1, giving kd = 0.165. That yields DNI up to 1654 W m-2, which is above the extraterrestrial value. This only touches 9 half-hours above 1500 and has a negligible mean effect.
* **LW sky consistency.** J_LWout uses a grey sky (delta_a B(Ta)) for the reflected part, while J_LWin stays as Article 1's blackbody T_sky. Rows 5 and 6 of b1 bound this inconsistency at +0.0005 / +0.004 in sigma. Keeping it as a sensitivity is fine.
* **Claim 4 robustness.** Article 1's J_SWout/J_SWin validity masks are reproduced: the same day, SW_IN > 0 and SW_OUT > 0 mask. There are 12,219 night half-hours with SW_OUT > 0 (offset, mean 1.7 W m-2), and they are excluded in both A1 and A2.
