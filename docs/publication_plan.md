# Publication plan: a two-figure short report

Working title: **Neighbourhood continuity as a reference-free quality indicator for
3D cell tracking**

Format: short report / brief communication / application note (bioRxiv first;
candidates: *Bioinformatics* Application Note, *Nature Methods* Brief
Communication if the zebrafish result is strong, *PLOS Comput Biol* Software
paper, *J Open Source Software* for the code alone). Two main figures, one or two
supplementary figures, methods, code release.

## The claim, in one sentence

Local neighbourhood continuity between consecutive frames, computed from tracks
alone, responds monotonically to tracking error rate, distinguishes error types
by its decomposition, and reproduces that behaviour on real ground truth, so it
can be used to assess and tune tracking where no ground truth exists.

Everything in the paper serves that sentence. Anything that does not is
supplementary or cut.

## What is new

1. A reference-free metric family (WIO, positional WIO, rank correlation, the
   vanished / displaced / newcomer decomposition) for dense 3D tracking.
2. A simulation framework with calibrated tissue and controlled injection of
   the five standard error types, used to obtain response curves.
3. Validation that the response transfers to real ground truth with divisions.
4. A demonstration of reference-free parameter selection on real light-sheet
   data.

## Figure 1: concept and calibration (simulation)

**Message:** the metrics respond predictably and differently to each error type.

| panel | content | data | status |
|---|---|---|---|
| 1a | Schematic: cell i, its k neighbours at t and t+1, what retained / vanished / displaced / newcomer mean, WIO weights by 1/distance. One cartoon, two frames. | drawn | to do |
| 1b | Response curves: mean WIO (and PWIO) vs injected error rate, one line per error type (miss, FP, switch, merge, split), simulated tissue, 2 frames, N ≈ 1000, 20 seeds, shaded 95% band. x-axis in events per cell per frame so types are comparable. | `simulate_sequence` + `apply_tracking_errors` (cross-frame switch rule) | pipeline exists in original repo (`run_error_sweep.py`); re-run with 0.3.0 sampler |
| 1c | Error fingerprint: stacked bar or small-multiple of the decomposition (retained, vanished, displaced, newcomer, WIO − PWIO gap) at a matched error rate for each type. Shows misses load on "vanished", switches on "displaced" with PWIO ≈ WIO baseline, FPs/splits on "newcomer". | same runs as 1b, `extra_metrics=True` | to do |
| 1d (optional) | Robustness: 1b repeated at two densities or with lognormal speeds, collapsed to one line per type to show the curves are stable. Could go to Supplement. | same | to do |

Methods points for Fig 1: tissue calibration source (spacing, speed, density from
real data), sampler with exact floor, Delaunay vs kNN choice (pick one for main
text, other in Supplement), border handling.

## Figure 2: validation on real ground truth and application

**Message:** the behaviour holds on real cells with divisions, and it lets you
choose tracker parameters without ground truth.

| panel | content | data | status |
|---|---|---|---|
| 2a | GT baseline on Fluo-N3DH-CE: per-frame-pair mean WIO over the movie under the "ignore" and "collapse" division policies, with division count per pair as a bar underneath. Shows the division artefact and that collapse removes it. | CE 01_GT/TRA via `read_ctc_tracks` | needs division policy in metrics |
| 2b | Transfer of response: inject the same error types into CE GT tracks (real geometry, real motion, real divisions) and overlay on the simulated curves from 1b. Expect same ordering and similar slopes. | CE GT + variable-N `apply_tracking_errors` | needs variable-N errors |
| 2c | Real tracker errors: run a simple linker (greedy nearest-neighbour or LAP, several max-link distances) on CE GT detections with 5% detection dropout; label every link with `traccuracy` (correct / FP / FN / identity switch); scatter per-frame-pair WIO drop against true error count per pair, one point per pair, colour by link distance. Report Spearman ρ. | CE GT + linker + traccuracy | linker exists in original repo (`node1_tracks_nn.csv` pipeline) or write a 40-line one; install traccuracy |
| 2d | Application without GT: your own light-sheet data, WIO and decomposition across the existing tracker parameter sweep (θ / confidence / max-distance variants in `real_data/`). Mark the parameter set the metric selects. If you have any manual spot-checks, mark the one you would have chosen by eye. | `real_data/*.csv` | data exists; run metrics |

Stretch (only if time): replace or complement 2c with linajea zebrafish, where
TGMM and linajea predicted tracks ship with sparse GT. This adds a second organism
and a flow-dominated motion regime, and is BSD-licensed. It needs the
evaluated-cell mask (metrics over GT cells, neighbourhoods from all predicted
cells). If it works it is the strongest panel; if not, nothing else depends on it.

## Supplementary

- S1: kNN vs Delaunay versions of Fig 1b and 2a.
- S2: division window sensitivity (0, 1, 2, 3 frames) on CE rank correlation.
- S3: switch rule comparison (same-frame vs cross-frame) and effective rates.
- S4: sampler nearest-neighbour distribution vs real data (hard floor vs soft).
- Table S1: datasets, voxel sizes, frame counts, division counts, licence.

## Code work, in order

Each item unblocks the next; estimates are working days for one person.

1. **Division policy in `metrics.py`** (collapse + `is_birth` mask + lineage
   survival + optional window). Tests with a synthetic division. Unblocks 2a. ~1 day.
2. **Variable cell count through `apply_tracking_errors`**, carrying
   `parent_ids`. Port the variable-N miss function; make merges/splits/FPs/
   switches operate on list-of-dict frames. Unblocks 2b. ~1 day.
3. **Simulated divisions** by making splits persistent with a parent link.
   Gives a controllable division rate to test item 1 before CE. Optional but
   cheap. ~0.5 day.
4. **Sweep driver** in this repo (port `run_error_sweep.py` essentials): error
   type × rate × seed → long-format CSV of per-cell metrics. Used for 1b, 1c, 2b. ~1 day.
5. **Linker + traccuracy wrapper** for 2c: greedy NN / LAP on detections with
   dropout, export CTC-format result, run traccuracy, join error labels to
   frame pairs. ~1–2 days.
6. **Figure scripts**: one script per figure, reading the CSVs, deterministic. ~2 days.
7. (Stretch) **Evaluated-cell mask** and linajea reader (tab-separated
   `t z y x cell_id parent_id track_id`). ~1–2 days.

Total for the two-figure version without the stretch: roughly two weeks of
focused work plus writing.

## Data and licence actions (start now, they take calendar time)

- Email the CTC organisers for permission to use Fluo-N3DH-CE training data in a
  non-challenge publication. Their terms require it. Mention no redistribution.
- Download CE 01 and 02 (training sets with GT).
- Decide whether your own light-sheet data in 2d can be shown; if not, use a
  crop and state it.
- If going for the stretch: download linajea zebrafish GT, predictions and
  TGMM baseline from Janelia figshare (BSD-3); image data not needed.

## Statistics

- Response curves: mean over cells per run, then mean and 95% CI over seeds.
  Never pool cells across seeds as if independent.
- 2c: Spearman ρ between per-pair WIO drop and per-pair error count, with a
  permutation p-value; also report the slope from a linear fit for the
  "one switch costs this much WIO" statement.
- State the baseline WIO on perfect GT and define the quantity of interest as
  the drop from it, not the absolute value.

## Methods must state

- Positions in microns; CE anisotropy 0.09 × 0.09 × 1.0 µm.
- Neighbourhood definition (k or Delaunay, hull exclusion), border margin.
- Division policy, exactly as in `docs/division_policy.md`, including the
  one-pair delay for false divisions.
- Switch rule used (cross-frame, with cap) and the effective per-jump rate.
- Simulator calibration parameters and their source.
- Sampler floor (exact) and version 0.3.0 of the code.

## Risks and the pre-committed response

| risk | response |
|---|---|
| CE early frames have too few cells for neighbourhoods | evaluate from the frame where N ≥ 50; say so |
| 2b curves differ from 1b in slope | that is a result, not a failure: report the ratio and attribute to real motion / anisotropy |
| 2c correlation weak because errors are rare at good link distances | increase dropout or include a bad link distance so the error range is wide |
| collapse policy hides false divisions for one pair | state it; it does not affect 2a–2d |
| reviewers ask for a second organism | point to S-figure if zebrafish worked; otherwise state as future work |
| CTC permission delayed | start with simulation + own data; CE panels slot in later |

## Writing outline (≈ 1500–2000 words main text)

1. Problem: no ground truth in most 3D tracking experiments; existing metrics
   need it. (1 paragraph)
2. Idea: neighbourhoods change smoothly in tissue, abruptly under tracking
   error. Define WIO, PWIO, decomposition. (2 paragraphs, Fig 1a)
3. Calibration: response curves and fingerprints. (2 paragraphs, Fig 1b–c)
4. Validation: CE baseline, transfer, real tracker errors. (3 paragraphs, Fig 2a–c)
5. Application: parameter selection without GT. (1 paragraph, Fig 2d)
6. Limitations: sister cells after division, coherent flow, segmentation vs
   linking. (1 paragraph)
7. Availability: code, version, how to run on a CTC folder or a CSV.
