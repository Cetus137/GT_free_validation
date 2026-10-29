# Error unmixing: method and status

Status at 2026-10-29.  Simulation only.

## Goal

Estimate how often each of four tracking errors occurs, from the tracks alone: missed detections,
false positives, track fragmentations and identity switches.  No ground truth and no simulation-based
calibration may be used on the data being assessed.

## The method (production)

Code: `unmix_production` in `src/tracking_validation/unmixing.py`.

**Seven measurements**, pooled over all consecutive frame pairs of the movie:

| # | Measurement | What it measures | Main job |
|---|---|---|---|
| 1 | Vanished (WIO) | Share of a cell's neighbour weight at t whose labels are gone at t+1 | Missed + false detections |
| 2 | Newcomer (WIO) | Share of neighbour weight at t+1 whose labels were not there at t | Missed + false detections |
| 3 | Displaced (WIO) | Share of neighbour weight at t whose labels persist but are no longer neighbours | Switches; separates fragmentations |
| 4 | PWIO | Share of neighbour weight at t whose *positions* are still occupied at t+1 (labels ignored) | Missed vs false detections |
| 5 | Rank correlation | How well the ranking of neighbour distances is kept from t to t+1 | Switches |
| 6 | Relabelled | A label ends and a new label begins at its predicted position | Fragmentations |
| 7 | Two-way flag, occupied spot | The detections nearest a label's forward and backward predictions carry other labels, and both lie within 0.5 spacings of the predictions | Switches |

Predicted position = own position + median step of the Delaunay neighbours.

**Error-free level.**  Measurements 1, 2, 6 and 7 are zero on error-free tracks by construction.  For
3, 4 and 5 the error-free level depends on the motion, so each is taken relative to its mean over the
movie's *clean* cells: no label ended or began in the neighbourhood (vanished = newcomer = 0) and no
flagged label within 1.5 spacings.

**Calibration by standard addition.**  For each error type, add 2% of that type to the same movie
(every frame but the first), recompute the seven measurements, and take the rise per added error as
that type's column of the 7 x 4 response matrix.

**Solve.**  measurements = matrix x rates, rates >= 0 (non-negative least squares), each measurement
scaled by its largest response so that weak but specific ones (PWIO) count.

**Why each piece is needed** (each was found by removing it):
- Without PWIO, missed detections and false positives are confused once errors persist through a
  movie: a missed cell disappears then reappears, a false detection appears then disappears, and the
  label measurements are symmetric in time.
- Without the relabelled count, fragmentations are absorbed into false positives when all types are
  present.
- Without the occupied-spot test, correctly tracked cells that move much faster than their neighbours
  are flagged as switches, and are dropped from the clean cells that set the error-free level.

## Testing ladder

| Step | Test | Status | Result |
|---|---|---|---|
| 1 | Simulated motion (no errors), six motion types, three to five speeds | Done | `src/tracking_validation/dynamics.py` |
| 2 | Errors injected into each frame pair separately | Done | Superseded by 2b (errors never persist) |
| 2b | Errors injected into the whole 100-frame movie, so they persist | Done, final-settings run in progress | See below |
| 3 | A tracker makes the errors (synthetic detections, LAP / motile) | Not started | |

Step 2b results (100-frame movies, 1000 cells):
- One error type at a time, persistent motion at 0.16 spacings per frame: all four types recovered at
  1.00-1.06 x true, at most 0.09% attributed to the wrong type.
- Mixes of all four types: RMSE 0.23-0.28 percentage points for random, persistent and streaming
  motion at 0.05-0.30 spacings per frame.
- Mixed speeds (per-cell speeds lognormal, sigma 0.75) broke at 0.23 spacings per frame and above
  before the occupied-spot test was added.  With the test, on error-free mixed-speed movies at 0.23
  spacings per frame: two-way flags 1.64% -> 0.03%, displaced +0.010 -> +0.0004, rank correlation
  -0.027 -> -0.001.
- In progress: `scripts/production_recovery.py`, recovery curves with the final settings for all six
  motion types at three speeds.

## Ruled out

| Idea | Why |
|---|---|
| Figure 1 unmixing (response matrix and error-free level from a simulated calibration tissue) | The error-free level drifts with motion: 1-17 points of error away from the calibration motion |
| Label counts only (labels ending / beginning, relabelled, two-way flag) + standard addition | Fine in short movies; in long movies false positives are read as missed detections (0.40 x true) |
| WIO components + rank correlation + two-way flag (no PWIO, no relabelled) | Confuses missed detections with false positives, and fragmentations with false positives in mixes |
| False-flag rate from shuffling the movie's own residuals | Overestimates 5-7 x: real motion is not random in direction |
| Error-free level matched to neighbourhood speed | No improvement: a fast cell among slow neighbours looks like a switch, which the matching cannot see |

Parked: calibration by deleting and adding *detections* and re-running the tracker (for step 3,
where the tracker's own kind of errors must be added).

## Known limitations and open questions

- The response matrix is measured with the same injection code that makes the errors in steps 2 and
  2b, so those results partly confirm themselves.  Step 3 is the real test.
- A real tracker's errors cluster where cells move fast; the clean-neighbourhood error-free level may
  then be biased.
- Injected switches are swaps only (two neighbours exchange labels).  Hops, a missed cell's label
  jumping to the nearest other cell, are what a frame-to-frame tracker mostly makes and are not yet
  injected: `apply_single_hops` in `errors.py` works on a frame pair only, and the unmixing has no hop
  column.  The occupied-spot test does not flag a hop (the hopping label's own cell is missing, so its
  spot is empty); rank correlation should still see it.
- Cell divisions are not simulated: a daughter appearing next to its sister resembles a false
  positive.
- False detections are currently placed next to real cells; detections in empty tissue are untested.
- Above ~0.45 spacings per frame of motion relative to neighbours, switches are overestimated.

## Files

Core:
- `src/tracking_validation/dynamics.py`: the simulator (reflective box, soft spacing, persistent motion,
  optional alignment).
- `src/tracking_validation/unmixing.py`: the method (`unmix_production`), plus the counts-only version.

Current scripts:
- `scripts/production_recovery.py`: recovery curves, final settings, all motions.
- `scripts/movie_testbed.py`: step 2b (whole-movie injection; recovery and mix figures per method).
- `scripts/make_fig1_v2.py`: Figure 1 v2, still built on the counts-only version; to be rebuilt.

Superseded or exploratory (kept for the record): `scripts/motion_testbed.py` (step 2; also provides
the movies the other scripts use), `scripts/tracker_testbed.py` (first look at step 3),
`scripts/deletion_calibration.py`, `scripts/nb_shuffle_null.py`, `scripts/plot_switch_channels_real.py`.

Movies: `motion_testbed.movie(motion, speed, seed, frames)`; 1000 cells in a 46 um reflective cube,
spring spacing 4.5 um (median nearest neighbour ~4.2 um), 0.05 time-unit steps.  Motion types:
random (persistence 0.2 frames), persistent (5), straight (100), mixed speeds (persistence 5,
lognormal speeds sigma 0.75), streams (alignment 1), flock (alignment 5).

## Running

    python scripts/production_recovery.py --workers N        # resumes; --stage figures to redraw
    python scripts/movie_testbed.py --stage mixes --frames 100 --workers N
    python -m pytest tests
