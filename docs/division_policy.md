# Division policy for neighbourhood-continuity metrics

Status: **proposal, not implemented.** The reader (`ctc.py`) already emits the
`parent_ids` field the policy needs. Nothing in `metrics.py` consumes it yet.

## The problem

The metrics were designed for a population with no appearance or disappearance.
Real ground truth has divisions. At the frame pair (t, t+1) in which mother M
divides into daughters D1 and D2, the current code sees:

| viewpoint | what the code sees now | what it should see |
|---|---|---|
| M as focal cell | M absent at t+1: NaN, or 0 with `penalise_absence` | lineage continues; not a loss |
| `track_survival` | M counted as lost | M survived (through D1, D2) |
| neighbours of M | M is a *vanished* neighbour (ID gone), weight lost from WIO | neighbour still there, as two cells |
| neighbours of D1, D2 | D1, D2 are *newcomers* (IDs absent at t) | not newcomers |
| D1, D2 as focal cells | no metric at (t, t+1): absent at t | continuity from M's neighbourhood |

So on perfect ground truth every division looks like one missed detection plus
two false positives. In Fluo-N3DH-CE roughly every cell divides within the
sequence, so the baseline WIO on GT would be well below 1 and the "vanished" and
"newcomer" diagnostics would be dominated by biology rather than tracking error.

A second, smaller effect: for a few frames after division the two sisters are
each other's nearest neighbour at an abnormally short distance. This perturbs
rank order locally even when nothing is wrong.

## What a tracker gets wrong at divisions

The policy has to keep these detectable, because they are the errors we want the
metrics to flag:

1. **Missed division link.** Tracker ends M's track and starts D1, D2 as new
   tracks with no parent. In CTC terms: two missing parent edges.
2. **Wrong daughter.** Tracker links M to D1 and to some unrelated cell X, while
   the real D2 starts a parentless track.
3. **False division.** Tracker splits a non-dividing cell (an over-segmentation
   that it then "explains" with a parent link).
4. **Sister swap after division.** D1 and D2 exchange labels a frame or two
   later while still very close. Hard for any metric to see; we should at least
   not make it invisible by design.

## Options considered

### A. Ignore (status quo)

Simple, but confounds biology with error as described above. The baseline on GT
is informative as a "biological floor", but it is not what we want to compare
tracker output against.

### B. Exclude dividing cells for the pair

Remove M from frame t and D1, D2 from frame t+1 before building neighbourhoods,
for that frame pair only. Both frames lose the same hole, so the pair is
self-consistent. Downsides: a hole in the point set pulls farther cells into the
kNN sets of M's neighbours at both times, which is a mild distortion; and it
throws away exactly the frames where tracker errors concentrate (1–3 above all
happen at the division). Errors of type 1 and 2 become invisible.

### C. Lineage collapse (recommended)

Treat a newborn daughter as carrying its mother's identity for the single frame
pair in which it is born. Concretely, for pair (t, t+1):

* **Neighbour identity key** at t+1 is `parent_id` if the detection is in the
  first frame of a track with a parent, else its own label. M's neighbours
  therefore see "M" retained at t+1 (as a set, two detections collapse to one
  key). D1, D2 are not newcomers. For rank correlation, M's rank at t+1 is the
  rank of the nearer daughter.
* **Survival and absence penalty.** A cell present at t whose label is the
  parent of at least one detection born at t+1 is counted as surviving and is
  not penalised.
* **Focal metrics for the daughters at their birth pair.** Compare M's
  neighbourhood at t with each daughter's neighbourhood at t+1, and store the
  result in the daughter's column at that pair (which is otherwise NaN). A
  boolean mask `is_birth` is returned so these entries can be included or
  excluded in analysis.
* **Focal metric for M** at (t, t+1): NaN, as now, but not penalised. M's
  column simply ends; its continuity lives on in the daughters' birth entries.

How the tracker errors show up under C:

| error | effect under C |
|---|---|
| 1. missed link | D1, D2 have no parent in the *result*, so they are newcomers and M vanishes. Penalised exactly as a miss + 2 FPs, which is what it is to a downstream user. |
| 2. wrong daughter | X is collapsed into M, so M's neighbours see M "retained" at X's position: WIO holds but rank correlation and PWIO drop. X's own true neighbours lose X (vanished). D2 is a newcomer. Detected, more weakly than 1. |
| 3. false division | two detections collapse into one key; the neighbours see nothing wrong at (t, t+1). The spurious track then persists with its own label, so from (t+1, t+2) onward it is an extra cell (newcomer effect) and PWIO sees a density anomaly. Detected one pair late. |
| 4. sister swap | identical to an ordinary switch at very short range. Not addressed by the policy; see "post-division window". |

### D. Post-division window (optional, orthogonal)

A parameter `division_window` (frames, default 0). For `w > 0`, daughters are
excluded **as focal cells** for `w` frames after birth, but remain as
neighbours of other cells. This removes the sister-proximity rank artefact
without punching holes in the point set. It should be a sensitivity setting,
not a default, until CE data shows whether the artefact is material.

## Recommendation

Implement C with D as an option. Decision points for you:

1. **Which key for the mother's rank at t+1:** nearer daughter (proposed) vs mean
   of both ranks. Nearer daughter keeps the rank integer-valued and is what a
   human would say "M is still my 3rd neighbour".
2. **Daughter birth entries:** store under the daughter's column (proposed) vs
   under the mother's. Daughter's column keeps every track's row contiguous in
   time; mother's would give one cell two values at one pair.
3. **Whether `track_survival` should also report a separate `lineage_survival`**
   so the raw "label present at t+1" fraction is still available. I would keep
   both: `track_survival` becomes lineage-aware under the policy, and a
   `raw_track_survival` is returned alongside.

## Implementation sketch

In `compute_neighbour_continuity_observed`:

* New argument `division_policy: Literal["ignore", "collapse"] = "ignore"` and
  `division_window: int = 0`. `"collapse"` requires `parent_ids` in every frame
  dict; raise if absent.
* Build `key1 = where(parent_ids1 >= 0, parent_ids1, assigned_ids1)`.
  Neighbour lists at t+1 are expressed in `key1`; neighbour lists at t stay in
  `assigned_ids0`. `_frame_neighbour_map` grows a `keys` argument so the focal
  cell is still looked up by its own label but its neighbours are reported by
  key.
* `set1_keys = set(key1)`. Survival and `absent_eligible` use `set1_keys`.
  "Vanished" tests `nb not in set1_keys`.
* For each daughter d born at t+1 with parent m present at t: compute all
  per-cell quantities with `neighbours0 = nb0[m]`, `neighbours1 = nb1[d]` and
  write to column `d`; set `is_birth[ti, col_d] = True`.
* `rank_correlation` with duplicate keys at t+1: when building
  `nb1_rank`, keep the first (nearest) occurrence of a key.

In `errors.py`, `apply_tracking_errors` must carry `parent_ids` through every
step and handle variable cell counts per frame, so that errors can be injected
into real GT with divisions. The variable-N miss function from the original
project's `inject_errors_real_data.py` is the starting point.

## Validation plan on Fluo-N3DH-CE

1. Read `01_GT/TRA` with `read_ctc_tracks(..., dataset="Fluo-N3DH-CE")`.
2. Run the observed metric with `division_policy="ignore"` and `"collapse"`.
   The difference in mean WIO, `wio_vanished` and `newcomer_fraction` is the
   size of the division artefact. Expect `"collapse"` to bring GT WIO close
   to the value seen in non-dividing frame pairs.
3. Plot per-pair WIO against the number of divisions in that pair under both
   policies. Under `"collapse"` the correlation should vanish.
4. Repeat with `division_window = 1, 2, 3` to see whether the sister-proximity
   artefact is visible in rank correlation.
5. Only then inject errors and compare to the simulated response curves.
