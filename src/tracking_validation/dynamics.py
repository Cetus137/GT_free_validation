"""Persistent cell motion in a reflective box with soft spacing.

Each cell carries an active velocity that follows an Ornstein-Uhlenbeck process: it keeps its
direction for about ``persistence_time`` and its root-mean-square speed is ``speed``.  Collective
motion (``alignment`` > 0): each cell also turns its velocity towards the mean velocity of the
cells within ``align_radius``, at that rate; only the direction is steered, so the speeds evolve
exactly as without alignment and only how much neighbours move together changes.  Positions
move with that velocity, plus the push of overlapping neighbours (a spring acting only closer
than ``spacing``) and optional diffusion.  Walls reflect the position and reverse the normal
velocity component.  The equations are integrated with step ``dt`` and a frame is recorded every
``frame_interval``, so the frame rate is a sampling choice, independent of the dynamics.

``simulate_sequence`` (simulator.py) is the older one-step-per-frame model used by the
calibration sweeps; this module does not change it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .errors import frames_from_array


@dataclass(slots=True)
class DynamicsConfig:
    num_cells: int = 300
    box_size: float | tuple[float, float, float] = 50.0   # cube side, or (x, y, z)
    spacing: float = 5.0             # springs act between centres closer than this
    stiffness: float = 2.0           # spring rate (1 / time): an overlap relaxes in ~1/stiffness
    speed: float = 1.0               # rms speed of the active velocity (length / time)
    speed_sigma: float = 0.0         # >0: per-cell speeds lognormal, median ``speed``, this shape
    persistence_time: float = 5.0    # correlation time of the active velocity
    persistence_sigma: float = 0.0   # >0: per-cell persistence times lognormal, this shape
    diffusion: float = 0.0           # extra positional diffusion (length^2 / time)
    alignment: float = 0.0           # collective motion: rate (1 / time) of turning towards neighbours
    align_radius: float | None = None  # neighbours that align; None = 1.5 x spacing
    dt: float = 0.05
    frame_interval: float = 1.0
    num_frames: int = 10
    burn_in: float = 20.0            # run before the first frame: relaxes packing, decorrelates start
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.stiffness * self.dt > 0.1:
            raise ValueError(f"stiffness * dt = {self.stiffness * self.dt:.3f} > 0.1: packed cells "
                             "have ~12 contacts and the explicit step oscillates; lower dt")
        if self.alignment * self.dt > 0.5:
            raise ValueError(f"alignment * dt = {self.alignment * self.dt:.3f} > 0.5: lower dt")
        if abs(self.frame_interval / self.dt - round(self.frame_interval / self.dt)) > 1e-9:
            raise ValueError("frame_interval must be a whole number of dt steps")


@dataclass(slots=True)
class DynamicsResult:
    frames: np.ndarray             # (T, N, 3) positions at the recorded frames
    velocities: np.ndarray         # (T, N, 3) active velocities at the recorded frames
    speeds: np.ndarray             # (N,) per-cell rms speed
    persistence_times: np.ndarray  # (N,)
    times: np.ndarray              # (T,) time of each frame, burn-in excluded
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_frames(self) -> list[dict]:
        """Per-frame dicts for the error, linking and metric functions (ids = 0..N-1)."""
        return frames_from_array(self.frames)


def repulsion(positions: np.ndarray, spacing: float, stiffness: float, rng: np.random.Generator) -> np.ndarray:
    """Velocity from the springs: each pair closer than ``spacing`` is pushed apart along its axis."""
    pairs = cKDTree(positions).query_pairs(spacing, output_type="ndarray")
    push = np.zeros_like(positions)
    if len(pairs) == 0:
        return push
    i, j = pairs[:, 0], pairs[:, 1]
    delta = positions[i] - positions[j]
    dist = np.linalg.norm(delta, axis=1)
    same = dist < 1e-9                              # coincident centres: push along a random axis
    if same.any():
        delta[same] = rng.normal(size=(int(same.sum()), positions.shape[1]))
        dist[same] = np.linalg.norm(delta[same], axis=1)
    f = (stiffness * (spacing - np.minimum(dist, spacing)) / dist)[:, None] * delta
    np.add.at(push, i, f)
    np.add.at(push, j, -f)
    return push


def neighbour_mean_velocity(positions: np.ndarray, velocities: np.ndarray, radius: float) -> np.ndarray:
    """Mean velocity of the other cells within ``radius`` of each cell (NaN where there are none)."""
    pairs = cKDTree(positions).query_pairs(radius, output_type="ndarray")
    total, count = np.zeros_like(velocities), np.zeros(len(positions))
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        np.add.at(total, i, velocities[j]); np.add.at(total, j, velocities[i])
        np.add.at(count, i, 1); np.add.at(count, j, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count[:, None]


def steer(velocities: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    """Turn each velocity ``fraction`` of the way towards ``target``, keeping its length."""
    has = np.isfinite(target).all(axis=1)
    w = velocities.copy()
    w[has] += fraction * (target[has] - velocities[has])
    norm = np.linalg.norm(w, axis=1, keepdims=True)
    return np.where(norm > 1e-12, w / np.maximum(norm, 1e-12) * np.linalg.norm(velocities, axis=1, keepdims=True), velocities)


def velocity_order(positions: np.ndarray, velocities: np.ndarray, radius: float) -> tuple[float, float]:
    """(mean cosine between the directions of cells within ``radius``, global polarisation)."""
    u = velocities / np.maximum(np.linalg.norm(velocities, axis=1, keepdims=True), 1e-12)
    pairs = cKDTree(positions).query_pairs(radius, output_type="ndarray")
    local = float(np.mean(np.sum(u[pairs[:, 0]] * u[pairs[:, 1]], axis=1))) if len(pairs) else float("nan")
    return local, float(np.linalg.norm(u.mean(axis=0)))


def reflect(positions: np.ndarray, velocities: np.ndarray, box: np.ndarray) -> None:
    """Mirror positions back into [0, box] and reverse the velocity component, in place."""
    low, high = positions < 0.0, positions > box
    positions[low] = -positions[low]
    positions[high] = (2.0 * box - positions)[high]
    velocities[low | high] *= -1.0
    np.clip(positions, 0.0, box, out=positions)     # a step longer than the box: stay inside


def simulate_dynamics(cfg: DynamicsConfig) -> DynamicsResult:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.num_cells
    box = np.broadcast_to(np.asarray(cfg.box_size, dtype=float), (3,)).copy()
    speeds = cfg.speed * (rng.lognormal(0.0, cfg.speed_sigma, n) if cfg.speed_sigma > 0 else np.ones(n))
    tau = cfg.persistence_time * (rng.lognormal(0.0, cfg.persistence_sigma, n) if cfg.persistence_sigma > 0 else np.ones(n))

    x = rng.uniform(0.0, 1.0, size=(n, 3)) * box     # overlaps are resolved during the burn-in
    sd = (speeds / np.sqrt(3.0))[:, None]             # per-axis sd, so that <|v|^2> = speed^2
    v = rng.normal(size=(n, 3)) * sd                  # stationary start
    keep = np.exp(-cfg.dt / tau)[:, None]
    kick = sd * np.sqrt(1.0 - keep ** 2)              # exact OU update over one step
    diff = np.sqrt(2.0 * cfg.diffusion * cfg.dt)
    radius = cfg.align_radius if cfg.align_radius is not None else 1.5 * cfg.spacing

    burn = int(round(cfg.burn_in / cfg.dt))
    every = int(round(cfg.frame_interval / cfg.dt))
    frames, vels = [], []
    for step in range(burn + every * (cfg.num_frames - 1) + 1):
        if step >= burn and (step - burn) % every == 0:
            frames.append(x.copy()); vels.append(v.copy())
            if len(frames) == cfg.num_frames:
                break
        target = neighbour_mean_velocity(x, v, radius) if cfg.alignment > 0 else None
        v = keep * v + kick * rng.normal(size=(n, 3))
        if target is not None:
            v = steer(v, target, cfg.alignment * cfg.dt)
        x = x + (v + repulsion(x, cfg.spacing, cfg.stiffness, rng)) * cfg.dt
        if diff > 0:
            x = x + diff * rng.normal(size=(n, 3))
        reflect(x, v, box)

    frames, vels = np.asarray(frames), np.asarray(vels)
    nn = np.array([np.median(cKDTree(f).query(f, k=2)[0][:, 1]) for f in frames])
    order = np.array([velocity_order(f, u, radius) for f, u in zip(frames, vels)])
    return DynamicsResult(frames=frames, velocities=vels, speeds=speeds, persistence_times=tau,
                          times=np.arange(cfg.num_frames) * cfg.frame_interval,
                          metadata=dict(seed=cfg.seed, median_nn=float(np.median(nn)), density=n / float(np.prod(box)),
                                        neighbour_alignment=float(np.nanmean(order[:, 0])) if np.isfinite(order[:, 0]).any() else float("nan"),
                                        polarisation=float(order[:, 1].mean())))
