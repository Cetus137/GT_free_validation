from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def export_tracks_csv(
    frames: np.ndarray,
    speeds: np.ndarray,
    persistence: np.ndarray,
    output_path: str | Path,
    frame_speeds: np.ndarray | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "cell_id", "x", "y", "z", "speed", "persistence"])
        t, n, _ = frames.shape
        for frame_idx in range(t):
            for cell_id in range(n):
                x, y, z = frames[frame_idx, cell_id]
                if frame_speeds is not None:
                    speed_value = float(frame_speeds[frame_idx, cell_id])
                elif speeds.ndim == 2:
                    speed_value = float(speeds[frame_idx, cell_id])
                else:
                    speed_value = float(speeds[cell_id])
                writer.writerow(
                    [
                        frame_idx,
                        cell_id,
                        float(x),
                        float(y),
                        float(z),
                        speed_value,
                        float(persistence[cell_id]),
                    ]
                )

    return output


def export_frames_npy(frames: np.ndarray, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, frames)
    return output


def export_metadata_json(metadata: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return output
