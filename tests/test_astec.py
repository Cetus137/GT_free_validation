"""Tests for the ASTEC / MorphoNet properties reader, on a synthetic XML."""
from __future__ import annotations

import numpy as np
import pytest

from tracking_validation.astec import read_astec_tracks, read_properties
from tracking_validation.lineage import inherit_parent_labels


def _xml(tmp_path):
    # time 1: cells 1,2 ; time 2: 1->1, 2 divides -> 2,3 ; time 3: all continue
    cells = {
        10001: ([0, 0, 0], [20001]), 10002: ([10, 0, 0], [20002, 20003]),
        20001: ([0.5, 0, 0], [30001]), 20002: ([9, 1, 0], [30002]), 20003: ([11, -1, 0], [30003]),
        30001: ([1, 0, 0], None), 30002: ([8.5, 1.5, 0], None), 30003: ([11.5, -1.5, 0], None),
    }
    bary = "\n".join(f'    <cell cell-id="{c}">[{p[0]}, {p[1]}, {p[2]}]</cell>' for c, (p, _) in cells.items())
    lin = "\n".join(f'    <cell cell-id="{c}">{s}</cell>' for c, (_, s) in cells.items() if s)
    vol = "\n".join(f'    <cell cell-id="{c}">{100 + c % 7}</cell>' for c in cells)
    p = tmp_path / "Test_properties.xml"
    p.write_text(f"<data>\n  <cell_barycenter>\n{bary}\n  </cell_barycenter>\n  <cell_lineage>\n{lin}\n  </cell_lineage>\n"
                 f"  <cell_volume>\n{vol}\n  </cell_volume>\n  <embryo_name/>\n</data>\n")
    return p


def test_read_properties_parses_blocks(tmp_path):
    props = read_properties(_xml(tmp_path))
    assert set(props) == {"cell_barycenter", "cell_lineage", "cell_volume"}
    assert props["cell_lineage"][10002] == [20002, 20003]
    assert props["cell_barycenter"][30001] == [1, 0, 0]


def test_tracks_follow_lineage_and_scale(tmp_path):
    d = read_astec_tracks(_xml(tmp_path), voxel_size=0.3)
    assert d.frame_indices == [1, 2, 3] and d.num_frames == 3
    assert d.counts().tolist() == [2, 3, 3]
    # cell 1 keeps one label throughout
    lab1 = [f["assigned_ids"][f["astec_ids"] == t * 10000 + 1][0] for t, f in zip(d.frame_indices, d.frames)]
    assert len(set(lab1)) == 1
    # division: mother's label ends at t=1, two daughters born at t=2 with parent set
    f2 = d.frames[1]
    born = f2["parent_ids"] >= 0
    assert born.sum() == 2 and len(set(f2["parent_ids"][born])) == 1
    mother = f2["parent_ids"][born][0]
    assert d.tracks[mother].end == 1 and d.tracks[mother].begin == 1
    assert d.divisions() == [(mother, *sorted(f2["assigned_ids"][born]), 2)]
    # positions scaled
    np.testing.assert_allclose(d.frames[0]["positions"][0], [0, 0, 0]); np.testing.assert_allclose(d.frames[0]["positions"][1], [3.0, 0, 0])
    # birth frame only: at t=3 the daughters have parent -1
    assert np.all(d.frames[2]["parent_ids"] == -1)
    assert len(d.tracks) == 4


def test_frame_subset_and_inheritance(tmp_path):
    d = read_astec_tracks(_xml(tmp_path), frames=[2, 3])
    assert d.frame_indices == [2, 3] and d.counts().tolist() == [3, 3]
    inh = inherit_parent_labels(read_astec_tracks(_xml(tmp_path)))
    f2 = inh.frames[1]
    assert (f2["parent_ids"] >= 0).sum() == 1          # one daughter inherits, the other keeps a parent link
    assert len(set(f2["assigned_ids"].tolist())) == 3
