"""Convert the C. elegans images from Zenodo record 4630933 to CTC-style per-frame TIFFs.

The CTC data server (data.celltrackingchallenge.net) was unreachable, so the Fluo-N3DH-CE
training images were taken from Zenodo record 4630933 (Fluo-N3DH-CE_01.h5 / _02.h5,
BigDataViewer HDF5; md5 checked).  Only GT-covered frames are written, as
data/Fluo-N3DH-CE/{01,02}/tNNN.tif (uint16, zlib), next to the existing {01,02}_GT folders.
Verified: same shape and voxel size as GT; markers brightest at zero time offset.

Requirements / quirks:
  - HDF5 1.14.x fails to decode these files (scale-offset + gzip filter).  Use h5py 3.9
    (HDF5 1.12):  python -m venv h5old && h5old/bin/pip install "h5py==3.9.0" "numpy<2" tifffile imagecodecs
  - Voxels are stored as int16 but are uint16 data; reinterpret with .view(np.uint16).

    h5old/bin/python scripts/convert_zenodo_ce.py 01,02
"""
import sys, os, h5py, tifffile, numpy as np
root = "/Users/ewheeler/Documents/GT_metrics/data"
seqs = sys.argv[1].split(",")
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
for s in seqs:
    gtdir = f"{root}/Fluo-N3DH-CE/{s}_GT/TRA"
    frames = sorted(int(n[9:12]) for n in os.listdir(gtdir) if n.startswith("man_track") and n.endswith(".tif"))
    out = f"{root}/Fluo-N3DH-CE/{s}"; os.makedirs(out, exist_ok=True)
    f = h5py.File(f"{root}/_downloads/zenodo_ce/Fluo-N3DH-CE_{s}.h5", "r")
    for t in frames[:limit]:
        im = f[f"t{t:05d}/s00/0/cells"][()].view(np.uint16)       # stored as int16; reinterpret
        tifffile.imwrite(f"{out}/t{t:03d}.tif", im, compression="zlib", compressionargs={"level": 6})
    print(s, len(frames[:limit]), "frames ->", out, flush=True)
