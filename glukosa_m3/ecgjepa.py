"""ECG-JEPA beku untuk M3b Glukosa (Kim 2024, arXiv 2410.08559; kode sehunfromdaegu/ECG_JEPA komit d937ad2, MIT;
titik periksa multi-block `multiblock_epoch100.pth`). Dilatih-awal pada Chapman, Ningbo, CODE-15 — tanpa MIMIC-IV-ECG
(makalah §4; bentuk bobot cocok: 8 sadapan × 50 tambalan × 50 sampel = 250 Hz, 10 s).

Pra-proses = jalur linear_eval resmi: sinyal mV (setara wfdb p_signal), scipy.signal.resample 5.000 → 2.500 sampel,
sadapan I, II, V1–V6, tanpa filter dan tanpa normalisasi. Sampel hilang diisi 0. Embedding = encoder.representation
(rerata token, 768 dimensi), mode eval. Paket timm diganti modul minimal di timm_pengganti (DropPath, trunc_normal_)
karena timm 1.0.8 menarik torchvision; bobot dimuat ketat oleh load_encoder resmi.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample

KODE = Path("ext/ECG_JEPA")
PENGGANTI_TIMM = Path("ext/timm_pengganti")
BOBOT = Path("models/ECG_JEPA/multiblock_epoch100.pth")
SADAPAN = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
FS, N_SAMPEL_ASAL, N_SAMPEL = 500, 5000, 2500


def siapkan_sinyal(rekaman):
    header = rekaman.header
    if header.fs != FS or header.n_sampel != N_SAMPEL_ASAL or not set(SADAPAN) <= set(header.sadapan):
        return None
    x = np.nan_to_num(rekaman.sinyal_uv[[header.sadapan.index(s) for s in SADAPAN]].astype(np.float64) / 1000.0)
    return resample(x, N_SAMPEL, axis=1).astype(np.float32)


def muat_penanam(perangkat):
    sys.path[:0] = [str(PENGGANTI_TIMM), str(KODE)]
    from models import load_encoder
    encoder, _ = load_encoder(str(BOBOT))
    return encoder.to(perangkat).eval().representation
