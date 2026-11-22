"""ECG-FM beku untuk uji kontaminasi Glukosa (McKeen dkk. 2025, JAMIA Open, doi 10.1093/jamiaopen/ooaf122; kode
Jwoo5/fairseq-signals komit 6ffdec72, MIT; bobot wanglab/ecg-fm revisi 584219ea `mimic_iv_ecg_physionet_pretrained.pt`,
MIT, SHA-256 4d0142bc…9b7e). Dilatih-awal pada MIMIC-IV-ECG v1.0 (bagian `train` dari daftar split penulisnya) dan
PhysioNet 2021: model ini SENGAJA terkontaminasi; daftar split dipakai untuk memisahkan rekaman yang dilihat dan tidak.

Pra-proses = tutorial resmi infer_quickstart (ecg-transform 0.1.3): 12 sadapan urutan I, II, III, aVR, aVL, aVF, V1–V6,
500 Hz; per sadapan dikurangi rerata dan dibagi simpangan baku atas 10 s penuh (sadapan konstan menjadi 0); dipotong
dua segmen 5 s tanpa tumpang. Embedding = rerata token keluaran encoder (768 dimensi, `extract_features`, tanpa
masking, mode eval), lalu dirata-rata atas dua segmen. Butuh lingkungan ECG-FM
(README).
"""
from pathlib import Path

import numpy as np

from ecgfounder import susun_sadapan

BOBOT = Path("models/ECG-FM/mimic_iv_ecg_physionet_pretrained.pt")
N_SEGMEN, PANJANG_SEGMEN = 2, 2500
EPSILON = 1e-8


def siapkan_sinyal(rekaman):
    x = susun_sadapan(rekaman)
    if x is None:
        return None
    tengah = x - x.mean(axis=1, keepdims=True)
    return (tengah / (tengah.std(axis=1, keepdims=True) + EPSILON)).astype(np.float32)


def muat_penanam(perangkat):
    from fairseq_signals.models import build_model_from_checkpoint
    model = build_model_from_checkpoint(checkpoint_path=str(BOBOT)).to(perangkat).eval()

    def tanam(x):
        segmen = x.reshape(len(x), x.shape[1], N_SEGMEN, PANJANG_SEGMEN).permute(0, 2, 1, 3)
        token = model.extract_features(segmen.reshape(-1, x.shape[1], PANJANG_SEGMEN), padding_mask=None)["x"]
        return token.mean(dim=1).reshape(len(x), N_SEGMEN, -1).mean(dim=1)

    return tanam
