"""ECGFounder beku untuk M3 Glukosa: pemuatan bobot dan pra-proses resmi, sama dengan M3a naskah OMI
(omi_m3_ecgfounder.py, kode Net1D NickLJLee/ECGFounder komit 67367071, MIT; bobot PKUDigitalHealth/ECGFounder
`12_lead_ECGFounder.pth`, MIT). ECGFounder dilatih pada Harvard-Emory ECG Database, bukan MIMIC-IV-ECG.

Pra-proses: 12 × 5.000 sampel (500 Hz) dalam mV, urutan I, II, III, aVR, aVL, aVF, V1–V6; notch 50 Hz (Q 30);
Butterworth orde 4 pita 0,67–40 Hz (filtfilt); dikurangi median bergerak 201 sampel (bantalan nol); z-score atas
seluruh larik. Embedding = `deep_features` Net1D (1.024 dimensi), mode eval, tanpa gradien.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, iirnotch

KODE = Path("ext/ECGFounder")
BOBOT = Path("models/ECGFounder/12_lead_ECGFounder.pth")
SADAPAN_MODEL = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS, N_SAMPEL, DIMENSI = 500, 5000, 1024
JENDELA_MEDIAN = int(0.4 * FS) + 1
NOTCH = iirnotch(50, 30, FS)
PITA = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=FS)


def susun_sadapan(rekaman):
    header = rekaman.header
    if header.fs != FS or header.n_sampel != N_SAMPEL or not set(SADAPAN_MODEL) <= set(header.sadapan):
        return None
    x = rekaman.sinyal_uv[[header.sadapan.index(s) for s in SADAPAN_MODEL]] / 1000.0
    return np.nan_to_num(x.astype(np.float64))


def praproses(x):
    f = filtfilt(*NOTCH, x, axis=1)
    f = filtfilt(*PITA, f, axis=1)
    f = f - median_filter(f, size=(1, JENDELA_MEDIAN), mode="constant", cval=0.0)
    return ((f - f.mean()) / (f.std() + 1e-8)).astype(np.float32)


def siapkan_sinyal(rekaman):
    x = susun_sadapan(rekaman)
    return None if x is None else praproses(x)


def muat_penanam(perangkat):
    model = muat_model(perangkat)
    return lambda x: model(x)[1]


def muat_model(perangkat):
    import torch
    sys.path.insert(0, str(KODE))
    from net1d import Net1D
    model = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                  m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16, verbose=False,
                  use_bn=False, use_do=False, n_classes=150, return_features=True)
    hilang, _ = model.load_state_dict(baca_titik_periksa(torch)["state_dict"], strict=False)
    if [k for k in hilang if not k.startswith("dense.")]:
        raise SystemExit(f"Bobot ECGFounder tidak termuat: {hilang[:4]}")
    return model.to(perangkat).eval()


def baca_titik_periksa(torch):
    import numpy.core.multiarray as nca
    izin = ([(nca.scalar, "numpy.core.multiarray.scalar"), np.dtype]
            + [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")])
    with torch.serialization.safe_globals(izin):
        return torch.load(BOBOT, map_location="cpu", weights_only=True)
