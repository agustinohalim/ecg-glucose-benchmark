"""ECG-FM fine-tuned beku sebagai kontrol positif uji kontaminasi Glukosa (protokol §F langkah 11). Bobot wanglab/ecg-fm
revisi 584219ea `mimic_iv_ecg_finetuned.pt`, MIT, SHA-256 f3e07588…7b3d: dilatih-lanjut dari checkpoint pretrained dengan
label diagnosis MIMIC-IV-ECG bagian `train` dari daftar split penulisnya (label_def.csv repo ECG-FM). Pra-proses dan
embedding sama dengan ecgfm.py; embedding = rerata token `encoder.extract_features` (tanpa kepala klasifikasi), dirata-rata
atas dua segmen 5 s. Butuh lingkungan ECG-FM (README).
"""
from pathlib import Path

from ecgfm import N_SEGMEN, PANJANG_SEGMEN, siapkan_sinyal  # noqa: F401

BOBOT = Path("models/ECG-FM/mimic_iv_ecg_finetuned.pt")


def muat_penanam(perangkat):
    from fairseq_signals.models import build_model_from_checkpoint
    model = build_model_from_checkpoint(checkpoint_path=str(BOBOT)).to(perangkat).eval()

    def tanam(x):
        segmen = x.reshape(len(x), x.shape[1], N_SEGMEN, PANJANG_SEGMEN).permute(0, 2, 1, 3)
        token = model.encoder.extract_features(segmen.reshape(-1, x.shape[1], PANJANG_SEGMEN), padding_mask=None)["x"]
        return token.mean(dim=1).reshape(len(x), N_SEGMEN, -1).mean(dim=1)

    return tanam
