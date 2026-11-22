"""Berkas prediksi per pasangan (di luar repo, drive terenkripsi): y, subject_id, status obat, nama model, dan matriks
peluang (baris × model). Dipakai agar model baru dibandingkan dengan model lama tanpa melatih ulang; berkas tidak
pernah dibuka di sesi asisten."""
import numpy as np


def simpan_prediksi(jalur, bagian, peluang):
    np.savez_compressed(jalur, y=bagian.y.values, subject_id=bagian.subject_id.values,
                        intervention_status=bagian.intervention_status.values.astype(str),
                        model=np.array(list(peluang)), peluang=np.column_stack(list(peluang.values())))


def muat_prediksi(jalur, bagian):
    tersimpan = np.load(jalur)
    if not (np.array_equal(tersimpan["y"], bagian.y.values)
            and np.array_equal(tersimpan["subject_id"], bagian.subject_id.values)):
        raise SystemExit(f"Baris {jalur.name} tidak sejajar dengan pasangan sekarang.")
    return dict(zip(tersimpan["model"].tolist(), tersimpan["peluang"].T))
