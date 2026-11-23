"""M2 analisis utama dirata-rata atas 3 benih (keputusan penulis 21 Nov 2026, Hasil_Glukosa_5.md): CNN dilatih di
TRAIN dengan BENIH (sama dengan benih validasi silang ICU), peluang VALIDATION dirata-rata. Prediksi M0, M1, M2 satu
benih (Hasil 4), dan M3 (Hasil 5) dibaca dari --prediksi dan dicocokkan baris demi baris; hanya M2 yang dilatih ulang.
Dicetak: AUROC per benih, tabel metrik semua model, selisih berpasangan. Prediksi rerata disimpan ke
utama_<tugas>_m2b3.npz. Keluaran stdout hanya agregat. Split TEST tidak disentuh.

Pakai lingkungan utama (README), dari akar repo:
    python glukosa_m2/jalankan_m2_benih.py [--boot 1000] [--epoch 20]
        [--prediksi data/mimic-iv-ext-ecg-glucose/prediksi]
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "glukosa_m1"))
from cetak import cetak_tabel_metrik  # noqa: E402
from jalankan_m2 import NAMA_M2, pengurai_dasar, siapkan  # noqa: E402
from muat import TUGAS_BINER, pasangan_tugas, per_split  # noqa: E402
from pemodelan import BENIH  # noqa: E402
from prediksi import muat_prediksi, simpan_prediksi  # noqa: E402
from selisih import Perbandingan, baris_selisih, cetak_tabel_selisih  # noqa: E402
from ukuran_kinerja import Bootstrap, Penilaian  # noqa: E402

KODE_TUGAS = ["hipo", "hiper_berat"]
NAMA_M2_BENIH = f"M2 CNN gelombang, rerata {len(BENIH)} benih"
NAMA_M3 = "M3 ECGFounder beku + probe L2"
PERBANDINGAN = [
    Perbandingan(NAMA_M2_BENIH, "M1 ukuran EKG + M0 (GB)", "M2b − M1 (GB)"),
    Perbandingan(NAMA_M2_BENIH, "M0 usia + jenis kelamin (LR)", "M2b − M0"),
    Perbandingan(NAMA_M3, NAMA_M2_BENIH, "M3 − M2b"),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pengurai = pengurai_dasar()
    pengurai.add_argument("--prediksi", default="data/mimic-iv-ext-ecg-glucose/prediksi")
    argumen = pengurai.parse_args()
    mulai = time.time()
    pasangan, model = siapkan(argumen)
    bootstrap = Bootstrap(argumen.boot, np.random.default_rng(BENIH[0]))
    print(f"\nEpoch {argumen.epoch}; benih {BENIH}; bootstrap {argumen.boot} bergugus pasien.\n")
    for kode, tugas in zip(KODE_TUGAS, TUGAS_BINER):
        latih, nilai = per_split(pasangan_tugas(pasangan[pasangan.bukan_exclude], tugas.kelas_positif))
        p2 = rerata_benih(latih, nilai, model[NAMA_M2])
        folder = Path(argumen.prediksi)
        simpan_prediksi(folder / f"utama_{kode}_m2b3.npz", nilai, {NAMA_M2_BENIH: p2})
        peluang = {**muat_prediksi(folder / f"utama_{kode}.npz", nilai),
                   **muat_prediksi(folder / f"utama_{kode}_m3.npz", nilai), NAMA_M2_BENIH: p2}
        cetak_tugas(tugas, nilai, Penilaian(nilai.y.values, nilai.subject_id.values, peluang), bootstrap)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def rerata_benih(latih, nilai, kolom_buat):
    kolom, buat = kolom_buat
    per_benih = []
    for benih in BENIH:
        model = buat(benih).fit(latih[kolom].values, latih.y.values)
        per_benih.append(model.predict_proba(nilai[kolom].values)[:, 1])
        print(f"- benih {benih}: AUROC VALIDATION {roc_auc_score(nilai.y.values, per_benih[-1]):.3f}", flush=True)
    return np.mean(per_benih, axis=0)


def cetak_tugas(tugas, nilai, penilaian, bootstrap):
    print(f"\n## {tugas.nama} — latih TRAIN, nilai VALIDATION (non-EXCLUDE)\n")
    cetak_tabel_metrik(penilaian, bootstrap)
    isi = [baris_selisih("VALIDATION", penilaian, PERBANDINGAN, bootstrap),
           baris_selisih("VALIDATION, konservatif", penilaian.bagian(nilai.konservatif.values), PERBANDINGAN,
                         bootstrap)]
    cetak_tabel_selisih(isi, PERBANDINGAN)


if __name__ == "__main__":
    main()
