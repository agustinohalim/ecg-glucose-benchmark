"""Selisih AUROC berpasangan M2 − M1 dan M2 − M0 untuk artikel Glukosa (Hasil_Glukosa_3.md catatan 1, 2, 4).

Model, pasangan, dan susunan sama dengan jalankan_m2.py: latih TRAIN–nilai VALIDATION (non-EXCLUDE), dan validasi
silang bergugus pasien di subset ICU setara (Hasil_Glukosa_2.md) dengan selisih per status obat. Tiga kelas tidak
diulang. CNN dilatih ulang; cuDNN tidak deterministik, jadi AUROC M2 bisa bergeser sedikit dari Hasil 3 dan dicetak
lagi. Prediksi per pasangan disimpan ke --simpan (di luar repo, drive terenkripsi) agar M3 nanti dibandingkan tanpa
melatih ulang; berkas itu tidak dibuka di sesi asisten. Keluaran stdout hanya agregat. Split TEST tidak disentuh.

Pakai venv-gpu, dari akar repo:
    python glukosa_m2/jalankan_selisih.py [--boot 1000] [--epoch 20]
        [--simpan data/mimic-iv-ext-ecg-glucose/prediksi]
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "glukosa_m1"))
from analisis_obat import strata_obat, subset_icu_setara  # noqa: E402
from cetak import cetak_ukuran_set  # noqa: E402
from jalankan_m2 import NAMA_M2, pengurai_dasar, siapkan  # noqa: E402
from muat import TUGAS_BINER, pasangan_tugas, per_split  # noqa: E402
from pemodelan import BENIH, LIPATAN, latih_dan_prediksi, peluang_luar_lipatan  # noqa: E402
from prediksi import simpan_prediksi  # noqa: E402
from selisih import Perbandingan, baris_selisih, cetak_tabel_selisih  # noqa: E402
from ukuran_kinerja import Bootstrap, Penilaian  # noqa: E402

KODE_TUGAS = ["hipo", "hiper_berat"]
PERBANDINGAN = [
    Perbandingan(NAMA_M2, "M1 ukuran EKG + M0 (GB)", "M2 − M1 (GB)"),
    Perbandingan(NAMA_M2, "M1 ukuran EKG + M0 (LR)", "M2 − M1 (LR)"),
    Perbandingan(NAMA_M2, "M0 usia + jenis kelamin (LR)", "M2 − M0"),
]


@dataclass(frozen=True)
class Konteks:
    model: dict
    bootstrap: Bootstrap
    simpan: Path


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen_selisih()
    mulai = time.time()
    pasangan, model = siapkan(argumen)
    konteks = Konteks(model, Bootstrap(argumen.boot, np.random.default_rng(BENIH[0])), Path(argumen.simpan))
    konteks.simpan.mkdir(parents=True, exist_ok=True)
    print(f"\nEpoch {argumen.epoch}; bootstrap {argumen.boot} bergugus pasien.\n")
    for kode, tugas in zip(KODE_TUGAS, TUGAS_BINER):
        selisih_utama(pasangan, tugas, kode, konteks)
        selisih_icu(pasangan, tugas, kode, konteks)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def baca_argumen_selisih():
    pengurai = pengurai_dasar()
    pengurai.add_argument("--simpan", default="data/mimic-iv-ext-ecg-glucose/prediksi")
    return pengurai.parse_args()


def selisih_utama(pasangan, tugas, kode, konteks):
    latih, nilai = per_split(pasangan_tugas(pasangan[pasangan.bukan_exclude], tugas.kelas_positif))
    peluang = {m: latih_dan_prediksi(latih, nilai, kolom, buat)[:, 1] for m, (kolom, buat) in konteks.model.items()}
    penilaian = Penilaian(nilai.y.values, nilai.subject_id.values, peluang)
    simpan_prediksi(konteks.simpan / f"utama_{kode}.npz", nilai, peluang)
    print(f"## {tugas.nama} — latih TRAIN, nilai VALIDATION (non-EXCLUDE)\n")
    isi = [baris_selisih("VALIDATION", penilaian, PERBANDINGAN, konteks.bootstrap),
           baris_selisih("VALIDATION, konservatif", penilaian.bagian(nilai.konservatif.values), PERBANDINGAN,
                         konteks.bootstrap)]
    cetak_tabel_selisih(isi, PERBANDINGAN)


def selisih_icu(pasangan, tugas, kode, konteks):
    bagian = pasangan_tugas(subset_icu_setara(pasangan), tugas.kelas_positif)
    print(f"## {tugas.nama} — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; validasi silang {LIPATAN} lipatan bergugus "
          f"pasien × {len(BENIH)} benih, selisih di dalam tiap status obat\n")
    cetak_ukuran_set(bagian, "subset")
    print()
    peluang = {m: peluang_luar_lipatan(bagian, kolom, buat) for m, (kolom, buat) in konteks.model.items()}
    penilaian = Penilaian(bagian.y.values, bagian.subject_id.values, peluang)
    simpan_prediksi(konteks.simpan / f"icu_{kode}.npz", bagian, peluang)
    isi = [baris_selisih(status, penilaian.bagian(pilih), PERBANDINGAN, konteks.bootstrap)
           for status, pilih in strata_obat(bagian).items()]
    cetak_tabel_selisih(isi, PERBANDINGAN)


if __name__ == "__main__":
    main()
