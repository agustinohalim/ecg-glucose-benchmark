"""Analisis subkelompok demografis TEST, deskriptif dan sesudah protokol (keputusan penulis 22 Nov 2026,
Tinjauan_Glukosa_2026-11-22.md): AUROC M3b dan selisih M3b − M1-GB per kelompok usia dan jenis kelamin.

Tanpa pelatihan: prediksi TEST dibaca dari test_<tugas>.npz (jalankan_test.py, Hasil 8) dan dicocokkan baris demi baris.
Kelompok usia ditetapkan sebelum dijalankan: 18–44, 45–64, 65–79, ≥ 80 tahun. Ras tidak dianalisis (hilang bila hadm_id
kosong, 53,5 % pasangan). Keluaran stdout hanya agregat; hitungan 1–10 dicetak "<11", metrik disembunyikan bila positif
< 11.

Pakai lingkungan utama (README), dari akar repo:
    python glukosa_test/jalankan_subkelompok.py [--boot 1000]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / "glukosa_m3"), str(AKAR_EKG / "glukosa_m2"), str(AKAR_EKG / "glukosa_m1")]
from jalankan_m2 import pasang_gelombang  # noqa: E402
from jalankan_test import KODE_TUGAS, UTAMA, baca_argumen, per_tiga_split  # noqa: E402
from muat import BERKAS, TUGAS_BINER, muat_pasangan, pasangan_tugas  # noqa: E402
from pemodelan import BENIH  # noqa: E402
from prediksi import muat_prediksi  # noqa: E402
from selisih import baris_selisih, cetak_tabel_selisih  # noqa: E402
from singgahan import Singgahan  # noqa: E402
from ukuran_kinerja import Bootstrap, Penilaian  # noqa: E402

BATAS_USIA = [18, 45, 65, 80, np.inf]
LABEL_USIA = ["usia 18–44", "usia 45–64", "usia 65–79", "usia ≥ 80"]
LABEL_JENIS_KELAMIN = {"F": "perempuan", "M": "laki-laki"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    pasangan = pasang_gelombang(muat_pasangan(Path(argumen.folder) / BERKAS, ["waveform_path"]),
                                Singgahan.buka(Path(argumen.singgahan)))
    bootstrap = Bootstrap(argumen.boot, np.random.default_rng(BENIH[0]))
    print(f"\nSubkelompok TEST, deskriptif, sesudah protokol; bootstrap {argumen.boot} bergugus pasien.\n")
    for kode, tugas in zip(KODE_TUGAS, TUGAS_BINER):
        test = per_tiga_split(pasangan_tugas(pasangan[pasangan.bukan_exclude], tugas.kelas_positif))[2]
        peluang = muat_prediksi(Path(argumen.prediksi) / f"test_{kode}.npz", test)
        penilaian = Penilaian(test.y.values, test.subject_id.values, peluang)
        print(f"## {tugas.nama} — TEST (non-EXCLUDE), per subkelompok\n")
        isi = [baris_selisih("TEST, semua", penilaian, UTAMA, bootstrap)]
        for label, pilih in subkelompok(test):
            isi.append(baris_selisih(label, penilaian.bagian(pilih), UTAMA, bootstrap))
        cetak_tabel_selisih(isi, UTAMA)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def subkelompok(test):
    kelompok_usia = pd.cut(test.age, BATAS_USIA, right=False, labels=LABEL_USIA)
    yield from ((label, (kelompok_usia == label).values) for label in LABEL_USIA)
    yield from ((label, test.gender.eq(kode).values) for kode, label in LABEL_JENIS_KELAMIN.items())


if __name__ == "__main__":
    main()
