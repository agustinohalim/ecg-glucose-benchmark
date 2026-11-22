"""Analisis kepekaan TEST atas temuan telaah internal pertama (22 Nov 2026), deskriptif dan sesudah protokol.

Tanpa pelatihan: prediksi TEST dibaca dari test_<tugas>.npz (jalankan_test.py, Hasil 8). Tiga bagian, ditetapkan sebelum
dijalankan:
1. AUROC dalam pasien: hanya pasien TEST yang punya pasangan positif dan euglikemik; tiap pasangan positif dibandingkan
   dengan pasangan euglikemik pasien yang sama saja, sehingga perbedaan antarpasien tidak ikut terhitung.
2. Tempat perawatan: tanpa rawat inap (hadm_id kosong: IGD atau rawat jalan), rawat inap di luar ICU, ICU.
3. Waktu EKG terhadap pengambilan glukosa: arah (sebelum, sesudah, serentak) dan jarak mutlak (≤ 30, 31–120,
   121–240 menit).
Keluaran stdout hanya agregat; hitungan 1–10 dicetak "<11", metrik disembunyikan bila positif < 11.

Pakai venv-gpu, dari akar repo:
    python glukosa_test/jalankan_kepekaan_panel.py [--boot 1000]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / "glukosa_m3"), str(AKAR_EKG / "glukosa_m2"), str(AKAR_EKG / "glukosa_m1")]
from cetak import SEL_KECIL, dengan_selang, tabel, tampil_hitungan  # noqa: E402
from jalankan_m2 import pasang_gelombang  # noqa: E402
from jalankan_test import KODE_TUGAS, UTAMA, baca_argumen, per_tiga_split  # noqa: E402
from muat import BERKAS, TUGAS_BINER, muat_pasangan, pasangan_tugas  # noqa: E402
from pemodelan import BENIH  # noqa: E402
from prediksi import muat_prediksi  # noqa: E402
from selisih import baris_selisih, cetak_tabel_selisih  # noqa: E402
from singgahan import Singgahan  # noqa: E402
from ukuran_kinerja import Bootstrap, Penilaian  # noqa: E402

ARAH_WAKTU = {"ECG_BEFORE_GLUCOSE": "EKG sebelum glukosa", "ECG_AFTER_GLUCOSE": "EKG sesudah glukosa",
              "SIMULTANEOUS": "serentak (menit yang sama)"}
BATAS_JARAK = [0, 31, 121, 241]
LABEL_JARAK = ["jarak ≤ 30 menit", "jarak 31–120 menit", "jarak 121–240 menit"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    pasangan = pasang_gelombang(muat_pasangan(Path(argumen.folder) / BERKAS, ["waveform_path", "hadm_id"]),
                                Singgahan.buka(Path(argumen.singgahan)))
    print(f"\nKepekaan TEST atas telaah internal pertama, deskriptif, sesudah protokol; bootstrap {argumen.boot} bergugus pasien.\n")
    for kode, tugas in zip(KODE_TUGAS, TUGAS_BINER):
        test = per_tiga_split(pasangan_tugas(pasangan[pasangan.bukan_exclude], tugas.kelas_positif))[2]
        peluang = muat_prediksi(Path(argumen.prediksi) / f"test_{kode}.npz", test)
        penilaian = Penilaian(test.y.values, test.subject_id.values, peluang)
        print(f"## {tugas.nama} — TEST (non-EXCLUDE)\n")
        cetak_dalam_pasien(penilaian, Bootstrap(argumen.boot, np.random.default_rng(BENIH[0])))
        bootstrap = Bootstrap(argumen.boot, np.random.default_rng(BENIH[0]))
        isi = [baris_selisih("TEST, semua", penilaian, UTAMA, bootstrap)]
        for label, pilih in strata(test):
            isi.append(baris_selisih(label, penilaian.bagian(pilih), UTAMA, bootstrap))
        print("### Per tempat perawatan dan waktu EKG\n")
        cetak_tabel_selisih(isi, UTAMA)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def strata(test):
    tanpa_rawat_inap = test.hadm_id.isna().values
    yield "tanpa rawat inap (IGD/rawat jalan)", tanpa_rawat_inap
    yield "rawat inap di luar ICU", ~tanpa_rawat_inap & ~test.di_icu.values
    yield "ICU", test.di_icu.values
    yield from ((label, test.temporal_relationship.eq(kode).values) for kode, label in ARAH_WAKTU.items())
    jarak = pd.cut(test.abs_offset_minutes, BATAS_JARAK, right=False, labels=LABEL_JARAK)
    yield from ((label, (jarak == label).values) for label in LABEL_JARAK)


def banding_per_pasien(penilaian, model):
    hasil = {}
    for pasien in np.unique(penilaian.gugus):
        pilih = penilaian.gugus == pasien
        y = penilaian.y[pilih]
        if 0 < y.sum() < len(y):
            hasil[pasien] = [hitung_konkordan(penilaian.peluang[m][pilih], y) for m in model]
    return np.array(list(hasil.values()))


def hitung_konkordan(p, y):
    beda = p[y == 1][:, None] - p[y == 0][None, :]
    return (beda > 0).sum() + 0.5 * (beda == 0).sum(), beda.size


def cetak_dalam_pasien(penilaian, bootstrap):
    model = [UTAMA[0].model_a, UTAMA[0].model_b]
    calon = np.isin(penilaian.gugus, np.unique(penilaian.gugus[penilaian.y == 1]))
    banding = banding_per_pasien(penilaian.bagian(calon), model)
    n_pasien = len(banding)
    print(f"### AUROC dalam pasien (pasien dengan pasangan positif dan euglikemik: {tampil_hitungan(n_pasien)})\n")
    if n_pasien < SEL_KECIL:
        print("Pasien < 11; metrik tidak dicetak.\n")
        return
    titik = banding[:, :, 0].sum(axis=0) / banding[:, :, 1].sum(axis=0)
    ulangan = []
    for _ in range(bootstrap.n):
        ambil = banding[bootstrap.rng.integers(0, n_pasien, n_pasien)]
        ulangan.append(ambil[:, :, 0].sum(axis=0) / ambil[:, :, 1].sum(axis=0))
    ulangan = np.array(ulangan)
    selang = np.quantile(ulangan, [0.025, 0.975], axis=0)
    selang_beda = np.quantile(ulangan[:, 0] - ulangan[:, 1], [0.025, 0.975])
    tabel(["Ukuran", "Nilai [95 % CI]"],
          [[f"AUROC dalam pasien {model[0]}", dengan_selang(titik[0], selang[:, 0])],
           [f"AUROC dalam pasien {model[1]}", dengan_selang(titik[1], selang[:, 1])],
           [UTAMA[0].label, dengan_selang(titik[0] - titik[1], selang_beda)],
           ["Perbandingan positif–euglikemik dalam pasien", tampil_hitungan(banding[:, 0, 1].sum())]])
    print("\nBootstrap pasien atas pasien yang memenuhi syarat.\n")


if __name__ == "__main__":
    main()
