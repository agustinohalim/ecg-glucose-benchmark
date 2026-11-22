"""Selisih berpasangan antarmodel gelombang dari prediksi VALIDATION yang sudah disimpan (tanpa melatih apa pun):
M3b (ECG-JEPA) lawan M2b (rerata 3 benih, keputusan 21 Nov 2026) dan M3 (ECGFounder), analisis utama, dua tugas biner.

Prediksi dibaca dari utama_<tugas>.npz (M0–M2), _m2b3, _m3, _m3b di --prediksi dan dicocokkan baris demi baris
(y dan subject_id harus sama). Bootstrap bergugus pasien dengan benih yang sama dengan skrip lain. Keluaran stdout hanya
agregat. Split TEST tidak disentuh.

Dari akar repo:
    python glukosa_m3/bandingkan_simpanan.py [--boot 1000]
        [--folder data/mimic-iv-ext-ecg-glucose] [--singgahan data/mimic-iv-ext-ecg-glucose/gelombang]
        [--prediksi data/mimic-iv-ext-ecg-glucose/prediksi]
"""
import sys
import time
from pathlib import Path

import numpy as np

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / "glukosa_m2"), str(AKAR_EKG / "glukosa_m1")]
from cetak import cetak_tabel_metrik  # noqa: E402
from jalankan_m2 import pasang_gelombang  # noqa: E402
from jalankan_m3 import baca_argumen  # noqa: E402
from model_fondasi import ECGFOUNDER, ECGJEPA  # noqa: E402
from muat import BERKAS, TUGAS_BINER, muat_pasangan, pasangan_tugas, per_split  # noqa: E402
from pemodelan import BENIH  # noqa: E402
from prediksi import muat_prediksi  # noqa: E402
from selisih import Perbandingan, baris_selisih, cetak_tabel_selisih  # noqa: E402
from singgahan import Singgahan  # noqa: E402
from ukuran_kinerja import Bootstrap, Penilaian  # noqa: E402

KODE_TUGAS = ["hipo", "hiper_berat"]
AKHIRAN = ["", "_m2b3", ECGFOUNDER.akhiran, ECGJEPA.akhiran]
NAMA_M2_BENIH = f"M2 CNN gelombang, rerata {len(BENIH)} benih"
PERBANDINGAN = [
    Perbandingan(ECGJEPA.nama, NAMA_M2_BENIH, "M3b − M2b"),
    Perbandingan(ECGJEPA.nama, ECGFOUNDER.nama, "M3b − M3"),
    Perbandingan(NAMA_M2_BENIH, ECGFOUNDER.nama, "M2b − M3"),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    pasangan = pasang_gelombang(muat_pasangan(Path(argumen.folder) / BERKAS, ["waveform_path"]),
                                Singgahan.buka(Path(argumen.singgahan)))
    bootstrap = Bootstrap(argumen.boot, np.random.default_rng(BENIH[0]))
    print(f"\nPrediksi tersimpan, tanpa pelatihan; bootstrap {argumen.boot} bergugus pasien.\n")
    for kode, tugas in zip(KODE_TUGAS, TUGAS_BINER):
        _, nilai = per_split(pasangan_tugas(pasangan[pasangan.bukan_exclude], tugas.kelas_positif))
        peluang = {}
        for akhiran in AKHIRAN:
            peluang.update(muat_prediksi(Path(argumen.prediksi) / f"utama_{kode}{akhiran}.npz", nilai))
        penilaian = Penilaian(nilai.y.values, nilai.subject_id.values, peluang)
        print(f"## {tugas.nama} — latih TRAIN, nilai VALIDATION (non-EXCLUDE)\n")
        cetak_tabel_metrik(penilaian, bootstrap)
        isi = [baris_selisih("VALIDATION", penilaian, PERBANDINGAN, bootstrap),
               baris_selisih("VALIDATION, konservatif", penilaian.bagian(nilai.konservatif.values), PERBANDINGAN,
                             bootstrap)]
        cetak_tabel_selisih(isi, PERBANDINGAN)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


if __name__ == "__main__":
    main()
