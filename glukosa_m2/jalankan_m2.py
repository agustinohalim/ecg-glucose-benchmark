"""M2 (CNN 1D pada gelombang EKG) berdampingan dengan M0/M1 untuk artikel Glukosa.

Rancangan: Artikel_Glukosa_Rencana.md §A, §C2, §F langkah 5. Semua model dinilai pada pasangan yang sama: pasangan yang
gelombangnya tidak terbaca dibuang untuk semua model, jadi angka M0/M1 bisa sedikit berbeda dari Hasil_Glukosa_1/2.
Susunan analisis sama dengan M1: latih TRAIN–nilai VALIDATION (non-EXCLUDE) untuk dua tugas biner dan tiga kelas, lalu
kendali obat setara di ICU (Hasil_Glukosa_2.md). Split TEST tidak disentuh. Keluaran hanya agregat.

Butuh singgahan dari bangun_singgahan.py. Pakai lingkungan utama (README), dari akar repo:
    python glukosa_m2/jalankan_m2.py [--boot 1000] [--epoch 20]
        [--folder data/mimic-iv-ext-ecg-glucose] [--singgahan data/mimic-iv-ext-ecg-glucose/gelombang]
Uji jalan (angkanya tidak dipakai): --epoch 1 --boot 20.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "glukosa_m1"))
from analisis import analisis_tiga_kelas, analisis_utama  # noqa: E402
from analisis_obat import kendali_obat_setara  # noqa: E402
from cetak import tampil_hitungan  # noqa: E402
from jaringan import PERANGKAT, Resep, pembuat_cnn  # noqa: E402
from muat import BERKAS, TUGAS_BINER, muat_pasangan  # noqa: E402
from pemodelan import BENIH, MODEL  # noqa: E402
from singgahan import STATUS_DIPAKAI, Singgahan  # noqa: E402
from ukuran_kinerja import Bootstrap  # noqa: E402

NAMA_M2 = "M2 CNN gelombang"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    pasangan, model = siapkan(argumen)
    bootstrap = Bootstrap(argumen.boot, np.random.default_rng(BENIH[0]))
    print(f"\nPerangkat CNN: {PERANGKAT}; epoch {argumen.epoch}; bootstrap {argumen.boot} bergugus pasien.\n")
    for tugas in TUGAS_BINER:
        analisis_utama(pasangan, tugas, bootstrap, model)
    analisis_tiga_kelas(pasangan, bootstrap, model)
    for tugas in TUGAS_BINER:
        kendali_obat_setara(pasangan, tugas, bootstrap, model)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def baca_argumen():
    return pengurai_dasar().parse_args()


def pengurai_dasar():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--boot", type=int, default=1000)
    pengurai.add_argument("--epoch", type=int, default=Resep().epoch)
    pengurai.add_argument("--folder", default="data/mimic-iv-ext-ecg-glucose")
    pengurai.add_argument("--singgahan", default="data/mimic-iv-ext-ecg-glucose/gelombang")
    return pengurai


def siapkan(argumen):
    singgahan = Singgahan.buka(argumen.singgahan)
    pasangan = pasang_gelombang(muat_pasangan(Path(argumen.folder) / BERKAS, ["waveform_path"]), singgahan)
    model = {**MODEL, NAMA_M2: (["baris_gelombang"], pembuat_cnn(singgahan.sinyal, Resep(epoch=argumen.epoch)))}
    return pasangan, model


def pasang_gelombang(pasangan, singgahan):
    baris = pasangan.waveform_path.map(singgahan.baris).fillna(-1).astype(int).values
    dipakai = (baris >= 0) & np.isin(singgahan.status[np.maximum(baris, 0)], STATUS_DIPAKAI)
    print("# M0/M1/M2 — pasangan dengan gelombang terbaca\n")
    print(f"- pasangan: {tampil_hitungan(len(pasangan))}; gelombang terbaca {tampil_hitungan(dipakai.sum())}; "
          f"dibuang {tampil_hitungan((~dipakai).sum())}.")
    bagian = pasangan[dipakai].copy()
    bagian["baris_gelombang"] = baris[dipakai]
    return bagian


if __name__ == "__main__":
    main()
