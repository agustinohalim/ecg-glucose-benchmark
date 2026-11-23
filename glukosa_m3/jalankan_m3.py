"""M3 (model fondasi beku + probe L2) berdampingan dengan M0–M2 untuk artikel Glukosa; --model ecgfounder (M3,
Hasil_Glukosa_5.md) atau ecgjepa (M3b).

Rancangan: Artikel_Glukosa_Rencana.md §A dan §F langkah 5–6. Hanya probe yang dilatih; prediksi M0–M2 (jalankan_selisih.py,
Hasil_Glukosa_4.md) dan prediksi tambahan model (M3 untuk M3b) dibaca dari --prediksi dan dicocokkan baris demi baris
(y dan subject_id harus sama). Susunan: latih TRAIN–nilai VALIDATION (non-EXCLUDE) dan ICU setara dengan validasi silang
bergugus pasien, dua tugas biner; tiga kelas tidak. Selisih berpasangan terhadap pembanding utama model (M2 untuk M3, M3
untuk M3b), M1-GB, dan M0. Prediksi disimpan ke folder yang sama (di luar repo). Keluaran stdout hanya agregat. Split
TEST tidak disentuh.

Butuh emb_<model>.npy dari bangun_embedding.py. Pakai lingkungan utama (README), dari akar repo:
    python glukosa_m3/jalankan_m3.py [--model ecgfounder|ecgjepa]
        [--boot 1000] [--folder data/mimic-iv-ext-ecg-glucose]
        [--singgahan data/mimic-iv-ext-ecg-glucose/gelombang] [--prediksi data/mimic-iv-ext-ecg-glucose/prediksi]
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / "glukosa_m2"), str(AKAR_EKG / "glukosa_m1")]
from analisis_obat import cetak_per_strata, strata_obat, subset_icu_setara  # noqa: E402
from cetak import cetak_tabel_metrik, cetak_ukuran_set  # noqa: E402
from jalankan_m2 import pasang_gelombang  # noqa: E402
from model_fondasi import MODEL_FONDASI  # noqa: E402
from muat import BERKAS, TUGAS_BINER, muat_pasangan, pasangan_tugas, per_split  # noqa: E402
from pemodelan import BENIH, LIPATAN, latih_dan_prediksi, peluang_luar_lipatan  # noqa: E402
from prediksi import muat_prediksi, simpan_prediksi  # noqa: E402
from probe import pembuat_probe  # noqa: E402
from selisih import Perbandingan, baris_selisih, cetak_tabel_selisih  # noqa: E402
from singgahan import Singgahan  # noqa: E402
from ukuran_kinerja import Bootstrap, Penilaian  # noqa: E402

KODE_TUGAS = ["hipo", "hiper_berat"]
KOLOM_PROBE = ["baris_gelombang", "subject_id"]


@dataclass(frozen=True)
class Konteks:
    model: object
    buat: object
    bootstrap: Bootstrap
    prediksi: Path

    @property
    def perbandingan(self):
        m = self.model
        return [Perbandingan(m.nama, m.pembanding, f"{m.label} − {m.label_pembanding}"),
                Perbandingan(m.nama, "M1 ukuran EKG + M0 (GB)", f"{m.label} − M1 (GB)"),
                Perbandingan(m.nama, "M0 usia + jenis kelamin (LR)", f"{m.label} − M0")]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    model = MODEL_FONDASI[argumen.model]
    folder = Path(argumen.singgahan)
    pasangan = pasang_gelombang(muat_pasangan(Path(argumen.folder) / BERKAS, ["waveform_path"]),
                                Singgahan.buka(folder))
    konteks = Konteks(model, pembuat_probe(muat_embedding(folder, model)),
                      Bootstrap(argumen.boot, np.random.default_rng(BENIH[0])), Path(argumen.prediksi))
    print(f"\n{model.nama}; bootstrap {argumen.boot} bergugus pasien.\n")
    for kode, tugas in zip(KODE_TUGAS, TUGAS_BINER):
        probe_utama(pasangan, tugas, kode, konteks)
        probe_icu(pasangan, tugas, kode, konteks)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--model", choices=sorted(MODEL_FONDASI), default="ecgfounder")
    pengurai.add_argument("--boot", type=int, default=1000)
    pengurai.add_argument("--folder", default="data/mimic-iv-ext-ecg-glucose")
    pengurai.add_argument("--singgahan", default="data/mimic-iv-ext-ecg-glucose/gelombang")
    pengurai.add_argument("--prediksi", default="data/mimic-iv-ext-ecg-glucose/prediksi")
    return pengurai.parse_args()


def muat_embedding(folder, model):
    if not np.load(folder / model.berkas_selesai).all():
        raise SystemExit(f"Embedding {model.kode} belum lengkap; jalankan bangun_embedding.py dulu.")
    return np.load(folder / model.berkas_emb)


def probe_utama(pasangan, tugas, kode, konteks):
    latih, nilai = per_split(pasangan_tugas(pasangan[pasangan.bukan_exclude], tugas.kelas_positif))
    p = latih_dan_prediksi(latih, nilai, KOLOM_PROBE, konteks.buat)[:, 1]
    penilaian = gabung_prediksi(konteks, f"utama_{kode}", nilai, p)
    print(f"## {tugas.nama} — latih TRAIN, nilai VALIDATION (non-EXCLUDE)\n")
    cetak_tabel_metrik(penilaian, konteks.bootstrap)
    isi = [baris_selisih("VALIDATION", penilaian, konteks.perbandingan, konteks.bootstrap),
           baris_selisih("VALIDATION, konservatif", penilaian.bagian(nilai.konservatif.values),
                         konteks.perbandingan, konteks.bootstrap)]
    cetak_tabel_selisih(isi, konteks.perbandingan)


def probe_icu(pasangan, tugas, kode, konteks):
    bagian = pasangan_tugas(subset_icu_setara(pasangan), tugas.kelas_positif)
    print(f"## {tugas.nama} — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; validasi silang {LIPATAN} lipatan bergugus "
          f"pasien × {len(BENIH)} benih, dinilai per status obat\n")
    cetak_ukuran_set(bagian, "subset")
    print()
    p = peluang_luar_lipatan(bagian, KOLOM_PROBE, konteks.buat)
    penilaian = gabung_prediksi(konteks, f"icu_{kode}", bagian, p)
    strata = strata_obat(bagian)
    cetak_per_strata(penilaian, strata, konteks.bootstrap)
    isi = [baris_selisih(status, penilaian.bagian(pilih), konteks.perbandingan, konteks.bootstrap)
           for status, pilih in strata.items()]
    cetak_tabel_selisih(isi, konteks.perbandingan)


def gabung_prediksi(konteks, dasar, bagian, p):
    peluang = muat_prediksi(konteks.prediksi / f"{dasar}.npz", bagian)
    for akhiran in konteks.model.prediksi_tambahan:
        peluang.update(muat_prediksi(konteks.prediksi / f"{dasar}{akhiran}.npz", bagian))
    simpan_prediksi(konteks.prediksi / f"{dasar}{konteks.model.akhiran}.npz", bagian, {konteks.model.nama: p})
    return Penilaian(bagian.y.values, bagian.subject_id.values, {**peluang, konteks.model.nama: p})


if __name__ == "__main__":
    main()
