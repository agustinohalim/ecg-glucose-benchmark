"""Penilaian TEST artikel Glukosa menurut protokol yang dikunci sebelum TEST dibuka (Artikel_Glukosa_Rencana.md §F
langkah 7, komit 45ff2cb).

Semua model dilatih ulang pada TRAIN non-EXCLUDE dengan resep dan benih Hasil 3–7; prediksi VALIDATION dan TEST dari
model yang sama. M2b = rerata 3 benih, dikalibrasi ulang dengan regresi logistik tanpa penalti pada logit yang dipasang
di VALIDATION lalu diterapkan ke TEST. Uji utama per tugas: M3b − M1-GB di TEST; sisanya sekunder. ICU dan tiga kelas
tidak dinilai. Dicetak: AUROC VALIDATION tiap model (cek ulang terhadap Hasil 5–7), parameter kalibrasi ulang, tabel
metrik TEST, selisih utama dan sekunder. Prediksi disimpan ke --prediksi (di luar repo). Keluaran stdout hanya agregat.

Pakai lingkungan utama (README), dari akar repo:
    python glukosa_test/jalankan_test.py [--boot 1000] [--epoch 20]
        [--folder data/mimic-iv-ext-ecg-glucose] [--singgahan data/mimic-iv-ext-ecg-glucose/gelombang]
        [--prediksi data/mimic-iv-ext-ecg-glucose/prediksi]
Uji jalan (angkanya tidak dipakai): --epoch 1 --boot 20.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / "glukosa_m3"), str(AKAR_EKG / "glukosa_m2"), str(AKAR_EKG / "glukosa_m1")]
from cetak import cetak_tabel_metrik, cetak_ukuran_set  # noqa: E402
from jalankan_m2 import NAMA_M2, pengurai_dasar, siapkan  # noqa: E402
from model_fondasi import ECGFOUNDER, ECGJEPA  # noqa: E402
from muat import TUGAS_BINER, pasangan_tugas  # noqa: E402
from pemodelan import BENIH  # noqa: E402
from prediksi import simpan_prediksi  # noqa: E402
from probe import pembuat_probe  # noqa: E402
from selisih import Perbandingan, baris_selisih, cetak_tabel_selisih  # noqa: E402
from ukuran_kinerja import Bootstrap, Penilaian  # noqa: E402

KODE_TUGAS = ["hipo", "hiper_berat"]
KOLOM_PROBE = ["baris_gelombang", "subject_id"]
NAMA_M1_GB = "M1 ukuran EKG + M0 (GB)"
NAMA_M2B = f"M2 CNN gelombang, rerata {len(BENIH)} benih"
NAMA_M2B_KAL = f"{NAMA_M2B}, terkalibrasi ulang"
UTAMA = [Perbandingan(ECGJEPA.nama, NAMA_M1_GB, "M3b − M1 (GB)")]
SEKUNDER = [
    Perbandingan(NAMA_M2B, NAMA_M1_GB, "M2b − M1 (GB)"),
    Perbandingan(ECGFOUNDER.nama, NAMA_M1_GB, "M3 − M1 (GB)"),
    Perbandingan(ECGJEPA.nama, NAMA_M2B, "M3b − M2b"),
    Perbandingan(ECGJEPA.nama, ECGFOUNDER.nama, "M3b − M3"),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    pasangan, model = siapkan(argumen)
    pembuat_m2 = model.pop(NAMA_M2)[1]
    model.update(model_probe(Path(argumen.singgahan)))
    bootstrap = Bootstrap(argumen.boot, np.random.default_rng(BENIH[0]))
    print(f"\nTEST, protokol §F langkah 7; epoch {argumen.epoch}; bootstrap {argumen.boot} bergugus pasien.\n")
    for kode, tugas in zip(KODE_TUGAS, TUGAS_BINER):
        latih, validasi, test = per_tiga_split(pasangan_tugas(pasangan[pasangan.bukan_exclude], tugas.kelas_positif))
        print(f"## {tugas.nama} — latih TRAIN, nilai TEST (non-EXCLUDE)\n")
        cetak_ukuran_set(test, "TEST")
        pv, pt = latih_tunggal(latih, (validasi, test), model)
        m2_validasi, m2_test = latih_m2(latih, (validasi, test), pembuat_m2)
        pv.update(m2_validasi)
        pt.update(m2_test)
        pt[NAMA_M2B_KAL] = kalibrasi_ulang(pv[NAMA_M2B], validasi.y.values, pt[NAMA_M2B])
        simpan_prediksi(Path(argumen.prediksi) / f"test_{kode}.npz", test, pt)
        simpan_prediksi(Path(argumen.prediksi) / f"test_{kode}_validasi.npz", validasi, pv)
        cetak_test(test, Penilaian(test.y.values, test.subject_id.values, pt), bootstrap)
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def baca_argumen():
    pengurai = pengurai_dasar()
    pengurai.add_argument("--prediksi", default="data/mimic-iv-ext-ecg-glucose/prediksi")
    return pengurai.parse_args()


def model_probe(folder):
    hasil = {}
    for spesifikasi in (ECGFOUNDER, ECGJEPA):
        if not np.load(folder / spesifikasi.berkas_selesai).all():
            raise SystemExit(f"Embedding {spesifikasi.kode} belum lengkap.")
        hasil[spesifikasi.nama] = (KOLOM_PROBE, pembuat_probe(np.load(folder / spesifikasi.berkas_emb, mmap_mode="r")))
    return hasil


def per_tiga_split(bagian):
    return tuple(bagian[bagian.split == nama] for nama in ("TRAIN", "VALIDATION", "TEST"))


def latih_tunggal(latih, sasaran, model):
    pv, pt = {}, {}
    for nama, (kolom, buat) in model.items():
        terlatih = buat(BENIH[0]).fit(latih[kolom].values, latih.y.values)
        pv[nama], pt[nama] = (terlatih.predict_proba(s[kolom].values)[:, 1] for s in sasaran)
        cetak_auroc(nama, sasaran[0], pv[nama])
    return pv, pt


def latih_m2(latih, sasaran, buat):
    per_benih = []
    for benih in BENIH:
        terlatih = buat(benih).fit(latih[["baris_gelombang"]].values, latih.y.values)
        per_benih.append([terlatih.predict_proba(s[["baris_gelombang"]].values)[:, 1] for s in sasaran])
        cetak_auroc(f"{NAMA_M2} benih {benih}", sasaran[0], per_benih[-1][0])
    hasil = tuple({NAMA_M2: per_benih[0][i], NAMA_M2B: np.mean([p[i] for p in per_benih], axis=0)}
                  for i in range(len(sasaran)))
    cetak_auroc(NAMA_M2B, sasaran[0], hasil[0][NAMA_M2B])
    return hasil


def cetak_auroc(nama, validasi, p):
    print(f"- VALIDATION {nama}: AUROC {roc_auc_score(validasi.y.values, p):.3f}", flush=True)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def kalibrasi_ulang(p_validasi, y_validasi, p_test):
    model = LogisticRegression(C=1e10, max_iter=1000).fit(logit(p_validasi), y_validasi)
    print(f"- Kalibrasi ulang M2b (dipasang di VALIDATION): intersep {model.intercept_[0]:.3f}, "
          f"kemiringan {model.coef_[0, 0]:.3f}\n")
    return model.predict_proba(logit(p_test))[:, 1]


def cetak_test(test, penilaian, bootstrap):
    cetak_tabel_metrik(penilaian, bootstrap)
    konservatif = test.konservatif.values
    for judul, perbandingan in (("Uji utama", UTAMA), ("Sekunder (deskriptif)", SEKUNDER)):
        print(f"### {judul}\n")
        isi = [baris_selisih("TEST", penilaian, perbandingan, bootstrap),
               baris_selisih("TEST, konservatif", penilaian.bagian(konservatif), perbandingan, bootstrap)]
        cetak_tabel_selisih(isi, perbandingan)


if __name__ == "__main__":
    main()
