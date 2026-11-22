"""Uji kontaminasi pra-latih Glukosa menurut protokol Artikel_Glukosa_Rencana.md §F langkah 9 (komit 82bee79), dicatat
sebelum embedding ECG-FM dibangun.

M3c (ECG-FM, dilatih-awal pada bagian `train` MIMIC-IV-ECG menurut daftar split penulisnya) dan pembanding bersih M3b
(ECG-JEPA), M3 (ECGFounder), M1-GB (ukuran mesin) dilatih pada TRAIN non-EXCLUDE dengan resep Hasil 8, lalu dinilai pada
VALIDATION ∪ TEST non-EXCLUDE. Paparan per pasangan: EKG berstatus `train` di daftar split ECG-FM = dilihat; `valid`
atau `test` = tidak dilihat; tidak tercantum = dikeluarkan dari kontras. Estimand utama per tugas:
DiD = [AUROC M3c(dilihat) − AUROC M3c(tidak)] − [AUROC M3b(dilihat) − AUROC M3b(tidak)], CI bootstrap bergugus pasien.
Keluaran stdout hanya agregat (hitungan 1–10 dicetak "<11", metrik disembunyikan bila positif < 11); prediksi disimpan di
--prediksi (di luar repo). Dijalankan sekali.

Pakai lingkungan ECG-FM (README), dari akar repo:
    python glukosa_test/jalankan_kontaminasi.py [--boot 1000]
        [--split models/ECG-FM/meta_split_mimic_iv_ecg.csv]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / "glukosa_m3"), str(AKAR_EKG / "glukosa_m2"), str(AKAR_EKG / "glukosa_m1")]
from cetak import SEL_KECIL, dengan_selang, tabel, tampil_hitungan  # noqa: E402
from jalankan_m2 import pasang_gelombang, pengurai_dasar  # noqa: E402
from jalankan_test import KODE_TUGAS, KOLOM_PROBE, NAMA_M1_GB, cetak_auroc, per_tiga_split  # noqa: E402
from model_fondasi import ECGFM, ECGFOUNDER, ECGJEPA  # noqa: E402
from muat import BERKAS, TUGAS_BINER, muat_pasangan, pasangan_tugas  # noqa: E402
from pemodelan import BENIH, MODEL  # noqa: E402
from prediksi import simpan_prediksi  # noqa: E402
from probe import pembuat_probe  # noqa: E402
from selisih import Perbandingan, baris_selisih, cetak_tabel_selisih  # noqa: E402
from singgahan import Singgahan  # noqa: E402
from ukuran_kinerja import Bootstrap, Penilaian, indeks_bootstrap  # noqa: E402

DILIHAT, TIDAK_DILIHAT, TAK_TERCANTUM = "dilihat", "tidak dilihat", "tak tercantum"
PAPARAN_SPLIT = {"train": DILIHAT, "valid": TIDAK_DILIHAT, "test": TIDAK_DILIHAT}
TERKONTAMINASI = ECGFM.nama
PEMBANDING = [ECGJEPA.nama, ECGFOUNDER.nama, NAMA_M1_GB]
SELISIH_KESELURUHAN = [Perbandingan(ECGFM.nama, ECGJEPA.nama, "M3c − M3b")]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    pasangan = pasang_gelombang(muat_pasangan(Path(argumen.folder) / BERKAS, ["waveform_path"]),
                                Singgahan.buka(argumen.singgahan))
    pasangan["paparan"] = tandai_paparan(pasangan.waveform_path, Path(argumen.split))
    model = {NAMA_M1_GB: MODEL[NAMA_M1_GB], **model_probe(Path(argumen.singgahan))}
    print(f"\nUji kontaminasi, protokol §F langkah 9; bootstrap {argumen.boot} bergugus pasien.\n")
    for kode, tugas in zip(KODE_TUGAS, TUGAS_BINER):
        latih, validasi, test = per_tiga_split(pasangan_tugas(pasangan[pasangan.bukan_exclude], tugas.kelas_positif))
        nilai = pd.concat([validasi, test])
        print(f"## {tugas.nama} — latih TRAIN, nilai VALIDATION ∪ TEST (non-EXCLUDE)\n")
        peluang = latih_dan_nilai(latih, validasi, nilai, model)
        simpan_prediksi(Path(argumen.prediksi) / f"kontaminasi_{kode}.npz", nilai, peluang)
        penilaian = Penilaian(nilai.y.values, nilai.subject_id.values, peluang)
        cetak_kelompok(nilai)
        cetak_kontras(nilai, penilaian, Bootstrap(argumen.boot, np.random.default_rng(BENIH[0])))
        cetak_keseluruhan(nilai, penilaian, Bootstrap(argumen.boot, np.random.default_rng(BENIH[0])))
    print(f"\nWaktu jalan: {(time.time() - mulai) / 60:.1f} menit.")


def baca_argumen():
    pengurai = pengurai_dasar()
    pengurai.add_argument("--prediksi", default="data/mimic-iv-ext-ecg-glucose/prediksi")
    pengurai.add_argument("--split", default="models/ECG-FM/meta_split_mimic_iv_ecg.csv")
    return pengurai.parse_args()


def tandai_paparan(jalur_gelombang, berkas_split):
    split = pd.read_csv(berkas_split, usecols=["source_path", "split"]).set_index("source_path").split
    paparan = jalur_gelombang.map(split).map(PAPARAN_SPLIT).fillna(TAK_TERCANTUM)
    print("\n### Paparan pra-latih ECG-FM (semua pasangan)\n")
    tabel(["Paparan", "Pasangan"], [[k, tampil_hitungan(n)] for k, n in paparan.value_counts().items()])
    return paparan.values


def model_probe(folder):
    hasil = {}
    for spesifikasi in (ECGFOUNDER, ECGJEPA, ECGFM):
        if not np.load(folder / spesifikasi.berkas_selesai).all():
            raise SystemExit(f"Embedding {spesifikasi.kode} belum lengkap.")
        emb = np.load(folder / spesifikasi.berkas_emb, mmap_mode="r")
        print(f"- embedding {spesifikasi.kode}: baris {len(emb)}, semua terhingga {bool(np.isfinite(emb).all())}")
        hasil[spesifikasi.nama] = (KOLOM_PROBE, pembuat_probe(emb))
    return hasil


def latih_dan_nilai(latih, validasi, nilai, model):
    peluang = {}
    for nama, (kolom, buat) in model.items():
        terlatih = buat(BENIH[0]).fit(latih[kolom].values, latih.y.values)
        if hasattr(terlatih, "C_"):
            print(f"- {nama}: C terpilih {terlatih.C_}")
        cetak_auroc(nama, validasi, terlatih.predict_proba(validasi[kolom].values)[:, 1])
        peluang[nama] = terlatih.predict_proba(nilai[kolom].values)[:, 1]
    print()
    return peluang


def cetak_kelompok(nilai):
    print("### Kelompok paparan di VALIDATION ∪ TEST\n")
    isi = []
    for paparan, bagian in nilai.groupby("paparan"):
        isi.append([paparan, tampil_hitungan(len(bagian)), tampil_hitungan(bagian.subject_id.nunique()),
                    tampil_hitungan(bagian.y.sum()), f"{bagian.age.mean():.1f}", f"{100 * bagian.laki.mean():.1f}",
                    f"{100 * bagian.di_icu.mean():.1f}"])
    tabel(["Paparan", "Pasangan", "Pasien", "Positif", "Usia rerata", "Laki-laki %", "ICU %"], isi)
    print()


def auroc_kelompok(y, peluang, dilihat):
    return {(m, g): roc_auc_score(y[pilih], p[pilih])
            for m, p in peluang.items() for g, pilih in ((DILIHAT, dilihat), (TIDAK_DILIHAT, ~dilihat))}


def did(auroc, pembanding):
    beda = {m: auroc[(m, DILIHAT)] - auroc[(m, TIDAK_DILIHAT)] for m in [TERKONTAMINASI, pembanding]}
    return beda[TERKONTAMINASI] - beda[pembanding]


def cetak_kontras(nilai, penilaian, bootstrap):
    tercantum = nilai.paparan.ne(TAK_TERCANTUM).values
    bagian = penilaian.bagian(tercantum)
    dilihat = nilai.paparan.values[tercantum] == DILIHAT
    positif = [bagian.y[dilihat].sum(), bagian.y[~dilihat].sum()]
    print("### Kontras dilihat lawan tidak dilihat (DiD utama: pembanding M3b)\n")
    if min(positif) < SEL_KECIL:
        print("Positif < 11 pada salah satu kelompok; kontras tidak dicetak.\n")
        return
    titik = auroc_kelompok(bagian.y, bagian.peluang, dilihat)
    ulangan = []
    for indeks in indeks_bootstrap(bagian.gugus, bootstrap):
        y, d = bagian.y[indeks], dilihat[indeks]
        if 0 < y[d].sum() < d.sum() and 0 < y[~d].sum() < (~d).sum():
            ulangan.append(auroc_kelompok(y, {m: p[indeks] for m, p in bagian.peluang.items()}, d))
    selang = {k: np.quantile([u[k] for u in ulangan], [0.025, 0.975]) for k in titik}
    tabel(["Model", "AUROC dilihat", "AUROC tidak dilihat", "Dilihat − tidak"],
          [[m, dengan_selang(titik[(m, DILIHAT)], selang[(m, DILIHAT)]),
            dengan_selang(titik[(m, TIDAK_DILIHAT)], selang[(m, TIDAK_DILIHAT)]),
            dengan_selang(titik[(m, DILIHAT)] - titik[(m, TIDAK_DILIHAT)],
                          np.quantile([u[(m, DILIHAT)] - u[(m, TIDAK_DILIHAT)] for u in ulangan], [0.025, 0.975]))]
           for m in bagian.peluang])
    print()
    tabel(["DiD (M3c − pembanding)", "Nilai [95 % CI]"],
          [[f"{'utama' if k == 0 else 'sekunder'}: {p}",
            dengan_selang(did(titik, p), np.quantile([did(u, p) for u in ulangan], [0.025, 0.975]))]
           for k, p in enumerate(PEMBANDING)])
    print(f"\nUlangan bootstrap terpakai: {len(ulangan)}; positif dilihat {tampil_hitungan(positif[0])}, "
          f"tidak dilihat {tampil_hitungan(positif[1])}.\n")


def cetak_keseluruhan(nilai, penilaian, bootstrap):
    print("### M3c − M3b keseluruhan (sekunder)\n")
    test = nilai.split.eq("TEST").values
    isi = [baris_selisih("VALIDATION ∪ TEST", penilaian, SELISIH_KESELURUHAN, bootstrap),
           baris_selisih("TEST", penilaian.bagian(test), SELISIH_KESELURUHAN, bootstrap)]
    cetak_tabel_selisih(isi, SELISIH_KESELURUHAN)


if __name__ == "__main__":
    main()
