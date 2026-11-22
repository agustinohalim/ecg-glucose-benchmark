"""Hitung ulang label MIMIC-IV-Ext-ECG-Glucose v1.0.0 (Azam & Singh, PhysioNet, lisensi credentialed 1.5.0).

Membaca mimiciv_ecg_glucose_aligned.csv dari data/mimic-iv-ext-ecg-glucose/ (di luar repo) dan
mencetak tabel Markdown agregat yang disalin apa adanya ke Hasil_Glukosa_0.md. Tidak ada pemodelan.
Hitungan 1-10 dicetak "<11" (beserta persentasenya), karena keluaran ini dibaca asisten AI dan DUA
PhysioNet melarang data tingkat pasien keluar dari laptop.

Pakai:
    python glukosa_hitung_label.py [folder_data]
"""
import sys
from pathlib import Path

import pandas as pd

FOLDER = Path(sys.argv[1] if len(sys.argv) > 1 else "data/mimic-iv-ext-ecg-glucose")
BERKAS = "mimiciv_ecg_glucose_aligned.csv"
KOLOM = ["subject_id", "ecg_study_id", "split", "glucose_mg_dl", "glycemic_class", "during_icu_stay",
         "record_usability", "temporal_relationship", "intervention_status"]
KELAS = ["SEVERE_HYPO", "HYPO", "EUGLYCEMIC", "HYPER", "SEVERE_HYPER"]
KELAS_HIPO = ["SEVERE_HYPO", "HYPO"]
BATAS_BAWAH_KELAS_MG_DL = [54, 70, 181, 251]
SPLIT = ["TRAIN", "VALIDATION", "TEST"]
SEL_KECIL = 11
SUBSET = {
    "semua non-EXCLUDE": ["bukan_exclude"],
    "konservatif (anjuran README)": ["konservatif"],
    "ICU, non-EXCLUDE": ["bukan_exclude", "di_icu"],
    "ICU, konservatif (obat terverifikasi)": ["konservatif", "di_icu"],
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pasangan = muat_pasangan(FOLDER / BERKAS)
    cetak_ukuran(pasangan)
    cetak_cek_label(pasangan)
    cetak_silang("3. Kelas glikemik per split (semua baris)", pasangan.kelas, pasangan.split)

    bukan_exclude = pasangan[pasangan.bukan_exclude]
    cetak_silang("4. Kelas glikemik ICU vs luar ICU (non-EXCLUDE)", bukan_exclude.kelas, bukan_exclude.tempat)
    cetak_sasaran_utama(pasangan)
    cetak_silang("6. intervention_status ICU vs luar ICU (semua baris)",
                 pasangan.intervention_status, pasangan.tempat)


def muat_pasangan(jalur):
    if not jalur.exists():
        raise SystemExit(f"Berkas tidak ada: {jalur}")
    pasangan = pd.read_csv(jalur, usecols=KOLOM, dtype={"during_icu_stay": "string"})
    pasangan["kelas"] = pasangan.glycemic_class.astype(pd.CategoricalDtype(KELAS, ordered=True))
    pasangan["split"] = pasangan.split.astype(pd.CategoricalDtype(SPLIT, ordered=True))
    pasangan["di_icu"] = pasangan.during_icu_stay.str.lower().isin(["true"])
    pasangan["tempat"] = pasangan.di_icu.map({True: "ICU", False: "luar ICU"})
    pasangan["bukan_exclude"] = pasangan.record_usability.ne("EXCLUDE")
    pasangan["konservatif"] = (pasangan.record_usability.isin(["IDEAL", "USABLE"])
                               & pasangan.temporal_relationship.eq("ECG_BEFORE_GLUCOSE")
                               & pasangan.intervention_status.eq("CLEAN"))
    return pasangan


def cetak_ukuran(pasangan):
    print("## 1. Ukuran berkas\n")
    tabel(["Baris (pasangan)", "Pasien", "Studi EKG unik"],
          [[len(pasangan), pasangan.subject_id.nunique(), pasangan.ecg_study_id.nunique()]])
    pasien_lintas_split = int((pasangan.groupby("subject_id").split.nunique() > 1).sum())
    print(f"\nPasien di lebih dari satu split: {pasien_lintas_split}.\n")


def cetak_cek_label(pasangan):
    batas = [float("-inf"), *BATAS_BAWAH_KELAS_MG_DL, float("inf")]
    kelas_hitung = pd.cut(pasangan.glucose_mg_dl, bins=batas, right=False, labels=KELAS).astype("string")
    beda = int(kelas_hitung.ne(pasangan.glycemic_class.astype("string")).sum())
    print("## 2. Cek label\n")
    print(f"Kelas dihitung ulang dari `glucose_mg_dl` (batas bawah {BATAS_BAWAH_KELAS_MG_DL} mg/dL) berbeda "
          f"dari `glycemic_class` pada {beda} baris. Rentang glukosa {pasangan.glucose_mg_dl.min():g}–"
          f"{pasangan.glucose_mg_dl.max():g} mg/dL; glukosa kosong {int(pasangan.glucose_mg_dl.isna().sum())}.\n")


def cetak_silang(judul, kelompok_baris, kelompok_kolom):
    silang = pd.crosstab(kelompok_baris, kelompok_kolom)
    print(f"## {judul}\n")
    tabel([silang.index.name, *silang.columns, "Jumlah"],
          [[nama, *map(tampil_hitungan, isi), tampil_hitungan(isi.sum())] for nama, isi in silang.iterrows()])
    print()


def cetak_sasaran_utama(pasangan):
    print("## 5. Sasaran utama: hipoglikemia (< 70 mg/dL) vs euglikemia (70–180 mg/dL)\n")
    isi = []
    for nama, syarat in SUBSET.items():
        isi += baris_sasaran(nama, pasangan[pasangan[syarat].all(axis=1)])
    tabel(["Subset", "Split", "n hipo", "n eu", "% hipo", "Pasien dengan ≥ 1 hipo"], isi)
    print()


def baris_sasaran(nama, bagian):
    return [[nama, split, *ringkas_sasaran(bagian[bagian.split == split])] for split in SPLIT]


def ringkas_sasaran(bagian):
    hipo = bagian.glycemic_class.isin(KELAS_HIPO)
    n_hipo = int(hipo.sum())
    n_eu = int(bagian.glycemic_class.eq("EUGLYCEMIC").sum())
    persen_hipo = f"{100 * n_hipo / (n_hipo + n_eu):.2f}" if n_hipo >= SEL_KECIL else "—"
    return [tampil_hitungan(n_hipo), tampil_hitungan(n_eu), persen_hipo,
            tampil_hitungan(bagian.subject_id[hipo].nunique())]


def tabel(judul_kolom, isi_baris):
    print("| " + " | ".join(map(str, judul_kolom)) + " |")
    print("|" + " --- |" * len(judul_kolom))
    for baris in isi_baris:
        print("| " + " | ".join(map(str, baris)) + " |")


def tampil_hitungan(n):
    return f"<{SEL_KECIL}" if 0 < n < SEL_KECIL else str(int(n))


if __name__ == "__main__":
    main()
