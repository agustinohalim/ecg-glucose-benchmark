"""Unit segar langkah 19a (protokol Artikel_Glukosa_Rencana.md §F langkah 19): EKG kohort Glukosa yang ada di daftar
split ECG-FM, dari pasien di luar 44.116 pasien uji keanggotaan langkah 16, satu EKG per pasien (benih 20261120). Peran
`uji`: semua pasien tidak dilihat dan pasien dilihat sama banyak; peran `setel`: pasien dilihat lain sama banyak, untuk
penyetelan kepadatan 19c. Tabel disimpan di folder audit di luar repo dan tidak dibuka di sesi asisten."""
from pathlib import Path

import numpy as np
import pandas as pd

from jalankan_m2 import pasang_gelombang
from muat import BERKAS, muat_pasangan
from penugasan_kontrol import FOLDER_REPLIKASI, muat_unit, pilihan_satu_per_pasien
from singgahan import Singgahan
from statistik_keanggotaan import BENIH, DILIHAT
from tugas_pernyataan import tabel_ekg

FOLDER_AUDIT = Path("data/audit2")
BERKAS_UNIT_SEGAR = f"unit_segar_b{BENIH}.csv"
KOLOM_SIMPAN = ["subject_id", "baris_gelombang", "waveform_path", "split_ecgfm", "dilihat", "peran", "age", "laki",
                "heart_rate_bpm", "qrs_duration_ms"]


def tambah_argumen_unit(pengurai):
    pengurai.add_argument("--folder", default="data/mimic-iv-ext-ecg-glucose")
    pengurai.add_argument("--singgahan", default="data/mimic-iv-ext-ecg-glucose/gelombang")
    pengurai.add_argument("--split", default="models/ECG-FM/meta_split_mimic_iv_ecg.csv")
    pengurai.add_argument("--replikasi", default=str(FOLDER_REPLIKASI))
    pengurai.add_argument("--keluaran", default=str(FOLDER_AUDIT))
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    pengurai.add_argument("--pekerja", type=int, default=4)


def muat_atau_susun_unit(argumen):
    jalur = Path(argumen.keluaran) / BERKAS_UNIT_SEGAR
    if not jalur.exists():
        jalur.parent.mkdir(parents=True, exist_ok=True)
        susun_unit(ekg_kohort(argumen), pasien_langkah16(argumen.replikasi)).to_csv(jalur, index=False)
    return pd.read_csv(jalur)


def ekg_kohort(argumen):
    pasangan = pasang_gelombang(muat_pasangan(Path(argumen.folder) / BERKAS, ["waveform_path"]),
                                Singgahan.buka(argumen.singgahan))
    split = pd.read_csv(argumen.split, usecols=["source_path", "split"]).set_index("source_path").split
    pasangan["split_ecgfm"] = pasangan.waveform_path.map(split).values
    return tabel_ekg(pasangan.dropna(subset=["split_ecgfm"]))


def pasien_langkah16(folder_replikasi):
    return set(pilihan_satu_per_pasien(muat_unit(folder_replikasi)).subject_id)


def susun_unit(ekg, dikecualikan):
    rng = np.random.default_rng(BENIH)
    calon = ekg[~ekg.subject_id.isin(dikecualikan)]
    satu = calon.iloc[rng.permutation(len(calon))].drop_duplicates("subject_id")
    satu = satu.assign(dilihat=satu.split_ecgfm.eq(DILIHAT))
    tidak, dilihat = satu[~satu.dilihat], satu[satu.dilihat]
    urutan = rng.permutation(len(dilihat))[:2 * len(tidak)]
    uji = pd.concat([tidak, dilihat.iloc[urutan[:len(tidak)]]]).assign(peran="uji")
    setel = dilihat.iloc[urutan[len(tidak):]].assign(peran="setel")
    return pd.concat([uji, setel]).sort_values("baris_gelombang")[KOLOM_SIMPAN].reset_index(drop=True)


def unit_uji(unit):
    return unit[unit.peran.eq("uji")].reset_index(drop=True)
