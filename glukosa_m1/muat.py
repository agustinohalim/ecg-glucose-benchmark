"""Memuat pasangan MIMIC-IV-Ext-ECG-Glucose v1.0.0 dan membentuk label tugas; tidak ada kolom turunan glukosa sebagai masukan."""
from collections import namedtuple

import numpy as np
import pandas as pd

BERKAS = "mimiciv_ecg_glucose_aligned.csv"
UKURAN_EKG = ["heart_rate_bpm", "rr_interval", "qrs_duration_ms", "qt_raw_ms", "pr_interval_ms",
              "jt_interval_ms", "p_axis", "qrs_axis", "t_axis", "qtc_bazett_ms", "qtc_fridericia_ms",
              "qtc_framingham_ms", "qtc_hodges_ms"]
DEMOGRAFI = ["age", "laki"]
KOLOM = ["subject_id", "split", "glycemic_class", "during_icu_stay", "record_usability",
         "temporal_relationship", "intervention_status", "abs_offset_minutes", "sqi_category", "gender",
         "age", *UKURAN_EKG]
KODE_TIDAK_TERUKUR = 10000
KELAS_HIPO = ["SEVERE_HYPO", "HYPO"]
KELAS_HIPER = ["HYPER", "SEVERE_HYPER"]
Tugas = namedtuple("Tugas", ["nama", "kelas_positif"])
TUGAS_BINER = [
    Tugas("Hipoglikemia (< 70) lawan euglikemia (70–180)", KELAS_HIPO),
    Tugas("Hiperglikemia berat (> 250) lawan euglikemia (70–180)", ["SEVERE_HYPER"]),
]


def muat_pasangan(jalur, kolom_tambahan=()):
    if not jalur.exists():
        raise SystemExit(f"Berkas tidak ada: {jalur}")
    pasangan = pd.read_csv(jalur, usecols=[*KOLOM, *kolom_tambahan], dtype={"during_icu_stay": "string"})
    ukuran = pasangan[UKURAN_EKG]
    pasangan[UKURAN_EKG] = ukuran.mask(ukuran.abs() >= KODE_TIDAK_TERUKUR)
    pasangan["rr_interval"] = pasangan.rr_interval.mask(pasangan.rr_interval <= 0)
    pasangan["laki"] = pasangan.gender.eq("M").astype(float)
    pasangan["di_icu"] = pasangan.during_icu_stay.str.lower().isin(["true"])
    pasangan["bukan_exclude"] = pasangan.record_usability.ne("EXCLUDE")
    pasangan["konservatif"] = (pasangan.record_usability.isin(["IDEAL", "USABLE"])
                               & pasangan.temporal_relationship.eq("ECG_BEFORE_GLUCOSE")
                               & pasangan.intervention_status.eq("CLEAN"))
    return pasangan


def pasangan_tugas(pasangan, kelas_positif):
    bagian = pasangan[pasangan.glycemic_class.isin([*kelas_positif, "EUGLYCEMIC"])].copy()
    bagian["y"] = bagian.glycemic_class.isin(kelas_positif).astype(int)
    return bagian


def pasangan_tiga_kelas(pasangan):
    bagian = pasangan.copy()
    bagian["y"] = np.select([bagian.glycemic_class.isin(KELAS_HIPO), bagian.glycemic_class.isin(KELAS_HIPER)],
                            [0, 2], default=1)
    return bagian


def per_split(bagian):
    return bagian[bagian.split == "TRAIN"], bagian[bagian.split == "VALIDATION"]
