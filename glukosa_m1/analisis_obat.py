import numpy as np
from sklearn.metrics import roc_auc_score

from cetak import SEL_KECIL, cetak_ukuran_set, dengan_selang, sel_auroc, tabel, tampil_hitungan
from muat import pasangan_tugas
from pemodelan import BENIH, LIPATAN, MODEL, peluang_luar_lipatan
from ukuran_kinerja import Penilaian, indeks_bootstrap

STATUS_BERSIH = "CLEAN"
STATUS_OBAT = [STATUS_BERSIH, "FLAG_MODERATE_RISK", "FLAG_HIGH_RISK", "EXCLUDE"]


def subset_icu_setara(pasangan):
    return pasangan[pasangan.di_icu
                    & pasangan.temporal_relationship.eq("ECG_BEFORE_GLUCOSE")
                    & pasangan.sqi_category.isin(["GOOD", "FAIR"])]


def strata_obat(bagian):
    status = bagian.intervention_status.values
    return {s: status == s for s in STATUS_OBAT}


def kendali_obat_setara(pasangan, tugas, bootstrap, model=MODEL):
    bagian = pasangan_tugas(subset_icu_setara(pasangan), tugas.kelas_positif)
    print(f"## {tugas.nama} — ICU, EKG sebelum glukosa, SQI GOOD/FAIR; satu model validasi silang "
          f"{LIPATAN} lipatan bergugus pasien × {len(BENIH)} benih, dinilai per status obat\n")
    cetak_ukuran_set(bagian, "subset")
    print()
    peluang = {m: peluang_luar_lipatan(bagian, kolom, buat) for m, (kolom, buat) in model.items()}
    penilaian = Penilaian(bagian.y.values, bagian.subject_id.values, peluang)
    strata = strata_obat(bagian)
    cetak_per_strata(penilaian, strata, bootstrap)
    cetak_selisih(penilaian, strata, bootstrap)


def cetak_per_strata(penilaian, strata, bootstrap):
    isi = []
    for nama, pilih in strata.items():
        bagian = penilaian.bagian(pilih)
        isi.append([nama, tampil_hitungan(len(bagian.y)), tampil_hitungan(bagian.y.sum()),
                    *sel_auroc(bagian, bootstrap)])
    tabel(["Status obat", "Pasangan", "Positif", *penilaian.peluang], isi)
    print("\nAUROC [95 % CI] di dalam tiap strata; \"—\" bila positif < 11.\n")


def cetak_selisih(penilaian, strata, bootstrap):
    y, bersih = penilaian.y, strata[STATUS_BERSIH]
    isi = []
    for status in STATUS_OBAT[1:]:
        obat = strata[status]
        if min(y[bersih].sum(), y[obat].sum()) < SEL_KECIL:
            isi.append([status, *(["—"] * len(penilaian.peluang))])
            continue
        selang = selang_selisih(penilaian, bersih, obat, bootstrap)
        isi.append([status, *(dengan_selang(selisih_auroc(y, p, bersih, obat), selang[m])
                              for m, p in penilaian.peluang.items())])
    tabel(["CLEAN − status", *penilaian.peluang], isi)
    print("\nSelisih AUROC [95 % CI]; bootstrap bergugus pasien atas seluruh subset, kedua AUROC pada ulangan "
          "yang sama; \"—\" bila positif < 11.\n")


def selisih_auroc(y, p, pilih_a, pilih_b):
    return roc_auc_score(y[pilih_a], p[pilih_a]) - roc_auc_score(y[pilih_b], p[pilih_b])


def selang_selisih(penilaian, pilih_a, pilih_b, bootstrap):
    y = penilaian.y
    ulangan = {m: [] for m in penilaian.peluang}
    for indeks in indeks_bootstrap(penilaian.gugus, bootstrap):
        a, b = pilih_a[indeks], pilih_b[indeks]
        if not (punya_dua_kelas(y[indeks][a]) and punya_dua_kelas(y[indeks][b])):
            continue
        for m, p in penilaian.peluang.items():
            ulangan[m].append(selisih_auroc(y[indeks], p[indeks], a, b))
    return {m: np.quantile(u, [0.025, 0.975]) for m, u in ulangan.items()}


def punya_dua_kelas(y):
    return 0 < y.sum() < len(y)
