from cetak import cetak_tabel_metrik, cetak_ukuran_set, dengan_selang, sel_auroc, tabel, tampil_hitungan
from muat import KELAS_HIPO, UKURAN_EKG, pasangan_tiga_kelas, pasangan_tugas, per_split
from pemodelan import BENIH, LIPATAN, MODEL, koefisien_terstandar, latih_dan_prediksi, peluang_luar_lipatan
from ukuran_kinerja import Penilaian, auroc_satu_lawan_sisa, selang_bootstrap


def cetak_ukuran_hilang(pasangan):
    bagian = pasangan[pasangan.bukan_exclude]
    print("## 0. Ukuran mesin EKG yang hilang (non-EXCLUDE, sesudah kode tidak-terukur dibuang)\n")
    tabel(["Ukuran", "Hilang", "% hilang"],
          [[k, tampil_hitungan(bagian[k].isna().sum()), f"{100 * bagian[k].isna().mean():.2f}"]
           for k in UKURAN_EKG])
    print()


def analisis_utama(pasangan, tugas, bootstrap, model=MODEL):
    latih, nilai = per_split(pasangan_tugas(pasangan[pasangan.bukan_exclude], tugas.kelas_positif))
    peluang = {m: latih_dan_prediksi(latih, nilai, kolom, buat)[:, 1] for m, (kolom, buat) in model.items()}
    print(f"## {tugas.nama} — latih TRAIN, nilai VALIDATION (non-EXCLUDE)\n")
    cetak_ukuran_set(latih, "TRAIN")
    cetak_ukuran_set(nilai, "VALIDATION")
    print()
    penilaian = Penilaian(nilai.y.values, nilai.subject_id.values, peluang)
    cetak_tabel_metrik(penilaian, bootstrap)
    cetak_kepekaan_validasi(nilai, penilaian, bootstrap)


def cetak_kepekaan_validasi(nilai, penilaian, bootstrap):
    subset = {
        "selisih ≤ 60 menit": nilai.abs_offset_minutes.le(60).values,
        "SQI GOOD": nilai.sqi_category.eq("GOOD").values,
        "konservatif (anjuran README)": nilai.konservatif.values,
    }
    print("Kepekaan (model yang sama, dinilai pada bagian VALIDATION), AUROC [95 % CI]:\n")
    isi = []
    for nama, pilih in subset.items():
        bagian = penilaian.bagian(pilih)
        isi.append([nama, tampil_hitungan(bagian.y.sum()), *sel_auroc(bagian, bootstrap)])
    tabel(["Subset", "n positif", *penilaian.peluang], isi)
    print()


def analisis_tiga_kelas(pasangan, bootstrap, model=MODEL):
    latih, nilai = per_split(pasangan_tiga_kelas(pasangan[pasangan.bukan_exclude]))
    y = nilai.y.values
    peluang = {m: latih_dan_prediksi(latih, nilai, kolom, buat) for m, (kolom, buat) in model.items()}
    selang = selang_bootstrap(Penilaian(y, nilai.subject_id.values, peluang), auroc_satu_lawan_sisa, bootstrap)
    print("## Tiga kelas (< 70 / 70–180 / > 180), sebanding Palanchuk 2026 — latih TRAIN, nilai VALIDATION\n")
    print(f"VALIDATION: hipo {tampil_hitungan((y == 0).sum())}, eu {tampil_hitungan((y == 1).sum())}, "
          f"hiper {tampil_hitungan((y == 2).sum())}.\n")
    kolom = ["hipo", "eu", "hiper", "makro"]
    isi = []
    for nama, P in peluang.items():
        titik = auroc_satu_lawan_sisa(y, P)
        isi.append([nama, *(dengan_selang(titik[k], selang[nama][k]) for k in kolom)])
    tabel(["Model", *(f"AUROC {k}" for k in kolom)], isi)
    print()


def kepekaan_obat_icu(pasangan, tugas, bootstrap):
    print(f"## Kepekaan kendali obat — {tugas.nama}: validasi silang {LIPATAN} lipatan bergugus pasien, "
          f"{len(BENIH)} benih (peluang dirata-rata), gabungan TRAIN/VALIDATION/TEST\n")
    subset = {
        "ICU, non-EXCLUDE (obat bisa aktif)": pasangan.di_icu & pasangan.bukan_exclude,
        "ICU, konservatif (obat terverifikasi bersih)": pasangan.di_icu & pasangan.konservatif,
    }
    for nama_subset, pilih in subset.items():
        bagian = pasangan_tugas(pasangan[pilih], tugas.kelas_positif)
        print(f"**{nama_subset}**\n")
        cetak_ukuran_set(bagian, "subset")
        print()
        peluang = {m: peluang_luar_lipatan(bagian, kolom, buat) for m, (kolom, buat) in MODEL.items()}
        cetak_tabel_metrik(Penilaian(bagian.y.values, bagian.subject_id.values, peluang), bootstrap)


def cetak_koefisien(pasangan):
    latih, _ = per_split(pasangan_tugas(pasangan[pasangan.bukan_exclude], KELAS_HIPO))
    koefisien = koefisien_terstandar(latih)
    print("## Koefisien terstandar M1 (LR), hipoglikemia lawan euglikemia, latih TRAIN\n")
    urut = sorted(koefisien, key=lambda k: -abs(koefisien[k]))
    tabel(["Ciri", "Koefisien (per 1 SD)"], [[k, f"{koefisien[k]:+.3f}"] for k in urut])
