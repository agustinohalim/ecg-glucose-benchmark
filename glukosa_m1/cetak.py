"""Pencetak tabel Markdown agregat; hitungan 1-10 dicetak "<11" dan metrik disembunyikan bila positif < 11."""
from sklearn.metrics import roc_auc_score

from ukuran_kinerja import metrik_lengkap, metrik_terurut, selang_bootstrap

SEL_KECIL = 11


def tabel(judul_kolom, isi_baris):
    print("| " + " | ".join(map(str, judul_kolom)) + " |")
    print("|" + " --- |" * len(judul_kolom))
    for baris in isi_baris:
        print("| " + " | ".join(map(str, baris)) + " |")


def tampil_hitungan(n):
    return f"<{SEL_KECIL}" if 0 < n < SEL_KECIL else str(int(n))


def dengan_selang(nilai, selang):
    return f"{nilai:.3f} [{selang[0]:.3f}, {selang[1]:.3f}]"


def cetak_ukuran_set(bagian, nama):
    print(f"- {nama}: {tampil_hitungan(len(bagian))} pasangan, {tampil_hitungan(bagian.subject_id.nunique())} "
          f"pasien, positif {tampil_hitungan(bagian.y.sum())} "
          f"(pasien dengan ≥ 1 positif {tampil_hitungan(bagian.subject_id[bagian.y == 1].nunique())}).")


def cetak_tabel_metrik(penilaian, bootstrap):
    y = penilaian.y
    if y.sum() < SEL_KECIL:
        print("Kasus positif < 11: metrik tidak dicetak.\n")
        return
    selang = selang_bootstrap(penilaian, metrik_terurut, bootstrap)
    isi = []
    for nama, p in penilaian.peluang.items():
        titik = metrik_lengkap(y, p)
        isi.append([nama, *(dengan_selang(titik[k], selang[nama][k]) for k in ("AUROC", "AUPRC")),
                    *(f"{titik[k]:.3f}" for k in ("Se@Sp90", "Brier", "O/E", "kemiringan"))])
    tabel(["Model", "AUROC [95 % CI]", "AUPRC [95 % CI]", "Se @ Sp 90 %", "Brier", "O/E", "Kemiringan kalibrasi"],
          isi)
    print(f"\nPrevalensi positif: {100 * y.mean():.2f} %. CI: bootstrap bergugus pasien.\n")


def sel_auroc(penilaian, bootstrap):
    if penilaian.y.sum() < SEL_KECIL:
        return ["—"] * len(penilaian.peluang)
    selang = selang_bootstrap(penilaian, metrik_terurut, bootstrap)
    return [dengan_selang(roc_auc_score(penilaian.y, p), selang[m]["AUROC"]) for m, p in penilaian.peluang.items()]
