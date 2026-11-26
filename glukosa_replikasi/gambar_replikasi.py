"""Gambar 1 naskah Glukosa sesudah telaah internal 8: DiD AUROC dilihat − tidak dilihat lawan ECG-JEPA untuk ECG-FM pretrained dan
ECG-FM fine-tuned (kontrol positif) di tiga sampel: kohort glukosa protokol kedua (per tugas dari Hasil_Glukosa_13.md,
rerata empat tugas berlabel dari Hasil_Glukosa_15.md), replikasi pra-registrasi seluruhnya, dan subset replikasi pasien
di luar kohort (Hasil_Glukosa_17.md). Hanya skala AUROC; skala logit tidak digambar karena memuat beda tingkat AUROC
kedua checkpoint. Tidak membaca data atau prediksi. Keluaran: Gambar_Glukosa_1b.png (300 dpi) dan .pdf di .

Pakai, dari akar repo:
    python glukosa_replikasi/gambar_replikasi.py
"""
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

AKAR_EKG = Path(__file__).resolve().parent.parent
KELUARAN = AKAR_EKG / "Gambar_Glukosa_1b"
BARIS = [("Hipoglikemia", "Hypoglycemia"), ("Hiperglikemia berat", "Severe hyperglycemia"),
         ("Fibrilasi atrium", "Atrial fibrillation"), ("LBBB", "LBBB"), ("RBBB", "RBBB"),
         ("AV blok derajat 1", "First-degree AV block"), ("LVH", "LVH (not a fine-tuning label)"),
         ("Rerata empat tugas berlabel", "Mean, four labeled")]
SERI = [("Pretrained ECG-FM − ECG-JEPA (pretraining exposure)", "#0072b2", "o"),
        ("Fine-tuned ECG-FM − ECG-JEPA (positive control)", "#d55e00", "s")]
NILAI_CI = re.compile(r"(-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\]")
BATAS_R2 = 0.002
GESER = 0.17


@dataclass(frozen=True)
class Panel:
    judul: str
    nilai: dict
    garis_r2: bool


def main():
    daftar_panel = [Panel("(a) Glucose cohort, second protocol", nilai_kohort(), False),
                    Panel("(b) Replication, pre-registered", nilai_replikasi("## Utama"), True),
                    Panel("(c) Replication, patients with\nno cohort ECG (descriptive)",
                          nilai_replikasi("## Pasien di luar kohort"), False)]
    fig, semua_sumbu = plt.subplots(1, 3, figsize=(7.6, 3.8), sharey=True)
    for sumbu, panel in zip(semua_sumbu, daftar_panel):
        gambar_panel(sumbu, panel)
    semua_sumbu[0].set_yticks(range(len(BARIS)), [label for _, label in BARIS])
    semua_sumbu[0].set_ylim(len(BARIS) - 0.5, -0.5)
    fig.tight_layout()
    fig.legend(*semua_sumbu[0].get_legend_handles_labels(), loc="upper center", ncol=1, fontsize=8, frameon=False,
               bbox_to_anchor=(0.6, 1.12))
    fig.savefig(KELUARAN.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(KELUARAN.with_suffix(".pdf"), bbox_inches="tight")


def nilai_kohort():
    teks_tugas = (AKAR_EKG / "Hasil_Glukosa_13.md").read_text(encoding="utf-8")
    nilai = {nama: [ci_baris(bagian(teks_tugas, f"#### {nama}"), jenis)[0] for jenis in ("utama", "kontrol positif")]
             for nama, _ in BARIS[:-1]}
    teks_rerata = (AKAR_EKG / "Hasil_Glukosa_15.md").read_text(encoding="utf-8")
    nilai[BARIS[-1][0]] = ci_baris(bagian(teks_rerata, "### Rerata empat tugas berlabel"), "AUROC")[:2]
    return nilai


def nilai_replikasi(judul):
    teks = bagian((AKAR_EKG / "Hasil_Glukosa_17.md").read_text(encoding="utf-8"), judul)
    return {nama: ci_baris(teks, nama, "AUROC")[:2] for nama, _ in BARIS[2:]}


def bagian(teks, judul):
    awal = teks.index(judul)
    akhir = teks.find("\n#", awal + len(judul))
    return teks[awal:akhir if akhir > 0 else None]


def ci_baris(teks, *sel_awal):
    for baris in teks.splitlines():
        sel = [s.strip() for s in baris.split("|")[1:]]
        if sel[:len(sel_awal)] == list(sel_awal):
            return [tuple(float(angka) for angka in nilai) for nilai in NILAI_CI.findall(baris)]
    raise SystemExit(f"baris {sel_awal} tidak ditemukan")


def gambar_panel(sumbu, panel):
    for i, (nama, _) in enumerate(BARIS):
        for k, nilai in enumerate(panel.nilai.get(nama, [])):
            gambar_titik(sumbu, i + (k - 0.5) * 2 * GESER, nilai, SERI[k] + (i == 2,))
    sumbu.axvline(0, color="#555555", linewidth=0.8)
    if panel.garis_r2:
        sumbu.axvline(BATAS_R2, color="#555555", linewidth=0.8, linestyle=":")
        sumbu.text(BATAS_R2, 0.5, " R2 threshold\n (four-task\n mean)", fontsize=7, color="#555555", va="center")
    sumbu.xaxis.set_major_locator(MaxNLocator(nbins=4))
    for posisi in (1.5, len(BARIS) - 1.5):
        sumbu.axhline(posisi, color="#bbbbbb", linewidth=0.6)
    sumbu.set_title(panel.judul, fontsize=8)
    sumbu.grid(axis="x", color="#dddddd", linewidth=0.6)
    sumbu.set_xlabel("DiD of AUROC, seen − unseen [95 % CI]", fontsize=7)
    sumbu.tick_params(labelsize=7)


def gambar_titik(sumbu, posisi, nilai, gaya):
    label, warna, penanda, beri_label = gaya
    titik, bawah, atas = nilai
    sumbu.errorbar(titik, posisi, xerr=[[titik - bawah], [atas - titik]], fmt=penanda, color=warna, markersize=4,
                   capsize=2, label=label if beri_label else None)


if __name__ == "__main__":
    main()
