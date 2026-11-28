"""Gambar 1 naskah Glukosa sesudah telaah internal 14: DiD AUROC dilihat − tidak dilihat lawan ECG-JEPA untuk ECG-FM pretrained dan
ECG-FM fine-tuned (kontrol positif) pada dua hasil konfirmatori di rekaman replikasi, dengan skala horizontal sama:
(a) empat pernyataan berplafon (Hasil_Glukosa_17.md, aturan R2) dan (b) keluarga lepas plafon (Hasil_Glukosa_23.md,
G21 dan H21). Tidak membaca data atau prediksi. Keluaran: Gambar_Glukosa_1c.png (300 dpi) dan .pdf di .

Pakai, dari akar repo:
    python glukosa_lepas_plafon/gambar_lepas_plafon.py
"""
import sys
from pathlib import Path

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / "glukosa_replikasi")]
from gambar_replikasi import GESER, SERI, bagian, ci_baris, gambar_titik, plt  # noqa: E402

KELUARAN = AKAR_EKG / "Gambar_Glukosa_1c"
PANEL = [("(a) Near the ceiling (rule R2)", "Hasil_Glukosa_17.md",
          [("Fibrilasi atrium", "Atrial fibrillation"), ("LBBB", "LBBB"), ("RBBB", "RBBB"),
           ("AV blok derajat 1", "First-degree AV block"), ("Rerata empat tugas berlabel", "Mean, four labeled")],
          [(0.002, ":", "R2 threshold")]),
         ("(b) Off the ceiling (rules G21, H21)", "Hasil_Glukosa_23.md",
          [("infark", "Infarction"), ("mutu", "Poor data quality"), ("Rerata keluarga utama", "Mean, family")],
          [(0.01, "--", "margin ε")])]


def main():
    fig, semua_sumbu = plt.subplots(2, 1, figsize=(5.2, 4.6), sharex=True,
                                    gridspec_kw={"height_ratios": [len(p[2]) for p in PANEL]})
    for sumbu, panel in zip(semua_sumbu, PANEL):
        gambar_panel(sumbu, panel)
    semua_sumbu[-1].set_xlabel("DiD of AUROC, seen − unseen, against ECG-JEPA [95 % CI]", fontsize=7)
    fig.tight_layout()
    fig.legend(*semua_sumbu[0].get_legend_handles_labels(), loc="upper center", ncol=1, fontsize=7, frameon=False,
               bbox_to_anchor=(0.55, 1.08))
    fig.savefig(KELUARAN.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(KELUARAN.with_suffix(".pdf"), bbox_inches="tight")


def gambar_panel(sumbu, panel):
    judul, berkas, baris, garis = panel
    teks = bagian((AKAR_EKG / berkas).read_text(encoding="utf-8"), "## Utama")
    for i, (nama, _) in enumerate(baris):
        for k, nilai in enumerate(ci_baris(teks, nama, "AUROC")[:2]):
            gambar_titik(sumbu, i + (k - 0.5) * 2 * GESER, nilai, SERI[k] + (i == 0,))
    sumbu.axvline(0, color="#555555", linewidth=0.8)
    for posisi, gaya, label in garis:
        sumbu.axvline(posisi, color="#555555", linewidth=0.8, linestyle=gaya)
        sumbu.text(posisi, -0.45, f" {label}", fontsize=6.5, color="#555555", va="top")
    sumbu.axhline(len(baris) - 1.5, color="#bbbbbb", linewidth=0.6)
    sumbu.set_yticks(range(len(baris)), [label for _, label in baris])
    sumbu.set_ylim(len(baris) - 0.5, -0.5)
    sumbu.set_title(judul, fontsize=8)
    sumbu.grid(axis="x", color="#dddddd", linewidth=0.6)
    sumbu.tick_params(labelsize=7)


if __name__ == "__main__":
    main()
