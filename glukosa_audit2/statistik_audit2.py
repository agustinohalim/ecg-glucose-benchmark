"""Statistik langkah 19 Glukosa (protokol Artikel_Glukosa_Rencana.md §F langkah 19b–e). Populasi segar: unit uji 19a,
kesesuaian segmen ECG-FM, ECG-FM-acak, ECG-JEPA, ECGFounder, dan log-likelihood inferensi dataset per model dari
embedding kohort langkah 11 (StandardScaler + PCA 64 + GaussianMixture 10 kovarians penuh, disetel sekali pada set
penyetelan). Populasi topeng sama: 22.058 pasien tidak dilihat langkah 17, Δℓ = ECG-FM+4 − ECG-FM asli. SMD berpenyebut
tetap dan selisih rerata dihitung per ulangan bootstrap oleh pemanggil; berkas skor tidak dibuka di sesi asisten."""
from pathlib import Path

import numpy as np
import pandas as pd

from statistik_audit import Populasi, auroc_kelompok, smd
from statistik_keanggotaan import BENIH, selisih_rerata

MODEL_FM = ("ecgfm", "ecgfm_acak")
BERSIH = ("ecgjepa", "ecgfounder")
MODEL_DI = ("ecgfm", *BERSIH)
KOVARIAT_SEGAR = ("age", "laki", "heart_rate_bpm", "qrs_duration_ms")
KOMPONEN_PCA = 64
KOMPONEN_GMM = 10
FPR_SASARAN = 0.01


def muat_segar(unit, berkas_fm, berkas_bersih, folder_emb):
    uji = unit[unit.peran.eq("uji")].reset_index(drop=True)
    fm, bersih = np.load(berkas_fm), np.load(berkas_bersih)
    for simpanan in (fm, bersih):
        if not np.array_equal(simpanan["baris_gelombang"], uji.baris_gelombang.to_numpy()):
            raise SystemExit("Simpanan kesesuaian tidak sejajar dengan unit uji segar.")
    nilai = {**{m: fm[f"kosinus_{m}"].astype(float) for m in MODEL_FM},
             **{m: bersih[f"kosinus_{m}"].astype(float) for m in BERSIH}}
    setel = unit.baris_gelombang[unit.peran.eq("setel")].to_numpy()
    nilai.update({f"di {m}": kemungkinan_log(np.load(Path(folder_emb) / f"emb_{m}.npy", mmap_mode="r"), setel,
                                              uji.baris_gelombang.to_numpy()) for m in MODEL_DI})
    sah = np.isfinite(np.column_stack(list(nilai.values()))).all(axis=1)
    nilai.update({k: uji[k].to_numpy(float) for k in (*KOVARIAT_SEGAR, "baris_gelombang")})
    return Populasi({k: v[sah] for k, v in nilai.items()}, uji.dilihat.to_numpy(bool)[sah],
                    uji.split_ecgfm.to_numpy()[sah], int((~sah).sum()))


def kemungkinan_log(emb, baris_setel, baris_uji):
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    latih = np.asarray(emb[baris_setel], dtype=np.float64)
    latih = latih[np.isfinite(latih).all(axis=1)]
    pipa = make_pipeline(StandardScaler(), PCA(KOMPONEN_PCA, random_state=BENIH)).fit(latih)
    kepadatan = GaussianMixture(KOMPONEN_GMM, covariance_type="full", random_state=BENIH).fit(pipa.transform(latih))
    uji = np.asarray(emb[baris_uji], dtype=np.float64)
    terhingga = np.isfinite(uji).all(axis=1)
    skor = np.full(len(uji), np.nan)
    skor[terhingga] = kepadatan.score_samples(pipa.transform(uji[terhingga]))
    return skor


def statistik_segar(pop):
    dilihat = pop.kelompok
    hasil = {f"SMD {m}": smd(pop.nilai[m], dilihat) for m in (*MODEL_FM, *BERSIH)}
    for m in ("ecgfm_acak", *BERSIH):
        hasil[f"Delta {m}"] = lambda w, m=m: hasil["SMD ecgfm"](w) - hasil[f"SMD {m}"](w)
    hasil.update({f"SMD di {m}": smd(pop.nilai[f"di {m}"], dilihat) for m in MODEL_DI})
    for m in BERSIH:
        hasil[f"Delta di {m}"] = lambda w, m=m: hasil["SMD di ecgfm"](w) - hasil[f"SMD di {m}"](w)
    for bagian in ("valid", "test"):
        hasil[f"SMD ecgfm {bagian}"] = smd(pop.nilai["ecgfm"], dilihat, pop.split == bagian)
    return hasil


def muat_topeng_sama(pilihan, berkas_topeng):
    tidak = pilihan[~pilihan.dilihat].reset_index(drop=True)
    skor = np.load(berkas_topeng)
    if not np.array_equal(skor["baris_gelombang"], tidak.baris_gelombang.to_numpy()):
        raise SystemExit("Kehilangan bertopeng sama tidak sejajar dengan pasien tidak dilihat langkah 17.")
    dl = skor["kehilangan_plus4"].astype(float) - skor["kehilangan_asli"].astype(float)
    sah = np.isfinite(dl)
    return Populasi({"dl": dl[sah]}, tidak.disuntik.to_numpy(bool)[sah], tidak.split_ecgfm.to_numpy()[sah],
                    int((~sah).sum()))


def statistik_topeng_sama(pop):
    disuntik, ditahan, dl = pop.kelompok, ~pop.kelompok, pop.nilai["dl"]
    return {"Dp": lambda w: selisih_rerata(dl, w * ditahan, w * disuntik), "AUROC -dl": auroc_kelompok(-dl, disuntik)}


def daya_pisah(pop, kode):
    satu = np.ones(len(pop.kelompok))
    return pd.DataFrame({k: [auroc_kelompok(pop.nilai[k], pop.kelompok)(satu), tpr_pada_fpr(pop.nilai[k], pop.kelompok)]
                         for k in kode}, index=["AUROC", f"TPR pada FPR {FPR_SASARAN:.0%}"]).T


def tpr_pada_fpr(skor, positif):
    ambang = np.quantile(skor[~positif], 1 - FPR_SASARAN)
    return float((skor[positif] > ambang).mean())
