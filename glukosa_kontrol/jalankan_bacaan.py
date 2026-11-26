"""Bacaan ECG-FM sesudah latih lanjut menurut protokol Artikel_Glukosa_Rencana.md §F langkah 17c: dengan bobot dosis
epoch 4 lalu epoch 1, (i) embedding mode eval (rerata token keluaran dua segmen, pra-proses langkah 15) untuk semua
rekaman `valid` ∪ `test` replikasi, dan (ii) skor keanggotaan langkah 16b (K = 4 undian, benih torch dan numpy 20261120
dipasang sesudah bobot dimuat) untuk satu rekaman per pasien tidak dilihat (undian langkah 16). Simpanan di
data/kontrol, di luar repo dan tidak dibuka di sesi asisten; stdout hanya agregat. Bacaan yang sudah tersimpan
dilewati.

Pakai venv-ecgfm (GPU), dari akar repo, sesudah jalankan_latih.py:
    python -u glukosa_kontrol/jalankan_bacaan.py [--dosis 4 1]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_replikasi", "glukosa_perluasan", "glukosa_test", "glukosa_m3",
                                            "glukosa_m2", "glukosa_m1")]
from baca_keanggotaan import baca_berurutan  # noqa: E402
from ecgfm import BOBOT, N_SEGMEN, PANJANG_SEGMEN  # noqa: E402
from hitung_keanggotaan import GAGAL, simpan_skor  # noqa: E402
from penugasan_kontrol import FOLDER_KONTROL, FOLDER_REPLIKASI, muat_unit, pasien_disuntik  # noqa: E402
from penugasan_kontrol import pilihan_satu_per_pasien, unit_bacaan  # noqa: E402
from skor_keanggotaan import PenilaiKeanggotaan  # noqa: E402
from statistik_keanggotaan import BENIH  # noqa: E402

BATCH_GPU = 64
DIMENSI = 768
LAPOR_TIAP = 10000


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    unit = muat_unit(argumen.folder)
    pilihan = pilihan_satu_per_pasien(unit)
    bacaan = unit_bacaan(unit, pasien_disuntik(pilihan))
    tidak = pilihan[~pilihan.dilihat].reset_index(drop=True)
    folder = Path(argumen.keluaran)
    for dosis in argumen.dosis:
        jalur_emb, jalur_skor = folder / f"emb_plus{dosis}.npz", folder / f"keanggotaan_plus{dosis}_b{BENIH}.npz"
        if not jalur_emb.exists():
            bangun_embedding(muat_bobot(folder, dosis), bacaan, argumen, jalur_emb)
        if not jalur_skor.exists():
            hitung_skor_dosis(muat_bobot(folder, dosis), tidak, argumen, jalur_skor)
        cetak_ringkas(dosis, jalur_emb, jalur_skor)


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--folder", default=str(FOLDER_REPLIKASI))
    pengurai.add_argument("--keluaran", default=str(FOLDER_KONTROL))
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    pengurai.add_argument("--dosis", type=int, nargs="+", default=[4, 1])
    pengurai.add_argument("--pekerja", type=int, default=6)
    return pengurai.parse_args()


def muat_bobot(folder, dosis):
    import torch
    return torch.load(folder / f"ecgfm_plus_e{dosis}.pt", map_location="cpu", weights_only=False)["model"]


def bangun_embedding(bobot, bacaan, argumen, jalur):
    import torch
    from fairseq_signals.models import build_model_from_checkpoint
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint(checkpoint_path=str(BOBOT)).to(perangkat).eval()
    model.load_state_dict(bobot)
    emb = np.full((len(bacaan), DIMENSI), np.nan, np.float32)
    baris, batch = [], []
    for nomor, sinyal in enumerate(baca_berurutan(argumen.zip, list(bacaan.waveform_path), argumen.pekerja)):
        if sinyal is not None:
            baris.append(nomor)
            batch.append(sinyal)
        if len(batch) == BATCH_GPU or (nomor == len(bacaan) - 1 and batch):
            emb[baris] = tanam(model, np.stack(batch), perangkat)
            baris, batch = [], []
        if (nomor + 1) % LAPOR_TIAP == 0:
            print(f"{time.strftime('%H:%M')} embedding {nomor + 1}", flush=True)
    np.savez(jalur, baris_gelombang=bacaan.baris_gelombang.to_numpy(), emb=emb)


def tanam(model, sinyal, perangkat):
    import torch
    x = torch.from_numpy(sinyal).to(perangkat)
    segmen = x.reshape(len(x), x.shape[1], N_SEGMEN, PANJANG_SEGMEN).permute(0, 2, 1, 3)
    with torch.no_grad():
        token = model.extract_features(segmen.reshape(-1, x.shape[1], PANJANG_SEGMEN), padding_mask=None)["x"]
        return token.mean(dim=1).reshape(len(x), N_SEGMEN, -1).mean(dim=1).cpu().numpy()


def hitung_skor_dosis(bobot, tidak, argumen, jalur):
    import torch
    penilai = PenilaiKeanggotaan(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    penilai.model.load_state_dict(bobot)
    torch.manual_seed(BENIH)
    np.random.seed(BENIH)
    hasil = []
    for nomor, sinyal in enumerate(baca_berurutan(argumen.zip, list(tidak.waveform_path), argumen.pekerja), 1):
        hasil.append(GAGAL if sinyal is None else penilai.nilai(sinyal))
        if nomor % LAPOR_TIAP == 0:
            print(f"{time.strftime('%H:%M')} skor {nomor}", flush=True)
    skor = pd.DataFrame({"terbaca": [h is not GAGAL for h in hasil], "kehilangan": [h.kehilangan for h in hasil],
                         "kosinus": [h.kosinus for h in hasil]})
    simpan_skor(jalur, tidak, skor)


def cetak_ringkas(dosis, jalur_emb, jalur_skor):
    emb, skor = np.load(jalur_emb), np.load(jalur_skor)
    terhingga = int(np.isfinite(emb["emb"]).all(axis=1).sum())
    print(f"Dosis epoch {dosis}: embedding terhingga {terhingga}/{len(emb['emb'])}; skor terbaca "
          f"{int(skor['terbaca'].sum())}/{len(skor['terbaca'])}.", flush=True)


if __name__ == "__main__":
    main()
