"""Latih lanjut ECG-FM untuk kontrol positif swaawas Glukosa menurut protokol Artikel_Glukosa_Rencana.md §F langkah 17b:
empat epoch atas rekaman pasien disuntik ditambah putar-ulang `train` (penugasan_kontrol.py), batch mikro 9 rekaman ×
19 akumulasi = 171 rekaman per pembaruan, urutan acak per epoch dari satu generator benih 20261120. Sesudah tiap epoch
keadaan lengkap disimpan untuk dilanjutkan bila terputus; bobot dosis epoch 1 dan 4 disimpan terpisah. Semua simpanan di
luar repo (data/kontrol); stdout hanya agregat (hitungan 1–10 disamarkan).

Pakai venv-ecgfm (GPU), dari akar repo, sekali:
    python -u glukosa_kontrol/jalankan_latih.py [--pekerja 8]
"""
import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AKAR_EKG = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AKAR_EKG / d) for d in ("glukosa_replikasi", "glukosa_perluasan", "glukosa_test", "glukosa_m3",
                                            "glukosa_m2", "glukosa_m1")]
from baca_keanggotaan import baca_berurutan  # noqa: E402
from cetak import tabel, tampil_hitungan  # noqa: E402
from pelatih_lanjut import PelatihLanjut  # noqa: E402
from penugasan_kontrol import FOLDER_KONTROL, FOLDER_REPLIKASI, muat_unit, pasien_disuntik  # noqa: E402
from penugasan_kontrol import pilihan_satu_per_pasien, rekaman_latih  # noqa: E402
from statistik_keanggotaan import BENIH  # noqa: E402

DOSIS_DISIMPAN = (1, 4)
BATCH_MIKRO = 9
AKUMULASI = 19
LAPOR_TIAP = 100
BERKAS_LANJUT = "lanjut_terakhir.pt"


@dataclass
class Akumulator:
    pelatih: PelatihLanjut
    batch: list = field(default_factory=list)
    mikro: int = 0
    bertopeng: float = 0.0
    total_bertopeng: float = 0.0
    jumlah_log: dict = field(default_factory=dict)
    gagal: int = 0

    def tambah(self, sinyal):
        if sinyal is None:
            self.gagal += 1
            return
        self.batch.append(sinyal)
        if len(self.batch) == BATCH_MIKRO:
            self.langkah()
        if self.mikro == AKUMULASI:
            self.perbarui()

    def langkah(self):
        bertopeng, log = self.pelatih.langkah_mikro(np.stack(self.batch))
        self.batch, self.mikro = [], self.mikro + 1
        self.bertopeng += bertopeng
        self.total_bertopeng += bertopeng
        for kunci, nilai in log.items():
            self.jumlah_log[kunci] = self.jumlah_log.get(kunci, 0.0) + nilai

    def perbarui(self):
        self.pelatih.perbarui(self.bertopeng)
        self.mikro, self.bertopeng = 0, 0.0
        if self.pelatih.pembaruan % LAPOR_TIAP == 0:
            print(f"{time.strftime('%H:%M')} pembaruan {self.pelatih.pembaruan}; kehilangan rerata per langkah "
                  f"bertopeng {self.rerata().get('loss', float('nan')):.4f}", flush=True)

    def tutup(self):
        if self.batch:
            self.langkah()
        if self.mikro:
            self.perbarui()

    def rerata(self):
        return {kunci: nilai / max(self.total_bertopeng, 1.0) for kunci, nilai in self.jumlah_log.items()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    unit = muat_unit(argumen.folder)
    disuntik = pasien_disuntik(pilihan_satu_per_pasien(unit))
    latih = rekaman_latih(unit, disuntik)
    cetak_penugasan(disuntik, latih)
    import torch
    folder = Path(argumen.keluaran)
    folder.mkdir(parents=True, exist_ok=True)
    pelatih = PelatihLanjut(torch.device("cuda" if torch.cuda.is_available() else "cpu"), BENIH)
    urutan = np.random.default_rng(BENIH)
    awal = pelatih.lanjutkan(folder / BERKAS_LANJUT, urutan) if (folder / BERKAS_LANJUT).exists() else 0
    print(f"Mulai dari pembaruan {pelatih.pembaruan}, epoch {awal + 1}.", flush=True)
    for epoch in range(awal + 1, argumen.epoch + 1):
        jalur = list(latih.waveform_path.to_numpy()[urutan.permutation(len(latih))])
        akumulator = jalankan_epoch(pelatih, jalur, argumen)
        pelatih.simpan(folder / BERKAS_LANJUT, epoch, urutan)
        if epoch in DOSIS_DISIMPAN:
            simpan_dosis(pelatih, folder / f"ecgfm_plus_e{epoch}.pt")
        cetak_epoch(epoch, akumulator)


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--folder", default=str(FOLDER_REPLIKASI))
    pengurai.add_argument("--keluaran", default=str(FOLDER_KONTROL))
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    pengurai.add_argument("--epoch", type=int, default=4)
    pengurai.add_argument("--pekerja", type=int, default=8)
    return pengurai.parse_args()


def cetak_penugasan(disuntik, latih):
    print(f"\nKontrol positif swaawas langkah 17; benih {BENIH}; batch {BATCH_MIKRO} × {AKUMULASI} rekaman.\n")
    tabel(["Kelompok latih", "Pasien", "Rekaman"],
          [["disuntik (tidak dilihat)", tampil_hitungan(len(disuntik)), tampil_hitungan(latih.disuntik.sum())],
           ["putar-ulang (train)", tampil_hitungan(latih.subject_id[~latih.disuntik].nunique()),
            tampil_hitungan((~latih.disuntik).sum())]])
    print(flush=True)


def jalankan_epoch(pelatih, jalur, argumen):
    akumulator = Akumulator(pelatih)
    for sinyal in baca_berurutan(argumen.zip, jalur, argumen.pekerja):
        akumulator.tambah(sinyal)
    akumulator.tutup()
    return akumulator


def simpan_dosis(pelatih, jalur):
    import torch
    torch.save({"model": pelatih.model.state_dict(), "pembaruan": pelatih.pembaruan}, jalur)


def cetak_epoch(epoch, akumulator):
    rerata = ", ".join(f"{k} {v:.4f}" for k, v in sorted(akumulator.rerata().items()))
    print(f"\n{time.strftime('%H:%M')} epoch {epoch} selesai: pembaruan {akumulator.pelatih.pembaruan}; gagal dibaca "
          f"{tampil_hitungan(akumulator.gagal)}; rerata per langkah bertopeng: {rerata}\n", flush=True)


if __name__ == "__main__":
    main()
