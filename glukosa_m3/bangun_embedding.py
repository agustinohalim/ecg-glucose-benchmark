"""Embedding model fondasi beku (M3: ECGFounder; M3b: ECG-JEPA) untuk semua rekaman di singgahan gelombang M2, dibaca
ulang dari ZIP MIMIC-IV-ECG v1.0 dalam bentuk asli 12 sadapan × 500 Hz (bukan dari singgahan 8 sadapan × 250 Hz), agar
pra-proses resmi tiap model dipakai apa adanya (lihat ecgfounder.py, ecgjepa.py).

Keluaran di folder singgahan (di luar repo): emb_<model>.npy (n × dimensi, float32, baris sejajar gelombang_jalur.txt)
dan emb_<model>_selesai.npy. Bila terputus, jalankan ulang; baris selesai dilewati. Pra-proses di proses pekerja CPU,
model di GPU. Yang dicetak hanya kemajuan dan hitungan agregat.

Pakai lingkungan utama (README), dari akar repo:
    python glukosa_m3/bangun_embedding.py [--model ecgfounder|ecgjepa]
        [--zip data/mimic-iv-ecg-1.0.zip] [--singgahan data/mimic-iv-ext-ecg-glucose/gelombang]
        [--pekerja 12]
"""
import argparse
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "glukosa_m2"))
from bangun_singgahan import lokasi_rekaman  # noqa: E402
from model_fondasi import MODEL_FONDASI  # noqa: E402
from singgahan import BERKAS_JALUR  # noqa: E402
from zip_wfdb import baca_rekaman, indeks_anggota  # noqa: E402

UKURAN_POTONGAN = 256
BATCH_GPU = 64
SIMPAN_TIAP = 20


@dataclass(frozen=True)
class PotonganEmbed:
    jalur_zip: str
    kode_model: str
    tugas: list


@dataclass(frozen=True)
class Keluaran:
    emb: np.ndarray
    selesai: np.ndarray
    jalur_selesai: Path

    @classmethod
    def siapkan(cls, folder, n, model):
        jalur_emb, jalur_selesai = folder / model.berkas_emb, folder / model.berkas_selesai
        if not jalur_emb.exists():
            np.lib.format.open_memmap(jalur_emb, mode="w+", dtype=np.float32, shape=(n, model.dimensi)).flush()
            np.save(jalur_selesai, np.zeros(n, dtype=bool))
        return cls(np.load(jalur_emb, mmap_mode="r+"), np.load(jalur_selesai), jalur_selesai)

    def simpan(self, keterangan):
        self.emb.flush()
        np.save(self.jalur_selesai, self.selesai)
        print(f"{time.strftime('%H:%M')} potongan {keterangan}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argumen = baca_argumen()
    mulai = time.time()
    model = MODEL_FONDASI[argumen.model]
    folder = Path(argumen.singgahan)
    jalur_rekaman = folder.joinpath(BERKAS_JALUR).read_text(encoding="ascii").split()
    keluaran = Keluaran.siapkan(folder, len(jalur_rekaman), model)
    sisa = np.flatnonzero(~keluaran.selesai)
    print(f"{model.nama}: rekaman {len(jalur_rekaman)}; belum {len(sisa)}.", flush=True)
    if len(sisa):
        indeks = indeks_anggota(argumen.zip, [jalur_rekaman[i] for i in sisa])
        tugas = [(int(i), jalur_rekaman[i], lokasi_rekaman(indeks, jalur_rekaman[i])) for i in sisa]
        potongan = [PotonganEmbed(argumen.zip, model.kode, tugas[k:k + UKURAN_POTONGAN])
                    for k in range(0, len(tugas), UKURAN_POTONGAN)]
        jalankan_semua(potongan, keluaran, model, argumen.pekerja)
    terhingga = int(np.isfinite(keluaran.emb).all(axis=1).sum())
    print(f"Selesai {int(keluaran.selesai.sum())}/{len(keluaran.selesai)}; baris embedding terhingga {terhingga}; "
          f"waktu {(time.time() - mulai) / 60:.1f} menit.")


def baca_argumen():
    pengurai = argparse.ArgumentParser()
    pengurai.add_argument("--model", choices=sorted(MODEL_FONDASI), default="ecgfounder")
    pengurai.add_argument("--zip", default="data/mimic-iv-ecg-1.0.zip")
    pengurai.add_argument("--singgahan", default="data/mimic-iv-ext-ecg-glucose/gelombang")
    pengurai.add_argument("--pekerja", type=int, default=12)
    return pengurai.parse_args()


def praproses_potongan(potongan):
    siapkan_sinyal = MODEL_FONDASI[potongan.kode_model].modul.siapkan_sinyal
    baris, sinyal = [], []
    with open(potongan.jalur_zip, "rb") as berkas:
        for nomor, jalur_rekaman, indeks in potongan.tugas:
            x = siapkan_sinyal(baca_rekaman(berkas, indeks, jalur_rekaman))
            if x is None:
                raise ValueError(f"bentuk rekaman tak sesuai {potongan.kode_model}: baris {nomor}")
            baris.append(nomor)
            sinyal.append(x)
    return np.array(baris), np.stack(sinyal)


def jalankan_semua(daftar_potongan, keluaran, model, pekerja):
    import torch
    torch.set_num_threads(4)
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    penanam = model.modul.muat_penanam(perangkat)
    antre, berikut, rampung = set(), 0, 0
    with ProcessPoolExecutor(pekerja) as kolam:
        while berikut < len(daftar_potongan) or antre:
            while berikut < len(daftar_potongan) and len(antre) < 2 * pekerja:
                antre.add(kolam.submit(praproses_potongan, daftar_potongan[berikut]))
                berikut += 1
            beres, antre = wait(antre, return_when=FIRST_COMPLETED)
            for tugas in beres:
                baris, sinyal = tugas.result()
                keluaran.emb[baris] = tanam(penanam, sinyal, perangkat)
                keluaran.selesai[baris] = True
                rampung += 1
                if rampung % SIMPAN_TIAP == 0 or rampung == len(daftar_potongan):
                    keluaran.simpan(f"{rampung}/{len(daftar_potongan)}")


def tanam(penanam, sinyal, perangkat):
    import torch
    hasil = []
    with torch.no_grad():
        for awal in range(0, len(sinyal), BATCH_GPU):
            hasil.append(penanam(torch.from_numpy(sinyal[awal:awal + BATCH_GPU]).to(perangkat)).cpu().numpy())
    return np.concatenate(hasil)


if __name__ == "__main__":
    main()
