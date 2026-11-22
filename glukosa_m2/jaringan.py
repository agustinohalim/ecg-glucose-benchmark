"""M2: CNN 1D pada gelombang EKG 10 s, dibungkus sebagai penaksir bergaya scikit-learn agar pemodelan M0/M1 (latih
TRAIN–nilai VALIDATION, validasi silang bergugus pasien) dipakai apa adanya. Masukan X adalah satu kolom berisi nomor
baris singgahan gelombang.

Resep sama dengan M2 naskah OMI (omi_m2.py): empat blok Conv1d(7) + BatchNorm + ReLU + MaxPool(4), kanal 32-64-64-128,
rerata global, dropout 0,3, linear; Adam 1e-3, epoch tetap, batch 64, bobot kelas seimbang, model epoch terakhir.
Bedanya: 8 sadapan independen, bukan 12, dan data dialirkan per batch dari RAM karena tidak muat di GPU 8 GB.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

PERANGKAT = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KANAL = (32, 64, 64, 128)
CONTOH_SKALA = 2000
BATCH_PREDIKSI = 512


@dataclass(frozen=True)
class Resep:
    epoch: int = 20
    batch: int = 64
    laju: float = 1e-3


class JaringanEKG(nn.Module):
    def __init__(self, n_sadapan, keluaran):
        super().__init__()
        blok, masuk = [], n_sadapan
        for kanal in KANAL:
            blok += [nn.Conv1d(masuk, kanal, 7, padding=3), nn.BatchNorm1d(kanal), nn.ReLU(), nn.MaxPool1d(4)]
            masuk = kanal
        self.fitur = nn.Sequential(*blok)
        self.kepala = nn.Sequential(nn.Dropout(0.3), nn.Linear(KANAL[-1], keluaran))

    def forward(self, x):
        return self.kepala(self.fitur(x).mean(dim=2))


class SinyalBaris:
    def __init__(self, sinyal, baris):
        unik, self.balik = np.unique(baris, return_inverse=True)
        self.data = np.asarray(sinyal[unik])

    def batch(self, posisi):
        return torch.from_numpy(self.data[self.balik[posisi]]).to(PERANGKAT).float()


class PenggolongCNN:
    def __init__(self, sinyal, benih, resep):
        self.sinyal, self.benih, self.resep = sinyal, benih, resep

    def fit(self, X, y):
        torch.manual_seed(self.benih)
        latih = SinyalBaris(self.sinyal, X[:, 0].astype(np.int64))
        self.skala_ = skala_global(latih.data, self.benih)
        self.kelas_ = np.unique(y)
        self.jaringan_ = JaringanEKG(latih.data.shape[1], 1 if self.biner else len(self.kelas_)).to(PERANGKAT)
        self.latih_epoch(latih, np.searchsorted(self.kelas_, y))
        return self

    @property
    def biner(self):
        return len(self.kelas_) == 2

    def latih_epoch(self, latih, y):
        optimasi = torch.optim.Adam(self.jaringan_.parameters(), lr=self.resep.laju)
        rugi = fungsi_rugi(y, self.biner)
        target = torch.tensor(y, device=PERANGKAT, dtype=torch.float32 if self.biner else torch.long)
        acak = torch.Generator().manual_seed(self.benih)
        self.jaringan_.train()
        for _ in range(self.resep.epoch):
            urutan = torch.randperm(len(y), generator=acak).numpy()
            for awal in range(0, len(y) - 1, self.resep.batch):
                posisi = urutan[awal:awal + self.resep.batch]
                keluaran = self.jaringan_(latih.batch(posisi) / self.skala_)
                optimasi.zero_grad()
                rugi(keluaran.squeeze(1) if self.biner else keluaran, target[posisi]).backward()
                optimasi.step()

    def predict_proba(self, X):
        nilai = SinyalBaris(self.sinyal, X[:, 0].astype(np.int64))
        self.jaringan_.eval()
        hasil = []
        n = len(nilai.balik)
        with torch.no_grad():
            for awal in range(0, n, BATCH_PREDIKSI):
                hasil.append(self.peluang_batch(nilai.batch(np.arange(awal, min(awal + BATCH_PREDIKSI, n)))))
        P = np.concatenate(hasil)
        return np.column_stack([1 - P[:, 0], P[:, 0]]) if self.biner else P

    def peluang_batch(self, x):
        keluaran = self.jaringan_(x / self.skala_)
        return (torch.sigmoid(keluaran) if self.biner else torch.softmax(keluaran, 1)).cpu().numpy()


def skala_global(data, benih):
    pilih = np.random.default_rng(benih).choice(len(data), min(CONTOH_SKALA, len(data)), replace=False)
    return float(data[pilih].astype(np.float32).std()) + 1e-6


def fungsi_rugi(y, biner):
    if biner:
        p = y.mean()
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - p) / p, dtype=torch.float32, device=PERANGKAT))
    n_kelas = np.bincount(y)
    bobot = torch.tensor(len(y) / (len(n_kelas) * n_kelas), dtype=torch.float32, device=PERANGKAT)
    return nn.CrossEntropyLoss(weight=bobot)


def pembuat_cnn(sinyal, resep):
    return lambda benih: PenggolongCNN(sinyal, benih, resep)
