"""Kesesuaian segmen pada model bersih (protokol Artikel_Glukosa_Rencana.md §F langkah 18b). ECGFounder: pra-proses resmi
atas 10 s penuh, dua paruh 2.500 sampel masuk Net1D terpisah, kosinus dua `deep_features`. ECG-JEPA: masukan resmi 10 s
(posisi tetap 50 tambalan per sadapan), satu forward pass, kosinus rerata token tambalan 1–25 lawan 26–50 atas delapan
sadapan. Pembaca menyerahkan masukan kedua model per rekaman dalam urutan masukan; gagal baca diserahkan sebagai None."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from baca_keanggotaan import GALAT_BACA, susun_potongan
from ecgfounder import siapkan_sinyal as siapkan_founder
from ecgjepa import siapkan_sinyal as siapkan_jepa
from zip_wfdb import baca_rekaman

PARUH_FOUNDER = 2500
TAMBALAN_PER_SADAPAN = 50
SAMPEL_PER_TAMBALAN = 50


def baca_dua_model(jalur_zip, jalur_rekaman, pekerja):
    with ProcessPoolExecutor(pekerja) as kolam:
        antre = deque()
        for potongan in susun_potongan(jalur_zip, jalur_rekaman):
            antre.append(kolam.submit(baca_potongan_dua, potongan))
            if len(antre) >= 2 * pekerja:
                yield from antre.popleft().result()
        while antre:
            yield from antre.popleft().result()


def baca_potongan_dua(potongan):
    with open(potongan.jalur_zip, "rb") as berkas:
        return [masukan_dua(berkas, lokasi, jalur) for jalur, lokasi in potongan.tugas]


def masukan_dua(berkas, lokasi, jalur):
    try:
        rekaman = baca_rekaman(berkas, lokasi, jalur)
    except GALAT_BACA:  # protokol 18b: rekaman gagal baca dihitung pemanggil, seperti langkah 16
        return None
    jepa, founder = siapkan_jepa(rekaman), siapkan_founder(rekaman)
    return None if jepa is None or founder is None else (jepa, founder)


class PenilaiKesesuaian:
    def __init__(self, perangkat):
        from ecgfounder import muat_model
        from ecgjepa import muat_penanam
        self.encoder_jepa = muat_penanam(perangkat).__self__
        self.founder = muat_model(perangkat)
        self.perangkat = perangkat

    def nilai(self, batch_jepa, batch_founder):
        import torch
        with torch.no_grad():
            jepa = self.kosinus_jepa(torch.from_numpy(batch_jepa).to(self.perangkat))
            founder = self.kosinus_founder(torch.from_numpy(batch_founder).to(self.perangkat))
        return jepa.cpu().numpy(), founder.cpu().numpy()

    def kosinus_jepa(self, x):
        import torch
        token = token_jepa(self.encoder_jepa, x)
        per_sadapan = token.reshape(x.shape[0], x.shape[1], TAMBALAN_PER_SADAPAN, -1)
        tengah = TAMBALAN_PER_SADAPAN // 2
        awal, akhir = per_sadapan[:, :, :tengah].mean(dim=(1, 2)), per_sadapan[:, :, tengah:].mean(dim=(1, 2))
        return torch.nn.functional.cosine_similarity(awal, akhir, dim=1)

    def kosinus_founder(self, x):
        import torch
        awal = self.founder(x[..., :PARUH_FOUNDER].contiguous())[1]
        akhir = self.founder(x[..., PARUH_FOUNDER:].contiguous())[1]
        return torch.nn.functional.cosine_similarity(awal, akhir, dim=1)


def token_jepa(encoder, x):
    pos_embed = encoder.pos_embed
    attention_mask = encoder._cross_attention_mask().to(x.device)
    if len(encoder.leads) < encoder.c:
        pos_embed = encoder.restrict_leads(pos_embed, type="vector")
        attention_mask = encoder.restrict_leads(attention_mask, type="matrix")
    token = encoder.W_P(x.reshape(x.shape[0], -1, SAMPEL_PER_TAMBALAN))
    token = encoder.encoder_blocks(token, pos_embed, attention_mask)
    return token if encoder.norm is None else encoder.norm(token)


def tumpuk(daftar):
    return np.stack(daftar).astype(np.float32)
