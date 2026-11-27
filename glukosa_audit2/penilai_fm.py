"""Penilai ECG-FM langkah 19b dan 19d: kesesuaian segmen ECG-FM dan ECG-FM-acak (isi tiap tensor bobot checkpoint
pretrained diacak urutannya, benih 20261120; arsitektur, pra-proses, dan pengukuran sama), dan kehilangan langkah 16b
dengan benih torch dan numpy dipasang per rekaman, sehingga undian topeng dan negatif sama antar-checkpoint."""
import numpy as np

from skor_keanggotaan import PenilaiKeanggotaan, potong_segmen
from statistik_keanggotaan import BENIH


def penilai_acak(perangkat):
    import torch
    penilai = PenilaiKeanggotaan(perangkat)
    pembangkit = torch.Generator().manual_seed(BENIH)
    with torch.no_grad():
        for parameter in penilai.model.parameters():
            datar = parameter.detach().flatten()
            urutan = torch.randperm(datar.numel(), generator=pembangkit).to(datar.device)
            parameter.copy_(datar[urutan].view_as(parameter))
    return penilai


def penilai_dengan_bobot(perangkat, bobot):
    penilai = PenilaiKeanggotaan(perangkat)
    penilai.model.load_state_dict(bobot)
    return penilai


def kosinus_rekaman(penilai, sinyal):
    import torch
    with torch.no_grad():
        return penilai.kosinus(potong_segmen(torch.from_numpy(sinyal).to(penilai.perangkat)))


def kehilangan_bertopeng_tetap(penilai, sinyal, nomor):
    import torch
    torch.manual_seed(BENIH + nomor)
    np.random.seed(BENIH + nomor)
    with torch.no_grad():
        return penilai.kehilangan(potong_segmen(torch.from_numpy(sinyal).to(penilai.perangkat)))
