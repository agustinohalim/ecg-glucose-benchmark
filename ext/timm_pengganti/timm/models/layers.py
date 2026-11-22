"""Pengganti minimal timm.models.layers untuk inferensi beku ECG-JEPA: hanya DropPath dan trunc_normal_ yang diimpor
ecg_jepa.py. DropPath identitas di mode eval (sama dengan timm); trunc_normal_ hanya inisialisasi yang ditimpa bobot."""
from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        tetap = 1 - self.drop_prob
        acak = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(tetap)
        return x * acak / tetap


def trunc_normal_(tensor, *posisi, **pilihan):
    return nn.init.trunc_normal_(tensor, *posisi, **pilihan)
