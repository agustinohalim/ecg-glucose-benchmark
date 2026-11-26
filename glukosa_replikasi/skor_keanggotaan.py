"""Skor keanggotaan pra-latih ECG-FM (protokol langkah 16b): kehilangan InfoNCE wav2vec 2.0 milik checkpoint pretrained
sendiri (kuantisator dan proyeksi bawaan, mode eval; topeng 0,65 × 10 langkah dan 100 negatif dari segmen yang sama,
semuanya dari konfigurasi checkpoint), dirata-rata atas langkah bertopeng dua segmen 5 s dan K undian topeng dan
negatif; ditambah kesesuaian segmen (kosinus rerata token keluaran kedua segmen tanpa topeng, pendekatan suku CMSC).
Benih torch dan numpy dipasang sekali oleh pemanggil sebelum rekaman pertama."""
from dataclasses import dataclass

from ecgfm import BOBOT, N_SEGMEN, PANJANG_SEGMEN

K_UNDIAN = 4


@dataclass(frozen=True)
class SkorRekaman:
    kehilangan: float
    kosinus: float


class PenilaiKeanggotaan:
    def __init__(self, perangkat):
        from fairseq_signals.models import build_model_from_checkpoint
        self.model = build_model_from_checkpoint(checkpoint_path=str(BOBOT)).to(perangkat).eval()
        self.perangkat = perangkat

    def nilai(self, sinyal):
        import torch
        segmen = potong_segmen(torch.from_numpy(sinyal).to(self.perangkat))
        with torch.no_grad():
            return SkorRekaman(self.kehilangan(segmen), self.kosinus(segmen))

    def kehilangan(self, segmen):
        import torch
        logit = self.model(source=segmen.repeat(K_UNDIAN, 1, 1), mask=True)["x"].float()
        return float(-torch.log_softmax(logit, dim=0)[0].mean())

    def kosinus(self, segmen):
        import torch
        token = self.model.extract_features(segmen, padding_mask=None)["x"].mean(dim=1)
        return float(torch.nn.functional.cosine_similarity(token[0], token[1], dim=0))


def potong_segmen(sinyal):
    return sinyal.reshape(sinyal.shape[0], N_SEGMEN, PANJANG_SEGMEN).permute(1, 0, 2).contiguous()
