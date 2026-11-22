"""M3: probe regresi logistik L2 di atas embedding ECGFounder beku, dibungkus sebagai penaksir bergaya scikit-learn.

X berisi dua kolom: nomor baris embedding dan subject_id (untuk CV dalam bergugus pasien). C dipilih dari kisi naskah
OMI (omi_m01.C_GRID) dengan log-loss rata-rata pada 3 lipatan dalam StratifiedGroupKFold di data latih saja (OMI: 5
lipatan; dikurangi karena data latih Glukosa ±17 kali lebih besar).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KISI_C = [1e-3, 1e-2, 1e-1, 1, 10, 100]
LIPATAN_DALAM = 3


def pipa(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def pilih_C(E, y, gugus, benih):
    pembagi = StratifiedGroupKFold(LIPATAN_DALAM, shuffle=True, random_state=benih)
    lipatan = list(pembagi.split(E, y, gugus))
    skor = {C: np.mean([log_loss(y[uji], pipa(C).fit(E[latih], y[latih]).predict_proba(E[uji]), labels=[0, 1])
                        for latih, uji in lipatan])
            for C in KISI_C}
    return min(skor, key=skor.get)


class ProbeLinier:
    def __init__(self, emb, benih):
        self.emb, self.benih = emb, benih

    def fit(self, X, y):
        E = self.embedding(X)
        self.C_ = pilih_C(E, y, X[:, 1], self.benih)
        self.pipa_ = pipa(self.C_).fit(E, y)
        return self

    def predict_proba(self, X):
        return self.pipa_.predict_proba(self.embedding(X))

    def embedding(self, X):
        return np.asarray(self.emb[X[:, 0].astype(np.int64)])


def pembuat_probe(emb):
    return lambda benih: ProbeLinier(emb, benih)
