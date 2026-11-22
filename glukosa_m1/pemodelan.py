import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from muat import DEMOGRAFI, UKURAN_EKG

BENIH = [20261120, 20261121, 20261122]
LIPATAN = 5
KUANTIL_PANGKAS = (0.005, 0.995)


class Pangkas(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.bawah_, self.atas_ = np.nanquantile(X, KUANTIL_PANGKAS, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.bawah_, self.atas_)


def buat_lr(benih):
    return make_pipeline(Pangkas(), SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=3000))


def buat_gb(benih):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, early_stopping=True,
                                          validation_fraction=0.1, random_state=benih)


MODEL = {
    "M0 usia + jenis kelamin (LR)": (DEMOGRAFI, buat_lr),
    "M1 ukuran EKG + M0 (LR)": (DEMOGRAFI + UKURAN_EKG, buat_lr),
    "M1 ukuran EKG + M0 (GB)": (DEMOGRAFI + UKURAN_EKG, buat_gb),
}


def latih_dan_prediksi(latih, nilai, kolom, buat):
    model = buat(BENIH[0]).fit(latih[kolom].values, latih.y.values)
    return model.predict_proba(nilai[kolom].values)


def peluang_luar_lipatan(bagian, kolom, buat):
    X, y, gugus = bagian[kolom].values, bagian.y.values, bagian.subject_id.values
    rata = np.zeros(len(y))
    for benih in BENIH:
        pembagi = StratifiedGroupKFold(LIPATAN, shuffle=True, random_state=benih)
        for latih, uji in pembagi.split(X, y, gugus):
            rata[uji] += buat(benih).fit(X[latih], y[latih]).predict_proba(X[uji])[:, 1] / len(BENIH)
    return rata


def koefisien_terstandar(latih):
    kolom = DEMOGRAFI + UKURAN_EKG
    model = buat_lr(BENIH[0]).fit(latih[kolom].values, latih.y.values)
    return dict(zip(kolom, model[-1].coef_[0]))
