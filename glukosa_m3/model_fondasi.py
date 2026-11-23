"""Daftar model fondasi beku untuk M3: nama di tabel, dimensi embedding, akhiran berkas prediksi, dan pembanding utama.
Modul tiap model menyediakan siapkan_sinyal(rekaman) dan muat_penanam(perangkat)."""
import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFondasi:
    kode: str
    nama: str
    label: str
    dimensi: int
    pembanding: str
    label_pembanding: str
    prediksi_tambahan: tuple

    @property
    def modul(self):
        return importlib.import_module(self.kode)

    @property
    def berkas_emb(self):
        return f"emb_{self.kode}.npy"

    @property
    def berkas_selesai(self):
        return f"emb_{self.kode}_selesai.npy"

    @property
    def akhiran(self):
        return f"_{self.label.lower()}"


ECGFOUNDER = ModelFondasi("ecgfounder", "M3 ECGFounder beku + probe L2", "M3", 1024, "M2 CNN gelombang", "M2", ())
ECGJEPA = ModelFondasi("ecgjepa", "M3b ECG-JEPA beku + probe L2", "M3b", 768, ECGFOUNDER.nama, "M3", ("_m3",))
ECGFM = ModelFondasi("ecgfm", "M3c ECG-FM beku + probe L2 (terkontaminasi)", "M3c", 768, ECGJEPA.nama, "M3b", ())
ECGFM_FT = ModelFondasi("ecgfm_ft", "ECG-FM fine-tuned beku + probe L2 (kontrol positif)", "FT", 768, ECGJEPA.nama,
                        "M3b", ())
MODEL_FONDASI = {m.kode: m for m in (ECGFOUNDER, ECGJEPA, ECGFM, ECGFM_FT)}
