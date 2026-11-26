"""Pembaca gelombang untuk uji keanggotaan (protokol langkah 16a–b): rekaman dibaca dari ZIP MIMIC-IV-ECG v1.0 di proses
pekerja dengan pra-proses ECG-FM (ecgfm.siapkan_sinyal) dan diserahkan satu per satu dalam urutan masukan, agar undian
topeng mengikuti urutan rekaman yang tetap. Rekaman yang gagal dibaca diserahkan sebagai None."""
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from bangun_singgahan import lokasi_rekaman
from ecgfm import siapkan_sinyal
from zip_wfdb import baca_rekaman, indeks_anggota

UKURAN_POTONGAN = 128
GALAT_BACA = (KeyError, IndexError, ValueError, zlib.error)


@dataclass(frozen=True)
class PotonganBaca:
    jalur_zip: str
    tugas: list


def baca_berurutan(jalur_zip, jalur_rekaman, pekerja):
    with ProcessPoolExecutor(pekerja) as kolam:
        antre = deque()
        for potongan in susun_potongan(jalur_zip, jalur_rekaman):
            antre.append(kolam.submit(baca_potongan, potongan))
            if len(antre) >= 2 * pekerja:
                yield from antre.popleft().result()
        while antre:
            yield from antre.popleft().result()


def susun_potongan(jalur_zip, jalur_rekaman):
    indeks = indeks_anggota(jalur_zip, jalur_rekaman)
    tugas = [(jalur, lokasi_rekaman(indeks, jalur)) for jalur in jalur_rekaman]
    return [PotonganBaca(jalur_zip, tugas[k:k + UKURAN_POTONGAN]) for k in range(0, len(tugas), UKURAN_POTONGAN)]


def baca_potongan(potongan):
    with open(potongan.jalur_zip, "rb") as berkas:
        return [baca_satu(berkas, lokasi, jalur) for jalur, lokasi in potongan.tugas]


def baca_satu(berkas, lokasi, jalur):
    try:
        return siapkan_sinyal(baca_rekaman(berkas, lokasi, jalur))
    except GALAT_BACA:  # protokol 16a: rekaman yang gagal dibaca dibuang dan dihitung oleh pemanggil
        return None
