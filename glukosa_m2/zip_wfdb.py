"""Membaca rekaman WFDB format 16 langsung dari ZIP MIMIC-IV-ECG v1.0 (deflate, zip64) tanpa mengekstrak berkas.

Indeks anggota dibuat sekali dari direktori pusat ZIP; isi anggota dibaca dengan seek + zlib sehingga proses pekerja
tidak perlu memuat direktori pusat (±1,6 juta anggota) masing-masing.
"""
import re
import struct
import zipfile
import zlib
from dataclasses import dataclass

import numpy as np

AWALAN_ZIP = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
TANDA_HEADER_LOKAL = b"PK\x03\x04"
PANJANG_HEADER_LOKAL = 30
METODE_SIMPAN, METODE_DEFLATE = 0, 8
NILAI_HILANG = -32768
MODULUS_CHECKSUM = 65536
POLA_GAIN = re.compile(r"^([0-9.eE+-]+)(?:\((-?\d+)\))?")


@dataclass(frozen=True)
class LokasiAnggota:
    offset: int
    ukuran_mampat: int
    metode: int


@dataclass(frozen=True)
class Header:
    fs: float
    n_sampel: int
    sadapan: list
    gain: np.ndarray
    baseline: np.ndarray
    checksum: np.ndarray


@dataclass(frozen=True)
class Rekaman:
    sinyal_uv: np.ndarray
    header: Header
    checksum_cocok: bool


def nama_anggota(jalur_rekaman, akhiran):
    return f"{AWALAN_ZIP}{jalur_rekaman}.{akhiran}"


def indeks_anggota(jalur_zip, jalur_rekaman):
    dicari = {nama_anggota(j, akhiran) for j in jalur_rekaman for akhiran in ("hea", "dat")}
    with zipfile.ZipFile(jalur_zip) as arsip:
        return {info.filename: LokasiAnggota(info.header_offset, info.compress_size, info.compress_type)
                for info in arsip.infolist() if info.filename in dicari}


def baca_anggota(berkas, lokasi):
    berkas.seek(lokasi.offset)
    kepala = berkas.read(PANJANG_HEADER_LOKAL)
    if kepala[:4] != TANDA_HEADER_LOKAL:
        raise ValueError(f"header lokal ZIP rusak di offset {lokasi.offset}")
    panjang_nama, panjang_ekstra = struct.unpack("<HH", kepala[26:30])
    berkas.seek(lokasi.offset + PANJANG_HEADER_LOKAL + panjang_nama + panjang_ekstra)
    mampat = berkas.read(lokasi.ukuran_mampat)
    if lokasi.metode == METODE_SIMPAN:
        return mampat
    if lokasi.metode == METODE_DEFLATE:
        return zlib.decompress(mampat, -zlib.MAX_WBITS)
    raise ValueError(f"metode kompresi ZIP {lokasi.metode} tidak didukung")


def urai_header(teks):
    baris = [b for b in teks.splitlines() if b.strip() and not b.startswith("#")]
    medan_rekaman = baris[0].split()
    n_sinyal = int(medan_rekaman[1])
    sinyal = [b.split() for b in baris[1:1 + n_sinyal]]
    if any(s[1] != "16" for s in sinyal):
        raise ValueError("hanya format WFDB 16 yang didukung")
    gain, baseline = zip(*(urai_gain(s[2], int(s[4])) for s in sinyal))
    return Header(fs=float(medan_rekaman[2].split("/")[0]), n_sampel=int(medan_rekaman[3]),
                  sadapan=[s[8] for s in sinyal], gain=np.array(gain), baseline=np.array(baseline),
                  checksum=np.array([int(s[6]) for s in sinyal]))


def urai_gain(medan, adc_nol):
    cocok = POLA_GAIN.match(medan)
    if cocok is None:
        raise ValueError(f"medan gain tidak terbaca: {medan}")
    gain = float(cocok.group(1))
    baseline = int(cocok.group(2)) if cocok.group(2) is not None else adc_nol
    return gain, baseline


def urai_sinyal(data, header):
    n_sinyal = len(header.sadapan)
    if len(data) != 2 * n_sinyal * header.n_sampel:
        raise ValueError(f"ukuran .dat {len(data)} byte tidak sesuai header")
    adu = np.frombuffer(data, "<i2").reshape(header.n_sampel, n_sinyal).T
    jumlah = adu.sum(axis=1, dtype=np.int64) % MODULUS_CHECKSUM
    checksum_cocok = bool(np.array_equal(jumlah, header.checksum % MODULUS_CHECKSUM))
    uv = (adu - header.baseline[:, None]) / header.gain[:, None] * 1000.0
    return Rekaman(np.where(adu == NILAI_HILANG, np.nan, uv).astype(np.float32), header, checksum_cocok)


def baca_rekaman(berkas, indeks, jalur_rekaman):
    teks = baca_anggota(berkas, indeks[nama_anggota(jalur_rekaman, "hea")]).decode("ascii")
    header = urai_header(teks)
    return urai_sinyal(baca_anggota(berkas, indeks[nama_anggota(jalur_rekaman, "dat")]), header)
