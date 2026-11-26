"""Latih lanjut swaawas ECG-FM (protokol §F langkah 17b dan catatan prosesnya): model, kriteria `wav2vec2_with_cmsc`, dan
Adam diambil dari checkpoint pretrained sendiri (momen Adam dan suhu Gumbel dilanjutkan dari pembaruan terakhirnya);
penyamaran sadapan acak per segmen dengan `RandomLeadsMask` fairseq-signals; pasangan CMSC = dua segmen satu rekaman;
gradien satu pembaruan dibagi jumlah langkah bertopeng semua batch mikronya, seperti pelatih fairseq; fp32."""
import random

import numpy as np

from ecgfm import BOBOT, N_SEGMEN, PANJANG_SEGMEN


class PelatihLanjut:
    def __init__(self, perangkat, benih):
        import torch
        from fairseq_signals.criterions.wav2vec2_criterion import Wav2Vec2WithCMSCCriterion
        from fairseq_signals.data.ecg.augmentations import RandomLeadsMask
        from fairseq_signals.models import build_model_from_checkpoint
        titik = torch.load(BOBOT, map_location="cpu", weights_only=False)
        self.model = build_model_from_checkpoint(checkpoint_path=str(BOBOT)).to(perangkat).train()
        from omegaconf import OmegaConf
        self.kriteria = Wav2Vec2WithCMSCCriterion(OmegaConf.create(titik["cfg"]["criterion"]), None)
        tugas = titik["cfg"]["task"]
        self.penyamar = RandomLeadsMask(p=tugas["p"][0], mask_leads_selection=tugas["mask_leads_selection"],
                                        mask_leads_prob=tugas["mask_leads_prob"],
                                        mask_leads_condition=tugas["mask_leads_condition"])
        self.pengoptimal = adam_lanjutan(self.model, titik["last_optimizer_state"], perangkat)
        self.pembaruan = int(titik["optimizer_history"][-1]["num_updates"])
        self.model.set_num_updates(self.pembaruan)
        self.perangkat = perangkat
        pasang_benih(benih)

    def langkah_mikro(self, sinyal):
        import torch
        segmen = torch.from_numpy(sinyal).reshape(len(sinyal), sinyal.shape[1], N_SEGMEN, PANJANG_SEGMEN)
        segmen = segmen.permute(0, 2, 1, 3).reshape(-1, sinyal.shape[1], PANJANG_SEGMEN)
        rekaman = torch.arange(len(sinyal)).repeat_interleave(N_SEGMEN).to(self.perangkat)
        sampel = {"net_input": {"source": torch.stack([self.penyamar(s) for s in segmen]).to(self.perangkat)},
                  "id": rekaman, "patient_id": rekaman,
                  "segment": torch.arange(N_SEGMEN).repeat(len(sinyal)).to(self.perangkat)}
        kehilangan, langkah_bertopeng, log = self.kriteria(self.model, sampel)
        kehilangan.backward()
        return float(langkah_bertopeng), {k: float(v) for k, v in log.items() if k.startswith("loss")}

    def perbarui(self, langkah_bertopeng):
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(max(langkah_bertopeng, 1.0))
        self.pengoptimal.step()
        self.pengoptimal.zero_grad(set_to_none=True)
        self.pembaruan += 1
        self.model.set_num_updates(self.pembaruan)

    def simpan(self, jalur, epoch, urutan):
        import torch
        keadaan = {"model": self.model.state_dict(), "adam": self.pengoptimal.state_dict(), "pembaruan": self.pembaruan,
                   "epoch": epoch, "acak": keadaan_acak(), "urutan": urutan.bit_generator.state}
        sementara = jalur.with_suffix(".tmp")
        torch.save(keadaan, sementara)
        sementara.replace(jalur)

    def lanjutkan(self, jalur, urutan):
        import torch
        keadaan = torch.load(jalur, map_location="cpu", weights_only=False)
        self.model.load_state_dict(keadaan["model"])
        self.pengoptimal.load_state_dict(keadaan["adam"])
        self.pembaruan = keadaan["pembaruan"]
        self.model.set_num_updates(self.pembaruan)
        pulihkan_acak(keadaan["acak"])
        urutan.bit_generator.state = keadaan["urutan"]
        return keadaan["epoch"]


def adam_lanjutan(model, lama, perangkat):
    import torch
    parameter = [p for p in model.parameters() if p.requires_grad]
    grup = lama["param_groups"][0]
    if len(lama["state"]) != len(parameter):
        raise SystemExit(f"Keadaan Adam {len(lama['state'])} parameter, model {len(parameter)}.")
    adam = torch.optim.AdamW(parameter, lr=grup["lr"], betas=tuple(grup["betas"]), eps=grup["eps"],
                             weight_decay=grup["weight_decay"])
    for indeks, p in enumerate(parameter):
        lalu = lama["state"][indeks]
        if lalu["exp_avg"].shape != p.shape:
            raise SystemExit(f"Bentuk momen Adam parameter {indeks} tidak cocok.")
        adam.state[p] = {"step": torch.tensor(float(lalu["step"])), "exp_avg": lalu["exp_avg"].to(perangkat),
                         "exp_avg_sq": lalu["exp_avg_sq"].to(perangkat)}
    return adam


def pasang_benih(benih):
    import torch
    torch.manual_seed(benih)
    np.random.seed(benih)
    random.seed(benih)


def keadaan_acak():
    import torch
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(), "numpy": np.random.get_state(),
            "random": random.getstate()}


def pulihkan_acak(keadaan):
    import torch
    torch.set_rng_state(keadaan["torch"])
    torch.cuda.set_rng_state_all(keadaan["cuda"])
    np.random.set_state(keadaan["numpy"])
    random.setstate(keadaan["random"])
