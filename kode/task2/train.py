"""Task 2d - loop training model sequence, dengan checkpoint per epoch (resume Colab).

Keputusan:
- Tiap epoch mengambil `samples_per_epoch` window acak dari ~200 rb window train (bukan semua):
  window yang bertetangga hampir identik, jadi subsample per epoch jauh lebih hemat waktu
  dengan informasi yang hampir sama. Generator di-seed per epoch -> resume tetap reproducible.
- Loss MSE pada RUL/150. Optimizer AdamW + OneCycle LR. Early stopping pada RMSE val
  (val terpotong seperti test), bobot terbaik yang dipakai.
- AMP (float16) otomatis di GPU untuk kecepatan.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from kode.common.utils import load_checkpoint, save_checkpoint
from kode.task2.models import build

SCALE = 150.0


@torch.no_grad()
def predict(model, bank, idx, bs: int = 4096, mc: int = 0) -> np.ndarray:
    """mc=0: prediksi deterministik (dropout mati). mc>0: MC dropout, kembalikan (mc, N)."""
    model.eval()
    if mc:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    pos_all = torch.from_numpy(bank.positions(idx)).to(bank.device)
    reps = []
    for _ in range(max(mc, 1)):
        out = []
        for k in range(0, len(pos_all), bs):
            out.append(model(bank.get(pos_all[k:k + bs])).float() * SCALE)
        reps.append(torch.cat(out).clamp(0, SCALE).cpu().numpy())
    model.eval()
    return np.stack(reps) if mc else reps[0]


def train_model(name, bank_tr, idx_tr, bank_va, idx_va, cfg, ckpt_path, device, seed, log=print):
    torch.manual_seed(seed)
    model = build(name, bank_tr.bank.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps_per_epoch = cfg["samples_per_epoch"] // cfg["bs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], epochs=cfg["epochs"],
                                                steps_per_epoch=steps_per_epoch, pct_start=0.2)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pos_tr = bank_tr.positions(idx_tr)
    y_tr = torch.from_numpy(bank_tr.labels(idx_tr) / SCALE).to(device)
    y_va = bank_va.labels(idx_va)

    state = {"epoch": 0, "best_rmse": float("inf"), "best_state": None, "history": [], "bad": 0,
             "done": False, "train_sec": 0.0}
    ck = load_checkpoint(ckpt_path, map_location=device) if cfg.get("resume", True) else None
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        # pulihkan RNG (dropout) agar hasil resume identik dengan training tanpa putus
        torch.set_rng_state(ck["rng_cpu"])
        if ck.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        log(f"    resume {name} dari epoch {state['epoch']} (best RMSE {state['best_rmse']:.2f})")

    while not state["done"] and state["epoch"] < cfg["epochs"]:
        ep = state["epoch"]
        t0 = time.time()
        model.train()
        rng = np.random.default_rng(seed * 1000 + ep)
        sel = torch.from_numpy(rng.choice(len(pos_tr), steps_per_epoch * cfg["bs"], replace=False)).to(device)
        pos_t = torch.from_numpy(pos_tr).to(device)
        tot = 0.0
        for b in range(steps_per_epoch):
            j = sel[b * cfg["bs"]:(b + 1) * cfg["bs"]]
            x, y = bank_tr.get(pos_t[j]), y_tr[j]
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                loss = nn.functional.mse_loss(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        p_va = predict(model, bank_va, idx_va)
        r = float(np.sqrt(np.mean((p_va - y_va) ** 2)))
        state["train_sec"] += time.time() - t0
        state["history"].append({"epoch": ep + 1, "train_rmse": float(np.sqrt(tot / steps_per_epoch)) * SCALE,
                                 "val_rmse": r})
        if r < state["best_rmse"] - 1e-3:
            state["best_rmse"], state["bad"] = r, 0
            state["best_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state["bad"] += 1
        state["epoch"] = ep + 1
        if state["bad"] >= cfg["patience"]:
            state["done"] = True
        if state["epoch"] >= cfg["epochs"]:
            state["done"] = True
        save_checkpoint(ckpt_path, model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), scaler=scaler.state_dict(), state=state,
                        rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        log(f"    {name} ep {ep + 1:2d}/{cfg['epochs']} train RMSE {state['history'][-1]['train_rmse']:.2f} "
            f"val RMSE {r:.2f} ({time.time() - t0:.1f}s)")

    model.load_state_dict(state["best_state"])
    return model, state
