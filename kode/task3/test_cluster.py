"""Uji Task 3c-f pada embedding SINTETIS yang kebenarannya diketahui (CPU, ~1 menit).

  python -m kode.task3.test_cluster

Skenario (meniru dugaan struktur data asli):
  6 material = 3 komposisi x 2 tekstur.
  - embedding gambar   : hanya melihat TEKSTUR (2 kelompok) + pergeseran besar karena MIKROSKOP
  - embedding spektrum : hanya melihat KOMPOSISI (3 kelompok), tidak dipengaruhi mikroskop
Harapan:
  1. Uji mikroskop lolos -> koreksi dipakai; jika komposisi dibuat bergantung mikroskop -> TIDAK dipakai.
  2. Pemilihan k menemukan k = 6.
  3. IDEC pada latent FusionAE memulihkan 6 material (ARI > 0,9).
  4. Ablasi: gambar saja / spektrum saja kehilangan informasi (ARI jauh lebih rendah), dan
     pasangan material yang tidak bisa dibedakan muncul (fusion: tidak ada).
  5. Gambar saja tanpa koreksi mengelompok menurut mikroskop (Cramér's V tinggi).
"""
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from kode.task3 import cluster as C, fusion as Fu, select_k as K


def synth(n=3000, seed=0, mic_dep=False):
    rng = np.random.default_rng(seed)
    comp = rng.integers(0, 3, n)
    tex = rng.integers(0, 2, n)
    p_m2 = np.where(comp == 0, 0.8, 0.2) if mic_dep else np.full(n, 0.4)
    mic = np.where(rng.random(n) < p_m2, "M2", "M1")
    c_tex = rng.normal(size=(2, 256)) * 1.0
    shift = rng.normal(size=256) * 1.5
    img = c_tex[tex] + (mic == "M2")[:, None] * shift + rng.normal(size=(n, 256)) * 0.6
    c_comp = rng.normal(size=(3, 128)) * 1.0
    spec = c_comp[comp] + rng.normal(size=(n, 128)) * 0.6
    return {"gambar": img.astype(np.float32), "spektrum": spec.astype(np.float32)}, mic, comp * 2 + tex


def test_microscope_check():
    emb, mic, _ = synth(mic_dep=True)
    _, info = Fu.build_blocks(emb, mic, 0, log=lambda *_: None)
    assert not info["cek_mikroskop"]["koreksi_dipakai"], info["cek_mikroskop"]
    print(f"[OK] komposisi bergantung mikroskop -> koreksi TIDAK dipakai "
          f"(V maks {info['cek_mikroskop']['maks']:.2f})")


def test_pipeline():
    dev = torch.device("cpu")
    emb, mic, truth = synth()
    blocks, info = Fu.build_blocks(emb, mic, 0, log=lambda *_: None)
    assert info["cek_mikroskop"]["koreksi_dipakai"]
    X = np.concatenate([blocks["gambar"], blocks["spektrum"]], 1)
    splits = (blocks["gambar"].shape[1], blocks["spektrum"].shape[1])
    ae, ainfo = Fu.train_fusion_ae(X, splits, dev, 0, epochs=30, log=lambda *_: None)
    Z = Fu.encode(ae, X, dev)
    kdf = K.scan(Z, range(2, 10), 0, n_sil=2000, log=lambda *_: None)
    k, kdf = K.choose(kdf)
    assert k == 6, kdf
    print(f"[OK] pemilihan k = {k}")
    mdf, labels, agree, _ = C.run_methods({"concat": X, "FusionAE": Z}, X, ae, splits, k, mic, dev, 0,
                                          seeds=2, idec_epochs=20, log=lambda *_: None)
    final = labels[C.pick_final(mdf)]
    ari = adjusted_rand_score(truth, final)
    assert ari > 0.9, (ari, mdf)
    print(f"[OK] {C.pick_final(mdf)} memulihkan 6 material: ARI {ari:.3f}")
    abl, single = Fu.ablation({"gambar saja": blocks["gambar"], "spektrum saja": blocks["spektrum"],
                               "FusionAE": Z}, final, mic, k, 0, log=lambda *_: None)
    a = abl.set_index("representasi")
    for m in ("gambar saja", "spektrum saja"):
        assert a.loc[m, "ARI_vs_final"] < 0.7, a
        assert a.loc[m, "pasangan_final_tak_terpisah"] != "", a     # ada pasangan yang hilang
    assert a.loc["FusionAE", "pasangan_final_tak_terpisah"] == ""
    print("[OK] ablasi: ARI satu modalitas " +
          ", ".join(f"{m} {a.loc[m, 'ARI_vs_final']:.2f}" for m in ("gambar saja", "spektrum saja")) +
          f" vs FusionAE {a.loc['FusionAE', 'ARI_vs_final']:.2f}")
    # tanpa koreksi mikroskop, gambar saja mengelompok menurut alat
    z_raw, _ = Fu.prep_block(emb["gambar"], 0)
    from sklearn.cluster import KMeans
    v = Fu.cramers_v(KMeans(4, n_init=5, random_state=0).fit_predict(z_raw), mic)   # 2 tekstur x 2 mikroskop
    assert v > 0.5, v
    print(f"[OK] tanpa koreksi, cluster gambar saja vs mikroskop: Cramér's V {v:.2f}")


def test_refinement():
    """Blok gambar: 4 tekstur, tetapi label final menggabungkan tekstur 0 dan 1 (spektrum sama).
    Tekstur 3 terbagi dua oleh mikroskop (pergeseran besar). Harapan: 0|1 dipecah (k 3->4),
    lalu pemecahan berikutnya (tekstur 3 menurut mikroskop) DITOLAK karena Cramér's V tinggi."""
    rng = np.random.default_rng(0)
    n = 4000
    tex = rng.integers(0, 4, n)
    mic = np.where(rng.random(n) < 0.4, "M2", "M1")
    c = rng.normal(size=(4, 16)) * 3.0
    c[1] = c[0] + rng.normal(size=16) * 1.2           # tekstur 0 & 1 dekat, tetapi terpisah
    B = c[tex] + rng.normal(size=(n, 16)) * 0.3
    B[(tex == 3) & (mic == "M2")] += 2.0              # efek mikroskop hanya di tekstur 3
    final = np.select([tex <= 1, tex == 2], [0, 1], 2)
    lab, steps = C.refine_by_block(final, B, mic, 0, log=lambda *_: None)
    assert steps[0]["diterima"] and steps[0]["cramers_v_mikroskop"] < 0.1, steps
    assert adjusted_rand_score(tex, lab) > 0.98, adjusted_rand_score(tex, lab)
    assert not steps[-1]["diterima"] and len(steps) == 2, steps
    print(f"[OK] refinement: k 3->4 diterima (ARI {adjusted_rand_score(tex, lab):.3f}); "
          f"pemecahan menurut mikroskop ditolak (V {steps[-1]['cramers_v_mikroskop']:.2f})")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(False)
    test_microscope_check()
    test_refinement()
    test_pipeline()
