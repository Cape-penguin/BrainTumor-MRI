"""Detailed test-set analysis of a trained run (per-label / IDH / laterality / baselines).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python analyze_results.py runs/clip_hardmut
"""
import json, os, re, sys, collections
import numpy as np, torch
run = sys.argv[1] if len(sys.argv) > 1 else "runs/clip_hardmut"
sys.argv = sys.argv[:1]
import clip_mri_report as c
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer

dev = torch.device("cuda:0")
ck = torch.load(f"{run}/best.pt", map_location="cpu", weights_only=False)
a = ck["args"]
m = c.build_model_from_args(a); m.load_state_dict(ck["model"]); m.to(dev).eval()
tok = CLIPTokenizer.from_pretrained(a["clip_model"])
rec, split, _ = c.load_data(a, json.load(open(f"{run}/split.json")))
idh = lambda t: "WT" if "wildtype" in t else "MUT"
def side(t):
    t = t.lower(); l = bool(re.search(r"\bleft\b", t)); r = bool(re.search(r"\bright\b", t))
    return "both" if l and r else "L" if l else "R" if r else "none"

def ranks_for(name):
    ld = c.VolumeLoader(a["modalities"], a["num_slices"], a["fusion"], train=False)
    ds = c.MRIReportDataset([rec[i] for i in split[name]], a["image_root"], ld)
    img, ids, labels, reps = c.encode_loader(m, DataLoader(ds, batch_size=16, num_workers=8, collate_fn=c.collate), dev)
    txt = c.encode_texts(m, tok, reps, a["max_text_len"], dev)
    sim = img @ txt.t()
    r_i2t = ((sim > sim.diag()[:, None]).sum(1) + 1).cpu().numpy()
    return img, ids, labels.numpy(), reps, r_i2t, sim

img, ids, labels, reps, r, sim = ranks_for("test")
N = len(ids)
print(f"N={N}  random baseline: R@1={100/N:.1f} R@5={500/N:.1f} R@10={1000/N:.1f} MedR={(N+1)/2:.1f}")
print("i2t rank distribution:", {k: int((r <= k).sum()) for k in [1, 5, 10, 20, 50]}, "max", r.max())
print("\nper label (i2t, test gallery):")
for L in sorted(set(labels)):
    s = r[labels == L]
    print(f"  label {L}: n={len(s)} R@1={100*(s<=1).mean():.1f} R@5={100*(s<=5).mean():.1f} R@10={100*(s<=10).mean():.1f} MedR={np.median(s):.0f}")
iv = np.array([idh(t) for t in reps])
for k in ["WT", "MUT"]:
    s = r[iv == k]; print(f"  IDH {k}: n={len(s)} R@5={100*(s<=5).mean():.1f} MedR={np.median(s):.0f}")

# label of top-1 within test gallery
top1 = sim.argmax(1).cpu().numpy()
print("\nwithin-test top1 same label:", (labels[top1] == labels).mean())

# bank analysis
bank_ids = split["train"]; bank = [rec[i] for i in bank_ids]
be = c.encode_texts(m, tok, [b["report"] for b in bank], a["max_text_len"], dev)
bs = img @ be.t(); top = bs.topk(5, 1).indices.cpu().numpy()
bl = np.array([b["label"] for b in bank]); bi = np.array([idh(b["report"]) for b in bank]); bside = np.array([side(b["report"]) for b in bank])
print("\nbank: n=", len(bank), "label dist", collections.Counter(bl.tolist()), "IDH dist", collections.Counter(bi.tolist()))
cm = collections.Counter((iv[q], bi[top[q, 0]]) for q in range(N)); print("IDH confusion (gt, top1):", dict(cm))
lc = collections.Counter((int(labels[q]), int(bl[top[q, 0]])) for q in range(N)); print("label confusion (gt, top1):", dict(sorted(lc.items())))
p5 = (bl[top] == labels[:, None]).mean(1)
print("label P@5 by label:", {int(L): round(100*p5[labels == L].mean(), 1) for L in sorted(set(labels))})
pr = collections.Counter(bl.tolist()); print("random-pick label P@5 expected:", round(100*sum((labels == L).mean() * pr[L] / len(bl) for L in pr), 1))
i5 = (bi[top] == iv[:, None]).mean(1); print("IDH P@5:", round(100*i5.mean(), 1), "random expected:", round(100*sum((iv == k).mean() * (bi == k).mean() for k in ["WT", "MUT"]), 1))
# laterality
qs = np.array([side(t) for t in reps]); mask = np.isin(qs, ["L", "R"])
same = [(bside[top[q, 0]] == qs[q]) for q in range(N) if mask[q]]
same5 = [np.mean(bside[top[q]] == qs[q]) for q in range(N) if mask[q]]
print(f"\nlaterality (queries with only L or R: {mask.sum()}): top1 same side {100*np.mean(same):.1f}%  top5 same side {100*np.mean(same5):.1f}%  bank L/R dist", collections.Counter(bside.tolist()))
# score spread
print("\nbank cos sim: top1 mean %.3f, top5 mean %.3f, overall mean %.3f std %.3f" % (bs.max(1).values.mean(), bs.topk(5,1).values.mean(), bs.mean(), bs.std()))
# val vs test
log = json.load(open(f"{run}/log.json")); b = max(log, key=lambda x: x.get("mean_R@5", -1))
print("\nbest val epoch(0-idx)", b["epoch"], "val mean_R@5", round(b["mean_R@5"], 1), "val i2t_R@5", round(b["i2t_R@5"], 1), "| last epoch val mean_R@5", round(log[-1]["mean_R@5"], 1))
