"""
CLIP fine-tuning for brain-tumor MRI -> radiology report (findings) retrieval.

Data
  - metadata json: list of {"image": "AA0003", "meta": "...;num=30;label=0;...",
                            "conversations": [{"from": "human", ...}, {"from": "gpt", "value": <report>}]}
  - images: <image_root>/<case_id>/{T1,T2,FLAIR,T1C}/<case_id>_0001.png ... (224x224, grayscale stored as RGB)

Model
  - Each sampled axial slice is encoded by the CLIP ViT.
      fusion=composite : 3 modalities stacked as RGB channels per slice (default FLAIR,T1C,T2)
      fusion=separate  : each modality slice encoded separately (gray->RGB) + modality embedding
  - Slice tokens (+ slice position / modality embeddings) are aggregated by a small transformer
    into one case-level embedding, projected with CLIP's visual projection.
  - Text: CLIP text tower, positional embeddings stretched (Long-CLIP style) to 224 so that
    long reports (hardmut reports: median 90, max 214 tokens; IDH sentence at the end) are not truncated.
  - Loss: symmetric InfoNCE; oversampled duplicates of the same case in a batch are treated as positives.

Usage
  # train
  python clip_mri_report.py train --output_dir runs/clip_b16

  # top-k similar reports for a case (bank = train split reports by default)
  python clip_mri_report.py retrieve --ckpt runs/clip_b16/best.pt --case AA0003 --topk 5
  python clip_mri_report.py retrieve --ckpt runs/clip_b16/best.pt --case_dir /path/to/NEW_CASE --topk 5
  python clip_mri_report.py retrieve --ckpt runs/clip_b16/best.pt --split test --topk 5 --out test_top5.json
"""
import argparse
import json
import logging
import math
import os
import random
import re
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2 as T
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

DEFAULT_JSON = "/media/NAS66/huiseung/severance/medcap/findings_20250624_rewrite_split_all.json"
DEFAULT_TRAIN_JSON = "/media/NAS66/huiseung/severance/medcap/findings_20250624_rewrite_hardmut_split_train.json"
DEFAULT_TEST_JSON = "/media/NAS66/huiseung/severance/medcap/findings_20250624_rewrite_hardmut_split_test.json"
DEFAULT_IMG_ROOT = "/media/NAS66/huiseung/severance/1005_anony_png"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ----------------------------------------------------------------------------- data

def clean_report(text):
    text = re.sub(r"\s*End of radiology report\.?\s*$", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def parse_meta(meta):
    out = {}
    for kv in meta.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def load_records(json_path):
    with open(json_path) as f:
        data = json.load(f)
    records = []
    for x in data:
        report = next(c["value"] for c in x["conversations"] if c["from"] == "gpt")
        human = next((c["value"] for c in x["conversations"] if c["from"] == "human"), "")
        meta = parse_meta(x.get("meta", ""))
        age = re.search(r"Age:\s*(\d+)", human)
        sex = re.search(r"Sex:\s*(\w+)", human)
        records.append({
            "id": x["image"],
            "report": clean_report(report),
            "label": int(meta.get("label", -1)),
            "age": int(age.group(1)) if age else None,
            "sex": sex.group(1) if sex else None,
        })
    return records


def make_split(records, ratios=(0.8, 0.1, 0.1), seed=42):
    """Case-level split, stratified by label."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r["id"])
    split = {"train": [], "val": [], "test": []}
    for ids in by_label.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_val = round(len(ids) * ratios[1])
        n_test = round(len(ids) * ratios[2])
        split["test"] += ids[:n_test]
        split["val"] += ids[n_test:n_test + n_val]
        split["train"] += ids[n_test + n_val:]
    return {k: sorted(v) for k, v in split.items()}


def sample_slice_indices(n, k, train):
    """TSN-style: split the volume into k segments, take one slice per segment."""
    if n <= 0:
        raise ValueError("empty volume")
    edges = np.linspace(0, n, k + 1)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = int(math.floor(a)), max(int(math.floor(a)) + 1, int(math.ceil(b)))
        hi = min(hi, n)
        idx.append(random.randrange(lo, hi) if train else (lo + hi - 1) // 2)
    return idx


def list_slices(case_dir, modality):
    d = os.path.join(case_dir, modality)
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".png"))


def load_gray(path):
    return torch.from_numpy(np.array(Image.open(path).convert("L"), dtype=np.uint8))


def load_slice(case_dir, modality, files, i):
    """Load slice i; if the png is corrupt, fall back to the nearest readable neighbour."""
    for off in sorted(range(-len(files), len(files)), key=abs):
        j = i + off
        if 0 <= j < len(files):
            try:
                return load_gray(os.path.join(case_dir, modality, files[j]))
            except OSError:
                if off == 0:
                    print(f"[warn] corrupt slice {os.path.join(case_dir, modality, files[i])}")
    raise OSError(f"no readable slice in {case_dir}/{modality}")


class VolumeLoader:
    """Loads a case directory into a tensor of shape [S, M, 3, H, W] (float, CLIP-normalised)."""

    def __init__(self, modalities, num_slices, fusion, image_size=224, train=False):
        self.modalities = modalities
        self.num_slices = num_slices
        self.fusion = fusion
        self.train = train
        self.resize = T.Resize((image_size, image_size), antialias=True)
        # no horizontal flip: reports describe laterality (left/right)
        self.geo = T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))
        self.norm = T.Normalize(CLIP_MEAN, CLIP_STD)

    def __call__(self, case_dir):
        files = {m: list_slices(case_dir, m) for m in self.modalities}
        n = min(len(v) for v in files.values())
        idx = sample_slice_indices(n, self.num_slices, self.train)
        # vol: [S, M, H, W] uint8
        vol = torch.stack([
            torch.stack([load_slice(case_dir, m, files[m], i) for m in self.modalities])
            for i in idx
        ])
        S, M, H, W = vol.shape
        x = vol.float().div(255.0).view(S * M, 1, H, W)
        x = self.resize(x)
        if self.train:
            x = self.geo(x)  # same geometric transform for all slices / modalities of the case
            x = x.view(S, M, *x.shape[-2:])
            # per-modality intensity jitter
            gain = torch.empty(1, M, 1, 1).uniform_(0.9, 1.1)
            bias = torch.empty(1, M, 1, 1).uniform_(-0.05, 0.05)
            x = (x * gain + bias).clamp(0, 1).view(S * M, 1, *x.shape[-2:])
        h, w = x.shape[-2:]
        if self.fusion == "composite":
            x = x.view(S, 1, M, h, w)  # modalities -> RGB channels
        else:
            x = x.view(S, M, 1, h, w).expand(S, M, 3, h, w)
        return self.norm(x.reshape(-1, 3, h, w)).view(S, -1, 3, h, w)


class MRIReportDataset(Dataset):
    def __init__(self, records, image_root, loader):
        self.records = records
        self.image_root = image_root
        self.loader = loader

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return {"pixel_values": self.loader(os.path.join(self.image_root, r["id"])),
                "report": r["report"], "id": r["id"], "label": r["label"], "index": i}


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "report": [b["report"] for b in batch],
        "id": [b["id"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch]),
        "index": torch.tensor([b["index"] for b in batch]),
    }


# ----------------------------------------------------------------------------- model

def extend_text_positions(clip, new_len, keep=20):
    """Long-CLIP style: keep first `keep` positions, linearly stretch the rest."""
    emb = clip.text_model.embeddings
    old = emb.position_embedding.weight.data
    old_len, dim = old.shape
    if new_len <= old_len:
        return
    head = old[:keep]
    tail = F.interpolate(old[keep:].T.unsqueeze(0), size=new_len - keep, mode="linear",
                         align_corners=True).squeeze(0).T
    new = nn.Embedding(new_len, dim).to(old.device, old.dtype)
    new.weight.data.copy_(torch.cat([head, tail], 0))
    emb.position_embedding = new
    emb.register_buffer("position_ids", torch.arange(new_len).unsqueeze(0), persistent=False)
    clip.config.text_config.max_position_embeddings = new_len
    clip.text_model.config.max_position_embeddings = new_len


class MRIClip(nn.Module):
    def __init__(self, clip_name, num_slices, num_modalities, fusion, max_text_len=160,
                 agg_layers=2, unfreeze_vision_layers=4, unfreeze_text_layers=4, grad_ckpt=False):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        extend_text_positions(self.clip, max_text_len)
        self.fusion = fusion
        vdim = self.clip.config.vision_config.hidden_size
        n_img = 1 if fusion == "composite" else num_modalities

        self.slice_pos = nn.Parameter(torch.zeros(1, num_slices, 1, vdim))
        self.mod_emb = nn.Parameter(torch.zeros(1, 1, n_img, vdim))
        self.cls = nn.Parameter(torch.zeros(1, 1, vdim))
        nn.init.trunc_normal_(self.slice_pos, std=0.02)
        nn.init.trunc_normal_(self.mod_emb, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(vdim, nhead=8, dim_feedforward=vdim * 4, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.aggregator = nn.TransformerEncoder(layer, num_layers=agg_layers)
        self.agg_norm = nn.LayerNorm(vdim)

        self._set_trainable(unfreeze_vision_layers, unfreeze_text_layers)
        if grad_ckpt:
            self.clip.gradient_checkpointing_enable()

    def _set_trainable(self, n_vis, n_txt):
        for p in self.clip.parameters():
            p.requires_grad = False
        vm, tm = self.clip.vision_model, self.clip.text_model
        vis_layers = vm.encoder.layers
        txt_layers = tm.encoder.layers
        train_mods = [self.clip.visual_projection, self.clip.text_projection,
                      vm.post_layernorm, tm.final_layer_norm, tm.embeddings.position_embedding]
        if n_vis < 0 or n_vis >= len(vis_layers):
            train_mods.append(vm)
        else:
            train_mods += list(vis_layers[len(vis_layers) - n_vis:])
        if n_txt < 0 or n_txt >= len(txt_layers):
            train_mods.append(tm)
        else:
            train_mods += list(txt_layers[len(txt_layers) - n_txt:])
        for m in train_mods:
            for p in m.parameters():
                p.requires_grad = True
        self.clip.logit_scale.requires_grad = True

    def encode_image(self, pixel_values):
        # pixel_values: [B, S, M, 3, H, W]
        B, S, M = pixel_values.shape[:3]
        x = pixel_values.flatten(0, 2)
        feats = self.clip.vision_model(pixel_values=x).pooler_output  # [B*S*M, D]
        feats = feats.view(B, S, M, -1) + self.slice_pos[:, :S] + self.mod_emb[:, :, :M]
        tokens = torch.cat([self.cls.expand(B, -1, -1), feats.flatten(1, 2)], 1)
        pooled = self.agg_norm(self.aggregator(tokens)[:, 0])
        return F.normalize(self.clip.visual_projection(pooled), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(self.clip.text_projection(out.pooler_output), dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        img = self.encode_image(pixel_values)
        txt = self.encode_text(input_ids, attention_mask)
        return img, txt, self.clip.logit_scale.exp().clamp(max=100)


def clip_loss(img, txt, scale, case_ids=None):
    """Symmetric InfoNCE. Entries of the same case (oversampled duplicates) count as positives."""
    logits = scale * img @ txt.t()
    if case_ids is None:
        target = torch.arange(len(img), device=img.device)
        return (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2
    uniq = {c: i for i, c in enumerate(dict.fromkeys(case_ids))}
    idx = torch.tensor([uniq[c] for c in case_ids], device=img.device)
    pos = (idx[:, None] == idx[None, :]).float()
    target = pos / pos.sum(1, keepdim=True)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2


# ----------------------------------------------------------------------------- helpers

def tokenize(tokenizer, texts, max_len, device):
    tok = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return tok["input_ids"].to(device), tok["attention_mask"].to(device)


@torch.no_grad()
def encode_texts(model, tokenizer, texts, max_len, device, bs=64, amp_dtype=torch.bfloat16):
    out = []
    for i in range(0, len(texts), bs):
        ids, mask = tokenize(tokenizer, texts[i:i + bs], max_len, device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            out.append(model.encode_text(ids, mask).float())
    return torch.cat(out)


@torch.no_grad()
def encode_loader(model, loader, device, amp_dtype=torch.bfloat16):
    embs, ids, labels, reports = [], [], [], []
    for b in tqdm(loader, desc="encode images", leave=False):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            embs.append(model.encode_image(b["pixel_values"].to(device, non_blocking=True)).float())
        ids += b["id"]
        labels.append(b["label"])
        reports += b["report"]
    return torch.cat(embs), ids, torch.cat(labels), reports


def retrieval_metrics(sim, ks=(1, 5, 10)):
    """sim[i, j]: query i vs gallery j, ground truth on the diagonal."""
    ranks = (sim > sim.diag().unsqueeze(1)).sum(1) + 1
    out = {f"R@{k}": (ranks <= k).float().mean().item() * 100 for k in ks}
    out["MedR"] = ranks.float().median().item()
    out["MRR"] = (1.0 / ranks.float()).mean().item()
    return out


@torch.no_grad()
def evaluate(model, tokenizer, loader, args, device, bank=None):
    model.eval()
    img, ids, labels, reports = encode_loader(model, loader, device)
    txt = encode_texts(model, tokenizer, reports, args.max_text_len, device)
    sim = img @ txt.t()
    res = {f"i2t_{k}": v for k, v in retrieval_metrics(sim).items()}
    res.update({f"t2i_{k}": v for k, v in retrieval_metrics(sim.t()).items()})
    res["mean_R@5"] = (res["i2t_R@5"] + res["t2i_R@5"]) / 2
    if bank is not None:
        # realistic setting: retrieve from the training report bank, check label agreement
        bank_emb, bank_labels = bank
        topk = (img @ bank_emb.t()).topk(args.topk, dim=1).indices
        res[f"bank_label_P@{args.topk}"] = (
            (bank_labels[topk.cpu()] == labels.unsqueeze(1)).float().mean().item() * 100)
    return res


def build_model_from_args(a):
    return MRIClip(a["clip_model"], a["num_slices"], len(a["modalities"]), a["fusion"],
                   max_text_len=a["max_text_len"], agg_layers=a["agg_layers"],
                   unfreeze_vision_layers=a["unfreeze_vision_layers"],
                   unfreeze_text_layers=a["unfreeze_text_layers"], grad_ckpt=a.get("grad_ckpt", False))


# ----------------------------------------------------------------------------- train

def setup_logger(path):
    logger = logging.getLogger("clip_mri")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(path), logging.StreamHandler()):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def load_data(a, split=None):
    """Returns (rec_by_id: one record per case, split: case ids, train_entries).

    train_json/test_json given: test = test_json cases, val = `val_ratio` of the train_json cases
    (all their duplicate entries removed from training). train_entries keeps the oversampled
    duplicates of train_json. Otherwise falls back to `json` (+ optional test_json) as before.
    """
    if a.get("train_json"):
        entries = load_records(a["train_json"])
        test = load_records(a["test_json"])
        uniq = list({r["id"]: r for r in entries}.values())
        rec_by_id = {r["id"]: r for r in uniq + test}
        if split is None:
            sub = make_split(uniq, ratios=(1 - a["val_ratio"], a["val_ratio"], 0.0), seed=a["seed"])
            split = {"train": sub["train"], "val": sub["val"], "test": sorted(r["id"] for r in test)}
    else:
        entries = load_records(a["json"])
        rec_by_id = {r["id"]: r for r in entries}
        if split is None and a.get("test_json"):
            test_ids = {x["image"] for x in json.load(open(a["test_json"]))}
            pool = [r for r in entries if r["id"] not in test_ids]
            sub = make_split(pool, ratios=(1 - a["val_ratio"], a["val_ratio"], 0.0), seed=a["seed"])
            split = {"train": sub["train"], "val": sub["val"], "test": sorted(test_ids)}
        elif split is None:
            split = make_split(entries, seed=a["seed"])
    train_ids = set(split["train"])
    return rec_by_id, split, [r for r in entries if r["id"] in train_ids]


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def list_epoch_ckpts(ckpt_dir):
    fs = [f for f in os.listdir(ckpt_dir) if re.fullmatch(r"epoch_\d+\.pt", f)]
    return sorted(fs, key=lambda f: int(re.findall(r"\d+", f)[0]))


def fmt_metrics(res):
    return " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items())


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, "train.log"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # pick GPU via CUDA_VISIBLE_DEVICES
    if device.type == "cuda" and args.max_vram_gb > 0:
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_gb / total_gb), device)
        logger.info(f"GPU {torch.cuda.get_device_name(device)} ({total_gb:.0f}GB), "
                    f"limited to {args.max_vram_gb}GB")
    if args.fusion == "composite" and len(args.modalities) != 3:
        raise ValueError("fusion=composite needs exactly 3 modalities (RGB channels)")

    split = json.load(open(args.split_file)) if args.split_file else None
    rec_by_id, split, train_entries = load_data(vars(args), split)
    json.dump(split, open(os.path.join(args.output_dir, "split.json"), "w"), indent=1)
    json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"), indent=1)
    logger.info(f"args: {json.dumps(vars(args))}")
    logger.info(f"split (cases): { {k: len(v) for k, v in split.items()} }, "
                f"train entries (with oversampling): {len(train_entries)}")

    def make_loader(name, train_mode):
        loader = VolumeLoader(args.modalities, args.num_slices, args.fusion, train=train_mode)
        recs = train_entries if name == "train" else [rec_by_id[i] for i in split[name]]
        ds = MRIReportDataset(recs, args.image_root, loader)
        return DataLoader(ds, batch_size=args.batch_size if train_mode else args.eval_batch_size,
                          shuffle=train_mode, drop_last=train_mode, num_workers=args.num_workers,
                          collate_fn=collate, pin_memory=True, persistent_workers=args.num_workers > 0)

    train_loader = make_loader("train", True)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)

    tokenizer = CLIPTokenizer.from_pretrained(args.clip_model)
    model = build_model_from_args(vars(args)).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable params: {n_train / 1e6:.1f}M / "
                f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    backbone, new, no_decay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "logit_scale" in n or n in ("slice_pos", "mod_emb", "cls"):
            no_decay.append(p)
        elif n.startswith("clip."):
            backbone.append(p)
        else:
            new.append(p)
    optim = torch.optim.AdamW([
        {"params": backbone, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": new, "lr": args.lr_new, "weight_decay": args.weight_decay},
        {"params": no_decay, "lr": args.lr_new, "weight_decay": 0.0},
    ])
    total = args.epochs * len(train_loader) // args.grad_accum
    warmup = int(total * args.warmup_ratio)
    base_lrs = [g["lr"] for g in optim.param_groups]

    def set_lr(step):
        f = step / max(1, warmup) if step < warmup else \
            0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
        for g, lr in zip(optim.param_groups, base_lrs):
            g["lr"] = lr * f

    train_texts = [rec_by_id[i]["report"] for i in split["train"]]
    train_labels = torch.tensor([rec_by_id[i]["label"] for i in split["train"]])

    # ---- resume from the latest epoch checkpoint (server interruption)
    best, best_epoch, step, start_epoch, log = -1.0, -1, 0, 0, []
    existing = list_epoch_ckpts(ckpt_dir)
    if args.resume and existing:
        path = os.path.join(ckpt_dir, existing[-1])
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        best, best_epoch, step, log = ck["best"], ck["best_epoch"], ck["step"], ck["log"]
        start_epoch = ck["epoch"] + 1
        torch.set_rng_state(ck["rng"]["torch"])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        logger.info(f"resumed from {path} (next epoch {start_epoch}, best {best:.2f} @ {best_epoch})")

    writer = SummaryWriter(os.path.join(args.output_dir, "tb"), purge_step=step if step else None)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        running = 0.0
        for it, b in enumerate(pbar):
            ids, mask = tokenize(tokenizer, b["report"], args.max_text_len, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                img, txt, scale = model(b["pixel_values"].to(device, non_blocking=True), ids, mask)
            loss = clip_loss(img.float(), txt.float(), scale.float(), b["id"]) / args.grad_accum
            loss.backward()
            if (it + 1) % args.grad_accum == 0:
                set_lr(step)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    writer.add_scalar("train/loss", loss.item() * args.grad_accum, step)
                    writer.add_scalar("train/grad_norm", gnorm.item(), step)
                    writer.add_scalar("train/logit_scale", scale.item(), step)
                    writer.add_scalar("lr/backbone", optim.param_groups[0]["lr"], step)
                    writer.add_scalar("lr/new", optim.param_groups[1]["lr"], step)
            running += loss.item() * args.grad_accum
            pbar.set_postfix(loss=f"{running / (it + 1):.4f}", scale=f"{scale.item():.1f}")

        train_loss = running / len(train_loader)
        epoch_time = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.0
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/time_sec", epoch_time, epoch)
        writer.add_scalar("epoch/peak_vram_gb", peak_gb, epoch)
        res = {"epoch": epoch, "train_loss": train_loss}

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            bank = (encode_texts(model, tokenizer, train_texts, args.max_text_len, device), train_labels)
            val = evaluate(model, tokenizer, val_loader, args, device, bank)
            for k, v in val.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            res.update(val)
            if val["mean_R@5"] > best:
                best, best_epoch = val["mean_R@5"], epoch
                atomic_save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val": val},
                            os.path.join(args.output_dir, "best.pt"))
        log.append(res)
        logger.info(f"[epoch {epoch + 1}/{args.epochs}] {fmt_metrics(res)} | lr={optim.param_groups[0]['lr']:.2e} "
                    f"time={epoch_time:.0f}s peak_vram={peak_gb:.1f}GB | best mean_R@5={best:.2f} @ epoch {best_epoch}")
        writer.flush()
        json.dump(log, open(os.path.join(args.output_dir, "log.json"), "w"), indent=1)

        # ---- rolling checkpoints: keep the latest `keep_last` (+ best.pt separately)
        atomic_save({"model": model.state_dict(), "optim": optim.state_dict(), "args": vars(args),
                     "epoch": epoch, "step": step, "best": best, "best_epoch": best_epoch, "log": log,
                     "rng": {"torch": torch.get_rng_state(), "python": random.getstate(),
                             "numpy": np.random.get_state()}},
                    os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))
        for old in list_epoch_ckpts(ckpt_dir)[:-args.keep_last]:
            os.remove(os.path.join(ckpt_dir, old))

    ckpt = torch.load(os.path.join(args.output_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    bank = (encode_texts(model, tokenizer, train_texts, args.max_text_len, device), train_labels)
    res = evaluate(model, tokenizer, test_loader, args, device, bank)
    for k, v in res.items():
        writer.add_scalar(f"test/{k}", v, ckpt["epoch"])
    writer.close()
    logger.info(f"TEST (best val ckpt, epoch {ckpt['epoch']}): {fmt_metrics(res)}")
    json.dump(res, open(os.path.join(args.output_dir, "test_metrics.json"), "w"), indent=1)


# ----------------------------------------------------------------------------- retrieve

@torch.no_grad()
def retrieve(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    model = build_model_from_args(a)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(a["clip_model"])

    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    if not os.path.exists(os.path.join(run_dir, "split.json")):  # checkpoints/epoch_XXX.pt
        run_dir = os.path.dirname(run_dir)
    rec_by_id, split, _ = load_data(a, json.load(open(os.path.join(run_dir, "split.json"))))
    bank_ids = split["train"] if args.bank == "train" else list(rec_by_id)
    bank_recs = [rec_by_id[i] for i in bank_ids]

    cache = os.path.join(run_dir, f"bank_{args.bank}_{os.path.basename(args.ckpt)}")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(args.ckpt):
        bank_emb = torch.load(cache).to(device)
    else:
        bank_emb = encode_texts(model, tokenizer, [r["report"] for r in bank_recs], a["max_text_len"], device)
        torch.save(bank_emb.cpu(), cache)

    loader = VolumeLoader(a["modalities"], a["num_slices"], a["fusion"], train=False)
    if args.case_dir:
        queries = [(os.path.basename(os.path.normpath(args.case_dir)), args.case_dir)]
    elif args.case:
        queries = [(c, os.path.join(a["image_root"], c)) for c in args.case]
    else:
        queries = [(c, os.path.join(a["image_root"], c)) for c in split[args.split]]

    results = []
    for qid, qdir in tqdm(queries, desc="retrieve", disable=len(queries) < 5):
        x = loader(qdir).unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            q = model.encode_image(x).float()
        sim = (q @ bank_emb.t()).squeeze(0)
        if args.bank == "all":  # don't return the query's own report
            sim[[i for i, b in enumerate(bank_ids) if b == qid]] = -2
        scores, idx = sim.topk(args.topk)
        res = {"query": qid,
               "gt_report": rec_by_id[qid]["report"] if qid in rec_by_id else None,
               "topk": [{"rank": r + 1, "id": bank_ids[i], "score": round(s, 4),
                         "label": bank_recs[i]["label"], "report": bank_recs[i]["report"]}
                        for r, (s, i) in enumerate(zip(scores.tolist(), idx.tolist()))]}
        results.append(res)
        if len(queries) <= 5:
            print(f"\n=== {qid}")
            if res["gt_report"]:
                print(f"[GT] {res['gt_report']}")
            for t in res["topk"]:
                print(f"  #{t['rank']} ({t['score']:.3f}) [{t['id']}, label={t['label']}] {t['report']}")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
        print(f"saved {len(results)} results -> {args.out}")


# ----------------------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--train_json", default=DEFAULT_TRAIN_JSON,
                   help="train json (may contain oversampled duplicate cases); '' = use --json instead")
    t.add_argument("--json", default=DEFAULT_JSON, help="single json with all cases (used when --train_json '')")
    t.add_argument("--image_root", default=DEFAULT_IMG_ROOT)
    t.add_argument("--output_dir", default="runs/clip_mri")
    t.add_argument("--split_file", default=None, help="reuse an existing split.json")
    t.add_argument("--test_json", default=DEFAULT_TEST_JSON, help="test split json")
    t.add_argument("--val_ratio", type=float, default=0.1, help="val fraction of non-test cases")
    t.add_argument("--clip_model", default="openai/clip-vit-base-patch16")
    t.add_argument("--modalities", nargs="+", default=["FLAIR", "T1C", "T2"])
    t.add_argument("--fusion", choices=["composite", "separate"], default="composite")
    t.add_argument("--num_slices", type=int, default=12)
    t.add_argument("--max_text_len", type=int, default=224)
    t.add_argument("--agg_layers", type=int, default=2)
    t.add_argument("--unfreeze_vision_layers", type=int, default=4, help="-1 = all")
    t.add_argument("--unfreeze_text_layers", type=int, default=4, help="-1 = all")
    t.add_argument("--grad_ckpt", action="store_true")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--eval_batch_size", type=int, default=16)
    t.add_argument("--grad_accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=1e-5, help="CLIP backbone lr")
    t.add_argument("--lr_new", type=float, default=1e-4, help="aggregator / new params lr")
    t.add_argument("--weight_decay", type=float, default=0.05)
    t.add_argument("--warmup_ratio", type=float, default=0.05)
    t.add_argument("--eval_every", type=int, default=1)
    t.add_argument("--topk", type=int, default=5)
    t.add_argument("--num_workers", type=int, default=8)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--max_vram_gb", type=float, default=0, help="cap GPU memory for this process (0 = no cap)")
    t.add_argument("--keep_last", type=int, default=5, help="number of latest epoch checkpoints to keep")
    t.add_argument("--log_every", type=int, default=10, help="tensorboard logging interval (steps)")
    t.add_argument("--no_resume", dest="resume", action="store_false",
                   help="do not resume from output_dir/checkpoints")

    r = sub.add_parser("retrieve")
    r.add_argument("--ckpt", required=True)
    g = r.add_mutually_exclusive_group()
    g.add_argument("--case", nargs="+", help="case id(s) under image_root, e.g. AA0003")
    g.add_argument("--case_dir", help="path to a case folder with modality subfolders")
    g.add_argument("--split", choices=["train", "val", "test"], default="test")
    r.add_argument("--bank", choices=["train", "all"], default="train")
    r.add_argument("--topk", type=int, default=5)
    r.add_argument("--out", default=None)

    args = p.parse_args()
    train(args) if args.cmd == "train" else retrieve(args)


if __name__ == "__main__":
    main()
