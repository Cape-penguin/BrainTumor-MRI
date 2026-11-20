"""Export TensorBoard scalars of a run to PNG figures (for slides).

    python plot_tb.py runs/clip_hardmut            # -> runs/clip_hardmut/figures/
    python plot_tb.py runs/clip_hardmut --dpi 300

Outputs
  figures/all/<tag>.png      one figure per TensorBoard scalar tag (train/*, lr/*, epoch/*, val/*)
  figures/01_*.png ...       curated summary figures for presentation
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# reference palette (light mode), assigned in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
TEXT, TEXT2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "text.color": TEXT, "axes.titlecolor": TEXT, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 10, "lines.linewidth": 2, "font.family": "DejaVu Sans",
})

LABELS = {
    "train/loss": ("Training loss (per step)", "loss"),
    "train/grad_norm": ("Gradient norm (before clipping)", "L2 norm"),
    "train/logit_scale": ("Logit scale (1 / temperature)", "scale"),
    "lr/backbone": ("Learning rate: CLIP backbone", "lr"),
    "lr/new": ("Learning rate: new modules", "lr"),
    "epoch/train_loss": ("Training loss (epoch mean)", "loss"),
    "epoch/time_sec": ("Epoch wall time", "seconds"),
    "epoch/peak_vram_gb": ("Peak GPU memory", "GB"),
    "val/mean_R@5": ("Validation mean R@5 (model selection)", "%"),
    "val/bank_label_P@5": ("Validation label precision@5 (train report bank)", "%"),
}


def load(tb_dir):
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        ev = ea.Scalars(tag)
        out[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def smooth(y, w=0.6):
    s, out = y[0], []
    for v in y:
        s = w * s + (1 - w) * v
        out.append(s)
    return np.array(out)


def label_for(tag):
    if tag in LABELS:
        return LABELS[tag]
    split, name = tag.split("/", 1)
    unit = "rank" if "MedR" in name else ("MRR" if "MRR" in name else "%")
    d = {"i2t": "image→text", "t2i": "text→image"}
    pre, metric = (name.split("_", 1) + [""])[:2] if "_" in name else (name, "")
    title = f"{'Validation' if split == 'val' else split.title()} {d.get(pre, pre)} {metric}".strip()
    return title, unit


def finish(fig, ax, path, xlabel):
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=ARGS.dpi)
    plt.close(fig)
    print("saved", path)


def is_epoch_tag(tag):
    return tag.startswith(("epoch/", "val/", "test/"))


def xs(tag, steps):
    return steps + 1 if is_epoch_tag(tag) else steps  # epochs shown 1-indexed


def plot_single(tag, steps, vals, path, best_epoch=None):
    title, unit = label_for(tag)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = xs(tag, steps)
    if tag == "train/loss" or tag == "train/grad_norm":
        ax.plot(x, vals, color=SERIES[0], alpha=0.25, linewidth=1)
        ax.plot(x, smooth(vals), color=SERIES[0])
        ax.text(0.99, 0.97, "faint: raw · solid: EMA 0.6", transform=ax.transAxes,
                ha="right", va="top", color=MUTED, fontsize=9)
    else:
        ax.plot(x, vals, color=SERIES[0], marker="o" if len(x) <= 40 else None, markersize=4)
    if tag.startswith("lr/"):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if best_epoch is not None and is_epoch_tag(tag):
        ax.axvline(best_epoch + 1, color=MUTED, linestyle=":", linewidth=1.2)
        ax.text(best_epoch + 1, ax.get_ylim()[1], f" best epoch {best_epoch + 1}", color=TEXT2,
                fontsize=9, va="top")
    ax.set_title(title)
    ax.set_ylabel(unit)
    finish(fig, ax, path, "epoch" if is_epoch_tag(tag) else "optimizer step")


def plot_multi(series, title, ylabel, path, best_epoch=None, baseline=None, ylim=None):
    """series: list of (label, x(epoch 1-indexed), y). Direct labels at line ends + legend."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (lab, x, y) in enumerate(series):
        ax.plot(x, y, color=SERIES[i], marker="o", markersize=4, label=lab)
    if baseline is not None:
        ax.axhline(baseline[1], color=MUTED, linestyle="--", linewidth=1.2, label=baseline[0])
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    lo, hi = ax.get_ylim()  # final range: series + baseline
    # end-of-line value labels, pushed apart so they never overlap
    gap = 0.06 * (hi - lo)
    ends = sorted(((y[-1], i) for i, (_, _, y) in enumerate(series)))
    placed = []
    for v, i in ends:
        pos = max(v, placed[-1] + gap) if placed else v
        placed.append(pos)
        lab, x, y = series[i]
        ax.annotate(f"{y[-1]:.1f}" if abs(y[-1]) >= 1 else f"{y[-1]:.3f}", (x[-1], y[-1]),
                    xytext=(x[-1] + 0.6, pos), textcoords="data", va="center", color=TEXT2, fontsize=9)
        ax.plot([x[-1] + 0.15, x[-1] + 0.5], [y[-1], pos], color=SERIES[i], linewidth=1)
    if best_epoch is not None:
        ax.axvline(best_epoch + 1, color=MUTED, linestyle=":", linewidth=1.2)
        ax.text(best_epoch + 1, hi, f" best epoch {best_epoch + 1}", color=TEXT2, fontsize=9, va="top")
    ax.set_ylim(lo, hi)
    ax.set_xlim(right=series[0][1][-1] + 3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("epoch")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(series) + (baseline is not None))
    fig.tight_layout()
    fig.savefig(path, dpi=ARGS.dpi)
    plt.close(fig)
    print("saved", path)


def main():
    run = ARGS.run
    out = os.path.join(run, "figures")
    os.makedirs(os.path.join(out, "all"), exist_ok=True)
    S = load(os.path.join(run, "tb"))
    n_test = len(json.load(open(os.path.join(run, "split.json")))["test"])
    n_val = len(json.load(open(os.path.join(run, "split.json")))["val"])
    best_epoch = int(S["val/mean_R@5"][0][np.argmax(S["val/mean_R@5"][1])])

    # ---- one PNG per tag (test/* are single points -> shown in the summary bar chart instead)
    for tag, (st, v) in S.items():
        if tag.startswith("test/"):
            continue
        plot_single(tag, st, v, os.path.join(out, "all", tag.replace("/", "__").replace("@", "at") + ".png"),
                    best_epoch)

    ep = lambda t: S[t][0] + 1

    # 01: loss curves (two panels, shared x; no dual axis)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    st, v = S["train/loss"]
    a1.plot(st, v, color=SERIES[0], alpha=0.25, linewidth=1)
    a1.plot(st, smooth(v), color=SERIES[0])
    a1.set_title("Training loss")
    a1.set_ylabel("loss (per step)")
    a1.set_xlabel("optimizer step")
    a1.text(0.99, 0.95, "faint: raw · solid: EMA 0.6", transform=a1.transAxes, ha="right", va="top",
            color=MUTED, fontsize=9)
    a2.plot(ep("epoch/train_loss"), S["epoch/train_loss"][1], color=SERIES[0], marker="o", markersize=4)
    a2.set_ylabel("loss (epoch mean)")
    a2.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "01_train_loss.png"), dpi=ARGS.dpi)
    plt.close(fig)
    print("saved 01_train_loss.png")

    # 02: val R@5 both directions + mean
    plot_multi([("image→text", ep("val/i2t_R@5"), S["val/i2t_R@5"][1]),
                ("text→image", ep("val/t2i_R@5"), S["val/t2i_R@5"][1]),
                ("mean", ep("val/mean_R@5"), S["val/mean_R@5"][1])],
               f"Validation Recall@5 (n={n_val})", "Recall@5 (%)", os.path.join(out, "02_val_recall5.png"),
               best_epoch, baseline=(f"random {500 / n_val:.1f}%", 500 / n_val), ylim=(0, 100))

    # 03: val i2t R@1/5/10
    plot_multi([(f"R@{k}", ep(f"val/i2t_R@{k}"), S[f"val/i2t_R@{k}"][1]) for k in (1, 5, 10)],
               f"Validation image→text Recall@K (n={n_val})", "Recall (%)",
               os.path.join(out, "03_val_i2t_recallK.png"), best_epoch, ylim=(0, 100))

    # 04: val MedR
    plot_multi([("image→text", ep("val/i2t_MedR"), S["val/i2t_MedR"][1]),
                ("text→image", ep("val/t2i_MedR"), S["val/t2i_MedR"][1])],
               f"Validation median rank (lower is better, n={n_val})", "median rank",
               os.path.join(out, "04_val_medr.png"), best_epoch,
               baseline=(f"random {(n_val + 1) / 2:.1f}", (n_val + 1) / 2))

    # 05: val MRR
    plot_multi([("image→text", ep("val/i2t_MRR"), S["val/i2t_MRR"][1]),
                ("text→image", ep("val/t2i_MRR"), S["val/t2i_MRR"][1])],
               f"Validation mean reciprocal rank (n={n_val})", "MRR", os.path.join(out, "05_val_mrr.png"),
               best_epoch, ylim=(0, 1))

    # 06: train loss vs val mean R@5 (overfitting view), stacked panels with shared epoch axis
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(ep("epoch/train_loss"), S["epoch/train_loss"][1], color=SERIES[0], marker="o", markersize=4)
    a1.set_title("Training loss vs. validation retrieval")
    a1.set_ylabel("train loss")
    a2.plot(ep("val/mean_R@5"), S["val/mean_R@5"][1], color=SERIES[1], marker="o", markersize=4)
    a2.set_ylabel("val mean R@5 (%)")
    a2.set_xlabel("epoch")
    for a in (a1, a2):
        a.axvline(best_epoch + 1, color=MUTED, linestyle=":", linewidth=1.2)
    a2.text(best_epoch + 1, a2.get_ylim()[0], f" best epoch {best_epoch + 1}", color=TEXT2, fontsize=9,
            va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "06_loss_vs_val.png"), dpi=ARGS.dpi)
    plt.close(fig)
    print("saved 06_loss_vs_val.png")

    # 07: learning rate schedule (log scale, one axis)
    fig, axs = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    for i, (t, name) in enumerate([("lr/backbone", "CLIP backbone"), ("lr/new", "new modules (aggregator etc.)")]):
        axs[i].plot(S[t][0], S[t][1], color=SERIES[0])
        axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[i].set_ylabel("learning rate")
        axs[i].text(0.99, 0.92, name, transform=axs[i].transAxes, ha="right", va="top", color=TEXT2)
    axs[0].set_title("Learning rate schedule (5% warmup + cosine decay)")
    axs[1].set_xlabel("optimizer step")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "07_lr.png"), dpi=ARGS.dpi)
    plt.close(fig)
    print("saved 07_lr.png")

    # 08: test metrics bar chart
    if "test/i2t_R@1" in S:
        ks = ["R@1", "R@5", "R@10"]
        i2t = [S[f"test/i2t_{k}"][1][-1] for k in ks]
        t2i = [S[f"test/t2i_{k}"][1][-1] for k in ks]
        rnd = [100 * int(k[2:]) / n_test for k in ks]
        x = np.arange(len(ks))
        w = 0.36
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        b1 = ax.bar(x - w / 2 - 0.01, i2t, w, color=SERIES[0], label="image→text")
        b2 = ax.bar(x + w / 2 + 0.01, t2i, w, color=SERIES[1], label="text→image")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}",
                        ha="center", va="bottom", color=TEXT, fontsize=10)
        for xi, r in zip(x, rnd):
            ax.hlines(r, xi - w - 0.05, xi + w + 0.05, color=TEXT2, linestyle="--", linewidth=1.2)
        ax.plot([], [], color=TEXT2, linestyle="--", linewidth=1.2, label="random")
        ax.set_xticks(x, ks)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Recall (%)")
        ax.set_title(f"Test retrieval (n={n_test}, best-val checkpoint, epoch {best_epoch + 1})")
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "08_test_recall.png"), dpi=ARGS.dpi)
        plt.close(fig)
        print("saved 08_test_recall.png")

        # test summary table as image
        rows = [("Recall@1 (%)", "R@1"), ("Recall@5 (%)", "R@5"), ("Recall@10 (%)", "R@10"),
                ("Median rank", "MedR"), ("MRR", "MRR")]
        rand = {"R@1": f"{100 / n_test:.0f}", "R@5": f"{500 / n_test:.0f}", "R@10": f"{1000 / n_test:.0f}",
                "MedR": f"{(n_test + 1) / 2:.1f}", "MRR": "≈0.05"}
        cell = [[n, f"{S[f'test/i2t_{k}'][1][-1]:.3g}", f"{S[f'test/t2i_{k}'][1][-1]:.3g}", rand[k]]
                for n, k in rows]
        cell.append(["Label P@5, train bank (%)", f"{S['test/bank_label_P@5'][1][-1]:.1f}", "–", "55.7"])
        fig, ax = plt.subplots(figsize=(8.5, 3.2))
        ax.axis("off")
        tb = ax.table(cellText=cell, colLabels=["Metric", "image→text", "text→image", "random"],
                      loc="center", cellLoc="center", colLoc="center", colWidths=[0.36, 0.2, 0.2, 0.16])
        tb.auto_set_font_size(False)
        tb.set_fontsize(11)
        tb.scale(1, 1.6)
        for (r, c), cl in tb.get_celld().items():
            cl.set_edgecolor(GRID)
            if r == 0:
                cl.set_text_props(weight="bold", color=TEXT)
                cl.set_facecolor("#f0efec")
            if c == 0:
                cl.set_text_props(ha="left")
        ax.set_title(f"Test results (n={n_test})", loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "09_test_table.png"), dpi=ARGS.dpi)
        plt.close(fig)
        print("saved 09_test_table.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="runs/clip_hardmut")
    p.add_argument("--dpi", type=int, default=200)
    ARGS = p.parse_args()
    main()
