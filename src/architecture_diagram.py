"""Draws reports/figures/architecture.png (system design) - regenerated with: python -m src.architecture_diagram"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.config import FIG_DIR

INK, BLUE, ORANGE, TEAL, GREY = "#16213A", "#2F6DB5", "#C2571A", "#1E7F78", "#5A6270"


def box(ax, x, y, w, h, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=col, lw=1.8))
    ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top", fontsize=8.6, weight="bold", color=col)
    ax.text(x + w / 2, y + h - 0.47, sub, ha="center", va="top", fontsize=6.9, color=INK, linespacing=1.3)


def arrow(ax, a, b, col=GREY, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, color=col, lw=1.3, linestyle=ls))


def main():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(0, 11); ax.set_ylim(-0.3, 6.4); ax.axis("off")
    ax.text(0.1, 6.25, "OFFLINE  (training pipeline)", fontsize=9, weight="bold", color=GREY)
    ax.text(0.1, 3.05, "ONLINE / BATCH  (serving)", fontsize=9, weight="bold", color=GREY)
    ax.plot([0.1, 10.9], [3.25, 3.25], ls=":", c=GREY, lw=1)
    # offline row
    box(ax, 0.1, 4.3, 1.75, 1.7, "fetch_data.py", "public CSV\nSHA-256 verified", INK)
    box(ax, 2.15, 4.3, 2.0, 1.7, "data_pipeline.py", "clean (row-wise)\nstratified 60/20/20\nleakage audit", INK)
    box(ax, 4.45, 4.3, 2.1, 1.7, "train.py", "baselines + LR + LightGBM\nclass-weighted loss\nCV tuning, Platt calibration\ncost-based threshold (val)", BLUE)
    box(ax, 6.85, 4.3, 1.75, 1.7, "evaluate.py", "test set, used once\nPR-AUC, bootstrap CI\nnet benefit (USD)", BLUE)
    box(ax, 8.9, 4.3, 2.0, 1.7, "artifacts/", "churn_model.joblib\nmodel_metadata.json\n(threshold, fallback)\nreference_profile.json", TEAL)
    for a, b in [((1.85, 5.15), (2.15, 5.15)), ((4.15, 5.15), (4.45, 5.15)), ((6.55, 5.15), (6.85, 5.15)),
                 ((8.6, 5.15), (8.9, 5.15))]:
        arrow(ax, a, b)
    # online row
    box(ax, 0.1, 0.9, 1.75, 1.8, "Clients", "CRM / retention app\nPOST /predict\nPOST /predict/batch", INK)
    box(ax, 2.15, 0.9, 2.0, 1.8, "schemas.py", "Pydantic contract\nenums, ranges, extra=forbid\ncross-field rules\n422 / 413 errors", ORANGE)
    box(ax, 4.45, 0.9, 2.1, 1.8, "app.py (FastAPI)", "/health  /model-info\nrequest id + latency header\nJSON logs\nprediction log (JSONL)", ORANGE)
    box(ax, 6.85, 0.9, 1.75, 1.8, "model_service.py", "load once at start-up\nscore vectorised\nfallback rule if the\nmodel fails", ORANGE)
    box(ax, 8.9, 0.9, 2.0, 1.8, "monitoring.py", "PSI + KS per feature\nscore drift, unseen values\nexit 2 = alert ->\nretrain / page on-call", TEAL)
    for a, b in [((1.85, 1.8), (2.15, 1.8)), ((4.15, 1.8), (4.45, 1.8)), ((6.55, 1.8), (6.85, 1.8))]:
        arrow(ax, a, b)
    arrow(ax, (9.9, 4.3), (7.75, 2.7), TEAL, ls="--")
    ax.text(7.1, 3.55, "artifacts loaded at start-up\n(no training code in serving)", fontsize=6.8, color=TEAL, ha="right")
    arrow(ax, (8.6, 1.3), (8.9, 1.3), TEAL, ls="--")
    ax.text(8.75, 0.72, "prediction log -> drift monitor", fontsize=6.8, color=TEAL, ha="center")
    box(ax, 0.1, -0.25, 4.05, 0.85, "predict_batch.py (CLI)", "same schema + model_service; rejects file; ~24k records/s", ORANGE)
    fig.tight_layout(); FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "architecture.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
