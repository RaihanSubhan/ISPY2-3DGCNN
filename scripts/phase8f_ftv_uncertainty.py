"""
Phase 8F: Uncertainty quantification for the FTV pCR model.

The Phase 8C results were point estimates only. For an n=13 pilot, a reviewer will
ask for confidence intervals and a significance test. This adds both, computed from
the saved per-patient predictions (no retraining):
  - bootstrap 95% CIs for AUROC, AUPRC, Brier
  - a label-permutation p-value for AUROC (is it better than chance?)

This does NOT change the result; it quantifies how uncertain the pilot result is.
With 13 patients the CIs are wide and may include 0.5 / chance, which is the honest,
expected outcome and the evidence for 'pilot feasibility, not validation'.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

REPO = Path.home() / "ISPY2-3DGCNN"
REPORTS = REPO / "reports"
TABLES = REPORTS / "tables"
FIGS = REPORTS / "figures"
for p in [REPORTS, TABLES, FIGS]:
    p.mkdir(parents=True, exist_ok=True)

PRED = TABLES / "phase8c_ftv_model_predictions.csv"


def bootstrap_ci(y, p, metric_fn, needs_two_classes, rng, B=5000):
    n = len(y)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        ys, ps = y[idx], p[idx]
        if needs_two_classes and len(np.unique(ys)) < 2:
            continue
        try:
            vals.append(metric_fn(ys, ps))
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan, 0
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(vals)


def permutation_p_auroc(y, p, rng, B=10000):
    obs = roc_auc_score(y, p)
    cnt, tot = 0, 0
    for _ in range(B):
        ys = rng.permutation(y)
        if len(np.unique(ys)) < 2:
            continue
        tot += 1
        if roc_auc_score(ys, p) >= obs:
            cnt += 1
    return float(obs), float((cnt + 1) / (tot + 1)), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not PRED.exists():
        raise SystemExit("Missing reports/tables/phase8c_ftv_model_predictions.csv. Run Phase 8C first.")
    df = pd.read_csv(PRED)
    rng = np.random.default_rng(args.seed)

    models = ["SVM_RBF", "GradientBoosting", "RandomForest", "LogisticRegression"]
    models = [m for m in models if m in df["model"].unique()]

    ci_rows, perm_rows = [], []
    for model in models:
        d = df[df["model"] == model]
        y = d["true_label"].to_numpy().astype(int)
        p = d["predicted_probability"].to_numpy().astype(float)

        au = float(roc_auc_score(y, p))
        ap_ = float(average_precision_score(y, p))
        br = float(brier_score_loss(y, p))

        au_lo, au_hi, au_n = bootstrap_ci(y, p, roc_auc_score, True, rng, args.boot)
        ap_lo, ap_hi, ap_n = bootstrap_ci(y, p, average_precision_score, True, rng, args.boot)
        br_lo, br_hi, br_n = bootstrap_ci(y, p, brier_score_loss, False, rng, args.boot)
        obs_au, perm_p, perm_n = permutation_p_auroc(y, p, rng, args.perm)

        for metric, est, lo, hi, nv in [
            ("AUROC", au, au_lo, au_hi, au_n),
            ("AUPRC", ap_, ap_lo, ap_hi, ap_n),
            ("Brier", br, br_lo, br_hi, br_n),
        ]:
            ci_rows.append({"model": model, "metric": metric, "estimate": round(est, 4),
                            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                            "n_valid_bootstrap": nv})
        perm_rows.append({"model": model, "observed_AUROC": round(obs_au, 4),
                          "permutation_p_value": round(perm_p, 4), "n_valid_permutations": perm_n,
                          "significant_at_0.05": bool(perm_p < 0.05)})
        print(f"{model}: AUROC {au:.3f} [{au_lo:.3f},{au_hi:.3f}] perm_p={perm_p:.3f}")

    ci_df = pd.DataFrame(ci_rows)
    perm_df = pd.DataFrame(perm_rows)
    ci_df.to_csv(TABLES / "phase8f_ftv_metric_confidence_intervals.csv", index=False)
    perm_df.to_csv(TABLES / "phase8f_ftv_permutation_pvalues.csv", index=False)

    au = ci_df[ci_df["metric"] == "AUROC"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.8 * len(au) + 1.5))
    yps = np.arange(len(au))[::-1]
    for yp, (_, r) in zip(yps, au.iterrows()):
        ax.plot([r["ci95_low"], r["ci95_high"]], [yp, yp], color="#444", lw=2)
        ax.plot(r["estimate"], yp, "o", color="#d6455f", ms=9)
    ax.axvline(0.5, color="grey", ls="--", lw=1, label="chance (AUROC 0.5)")
    ax.set_yticks(yps)
    ax.set_yticklabels(au["model"])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("AUROC (point estimate, 95% bootstrap CI)")
    ax.set_title("Phase 8F: FTV pCR model AUROC with 95% bootstrap CI (n=13 pilot)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "270_phase8f_ftv_auroc_ci_forest.png", dpi=140)
    plt.close(fig)

    best = perm_df.sort_values("observed_AUROC", ascending=False).iloc[0]
    best_ci = au[au["model"] == best["model"]].iloc[0]
    includes_chance = best_ci["ci95_low"] <= 0.5
    sig = bool(best["permutation_p_value"] < 0.05)

    (REPORTS / "Phase8F_FTV_Uncertainty_Report.md").write_text(
        "# Phase 8F: FTV pCR Model Uncertainty\n\n"
        "Bootstrap 95% confidence intervals and a label-permutation p-value for the FTV pCR\n"
        "model, computed from the saved per-patient predictions (no retraining).\n\n"
        f"## Best model: {best['model']}\n"
        f"- AUROC {best_ci['estimate']} (95% CI {best_ci['ci95_low']} to {best_ci['ci95_high']})\n"
        f"- Permutation p-value (AUROC vs chance): {best['permutation_p_value']}\n"
        f"- 95% CI includes chance (0.5): {'YES' if includes_chance else 'NO'}\n"
        f"- Statistically significant at 0.05: {'YES' if sig else 'NO'}\n\n"
        "## Honest interpretation\n"
        + ("The 95% CI includes 0.5 and the permutation test is not significant, so with this\n"
           "cohort the model's discrimination cannot be distinguished from chance. Report the\n"
           "result strictly as a feasibility signal, never as a working pCR predictor. The wide\n"
           "interval is the direct consequence of n=13 and is the core argument for 'pilot, not\n"
           "validation'.\n" if (includes_chance or not sig) else
           "The 95% CI excludes chance, an encouraging pilot signal; still report as pilot given n.\n")
        + "\n## Outputs\n"
        "- reports/tables/phase8f_ftv_metric_confidence_intervals.csv\n"
        "- reports/tables/phase8f_ftv_permutation_pvalues.csv\n"
        "- reports/figures/270_phase8f_ftv_auroc_ci_forest.png\n\n"
        "## Limitation\nn=13 labeled FTV patients. CIs are wide; this is a pilot feasibility result.\n")

    print("[done] wrote Phase 8F tables, figure, and report.")


if __name__ == "__main__":
    main()
