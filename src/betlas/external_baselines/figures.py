from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from ..constants import FOLD_LABELS
from ..publication import colors as cz_colors
from ..publication import figure_style as cz_style


def _normalise_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norm, 1e-12)


def _balanced_sample(
    meta: pd.DataFrame,
    *,
    max_per_label: int,
    seed: int,
) -> pd.DataFrame:
    parts = []
    for _label, part in meta.groupby("fold_label_final", sort=True):
        n = min(max_per_label, len(part))
        parts.append(part.sample(n=n, random_state=seed) if len(part) > n else part)
    return pd.concat(parts, ignore_index=True)


def _label_palette(labels: list[str]) -> dict[str, str]:
    return cz_colors.fold_palette(labels)


def _display_fold_label(label: str) -> str:
    labels = {
        "beta_barrel": "beta barrel",
        "beta_prism": "beta prism",
        "beta_propeller": "beta propeller",
        "jelly_roll": "jelly roll",
        "beta_solenoid": "beta solenoid",
        "beta_sandwich": "beta sandwich",
        "tim_like_beta_alpha_barrel": "TIM-like beta-alpha barrel",
    }
    return labels.get(label, label.replace("_", " "))


def _publication_rc() -> dict[str, object]:
    return cz_style.publication_rc(pad_inches=0.018)


def _style_axis(ax: object) -> None:
    cz_style.style_axis(ax)


def _save_publication_figure(fig: object, path: Path) -> None:
    cz_style.save_figure(fig, path, formats=(path.suffix.lstrip(".") or "png",), png_dpi=450, tif_dpi=600, pad_inches=0.018)


def _mutual_knn_edges(sim: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    top = np.argsort(sim, axis=1)[:, ::-1][:, :k]
    top_sets = [set(row.tolist()) for row in top]
    edges: list[tuple[int, int, float]] = []
    for i, neighbors in enumerate(top):
        for j in neighbors:
            j = int(j)
            if i < j and i in top_sets[j]:
                edges.append((i, j, float(sim[i, j])))
    return edges


def _binned_quantiles(x: np.ndarray, y: np.ndarray, bins: int = 16) -> pd.DataFrame:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return pd.DataFrame(columns=["x", "q25", "median", "q75"])
    breaks = np.linspace(np.nanquantile(x, 0.005), np.nanquantile(x, 0.995), bins + 1)
    rows = []
    for lo, hi in zip(breaks[:-1], breaks[1:], strict=False):
        mask = (x >= lo) & (x < hi)
        if mask.sum() < 25:
            continue
        rows.append(
            {
                "x": float(np.nanmedian(x[mask])),
                "q25": float(np.nanquantile(y[mask], 0.25)),
                "median": float(np.nanmedian(y[mask])),
                "q75": float(np.nanquantile(y[mask], 0.75)),
            }
        )
    return pd.DataFrame(rows)


def _panel_label(ax: object, label: str, *, x: float = -0.12, y: float = 1.06) -> None:
    return None


def _model_display_name(model: str) -> str:
    names = {
        "xgboost_tuned": "Betlas XGBoost",
        "grammar_rules": "Geometry rules",
        "esmc_cosine_knn_k1": "ESM-C kNN, k=1",
        "esmc_cosine_knn_k5": "ESM-C kNN, k=5",
        "esmc_cosine_knn_k10": "ESM-C kNN, k=10",
        "foldseek_nearest_neighbor": "Foldseek structural NN",
    }
    return names.get(model, model.replace("_", " "))


def _model_palette() -> dict[str, str]:
    return cz_colors.MODEL_COLORS


def _short_fold_label(label: str, *, with_breaks: bool = False) -> str:
    labels = {
        "beta_barrel": "Beta barrel",
        "beta_prism": "Beta prism",
        "beta_propeller": "Beta propeller",
        "jelly_roll": "Jelly roll",
        "beta_solenoid": "Beta solenoid",
        "beta_sandwich": "Beta sandwich",
        "tim_like_beta_alpha_barrel": "TIM-like beta-alpha barrel",
    }
    value = labels.get(label, label.replace("_", " ").title())
    return value.replace(" beta-alpha ", "\nbeta-alpha ") if with_breaks else value


def _fold_abbrev(label: str) -> str:
    labels = {
        "beta_barrel": "Barrel",
        "beta_prism": "Prism",
        "beta_propeller": "Propeller",
        "jelly_roll": "Jelly roll",
        "beta_solenoid": "Solenoid",
        "beta_sandwich": "Sandwich",
        "tim_like_beta_alpha_barrel": "TIM barrel",
    }
    return labels.get(label, label.replace("_", " ").title())


def _clean_ablation_label(name: str) -> str:
    labels = {
        "sheet_sequence_topology": "Sheet sequence topology",
        "sheet_global": "Sheet global geometry",
        "composition": "Composition",
        "contact_graph": "Contact graph",
        "cz_sheet_pair_size_balance": "Sheet-pair size balance",
        "cz_sheet_order_adjacent_seq_step_mean": "Adjacent-strand sequence step",
        "sheet_global+sheet_sequence_topology": "Sheet global + sequence topology",
        "contact_graph+sheet_global+sheet_sequence_topology": "Contact graph + sheet global + sequence topology",
    }
    return labels.get(name, name.replace("cz_", "").replace("_", " ").replace("+", " + ").title())


def _contrast_text_color(value: float, cmap: object) -> str:
    r, g, b, _alpha = cmap(value)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return cz_colors.TEXT if luminance > 0.54 else "white"


def _nature_scalar_cmap(name: str = "nature_scalar") -> object:
    if name == "esmc_distance":
        return cz_colors.matplotlib_cmap("distance")
    return cz_colors.matplotlib_cmap("performance")


def _nature_density_cmap(name: str = "nature_density") -> object:
    return cz_colors.matplotlib_cmap("density")


def _metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    valid = pd.Series(y_pred).astype(str) != ""
    y_true_valid = y_true[valid.to_numpy()]
    y_pred_valid = y_pred[valid.to_numpy()]
    if len(y_true_valid) == 0:
        return float("nan")
    if metric == "macro_f1":
        return float(f1_score(y_true_valid, y_pred_valid, labels=list(FOLD_LABELS), average="macro", zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y_true_valid, y_pred_valid))
    raise ValueError(f"Unsupported metric: {metric}")


def _bootstrap_metric_ci(
    predictions: pd.DataFrame,
    *,
    metric: str,
    seed: int,
    bootstrap_n: int = 1000,
) -> tuple[float, float, float]:
    pred = predictions[["group", "true_label", "pred_label"]].copy()
    pred["group"] = pred["group"].astype(str)
    pred["pred_label"] = pred["pred_label"].fillna("").astype(str)
    pred = pred[pred["pred_label"] != ""].copy()
    label_to_index = {label: i for i, label in enumerate(FOLD_LABELS)}
    true_idx = pred["true_label"].astype(str).map(label_to_index)
    pred_idx = pred["pred_label"].astype(str).map(label_to_index)
    valid = true_idx.notna() & pred_idx.notna()
    pred = pred[valid].copy()
    true_arr = true_idx[valid].to_numpy(dtype=int)
    pred_arr = pred_idx[valid].to_numpy(dtype=int)
    group_codes, groups = pd.factorize(pred["group"], sort=False)
    group_cm = np.zeros((len(groups), len(FOLD_LABELS), len(FOLD_LABELS)), dtype=np.int64)
    np.add.at(group_cm, (group_codes, true_arr, pred_arr), 1)

    def score(cm: np.ndarray) -> float:
        if cm.sum() == 0:
            return float("nan")
        if metric == "accuracy":
            return float(np.trace(cm) / cm.sum())
        if metric == "macro_f1":
            tp = np.diag(cm).astype(float)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            denom = 2 * tp + fp + fn
            f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
            return float(np.mean(f1))
        raise ValueError(f"Unsupported metric: {metric}")

    estimate = score(group_cm.sum(axis=0))
    rng = np.random.default_rng(seed)
    values = np.empty(bootstrap_n, dtype=float)
    for i in range(bootstrap_n):
        sampled = rng.integers(0, len(groups), size=len(groups))
        weights = np.bincount(sampled, minlength=len(groups))
        values[i] = score(np.tensordot(weights, group_cm, axes=(0, 0)))
    return estimate, float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))


def _performance_ci_table(
    benchmark_dir: Path,
    benchmark_ci_csv: Path,
    external_dir: Path,
    *,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if benchmark_ci_csv.exists():
        ci = pd.read_csv(benchmark_ci_csv)
        ci = ci[ci["metric"] == "macro_f1"].copy()
        metrics = pd.read_csv(benchmark_dir / "metrics_summary.csv")
        coverage_by_model = {model: 1.0 for model in metrics["model"].astype(str)}
        n_by_model = dict(zip(metrics["model"].astype(str), metrics["n"].astype(int), strict=False))
        for row in ci.itertuples(index=False):
            rows.append(
                {
                    "model": str(row.model),
                    "model_label": _model_display_name(str(row.model)),
                    "metric": "macro_f1",
                    "estimate": float(row.estimate),
                    "ci_low": float(row.ci_low),
                    "ci_high": float(row.ci_high),
                    "n": int(n_by_model.get(str(row.model), 0)),
                    "coverage": float(coverage_by_model.get(str(row.model), 1.0)),
                    "ci_source": "group_bootstrap",
                }
            )
    external_files = [
        external_dir / "esmc_knn_predictions.csv",
        external_dir / "foldseek_nn_predictions.csv",
    ]
    for path in external_files:
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        for model, part in pred.groupby("model", sort=False):
            estimate, ci_low, ci_high = _bootstrap_metric_ci(part, metric="macro_f1", seed=seed)
            coverage = float((part["pred_label"].fillna("").astype(str) != "").mean())
            rows.append(
                {
                    "model": str(model),
                    "model_label": _model_display_name(str(model)),
                    "metric": "macro_f1",
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": int((part["pred_label"].fillna("").astype(str) != "").sum()),
                    "coverage": coverage,
                    "ci_source": "group_bootstrap",
                }
            )
    order = [
        "foldseek_nearest_neighbor",
        "xgboost_tuned",
        "esmc_cosine_knn_k1",
        "esmc_cosine_knn_k5",
        "esmc_cosine_knn_k10",
        "grammar_rules",
    ]
    table = pd.DataFrame(rows)
    table["_order"] = table["model"].map({model: i for i, model in enumerate(order)})
    return table.sort_values(["_order", "estimate"], ascending=[True, False]).drop(columns=["_order"])


def _per_class_f1_table(benchmark_dir: Path, external_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    per_class_path = benchmark_dir / "per_class_metrics.csv"
    if per_class_path.exists():
        per_class = pd.read_csv(per_class_path)
        per_class = per_class[per_class["model"].isin(["xgboost_tuned", "grammar_rules"])].copy()
        for _, row in per_class.iterrows():
            rows.append(
                {
                    "model": str(row["model"]),
                    "model_label": _model_display_name(str(row["model"])),
                    "fold_label": str(row["fold_label"]),
                    "fold_label_display": _short_fold_label(str(row["fold_label"])),
                    "f1": float(row["f1-score"]),
                    "support": int(float(row["support"])),
                }
            )
    for path in [external_dir / "esmc_knn_predictions.csv", external_dir / "foldseek_nn_predictions.csv"]:
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        for model, part in pred.groupby("model", sort=False):
            if model not in {"esmc_cosine_knn_k1", "foldseek_nearest_neighbor"}:
                continue
            valid = part[part["pred_label"].fillna("").astype(str) != ""]
            precision, recall, f1, support = precision_recall_fscore_support(
                valid["true_label"].astype(str),
                valid["pred_label"].astype(str),
                labels=list(FOLD_LABELS),
                zero_division=0,
            )
            for label, score, n in zip(FOLD_LABELS, f1, support, strict=False):
                rows.append(
                    {
                        "model": str(model),
                        "model_label": _model_display_name(str(model)),
                        "fold_label": label,
                        "fold_label_display": _short_fold_label(label),
                        "f1": float(score),
                        "support": int(n),
                    }
                )
    table = pd.DataFrame(rows)
    model_order = {
        "xgboost_tuned": 0,
        "foldseek_nearest_neighbor": 1,
        "esmc_cosine_knn_k1": 2,
        "grammar_rules": 3,
    }
    label_order = {label: i for i, label in enumerate(FOLD_LABELS)}
    table["_model_order"] = table["model"].map(model_order)
    table["_label_order"] = table["fold_label"].map(label_order)
    return table.sort_values(["_label_order", "_model_order"]).drop(columns=["_model_order", "_label_order"])


def _knn_purity_table(
    sim: np.ndarray,
    meta: pd.DataFrame,
    *,
    k_values: list[int],
    seed: int,
    bootstrap_n: int = 500,
) -> pd.DataFrame:
    labels = meta["fold_label_final"].astype(str).to_numpy()
    folds = meta["fold"].astype(str).to_numpy()
    scores = sim.copy()
    for fold in np.unique(folds):
        mask = folds == fold
        scores[np.ix_(mask, mask)] = -np.inf
    max_k = max(k_values)
    top_unsorted = np.argpartition(scores, -max_k, axis=1)[:, -max_k:]
    row = np.arange(len(scores))[:, None]
    order = np.argsort(scores[row, top_unsorted], axis=1)[:, ::-1]
    top = top_unsorted[row, order]
    same = labels[top] == labels[:, None]
    cumulative = same.cumsum(axis=1)
    purity = np.column_stack([cumulative[:, k - 1] / k for k in k_values])

    baseline_by_query = np.empty(len(meta), dtype=float)
    for i, label in enumerate(labels):
        candidate = folds != folds[i]
        baseline_by_query[i] = float(np.mean(labels[candidate] == label))

    label_indices = {label: np.flatnonzero(labels == label) for label in FOLD_LABELS}
    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap_n, len(k_values)), dtype=float)
    baseline_boot = np.empty(bootstrap_n, dtype=float)
    for b in range(bootstrap_n):
        class_means = []
        class_baselines = []
        for _label, idx in label_indices.items():
            if len(idx) == 0:
                continue
            sampled = rng.choice(idx, size=len(idx), replace=True)
            class_means.append(purity[sampled].mean(axis=0))
            class_baselines.append(baseline_by_query[sampled].mean())
        boot[b] = np.vstack(class_means).mean(axis=0)
        baseline_boot[b] = float(np.mean(class_baselines))

    class_means = []
    class_baselines = []
    for _label, idx in label_indices.items():
        if len(idx) == 0:
            continue
        class_means.append(purity[idx].mean(axis=0))
        class_baselines.append(baseline_by_query[idx].mean())
    estimate = np.vstack(class_means).mean(axis=0)
    baseline = float(np.mean(class_baselines))
    rows = []
    for i, k in enumerate(k_values):
        rows.append(
            {
                "k": int(k),
                "purity": float(estimate[i]),
                "purity_ci_low": float(np.nanquantile(boot[:, i], 0.025)),
                "purity_ci_high": float(np.nanquantile(boot[:, i], 0.975)),
                "random_baseline": baseline,
                "random_baseline_ci_low": float(np.nanquantile(baseline_boot, 0.025)),
                "random_baseline_ci_high": float(np.nanquantile(baseline_boot, 0.975)),
                "enrichment": float(estimate[i] / baseline),
                "enrichment_ci_low": float(np.nanquantile(boot[:, i] / np.maximum(baseline_boot, 1e-12), 0.025)),
                "enrichment_ci_high": float(np.nanquantile(boot[:, i] / np.maximum(baseline_boot, 1e-12), 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _risk_coverage_table(benchmark_dir: Path, *, model: str = "xgboost_tuned") -> pd.DataFrame:
    predictions_path = benchmark_dir / "oof_predictions.csv"
    if not predictions_path.exists():
        return pd.DataFrame(
            columns=[
                "threshold",
                "coverage",
                "review_fraction",
                "accepted_accuracy",
                "accepted_error_rate",
                "review_error_capture",
                "accepted_n",
                "review_n",
            ]
        )
    pred = pd.read_csv(predictions_path)
    pred = pred[pred["model"].astype(str) == model].copy()
    pred["pred_probability"] = pd.to_numeric(pred["pred_probability"], errors="coerce")
    pred = pred[pred["pred_probability"].notna()].copy()
    pred["correct"] = pred["true_label"].astype(str) == pred["pred_label"].astype(str)
    total_errors = int((~pred["correct"]).sum())
    thresholds = sorted(set([0.0, 0.9, 0.95, 0.99] + np.linspace(0.5, 0.995, 80).round(4).tolist()))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = pred["pred_probability"] >= threshold
        reviewed = ~accepted
        accepted_n = int(accepted.sum())
        review_n = int(reviewed.sum())
        accepted_accuracy = float(pred.loc[accepted, "correct"].mean()) if accepted_n else float("nan")
        reviewed_errors = int((~pred.loc[reviewed, "correct"]).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(accepted.mean()),
                "review_fraction": float(reviewed.mean()),
                "accepted_accuracy": accepted_accuracy,
                "accepted_error_rate": 1.0 - accepted_accuracy if math.isfinite(accepted_accuracy) else float("nan"),
                "review_error_capture": float(reviewed_errors / total_errors) if total_errors else float("nan"),
                "accepted_n": accepted_n,
                "review_n": review_n,
            }
        )
    return pd.DataFrame(rows)


def _ablation_delta_table(ablation_ci_csv: Path) -> pd.DataFrame:
    if not ablation_ci_csv.exists():
        return pd.DataFrame(
            columns=[
                "name",
                "label",
                "metric",
                "drop_estimate",
                "drop_ci_low",
                "drop_ci_high",
                "ablation_type",
            ]
        )
    table = pd.read_csv(ablation_ci_csv)
    table = table[table["metric"] == "boundary_macro_f1"].copy()
    table["label"] = table["name"].astype(str).map(_clean_ablation_label)
    table["drop_estimate"] = -pd.to_numeric(table["delta_estimate"], errors="coerce")
    table["drop_ci_low"] = -pd.to_numeric(table["delta_ci_high"], errors="coerce")
    table["drop_ci_high"] = -pd.to_numeric(table["delta_ci_low"], errors="coerce")
    table = table.sort_values("drop_estimate", ascending=True)
    return table[
        [
            "ablation_type",
            "name",
            "label",
            "metric",
            "baseline_estimate",
            "ablation_estimate",
            "delta_estimate",
            "delta_ci_low",
            "delta_ci_high",
            "drop_estimate",
            "drop_ci_low",
            "drop_ci_high",
        ]
    ]


def _plot_context_figures(
    emb: np.ndarray,
    emb_meta: pd.DataFrame,
    foldseek_hits: Path,
    figure_dir: Path,
    *,
    fold_assignments: pd.DataFrame,
    benchmark_dir: Path,
    benchmark_ci_csv: Path,
    ablation_ci_csv: Path,
    external_dir: Path,
    dataset: str,
    seed: int,
    max_per_label: int,
    network_k: int,
    scatter_pairs: int,
    deferred_figure_dir: Path | None = None,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator, PercentFormatter
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr

    figure_dir.mkdir(parents=True, exist_ok=True)
    deferred_dir = deferred_figure_dir or (figure_dir.parent / "model_reliability_ablation_deferred")
    deferred_dir.mkdir(parents=True, exist_ok=True)
    extra_outputs: dict[str, Path] = {}
    meta = emb_meta.reset_index(drop=True).copy()
    x = _normalise_embeddings(emb)
    fold_meta = meta.reset_index(names="embedding_index").merge(fold_assignments, on="record_id", how="inner")
    full_x = x[fold_meta["embedding_index"].to_numpy(dtype=int)]
    full_sim = full_x @ full_x.T
    np.fill_diagonal(full_sim, -np.inf)
    sample = _balanced_sample(meta.reset_index(names="embedding_index"), max_per_label=max_per_label, seed=seed)
    sample_idx = sample["embedding_index"].to_numpy(dtype=int)
    sample_labels = sample["fold_label_final"].astype(str).to_numpy()
    sample_x = x[sample_idx]
    dist = 1.0 - np.clip(sample_x @ sample_x.T, -1.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    model_colors = _model_palette()

    def text_halo(text_color: str) -> list[object]:
        halo_color = cz_colors.TEXT if text_color.lower() == "white" else "white"
        return [pe.withStroke(linewidth=0.9, foreground=halo_color)]

    with plt.rc_context(_publication_rc()):
        matrix_path = figure_dir / "A_esmc_cosine_distance_matrix.png"
        matrix_csv = figure_dir / "A_esmc_topology_distance_matrix.csv"
        class_dist = np.zeros((len(FOLD_LABELS), len(FOLD_LABELS)), dtype=float)
        for i, label_i in enumerate(FOLD_LABELS):
            mask_i = sample_labels == label_i
            for j, label_j in enumerate(FOLD_LABELS):
                mask_j = sample_labels == label_j
                values = dist[np.ix_(mask_i, mask_j)]
                if i == j and values.shape[0] > 1:
                    tri = np.triu_indices(values.shape[0], k=1)
                    values = values[tri]
                class_dist[i, j] = float(np.nanmean(values))
        pd.DataFrame(class_dist, index=FOLD_LABELS, columns=FOLD_LABELS).to_csv(matrix_csv)
        class_order_distance = class_dist.copy()
        np.fill_diagonal(class_order_distance, 0.0)
        class_order = leaves_list(linkage(squareform(class_order_distance, checks=False), method="average"))
        ordered_labels = [FOLD_LABELS[int(i)] for i in class_order]
        plot_dist = class_dist[np.ix_(class_order, class_order)]

        fig, ax = plt.subplots(figsize=(4.15, 3.8), constrained_layout=False)
        fig.subplots_adjust(left=0.22, right=0.88, bottom=0.20, top=0.97)
        vmin = float(np.nanmin(plot_dist)) if np.isfinite(plot_dist).any() else 0.0
        vmax = float(np.nanmax(plot_dist)) if np.isfinite(plot_dist).any() else 1.0
        im = ax.imshow(
            plot_dist,
            cmap=_nature_scalar_cmap("esmc_distance"),
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        for i in range(len(ordered_labels)):
            for j in range(len(ordered_labels)):
                value = float(plot_dist[i, j])
                norm_value = (value - vmin) / max(vmax - vmin, 1e-12)
                text_color = _contrast_text_color(norm_value, im.cmap)
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color=text_color,
                    path_effects=text_halo(text_color),
                )
        ax.set_xticks(np.arange(len(ordered_labels)), [_fold_abbrev(label) for label in ordered_labels], rotation=42, ha="right")
        ax.set_yticks(np.arange(len(ordered_labels)), [_fold_abbrev(label) for label in ordered_labels])
        ax.set_xticks(np.arange(-0.5, len(ordered_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ordered_labels), 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.65)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlabel("Topology")
        ax.set_ylabel("Topology")
        _style_axis(ax)
        ax.text(
            0.0,
            1.02,
            "Balanced sample; diagonal excludes self-pairs",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.1,
            color=cz_colors.TEXT,
        )
        colorbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
        colorbar.set_label("Mean ESM-C distance\n(observed range)", fontweight="bold")
        colorbar.ax.tick_params(labelsize=6.2, width=0.6, length=2.5)
        colorbar.outline.set_linewidth(0.4)
        colorbar.outline.set_edgecolor(cz_colors.NEUTRAL_DARK)
        _save_publication_figure(fig, matrix_path)
        plt.close(fig)

        purity_path = figure_dir / "B_esmc_knn_purity_enrichment.png"
        purity_csv = figure_dir / "B_esmc_knn_purity_enrichment.csv"
        k_values = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
        purity = _knn_purity_table(full_sim, fold_meta, k_values=k_values, seed=seed)
        purity.to_csv(purity_csv, index=False)
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(4.45, 3.35),
            height_ratios=[2.05, 1.1],
            sharex=True,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.16, right=0.90, bottom=0.15, top=0.97, hspace=0.07)
        k = purity["k"].to_numpy(dtype=float)
        esmc_color = model_colors["ESM-C kNN, k=1"]
        null_color = cz_colors.GRAY
        for axis in (ax_top, ax_bottom):
            axis.set_xscale("log")
            axis.set_xlim(0.85, 70)
            axis.xaxis.set_minor_locator(NullLocator())
        ax_top.fill_between(k, purity["purity_ci_low"], purity["purity_ci_high"], color=esmc_color, alpha=0.18, linewidth=0)
        ax_top.plot(k, purity["purity"], color=esmc_color, lw=1.35, marker="o", ms=3.0, label="ESM-C neighborhood")
        ax_top.fill_between(
            k,
            purity["random_baseline_ci_low"],
            purity["random_baseline_ci_high"],
            color=null_color,
            alpha=0.11,
            linewidth=0,
        )
        ax_top.plot(k, purity["random_baseline"], color=null_color, lw=0.9, ls=(0, (3, 2)), label="Random expectation")
        ax_top.set_ylabel("Same-topology\nneighbors")
        ax_top.set_ylim(0, 1.0)
        ax_top.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_top.grid(axis="y", color=cz_colors.GRID, lw=0.45)
        _style_axis(ax_top)
        ax_top.annotate(
            "ESM-C kNN",
            xy=(k[-1], float(purity["purity"].iloc[-1])),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=6.2,
            color=esmc_color,
            clip_on=False,
        )
        ax_top.annotate(
            "Fold-held-out random",
            xy=(k[-1], float(purity["random_baseline"].iloc[-1])),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=6.2,
            color=null_color,
            clip_on=False,
        )
        ax_bottom.fill_between(
            k,
            purity["enrichment_ci_low"],
            purity["enrichment_ci_high"],
            color=esmc_color,
            alpha=0.18,
            linewidth=0,
        )
        ax_bottom.plot(k, purity["enrichment"], color=esmc_color, lw=1.35, marker="o", ms=3.0)
        ax_bottom.axhline(1.0, color=null_color, lw=0.9, ls=(0, (3, 2)))
        ax_bottom.set_ylim(0.75, max(1.2, float(np.nanmax(purity["enrichment_ci_high"])) * 1.12))
        ax_bottom.set_xticks(k_values, [str(v) for v in k_values])
        ax_bottom.set_xlabel("Held-out ESM-C neighbors k (log scale)")
        ax_bottom.set_ylabel("Purity / random")
        ax_bottom.grid(axis="y", color=cz_colors.GRID, lw=0.45)
        ax_bottom.annotate(
            f"{purity['enrichment'].iloc[0]:.1f}x",
            xy=(k[0], float(purity["enrichment"].iloc[0])),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.1,
            color=esmc_color,
        )
        ax_bottom.annotate(
            f"{purity['enrichment'].iloc[-1]:.1f}x",
            xy=(k[-1], float(purity["enrichment"].iloc[-1])),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.1,
            color=esmc_color,
            clip_on=False,
        )
        _style_axis(ax_bottom)
        _save_publication_figure(fig, purity_path)
        plt.close(fig)

        performance_path = figure_dir / "C_model_macro_f1_ci.png"
        performance_csv = figure_dir / "C_model_macro_f1_ci.csv"
        performance = _performance_ci_table(benchmark_dir, benchmark_ci_csv, external_dir, seed=seed)
        performance.to_csv(performance_csv, index=False)
        perf = performance.sort_values("estimate", ascending=True).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(4.55, 2.65), constrained_layout=False)
        fig.subplots_adjust(left=0.35, right=0.78, bottom=0.17, top=0.98)
        y = np.arange(len(perf))
        colors = [model_colors.get(str(label), cz_colors.NEUTRAL_DARK) for label in perf["model_label"]]
        ax.barh(y, perf["estimate"], height=0.62, color=colors, edgecolor="white", linewidth=0.55, alpha=0.94, zorder=2)
        xerr = np.vstack(
            [
                perf["estimate"].to_numpy(dtype=float) - perf["ci_low"].to_numpy(dtype=float),
                perf["ci_high"].to_numpy(dtype=float) - perf["estimate"].to_numpy(dtype=float),
            ]
        )
        ax.errorbar(
            perf["estimate"],
            y,
            xerr=xerr,
            fmt="none",
            ecolor=cz_colors.TEXT,
            elinewidth=0.72,
            capsize=2.2,
            capthick=0.72,
            zorder=3,
        )
        ax.set_yticks(y, perf["model_label"])
        ax.set_xlim(0, 1.04)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Macro-F1")
        ax.grid(axis="x", color=cz_colors.GRID, lw=0.5, zorder=0)
        for yi, row in perf.iterrows():
            label = f"{row.estimate:.1%} [{row.ci_low:.1%}, {row.ci_high:.1%}]\nn={int(row.n):,}"
            ax.text(1.025, yi, label, va="center", ha="left", fontsize=5.8, linespacing=1.0, transform=ax.get_yaxis_transform())
        ax.text(
            0.0,
            1.02,
            "Error bars: 95% group-bootstrap CI; Foldseek is a structural retrieval reference",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.9,
            color=cz_colors.TEXT,
        )
        _style_axis(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        _save_publication_figure(fig, performance_path)
        plt.close(fig)

        confusion_path = figure_dir / "D_xgboost_normalized_confusion_matrix.png"
        confusion_csv = figure_dir / "D_xgboost_normalized_confusion_matrix.csv"
        cm_counts = pd.read_csv(benchmark_dir / "confusion_matrix_xgboost_tuned.csv", index_col=0)
        cm_counts = cm_counts.loc[list(FOLD_LABELS), list(FOLD_LABELS)]
        support = cm_counts.sum(axis=1).to_numpy(dtype=float)
        cm_norm = cm_counts.div(cm_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        cm_norm.to_csv(confusion_csv)
        fig, ax = plt.subplots(figsize=(4.55, 3.8), constrained_layout=False)
        fig.subplots_adjust(left=0.30, right=0.89, bottom=0.22, top=0.965)
        im = ax.imshow(cm_norm.to_numpy(), cmap=_nature_scalar_cmap("row_fraction"), vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(FOLD_LABELS)), [_fold_abbrev(label) for label in FOLD_LABELS], rotation=42, ha="right")
        ax.set_yticks(
            np.arange(len(FOLD_LABELS)),
            [f"{_fold_abbrev(label)} (n={int(n):,})" for label, n in zip(FOLD_LABELS, support, strict=False)],
        )
        ax.set_xlabel("Predicted topology")
        ax.set_ylabel("True topology")
        ax.set_xticks(np.arange(-0.5, len(FOLD_LABELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(FOLD_LABELS), 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        cmap = im.cmap
        for i in range(len(FOLD_LABELS)):
            for j in range(len(FOLD_LABELS)):
                value = float(cm_norm.iloc[i, j])
                count = int(cm_counts.iloc[i, j])
                if i == j or count >= 10 or value >= 0.035:
                    label = f"{value:.0%}\n{count:,}" if i == j else f"{value:.0%}\n{count:,}"
                    text_color = _contrast_text_color(value, cmap)
                    ax.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=6.1,
                        linespacing=0.9,
                        color=text_color,
                        path_effects=text_halo(text_color),
                    )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        colorbar = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.025)
        colorbar.set_label("Row fraction", fontweight="bold")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        colorbar.ax.tick_params(labelsize=6.2, width=0.6, length=2.5)
        colorbar.outline.set_linewidth(0.4)
        ax.text(
            0.0,
            1.02,
            "Rows sum to 100%; second line is cell count",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.0,
            color=cz_colors.TEXT,
        )
        _save_publication_figure(fig, confusion_path)
        plt.close(fig)

        per_class_path = figure_dir / "E_per_class_f1.png"
        per_class_csv = figure_dir / "E_per_class_f1.csv"
        per_class = _per_class_f1_table(benchmark_dir, external_dir)
        per_class.to_csv(per_class_csv, index=False)
        keep_models = ["Betlas XGBoost", "Foldseek structural NN", "ESM-C kNN, k=1", "Geometry rules"]
        support_parts = per_class[per_class["model_label"].isin(keep_models)]
        support_ranges = support_parts.groupby("fold_label")["support"].agg(["min", "max"]).reindex(FOLD_LABELS)
        f1_matrix = np.zeros((len(FOLD_LABELS), len(keep_models)), dtype=float)
        for col, model_label in enumerate(keep_models):
            part = per_class[per_class["model_label"] == model_label].set_index("fold_label").reindex(FOLD_LABELS)
            f1_matrix[:, col] = part["f1"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(4.75, 3.45), constrained_layout=False)
        fig.subplots_adjust(left=0.31, right=0.87, bottom=0.25, top=0.965)
        im = ax.imshow(f1_matrix, cmap=_nature_scalar_cmap("performance"), vmin=0, vmax=1, interpolation="nearest")
        model_labels = ["XGBoost", "Foldseek\nstruct.", "ESM-C\nk=1", "Geometry\nrules"]
        y_labels = []
        for label in FOLD_LABELS:
            lo = int(support_ranges.loc[label, "min"])
            hi = int(support_ranges.loc[label, "max"])
            support_label = f"n={lo:,}" if lo == hi else f"n={lo:,}-{hi:,}"
            y_labels.append(f"{_fold_abbrev(label)} ({support_label})")
        ax.set_xticks(np.arange(len(keep_models)), model_labels)
        ax.set_yticks(np.arange(len(FOLD_LABELS)), y_labels)
        ax.set_xlabel("Model", labelpad=3)
        ax.set_ylabel("Topology")
        ax.tick_params(axis="x", labelsize=5.6, pad=1.5)
        ax.set_xticks(np.arange(-0.5, len(keep_models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(FOLD_LABELS), 1), minor=True)
        ax.grid(which="minor", color="white", lw=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        cmap = im.cmap
        for i in range(len(FOLD_LABELS)):
            for j in range(len(keep_models)):
                value = float(f1_matrix[i, j])
                text_color = _contrast_text_color(value, cmap)
                ax.text(
                    j,
                    i,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    color=text_color,
                    path_effects=text_halo(text_color),
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
        colorbar.set_label("Per-class F1", fontweight="bold")
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        colorbar.ax.tick_params(labelsize=6.2, width=0.6, length=2.5)
        colorbar.outline.set_linewidth(0.4)
        ax.text(
            0.0,
            1.02,
            "Point estimates; rows show covered-domain support",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.0,
            color=cz_colors.TEXT,
        )
        _save_publication_figure(fig, per_class_path)
        plt.close(fig)

    hits = pd.read_csv(
        foldseek_hits,
        sep="\t",
        names=["query", "target", "evalue", "bits", "alntmscore", "qtmscore", "ttmscore", "prob"],
        dtype=str,
    )
    for column in ["alntmscore", "bits", "prob"]:
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    index_by_record = dict(zip(meta["record_id"], np.arange(len(meta)), strict=False))
    label_by_record = dict(zip(meta["record_id"], meta["fold_label_final"], strict=False))
    hits = hits[
        (hits["query"] != hits["target"])
        & hits["query"].isin(index_by_record)
        & hits["target"].isin(index_by_record)
        & hits["alntmscore"].notna()
    ].copy()
    if len(hits) > scatter_pairs:
        hits = hits.sample(n=scatter_pairs, random_state=seed)
    esm_dist = []
    same_label = []
    for row in hits.itertuples(index=False):
        qi = index_by_record[row.query]
        ti = index_by_record[row.target]
        esm_dist.append(float(1.0 - np.dot(x[qi], x[ti])))
        same_label.append(label_by_record[row.query] == label_by_record[row.target])
    hits["esmc_cosine_distance"] = esm_dist
    hits["same_fold_label"] = same_label
    hits.to_csv(figure_dir / "F_esmc_vs_foldseek_pairs.csv", index=False)
    rho = spearmanr(hits["esmc_cosine_distance"], hits["alntmscore"], nan_policy="omit").statistic if len(hits) else float("nan")

    with plt.rc_context(_publication_rc()):
        scatter_path = figure_dir / "F_esmc_distance_vs_foldseek_similarity.png"
        fig, ax = plt.subplots(figsize=(4.55, 3.15), constrained_layout=False)
        fig.subplots_adjust(left=0.14, right=0.88, bottom=0.16, top=0.985)
        hit_x = hits["esmc_cosine_distance"].to_numpy(dtype=float)
        hit_y = hits["alntmscore"].to_numpy(dtype=float)
        same = hits["same_fold_label"].to_numpy(dtype=bool)
        x_cap = float(np.nanquantile(hit_x, 0.995)) if len(hit_x) else 1.0
        plot_mask = np.isfinite(hit_x) & np.isfinite(hit_y) & (hit_x >= -0.02) & (hit_x <= x_cap)
        hb = ax.hexbin(
            hit_x[plot_mask],
            hit_y[plot_mask],
            gridsize=42,
            mincnt=1,
            cmap=_nature_density_cmap("foldseek_pair_density"),
            bins="log",
            linewidths=0,
            alpha=0.72,
            rasterized=True,
        )
        trend_same = _binned_quantiles(hit_x[plot_mask & same], hit_y[plot_mask & same], bins=14)
        trend_diff = _binned_quantiles(hit_x[plot_mask & ~same], hit_y[plot_mask & ~same], bins=14)
        same_color = cz_colors.BLUE
        diff_color = cz_colors.ORANGE
        if not trend_same.empty:
            ax.fill_between(trend_same["x"], trend_same["q25"], trend_same["q75"], color=same_color, alpha=0.12, linewidth=0)
            ax.plot(trend_same["x"], trend_same["median"], color=same_color, lw=1.45, label="Same topology")
            ax.annotate(
                "Same topology",
                xy=(float(trend_same["x"].iloc[-1]), float(trend_same["median"].iloc[-1])),
                xytext=(-7, 5),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=6.1,
                color=same_color,
                path_effects=[pe.withStroke(linewidth=1.1, foreground="white")],
            )
        if not trend_diff.empty:
            ax.fill_between(trend_diff["x"], trend_diff["q25"], trend_diff["q75"], color=diff_color, alpha=0.10, linewidth=0)
            ax.plot(trend_diff["x"], trend_diff["median"], color=diff_color, lw=1.45, ls=(0, (4, 2)), label="Different topology")
            ax.annotate(
                "Different topology",
                xy=(float(trend_diff["x"].iloc[-1]), float(trend_diff["median"].iloc[-1])),
                xytext=(-7, 5),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=6.1,
                color=diff_color,
                path_effects=[pe.withStroke(linewidth=1.1, foreground="white")],
            )
        ax.set_xlabel("ESM-C cosine distance")
        ax.set_ylabel("Foldseek TM-score")
        ax.set_xlim(-0.02, x_cap)
        ax.set_ylim(0.05, 1.02)
        ax.grid(color=cz_colors.GRID, lw=0.45)
        _style_axis(ax)
        ax.text(
            0.03,
            0.94,
            f"n={len(hits):,} sampled hits; Spearman rho = {rho:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color=cz_colors.TEXT,
            path_effects=[pe.withStroke(linewidth=1.1, foreground="white")],
        )
        ax.text(
            0.03,
            0.885,
            "Lines are binned medians; bands are IQR",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color=cz_colors.TEXT,
            path_effects=[pe.withStroke(linewidth=1.1, foreground="white")],
        )
        colorbar = fig.colorbar(hb, ax=ax, fraction=0.03, pad=0.02, shrink=0.82)
        colorbar.set_label("All pairs (log count)", fontweight="bold")
        colorbar.ax.yaxis.label.set_size(7.0)
        colorbar.ax.tick_params(labelsize=6.2, width=0.6, length=2.5)
        colorbar.outline.set_linewidth(0.4)
        _save_publication_figure(fig, scatter_path)
        plt.close(fig)

    with plt.rc_context(_publication_rc()):
        risk_path = deferred_dir / "G_xgboost_uncertainty_triage_curve.png"
        risk_csv = deferred_dir / "G_xgboost_uncertainty_triage_curve.csv"
        risk = _risk_coverage_table(benchmark_dir, model="xgboost_tuned")
        risk.to_csv(risk_csv, index=False)
        if not risk.empty:
            fig, ax = plt.subplots(figsize=(4.45, 3.0), constrained_layout=True)
            valid = risk["accepted_accuracy"].notna() & risk["review_error_capture"].notna()
            x_review = risk.loc[valid, "review_fraction"].to_numpy(dtype=float)
            ax.plot(
                x_review,
                risk.loc[valid, "accepted_accuracy"].to_numpy(dtype=float),
                color=model_colors["Betlas XGBoost"],
                lw=1.1,
            )
            ax.plot(
                x_review,
                risk.loc[valid, "review_error_capture"].to_numpy(dtype=float),
                color=cz_colors.RED,
                lw=1.1,
            )
            threshold_label_y = {0.90: 0.875, 0.95: 0.945}
            for threshold in (0.90, 0.95):
                row = risk[np.isclose(risk["threshold"], threshold)]
                if len(row):
                    row = row.iloc[0]
                    ax.scatter(
                        [row.review_fraction],
                        [row.accepted_accuracy],
                        s=15,
                        color=model_colors["Betlas XGBoost"],
                        edgecolors="white",
                        linewidths=0.35,
                        zorder=4,
                    )
                    ax.scatter(
                        [row.review_fraction],
                        [row.review_error_capture],
                        s=15,
                        color=cz_colors.RED,
                        edgecolors="white",
                        linewidths=0.35,
                        zorder=4,
                    )
                    ax.axvline(row.review_fraction, color=cz_colors.NEUTRAL, lw=0.5, ls=(0, (2, 2)), zorder=1)
                    ax.text(
                        row.review_fraction,
                        threshold_label_y[threshold],
                        f"confidence >= {threshold:.2f}",
                        ha="center",
                        va="center",
                        fontsize=4.8,
                        color=cz_colors.TEXT,
                    )
            ax.set_xlabel("Manual review fraction")
            ax.set_ylabel("Fraction")
            ax.set_xlim(0, min(1.0, max(0.35, float(np.nanmax(x_review)) * 1.02 if len(x_review) else 0.35)))
            ax.set_ylim(0, 1.02)
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(color=cz_colors.GRID, lw=0.35)
            _style_axis(ax)
            ax.text(
                0.97,
                0.92,
                "Accepted-set accuracy",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=4.9,
                color=model_colors["Betlas XGBoost"],
            )
            ax.text(
                0.97,
                0.82,
                "Fraction of total errors reviewed",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=4.9,
                color=cz_colors.RED,
            )
            _save_publication_figure(fig, risk_path)
            plt.close(fig)
            extra_outputs.update({"figure_g_uncertainty_triage": risk_path, "uncertainty_triage": risk_csv})

        ablation_path = deferred_dir / "H_ablation_boundary_macro_f1_delta.png"
        ablation_csv = deferred_dir / "H_ablation_boundary_macro_f1_delta.csv"
        ablation = _ablation_delta_table(ablation_ci_csv)
        ablation.to_csv(ablation_csv, index=False)
        if not ablation.empty:
            fig, ax = plt.subplots(figsize=(4.95, 3.05), constrained_layout=False)
            fig.subplots_adjust(left=0.42, right=0.80, bottom=0.16, top=0.98)
            y = np.arange(len(ablation))
            type_colors = {
            "drop_group_single": cz_colors.TEAL,
            "drop_group_combo_2": cz_colors.BROWN,
            "drop_group_combo_3": cz_colors.BROWN,
            "drop_feature_single": cz_colors.PURPLE,
            }
            type_labels = {
                "drop_group_single": "Single group",
                "drop_group_combo_2": "Group combo",
                "drop_group_combo_3": "Group combo",
                "drop_feature_single": "Single feature",
            }
            type_markers = {
                "drop_group_single": "o",
                "drop_group_combo_2": "s",
                "drop_group_combo_3": "s",
                "drop_feature_single": "D",
            }
            for yi, row in enumerate(ablation.itertuples(index=False)):
                color = type_colors.get(str(row.ablation_type), cz_colors.NEUTRAL_DARK)
                ax.plot([row.delta_ci_low, row.delta_ci_high], [yi, yi], color=color, lw=1.0, solid_capstyle="round")
                ax.scatter(
                    row.delta_estimate,
                    yi,
                    s=19,
                    marker=type_markers.get(str(row.ablation_type), "o"),
                    color=color,
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=3,
                )
            labels = [str(label).replace(" + ", " +\n") for label in ablation["label"]]
            ax.set_yticks(y, labels)
            ax.invert_yaxis()
            ax.axvline(0, color=cz_colors.NEUTRAL_DARK, lw=0.55, ls=(0, (2, 2)))
            xmin = float(np.nanmin(ablation["delta_ci_low"])) if len(ablation) else -0.1
            xmax = float(np.nanmax(ablation["delta_ci_high"])) if len(ablation) else 0.02
            ax.set_xlim(min(xmin * 1.08, -0.01), max(xmax * 1.18, 0.01))
            ax.xaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_xlabel("Delta boundary macro-F1 (ablation - full)")
            ax.grid(axis="x", color=cz_colors.GRID, lw=0.45)
            _style_axis(ax)
            handles = []
            seen: set[str] = set()
            for ablation_type in ablation["ablation_type"].astype(str):
                label = type_labels.get(ablation_type, ablation_type)
                if label in seen:
                    continue
                seen.add(label)
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="none",
                        markerfacecolor=type_colors.get(ablation_type, cz_colors.NEUTRAL_DARK),
                        markeredgecolor="white",
                        marker=type_markers.get(ablation_type, "o"),
                        markersize=4.5,
                        label=label,
                    )
                )
            ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.25)
            _save_publication_figure(fig, ablation_path)
            plt.close(fig)
            extra_outputs.update({"figure_h_ablation_delta": ablation_path, "ablation_boundary_delta": ablation_csv})

    deferred_caption_path = deferred_dir / "figure_caption.md"
    deferred_caption_parts = [
        "Deferred model reliability and ablation readouts from the "
        f"{dataset} benchmark.",
    ]
    if "figure_g_uncertainty_triage" in extra_outputs:
        deferred_caption_parts.append(
            "(G) XGBoost uncertainty-triage curve: increasing the manual-review fraction "
            "by filtering low maximum-class-probability predictions raises accepted-set "
            "accuracy and captures a larger fraction of total errors."
        )
    if "figure_h_ablation_delta" in extra_outputs:
        deferred_caption_parts.append(
            "(H) Paired bootstrap ablation forest plot showing delta boundary macro-F1 "
            "(ablation minus full model) after dropping selected feature groups or "
            "individual features."
        )
    deferred_caption_path.write_text(" ".join(deferred_caption_parts) + "\n", encoding="utf-8")
    extra_outputs["deferred_figure_caption"] = deferred_caption_path

    caption_path = figure_dir / "figure_caption.md"
    caption_parts = [
        "Figure. External baseline context and predictive performance on the "
        f"{dataset} benchmark using S35/homology group-aware folds.",
        "(A) Mean pairwise ESM-C cosine distance between topology classes, computed "
        f"from a balanced sample of domains (up to {max_per_label} per topology class); "
        "diagonal cells report within-topology, non-self distances, and colors are "
        "scaled over the observed class-mean distance range.",
        "(B) Class-balanced ESM-C k-nearest-neighbor topology purity and enrichment "
        "relative to fold-held-out random-neighbor expectation; purity is the "
        "same-topology fraction among k held-out neighbors, and shaded bands are "
        "query-bootstrap 95% intervals.",
        "(C) Macro-F1 model comparison with 95% group-bootstrap confidence intervals; "
        "Foldseek structural nearest-neighbor retrieval is reported as a structure-based "
        "reference rather than a de novo feature model; metrics use covered predictions.",
        "(D) Row-normalized confusion matrix for Betlas XGBoost; colors are row "
        "fractions, row support is shown beside each true topology, and annotated cells "
        "show row fraction and count.",
        "(E) Per-class F1 point estimates for Betlas XGBoost, Foldseek structural "
        "nearest-neighbor retrieval, ESM-C kNN, and geometry rules; support ranges "
        "reflect small coverage differences among methods.",
        "(F) ESM-C cosine distance versus Foldseek alignment TM-score for sampled "
        "Foldseek-hit pairs, capped at the 99.5th percentile of ESM-C distance; the "
        f"density layer pools {len(hits):,} sampled hit pairs, and lines show binned "
        "medians with interquartile bands for same-topology and different-topology "
        f"pairs. Spearman rho = {rho:.2f}.",
    ]
    caption_path.write_text(" ".join(caption_parts) + "\n", encoding="utf-8")

    return {
        "figure_a_distance_matrix": matrix_path,
        "figure_b_knn_purity_enrichment": purity_path,
        "figure_c_model_macro_f1_ci": performance_path,
        "figure_d_normalized_confusion_matrix": confusion_path,
        "figure_e_per_class_f1": per_class_path,
        "figure_f_distance_vs_foldseek": scatter_path,
        "figure_caption": caption_path,
        "esmc_topology_distance_matrix": matrix_csv,
        "knn_purity_enrichment": purity_csv,
        "model_macro_f1_ci": performance_csv,
        "normalized_confusion_matrix": confusion_csv,
        "per_class_f1": per_class_csv,
        "scatter_pairs": figure_dir / "F_esmc_vs_foldseek_pairs.csv",
        **extra_outputs,
    }
