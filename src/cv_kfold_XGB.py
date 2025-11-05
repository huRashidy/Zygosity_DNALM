import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------ Settings ------------------
chrs = [6]

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})
TEXTWIDTH_IN = 6.8

agg_order = ["last", "mean", "max"]
agg_labels = {"last": "Last", "mean": "Mean", "max": "Max"}
agg_colors = {"last": "#1f77b4", "mean": "#2ca02c", "max": "#ff7f0e"}

metrics = ["accuracy", "f1", "mcc", "auroc"]
x_order = ["AB", "A", "B"]

# ------------------ Helpers ------------------
def parse_embedding_type(et: str) -> pd.Series:
    et_low = str(et).lower()
    if et_low.endswith("_mat"):
        base = "A"
    elif et_low.endswith("_pat"):
        base = "B"
    else:
        base = "AB"

    if et_low.startswith("last"):
        agg = "last"
    elif et_low.startswith("mean"):
        agg = "mean"
    elif et_low.startswith("max"):
        agg = "max"
    else:
        agg = et_low.split("_")[0] if et_low else "last"

    return pd.Series({"x_label": base, "agg": agg})

# ------------------ Main plotting ------------------
for chr_num in chrs:
    csv_path = f"./results/results_chr{chr_num}_XGB_all_types_filtering_True_var1e-05_corr1_xgbparams.csv"
    out_pdf = f"./figures/for_thesis/chr{chr_num}_XGB_True_var1e-05_corr1_xgbparams.pdf"
    out_png = f"./figures/for_pres/chr{chr_num}_XGB_True_var1e-05_corr1_xgbparams.png"
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    df = pd.read_csv(csv_path)

    def parse_embedding_type(et):
        et = et.lower()
        if et.endswith("_mat"):
            base = "A"
        elif et.endswith("_pat"):
            base = "B"
        else:
            base = "AB"

        if et.startswith("last"):
            agg = "last"
        elif et.startswith("mean"):
            agg = "mean"
        elif et.startswith("max"):
            agg = "max"
        else:
            agg = "unknown"

        return pd.Series({"x_label": base, "agg": agg})

    df[["x_label", "agg"]] = df["embedding_type"].apply(parse_embedding_type)
    df = df[df["agg"].isin(agg_order)]
    df_long = df.melt(id_vars=["x_label", "agg"], value_vars=metrics,
                    var_name="Metric", value_name="Value")

    # Ensure category order
    df_long["x_label"] = pd.Categorical(df_long["x_label"], categories=x_order, ordered=True)
    df_long["agg"] = pd.Categorical(df_long["agg"], categories=agg_order, ordered=True)

    # ------------------ Plot ------------------
    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 6.5))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = df_long[df_long["Metric"] == metric]
        
        sns.boxplot(
            data=sub,
            x="x_label",
            y="Value",
            hue="agg",
            order=x_order,
            hue_order=agg_order,
            palette=agg_colors,
            ax=ax,
            width=0.9,
            gap=0.2,
            fliersize=2.5,
            linewidth=1.25,
        )

        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Value" if i % 2 == 0 else "")
        ax.tick_params(axis='x', rotation=0)
        if ax.get_legend():
            ax.get_legend().remove()

    # ------------------ Global Legend ------------------
    handles, labels = axes[0].get_legend_handles_labels()
    #add aggregations that are represented in the figure only
    unique_labels = df_long["agg"].unique()
    handles = [h for h, l in zip(handles, labels) if l in unique_labels]
    fig.legend(
        handles,
        [agg_labels.get(l, l.title()) for l in unique_labels],
        loc="center right",
        bbox_to_anchor=(1.15, 0.5),
        frameon=True,
        borderpad=0.4,
        handlelength=1.6,
    )

    plt.tight_layout(rect=[0, 0, 0.97, 1])

    # Save as pdf and png
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
