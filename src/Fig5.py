import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

chrs = [6]
models = ["XGB" ]

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

for model in models:
    for chr in chrs:
        # Load the data
        a = np.load(f"./results/f1_scores_{model}_max_chr{chr}_windows.npy")
        # shape: (n_windows, n_folds) -> make columns=windows for seaborn
        A = a.T  # now shape: (5 folds, 11 windows)

        fig, ax = plt.subplots(figsize=(6.5, 3))  # fits in thesis column
        # Use the first colour of the colorblind palette to colour the single series.
        palette = sns.color_palette("colorblind")
        sns.boxplot(data=A, ax=ax, width=0.6, fliersize=3, color="#ff7f0e")

        n_windows = a.shape[0]
        ax.set_xticks(np.arange(n_windows))
        ax.set_xticklabels([f"W{i+1}" for i in range(n_windows)], rotation=0)

        ax.set_xlabel(f"450 kb windows (chr{chr} 5 Mb)", fontweight="bold")
        ax.set_ylabel("F1 score", fontweight="bold")

        plt.tight_layout()
        plt.savefig(f"./figures/for_thesis/f1_chr{chr}_windows_{model}.pdf", format="pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"./figures/for_pres/f1_chr{chr}_windows_{model}.png", format="png", dpi=300, bbox_inches="tight")

        plt.show()
