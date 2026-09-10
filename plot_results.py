import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Clean, professional style for academic paper figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

def load_all_experiments(results_dir: str):
    """
    Scans results_dir for all summary.json and round_metrics.csv files.
    """
    summary_files = glob.glob(os.path.join(results_dir, "exp_*", "summary.json"))
    summaries = []
    round_dfs = []

    for s_file in summary_files:
        exp_folder = os.path.dirname(s_file)
        with open(s_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            summaries.append(data)

        csv_file = os.path.join(exp_folder, "round_metrics.csv")
        if os.path.exists(csv_file):
            rdf = pd.read_csv(csv_file)
            rdf["method"] = data.get("method")
            rdf["budget_tier"] = data.get("budget_tier")
            rdf["alpha"] = data.get("alpha")
            rdf["seed"] = data.get("seed")
            round_dfs.append(rdf)

    df_summary = pd.DataFrame(summaries)
    df_rounds = pd.concat(round_dfs, ignore_index=True) if round_dfs else pd.DataFrame()

    return df_summary, df_rounds

def generate_paper_tables(df_summary: pd.DataFrame, output_dir: str):
    """
    Computes Mean +/- Std across seeds and Performance Drop:
    Drop = Perf(alpha=1.0) - Perf(alpha=0.01)
    Generates Markdown and LaTeX tables.
    """
    if df_summary.empty:
        print("[Warning] No completed experiment summaries found to generate tables.")
        return

    # Group by method, budget_tier, alpha
    grouped = df_summary.groupby(["method", "budget_tier", "alpha"])["best_val_acc"].agg(["mean", "std", "count"]).reset_index()
    grouped["mean"] = grouped["mean"].round(2)
    grouped["std"] = grouped["std"].fillna(0.0).round(2)
    grouped["formatted"] = grouped.apply(lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1)

    # Pivot table: rows=(method, budget_tier), cols=alpha
    pivot = grouped.pivot(index=["method", "budget_tier"], columns="alpha", values="formatted").reset_index()

    # Calculate mean drop for robustness analysis
    mean_pivot = grouped.pivot(index=["method", "budget_tier"], columns="alpha", values="mean").reset_index()
    if 1.0 in mean_pivot.columns and 0.01 in mean_pivot.columns:
        mean_pivot["performance_drop"] = (mean_pivot[1.0] - mean_pivot[0.01]).round(2)
        pivot["performance_drop"] = mean_pivot["performance_drop"]

    # Save to Markdown
    md_path = os.path.join(output_dir, "summary_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Federated PEFT Experimental Results (AG News - RoBERTa-base)\n\n")
        f.write("Values show Mean ± Standard Deviation over 3 random seeds.\n\n")
        f.write(pivot.to_markdown(index=False))
        f.write("\n\n*Performance Drop = Accuracy(alpha=1.0) - Accuracy(alpha=0.01)*\n")
    print(f"[Table Saved] Markdown table written to: {md_path}")

    # Save to LaTeX
    tex_path = os.path.join(output_dir, "summary_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated LaTeX table for Federated PEFT Paper\n")
        f.write(pivot.to_latex(index=False, caption="Evaluation Accuracy across Parameter Budgets and Heterogeneity Levels", label="tab:fl_peft_results"))
    print(f"[Table Saved] LaTeX table written to: {tex_path}")

def plot_convergence(df_rounds: pd.DataFrame, output_dir: str):
    """
    Plots Accuracy vs. FL Rounds for all 4 PEFT methods under each alpha.
    """
    if df_rounds.empty:
        return

    alphas = sorted(df_rounds["alpha"].unique())
    budgets = sorted(df_rounds["budget_tier"].unique())

    fig, axes = plt.subplots(len(budgets), len(alphas), figsize=(4 * len(alphas), 3.5 * len(budgets)), sharey=True)
    if len(budgets) == 1 and len(alphas) == 1:
        axes = np.array([[axes]])
    elif len(budgets) == 1:
        axes = np.expand_dims(axes, axis=0)
    elif len(alphas) == 1:
        axes = np.expand_dims(axes, axis=1)

    palette = {"lora": "#1f77b4", "adapter": "#ff7f0e", "prefix": "#2ca02c", "ia3": "#d62728"}

    for i, b in enumerate(budgets):
        for j, a in enumerate(alphas):
            ax = axes[i, j]
            sub = df_rounds[(df_rounds["budget_tier"] == b) & (df_rounds["alpha"] == a)]
            if not sub.empty:
                sns.lineplot(
                    data=sub,
                    x="round",
                    y="val_acc",
                    hue="method",
                    palette=palette,
                    ax=ax,
                    errorbar="sd"
                )
            ax.set_title(f"Budget: {b.upper()} | α = {a}")
            ax.set_xlabel("FL Communication Rounds")
            ax.set_ylabel("Global Accuracy (%)" if j == 0 else "")
            if i > 0 or j > 0:
                leg = ax.get_legend()
                if leg is not None:
                    leg.remove()

    plt.tight_layout()
    out_file = os.path.join(output_dir, "convergence_curves.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[Figure Saved] Convergence plot written to: {out_file}")

def plot_robustness_analysis(df_summary: pd.DataFrame, output_dir: str):
    """
    Plots Accuracy vs Dirichlet alpha (Non-IID severity) to show robustness drop.
    """
    if df_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"lora": "#1f77b4", "adapter": "#ff7f0e", "prefix": "#2ca02c", "ia3": "#d62728"}

    # Use medium budget for the primary paper robustness figure
    medium_sub = df_summary[df_summary["budget_tier"] == "medium"]
    if medium_sub.empty:
        medium_sub = df_summary

    sns.lineplot(
        data=medium_sub,
        x="alpha",
        y="best_val_acc",
        hue="method",
        marker="o",
        markersize=8,
        palette=palette,
        ax=ax,
        errorbar="sd"
    )

    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.1, 0.5, 1.0])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Dirichlet Parameter α (Lower = Stronger Non-IID)")
    ax.set_ylabel("Global Validation Accuracy (%)")
    ax.set_title("Robustness to Statistical Heterogeneity across PEFT Methods")
    ax.invert_xaxis() # non-IID increases to the right

    plt.tight_layout()
    out_file = os.path.join(output_dir, "robustness_analysis.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[Figure Saved] Robustness curve written to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Plot and Analyze Federated PEFT Results")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing experiment results")
    parser.add_argument("--figures_dir", type=str, default="results/figures", help="Directory to save figures and tables")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    print(f"[Analyzer] Loading experiment results from: {args.results_dir}...")
    df_summary, df_rounds = load_all_experiments(args.results_dir)

    print(f"[Analyzer] Loaded {len(df_summary)} completed experiment runs.")
    if df_summary.empty:
        print("[Info] No results found yet. Run experiments with run_experiments.py first.")
        return

    generate_paper_tables(df_summary, args.figures_dir)
    plot_convergence(df_rounds, args.figures_dir)
    plot_robustness_analysis(df_summary, args.figures_dir)
    print("\n[Analyzer] All paper figures and tables successfully generated!")

if __name__ == "__main__":
    main()
