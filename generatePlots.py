import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def extract_model_size(model_name):
    match = re.search(r"(\d+(?:\.\d+)?B)", str(model_name), re.IGNORECASE)
    return match.group(1) if match else str(model_name)


def short_model_name(model_name):
    name = str(model_name)
    replacements = {
        "Llama-3.1-8B-Instruct": "Llama 3.1 8B",
        "gemma-3-12b-it": "Gemma 3 12B",
        "Qwen2.5-14B-Instruct": "Qwen 2.5 14B",
        "Mistral-Small-3.2-24B-Instruct-2506": "Mistral Small 3.2 24B",
        "Qwen2.5-32B-Instruct": "Qwen 2.5 32B",
    }
    return replacements.get(name, name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark PNG plots from a summary CSV."
    )
    parser.add_argument("summary_csv", help="Path to full_benchmark_summary CSV.")
    parser.add_argument(
        "--output-dir",
        help="Directory for PNG outputs. Defaults to <summary_csv stem>/plots_by_dataset.",
    )
    args = parser.parse_args()

    csv_path = Path(args.summary_csv)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else csv_path.with_suffix("") / "plots_by_dataset"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = [
        "Supercomputer",
        "Model",
        "Dataset",
        "Concurrency Level",
        "Number of GPUs",
        "Power Usage (W)",
        "Output Throughput (tokens/s)",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "Concurrency Level",
        "Number of GPUs",
        "Power Usage (W)",
        "Output Throughput (tokens/s)",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "Supercomputer",
            "Model",
            "Dataset",
            "Concurrency Level",
            "Output Throughput (tokens/s)",
        ]
    )
    df = df[df["Output Throughput (tokens/s)"] > 0].copy()
    df["ModelSize"] = df["Model"].map(extract_model_size)

    color_cycle = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    models = sorted(df["Model"].dropna().unique(), key=short_model_name)
    color_map = {
        model: color_cycle[index % len(color_cycle)]
        for index, model in enumerate(models)
    }

    for dataset_name, dataset_df in df.groupby("Dataset"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for (supercomputer, model), group in dataset_df.groupby(
            ["Supercomputer", "Model"]
        ):
            group = group.sort_values("Concurrency Level")
            model_size = extract_model_size(model)
            model_label = short_model_name(model)
            label = f"{supercomputer} - {model_label}"
            ax.plot(
                group["Concurrency Level"],
                group["Output Throughput (tokens/s)"],
                marker="o",
                linewidth=2,
                color=color_map.get(model),
                label=label,
            )
            last = group.iloc[-1]
            ax.annotate(
                model_label,
                (
                    last["Concurrency Level"],
                    last["Output Throughput (tokens/s)"],
                ),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=color_map.get(model),
            )

        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Output Throughput (tokens/s)")
        ax.set_title(f"{dataset_name} - Throughput vs Concurrency")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"throughput_vs_concurrency_{dataset_name}.png", dpi=160)
        plt.close(fig)

    energy_df = df.dropna(
        subset=["Power Usage (W)", "Number of GPUs", "Output Throughput (tokens/s)"]
    ).copy()
    if not energy_df.empty:
        energy_df["Total Power (W)"] = (
            energy_df["Power Usage (W)"] * energy_df["Number of GPUs"]
        )
        energy_df["Joule per Token (J/token)"] = (
            energy_df["Total Power (W)"] / energy_df["Output Throughput (tokens/s)"]
        )
        energy_df = energy_df[energy_df["Concurrency Level"] == 1000]

        for dataset_name, dataset_df in energy_df.groupby("Dataset"):
            fig, ax = plt.subplots(figsize=(10, 6))
            for _, row in dataset_df.iterrows():
                model_size = row["ModelSize"]
                ax.scatter(
                    row["Output Throughput (tokens/s)"],
                    row["Joule per Token (J/token)"],
                    s=90,
                    color=color_map.get(row["Model"]),
                    edgecolors="black",
                    label=f"{row['Supercomputer']} - {short_model_name(row['Model'])}",
                )
                ax.annotate(
                    short_model_name(row["Model"]),
                    (row["Output Throughput (tokens/s)"], row["Joule per Token (J/token)"]),
                    xytext=(8, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys())
            ax.set_xlabel("Output Throughput (tokens/s)")
            ax.set_ylabel("Energy per Token (J/token)")
            ax.set_title(f"{dataset_name} - Energy vs Throughput at Concurrency 1000")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                output_dir / f"j_per_token_vs_throughput_{dataset_name}_concurrency_1000.png",
                dpi=160,
            )
            plt.close(fig)

    print(f"Plots written to: {output_dir}")


if __name__ == "__main__":
    main()
