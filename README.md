# LLM Inference Benchmark

Benchmark launcher and report tooling for vLLM inference runs on Slurm clusters and a local Dell GB10 workstation.

For a deeper explanation of the project structure and benchmark concepts, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Setup

Create or activate an environment with vLLM, Ray, datasets, pandas, and matplotlib installed. On the GB10 setup used here:

```bash
conda activate vllm
```

Update the environment file for the target machine:

- `configs/a100env`
- `configs/h100env`
- `configs/gb10env`

Also update local dataset/model roots in:

- `configs/config_datasets_paths_map.json`
- `configs/model_type_directories_map.json`
- `configs/model_type_map.json`

## Datasets

Place datasets under `benchmarks/datasets` or update `configs/config_datasets_paths_map.json`.

- [ShareGPT_V3_unfiltered_cleaned_split](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json)
- [Sonnet](https://huggingface.co/datasets/zhyncs/sonnet)

## GB10 Model Set

The default `run_benchmark.sh` model list is sized for one Dell GB10 and covers small to upper-range local inference:

- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-3-12b-it`
- `Qwen/Qwen2.5-14B-Instruct`
- `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- `Qwen/Qwen2.5-32B-Instruct`

Download them into `benchmarks/models`:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "Qwen/Qwen2.5-32B-Instruct",
]

for model in models:
    local_dir = "benchmarks/models/" + model
    snapshot_download(repo_id=model, local_dir=local_dir, local_dir_use_symlinks=False)
    print("Downloaded:", model, "->", local_dir)
PY
```

Llama 3.1 70B and 405B are not part of the GB10 default run. The 70B BF16 checkpoint is not practical on a single GB10, and 405B is out of range.

## Running

Run directly:

```bash
bash run_benchmark.sh configs/gb10env
```

Or keep it running after logout:

```bash
nohup bash run_benchmark.sh configs/gb10env > ~/logbench 2>&1 &
```

Follow the parent launcher:

```bash
tail -f ~/logbench
```

Follow the active local vLLM launcher:

```bash
tail -f results/vllm/sharegpt/Llama-3.1-8B-Instruct/Nodes_1-GPUs_1-TP_1-PP_1/launch-1/run-local.out
```

The launcher skips a completed `launch-*` folder when its `run-local.out` contains `All concurrency runs completed successfully.`. This makes interrupted GB10 runs resumable without rerunning completed launches.

`REPEATS` in `run_benchmark.sh` controls how many times each exact model/dataset configuration is rerun. Use `REPEATS=1` for quick exploration and `REPEATS=3` for more stable benchmark numbers.

## Reports

Generate the summary CSV:

```bash
python generateSummaryTable-checkpoint.py configs/gb10env
```

Generate PNG plots:

```bash
python generatePlots.py "results/full_benchmark_summary_Dell GB10_gpu.csv"
```

On GB10, `nvidia-smi` may report `N/A` for `memory.used`. The summary script leaves GPU memory blank when unsupported but still averages power metrics and benchmark throughput.

## Existing Notebook

`bench_analysis.ipynb` is still available for older report workflows. For GB10 runs, prefer `generateSummaryTable-checkpoint.py` plus `generatePlots.py`.
