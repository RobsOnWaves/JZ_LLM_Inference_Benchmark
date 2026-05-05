# Architecture and Concepts

This project runs reproducible inference benchmarks against an OpenAI-compatible
vLLM server. It is organized around a small launcher layer, benchmark client
scripts, raw result folders, and report generation scripts.

## High-Level Flow

The standard flow is:

1. Configure the target machine and benchmark matrix.
2. Resolve model and dataset paths.
3. Start Ray locally or through the scheduler environment.
4. Start a vLLM OpenAI API server for one model.
5. Wait until the model is loaded and the API is ready.
6. Run the benchmark client for each concurrency level.
7. Record per-concurrency JSON results and GPU telemetry.
8. Stop vLLM and Ray.
9. Repeat for the next model, dataset, and repeat.
10. Generate summary CSV and plots from raw outputs.

For local GB10 runs, `run_benchmark.sh` is the parent orchestrator and
`scripts/vllm/run_ray_vllm.sh` is the per-launch vLLM/Ray worker.

## Main Components

### `run_benchmark.sh`

This is the top-level benchmark orchestrator. It defines the benchmark matrix:

- frameworks, currently `vllm`;
- datasets, currently `sharegpt` and `sonnet`;
- model list;
- number of nodes;
- repeat count.

For every combination, it computes:

- total GPUs and total nominal memory;
- estimated memory required from the model size;
- tensor parallelism (`TP`);
- pipeline parallelism (`PP`);
- output folder layout.

It then creates a launch folder and runs the framework-specific launcher.

The current GB10 workflow supports resuming interrupted runs. If a launch folder
already contains a `run-local.out` with:

```text
All concurrency runs completed successfully.
```

that launch is skipped and the script continues to the next one.

### `configs/*env`

Environment files describe the target system.

Important fields:

- `ENVIRONMENT_VLLM`: conda or virtualenv path used by the vLLM launcher.
- `SCRIPT_VLLM`: framework launcher script.
- `PORT`: vLLM API port.
- `BENCHMARK_FILE`: benchmark client script.
- `SUPCOMPUTER_NAME` and `PARTITION_NAME`: labels used in reports.
- `GPUS_PER_NODE`, `CPUS_PER_NODE`, `VRAM_PER_NODE`: capacity model used by the launcher.
- `LOCAL_EXECUTION`: when `true`, the benchmark runs without Slurm.

For the GB10, `LOCAL_EXECUTION=true`, one GPU is exposed, and unified memory is
treated as node memory.

### Config Maps

The JSON config maps decouple benchmark names from local paths:

- `configs/config_datasets_paths_map.json` maps dataset names to dataset files.
- `configs/model_type_map.json` maps Hugging Face model IDs to model types.
- `configs/model_type_directories_map.json` maps model types to local model roots.

The current launcher accepts relative paths in these JSON files and resolves them
from the repository root.

### `scripts/vllm/run_ray_vllm.sh`

This script runs one concrete model/dataset/repeat launch.

Its responsibilities are:

- activate the vLLM environment;
- start a local Ray head node;
- start `vllm.entrypoints.openai.api_server`;
- wait for `/v1/models` to return HTTP 200;
- send a small smoke-test inference request;
- run all concurrency levels;
- capture GPU telemetry with `nvidia-smi`;
- shut down vLLM and Ray.

On GB10, Ray's memory killer is disabled for local execution because unified
GPU memory is accounted as node memory and can otherwise trigger false-positive
worker kills. The launcher also lowers vLLM GPU memory utilization to leave more
headroom.

### `benchmarks/benchmark_serving.py`

This is the asynchronous benchmark client. It sends requests to the OpenAI-style
vLLM API and records latency and throughput metrics.

Key inputs:

- backend, usually `vllm`;
- host and port;
- model path;
- dataset name and dataset path;
- max concurrency;
- number of prompts.

Key output metrics:

- request throughput;
- output token throughput;
- time to first token (`TTFT`);
- inter-token latency (`ITL`);
- time per output token (`TPOT`);
- end-to-end latency.

## Result Layout

Raw results are written under:

```text
results/<framework>/<dataset>/<model>/<parallelism-folder>/launch-<repeat>/
```

Example:

```text
results/vllm/sharegpt/Llama-3.1-8B-Instruct/Nodes_1-GPUs_1-TP_1-PP_1/launch-1/
```

Each launch folder contains:

- `run-local.out`: full launcher log;
- `Concurrency_<N>.json`: benchmark metrics for concurrency `N`;
- `logs_benchmarking_<N>_concurrency.log`: client-side benchmark log;
- `gpu_metrics_<N>.csv`: sampled `nvidia-smi` telemetry during that run.

Concurrency levels currently run from 50 to 1000 in increments of 50.

## Report Generation

### `generateSummaryTable-checkpoint.py`

This script scans `results/`, reads every `Concurrency_*.json`, joins it with
the matching `gpu_metrics_*.csv`, and writes a summary CSV:

```text
results/full_benchmark_summary_<SUPCOMPUTER_NAME>_<PARTITION_NAME>.csv
```

It accepts an optional environment config:

```bash
python generateSummaryTable-checkpoint.py configs/gb10env
```

On GB10, `nvidia-smi` can return `N/A` for memory usage. The summary generator
keeps memory usage blank in that case but still averages power usage and
benchmark metrics.

### `generatePlots.py`

This script reads the summary CSV and generates PNG plots:

- throughput vs concurrency;
- energy per token vs throughput at concurrency 1000.

Example:

```bash
python generatePlots.py "results/full_benchmark_summary_Dell GB10_gpu.csv"
```

## Core Concepts

### Concurrency

Concurrency is the maximum number of in-flight requests sent by the benchmark
client. Increasing concurrency stresses the serving stack and shows where
throughput saturates or latency becomes unacceptable.

### Repeat

A repeat is a full rerun of the same configuration:

```text
model + dataset + framework + TP/PP + concurrency sweep
```

Repeats help measure variability from caching, compilation, thermals, background
system load, and runtime scheduling. Use `REPEATS=1` for exploration and
`REPEATS=3` or more for more stable numbers.

### Tensor Parallelism

Tensor parallelism splits model tensor computations across GPUs. It is useful
when a model does not fit on one GPU or when multiple GPUs can accelerate a
single model. On a single GB10, `TP=1`.

### Pipeline Parallelism

Pipeline parallelism splits model layers across nodes or GPUs. This project
computes `PP` from the estimated memory requirement, but local GB10 runs should
remain `PP=1`.

### Ray

Ray is used by vLLM as the distributed executor backend. Even for one local GPU,
the launcher starts a Ray head node so vLLM can use the same execution path as
multi-worker configurations.

### vLLM

vLLM serves the model through an OpenAI-compatible API. The benchmark client
targets endpoints such as `/v1/completions` and `/v1/chat/completions`.

### Power and Energy

`gpu_metrics_*.csv` records power draw while each benchmark is running. The
summary computes average power and derives energy per token as:

```text
energy_per_token = total_power_watts / output_tokens_per_second
```

On systems where memory telemetry is unsupported, power telemetry can still be
valid and useful.

## GB10-Specific Notes

The default GB10 model set is intended to cover useful capacity points without
requiring multi-node memory:

- 8B: small baseline;
- 12B/14B: mid-size models;
- 24B: large practical model;
- 32B: upper practical range for local testing.

Full BF16 70B and 405B models are not practical defaults for one GB10. Llama 4
Scout/Maverick are also not default GB10 targets because their total MoE weights
remain too large even though active parameters per token are smaller.

## Operational Tips

Run in the background:

```bash
nohup bash run_benchmark.sh configs/gb10env > ~/logbench 2>&1 &
```

Follow the parent process:

```bash
tail -f ~/logbench
```

Follow the active launch:

```bash
tail -f results/vllm/sharegpt/Llama-3.1-8B-Instruct/Nodes_1-GPUs_1-TP_1-PP_1/launch-1/run-local.out
```

Check active benchmark processes:

```bash
ps -eo pid,ppid,stat,etime,cmd | grep -E 'run_benchmark|run_ray_vllm|vllm|benchmark_serving|ray' | grep -v grep
```
