import argparse
import os
import json
import statistics
import csv
import re
import subprocess


def load_env_config(path):
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                continue
            if value and value[0] not in ("'", '"') and "#" in value:
                value = value.split("#", 1)[0].strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in ("'", '"')
            ):
                value = value[1:-1]
            os.environ[key] = os.path.expandvars(value)

parser = argparse.ArgumentParser(
    description="Generate a benchmark summary CSV from raw benchmark outputs."
)
parser.add_argument(
    "env_config",
    nargs="?",
    default="configs/gb10env",
    help="Environment config file used to name the output summary.",
)
args = parser.parse_args()

if not os.path.exists(args.env_config):
    raise FileNotFoundError(f"Environment config file not found: {args.env_config}")

load_env_config(args.env_config)



BASE_DIR =  os.getcwd() # Adjust if needed
BASE_DIR_RESULTS = os.path.join(BASE_DIR, "results")
SUPCOMPUTER_NAME = os.getenv("SUPCOMPUTER_NAME", "Add to .env file")
PARTITION_NAME = os.getenv("PARTITION_NAME", "Add to .env file")
MODEL_TYPE_MAP = json.load(open(os.path.join(BASE_DIR, "configs", "model_type_map.json")))
OUTPUT_FILE = f"results/full_benchmark_summary_{SUPCOMPUTER_NAME}_{PARTITION_NAME}.csv"
CSV_HEADERS = [
    "Supercomputer", "Partition", "Model", "Dataset/Model Type", "Dataset", "Framework",
    "Concurrency Level", "Number of GPUs", "GPU Memory Usage (GB)", "Power Usage (W)",
    "TTFT (ms)", "ITL (ms)", "TPOT (ms)", "Output Throughput (tokens/s)", "Request Throughput (requests/s)"
]

def extract_metrics(filepath):
    try:
        with open(filepath) as f:
            data = json.load(f)
        return {
            "ttft": data.get("mean_ttft_ms"),
            "itl": data.get("mean_itl_ms"),
            "tpot": data.get("mean_tpot_ms"),
            "output_throughput": data.get("output_throughput"),
            "request_throughput": data.get("request_throughput")
        }
    except Exception as e:
        print(f"Warning: Could not parse {filepath}: {e}")
        return None

def average_metrics(metrics_list):
    averaged = {}
    for k in metrics_list[0]:
        values = [
            m[k]
            for m in metrics_list
            if m.get(k) is not None and m.get(k) != ""
        ]
        averaged[k] = round(statistics.mean(values), 2) if values else ""
    return averaged

def extract_config_from_path(path):
    parts = path.split(os.sep)
    if len(parts) < 5:
        return None
    
    framework, dataset, model, node_gpu_part= parts[-6], parts[-5], parts[-4], parts[-3]
    nodes_match = re.search(r"Nodes_(\d+)", node_gpu_part)
    gpus_match = re.search(r"GPUs_(\d+)", node_gpu_part)
    
    if not nodes_match or not gpus_match:
        print("Nodes not match")
        return None

    nodes = int(nodes_match.group(1))
    gpus_per_node = int(gpus_match.group(1))
    total_gpus = nodes * gpus_per_node
    return framework, dataset, model, total_gpus

def parse_gpu_metrics_csv(csv_path):
    """
    Parse raw nvidia-smi CSV logs:
    timestamp,index,name,memory_used_MB,power_draw_W,utilization_gpu,utilization_memory
    (no header)
    """
    mems = []
    powers = []

    def parse_numeric(value):
        value = value.strip().strip("[]")
        if not value or value.upper() == "N/A":
            return None
        return float(value)

    try:
        with open(csv_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                mem = parse_numeric(parts[3])     # memory.used in MB
                power = parse_numeric(parts[4])   # power.draw in W
                if mem is not None:
                    mems.append(mem)
                if power is not None:
                    powers.append(power)
    except Exception as e:
        print(f"⚠️ Failed to parse GPU CSV {csv_path}: {e}")
        return "", ""

    avg_mem_gb = round(statistics.mean(mems) / 1024, 2) if mems else ""
    avg_power_w = round(statistics.mean(powers), 2) if powers else ""
    return avg_mem_gb, avg_power_w


# Stores: {(framework, dataset, model, total_gpus, concurrency): [metrics_dicts]}
data_index = {}
complete_launch_cache = {}

def is_complete_launch(root):
    if root not in complete_launch_cache:
        run_log = os.path.join(root, "run-local.out")
        complete = False
        try:
            with open(run_log) as f:
                complete = "All concurrency runs completed successfully." in f.read()
        except FileNotFoundError:
            complete = False
        complete_launch_cache[root] = complete
    return complete_launch_cache[root]

for root, dirs, files in os.walk(BASE_DIR_RESULTS):
    if any(file.startswith("Concurrency_") and file.endswith(".json") for file in files):
        if not is_complete_launch(root):
            continue

    for file in files:
        if not file.startswith("Concurrency_") or not file.endswith(".json"):
            continue
        
        file_path = os.path.join(root, file)
        config = extract_config_from_path(file_path)
        # print("file_path:", file_path, " config:", config)
        if config is None:
            continue
        
        # Extract metrics from CSV
        concurrency_level = int(file.split("_")[1].split(".")[0])
        metrics = extract_metrics(file_path)
        if metrics is None:
            continue
        
        # Extract GPU metrics from CSV
        csv_file = f"gpu_metrics_{concurrency_level}.csv"
        if os.path.exists(os.path.join(root, csv_file)):
            csv_path = os.path.join(root, csv_file)
            avg_mem_gb, avg_power_w = parse_gpu_metrics_csv(csv_path)

            # Add GPU memory and power usage to metrics
            metrics["gpu_memory_usage"] = avg_mem_gb
            metrics["power_usage"] = avg_power_w


        key = (*config, concurrency_level)
        # print("key:", key, "metrics:",metrics)
        if key not in data_index:
            data_index[key] = []
        data_index[key].append(metrics)



# Write the table
with open(OUTPUT_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(CSV_HEADERS)

    for (framework, dataset, model, total_gpus, concurrency), metrics_list in data_index.items():
        avg = average_metrics(metrics_list)
        model_type = next(
            (v for k, v in MODEL_TYPE_MAP.items() if k.split("/")[-1].lower() == model.strip().lower()),
            "Unknown: Add to model_type_map.json"
        )

        writer.writerow([
            SUPCOMPUTER_NAME, PARTITION_NAME, model, model_type, dataset, framework,
            concurrency, total_gpus, avg["gpu_memory_usage"] if "gpu_memory_usage" in avg else "", avg["power_usage"] if "power_usage" in avg else "", #"", "",  # GPU Mem / Power Usage not tracked here
            avg["ttft"], avg["itl"], avg["tpot"], avg["output_throughput"], avg["request_throughput"]
        ])

print(f"✅ Benchmark summary written to: {OUTPUT_FILE}")
