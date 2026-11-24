# noclear

Extracts the calibration tables from scanned **Ideal Scales** calibration forms (PDF) into CSV.

For every page it reads the **"Calibration Data of"** value (Aggregate, Cement, Water, …) and the
table underneath it, and writes one CSV per PDF:

| Page | Calibration Data of | S# | Standard Weight (Kg) | Loading value (Kg) | Unloading Value (Kg) | Average Error (%) |
|---|---|---|---|---|---|---|
| 1 | Aggregate | 2 | 20 | 20 | 20 | 0.00% |
| 2 | Cement | 8 | 100 | 100 | 99 | 0.50% |

**Average Error (%)** is calculated, not read from the form:
`(|loading − standard| + |unloading − standard|) / 2 / standard × 100`.

> The values come from handwriting recognition. Always check them against the scans before use;
> the last digit is the most common mistake (e.g. `905` read as `900`).

OCR is done by [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL): a small layout
model runs on the CPU, and the vision-language model runs on the GPU through a vLLM server. The
Docker image starts vLLM itself, waits until it is ready, then runs the extraction or the web
interface.

**Contents**

- [Requirements (all GPUs)](#requirements-all-gpus)
- [NVIDIA GPUs](#nvidia-gpus)
- [AMD GPUs](#amd-gpus)
- [Running without Docker](#running-without-docker)
- [Settings](#settings)
- [Troubleshooting](#troubleshooting)

---

## Requirements (all GPUs)

- Docker Engine with BuildKit (`docker buildx`).
- **Free disk space for Docker.** The vLLM base images are very large. Check with
  `df -h /var/lib/docker`; aim for 40 GB+ under **Avail** before the first build.
- A GPU with roughly **8 GB+ of VRAM**. The model weights are only ~2 GB. vLLM reserves memory up
  front and is capped at **50% of the GPU's VRAM** by default (`--gpu-memory-utilization 0.5`,
  adjustable, see [Settings](#settings)), leaving the rest free for other programs.

---

## NVIDIA GPUs

Uses `Dockerfile.cuda`, built on the official CUDA vLLM image (`vllm/vllm-openai`).

### 1. Host setup (once per machine)

Containers can only use an NVIDIA GPU through the **NVIDIA Container Toolkit**.

1. **Install the NVIDIA driver** on the host and check it works:

   ```bash
   nvidia-smi
   ```

   It should list your GPU. vLLM needs a GPU with compute capability 7.0 or newer (Volta / RTX 20xx
   and later) and a recent driver; if the container later reports that the driver is too old for
   its CUDA version, update the driver.

2. **Install the NVIDIA Container Toolkit.**

   - Arch / CachyOS / Manjaro:

     ```bash
     sudo pacman -S nvidia-container-toolkit
     ```

   - Ubuntu / Debian / Fedora / RHEL: add NVIDIA's package repository and install
     `nvidia-container-toolkit` by following the official guide:
     <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>

3. **Register it with Docker and restart Docker:**

   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

4. **Check that containers can see the GPU:**

   ```bash
   docker run --rm --gpus all ubuntu nvidia-smi
   ```

   This must print the same GPU table as step 1. If it fails, fix this before going further; see
   [Troubleshooting](#troubleshooting).

### 2. Build

From the project folder:

```bash
uv lock     # only needed after changing dependencies in pyproject.toml
docker buildx build -f Dockerfile.cuda -t noclear:cuda .
```

The first build downloads the base image and all Python packages and can take a while. Rebuilds
reuse a package cache.

### 3. Run: process PDFs

Stop any other vLLM container first (`docker ps`, `docker stop <id>`): two on one GPU will run out
of memory.

```bash
docker run --rm -it \
  --gpus all --ipc=host \
  -v "$PWD":/data \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.paddlex:/root/.paddlex \
  noclear:cuda
```

With no arguments, every PDF in the current folder is processed. To pick files, add them after the
image name using `/data/` paths:

```bash
docker run ...same options... noclear:cuda "/data/CCECC.pdf" "/data/Safemix P#1 11-18-2026 17.56.pdf"
```

CSVs are written to `./output/`, named after each PDF.

### 4. Run: web interface

```bash
docker run --rm -it \
  --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/output":/data/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.paddlex:/root/.paddlex \
  noclear:cuda web
```

When the log shows vLLM is ready and uvicorn is running, open <http://localhost:8000>. Upload one
or more PDFs, watch each page complete, preview the table and download the CSV. Stop with Ctrl+C.

### NVIDIA options explained

| Option | Purpose |
|---|---|
| `--gpus all` | Gives the container every NVIDIA GPU. To use one GPU only: `--gpus '"device=0"'`. |
| `--ipc=host` | Shared memory for vLLM's worker processes, as recommended by vLLM. Alternative: `--shm-size=8g`. |
| `-v "$PWD":/data` | Shares the PDFs; results go to `output/` inside it. |
| `-v ~/.cache/huggingface:…` | Keeps the VL model download between runs. |
| `-v ~/.paddlex:…` | Keeps the layout model download between runs. |
| `-p 127.0.0.1:8000:8000` | Web interface, reachable from this machine only. |

### GPU memory limit and smaller NVIDIA GPUs (8–12 GB)

vLLM uses at most 50% of the GPU's memory by default. To change the cap, set
`--gpu-memory-utilization` (a fraction of total VRAM) in `VLLM_EXTRA_ARGS`, e.g. 40%:

```bash
docker run ...same options... \
  -e VLLM_EXTRA_ARGS="--gpu-memory-utilization 0.4 --max-model-len 8192 --max-num-batched-tokens 8192" \
  noclear:cuda
```

On a small card a low cap may be too little: if vLLM fails at startup with an out-of-memory error
or `No available memory for the cache blocks`, raise the fraction (e.g. `0.6`) or lower
`--max-model-len` / `--max-num-batched-tokens` to `8192` as above. The same setting works for the
AMD image.

---

## AMD GPUs

Uses `Dockerfile.rocm`, built on AMD's ROCm vLLM image (`rocm/vllm`). Needs a ROCm-supported GPU
and the `amdgpu` kernel driver on the host (included in current Linux kernels); no extra container
toolkit is required. Tested on a 16 GB card.

### Build

```bash
uv lock     # only needed after changing dependencies in pyproject.toml
docker buildx build -f Dockerfile.rocm -t noclear:rocm .
```

### Run: process PDFs

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$PWD":/data \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.paddlex:/root/.paddlex \
  noclear:rocm
```

Add PDF paths after the image name (`"/data/CCECC.pdf"`) to process specific files.

### Run: web interface

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/output":/data/output \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.paddlex:/root/.paddlex \
  noclear:rocm web
```

`--device=/dev/kfd --device=/dev/dri --group-add video` gives the container the AMD GPU. The ROCm
image also sets the memory-efficient attention settings that RDNA cards need (see
[Troubleshooting](#troubleshooting)).

---

## Running without Docker

Requires [uv](https://docs.astral.sh/uv/). The VL model still needs a vLLM server for GPU use; run
it in a container and noclear on the host.

**1. Start vLLM** and leave it running until it prints `Application startup complete`.

NVIDIA:

```bash
docker run --rm -it --gpus all --ipc=host \
  -p 127.0.0.1:8118:8118 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint vllm \
  vllm/vllm-openai:latest \
  serve PaddlePaddle/PaddleOCR-VL --trust-remote-code \
  --served-model-name PaddleOCR-VL-0.9B --port 8118 \
  --trust-request-mm-kwargs \
  --gpu-memory-utilization 0.5 --max-model-len 16384 --max-num-batched-tokens 16384
```

AMD:

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
  -p 127.0.0.1:8118:8118 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  rocm/vllm \
  vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code \
  --served-model-name PaddleOCR-VL-0.9B --port 8118 \
  --trust-request-mm-kwargs \
  --gpu-memory-utilization 0.5 --max-model-len 16384 --max-num-batched-tokens 16384
```

**2. Run noclear:**

```bash
uv run noclear "path/to/file.pdf" "path/to/another.pdf"   # command line
uv run noclear-web                                          # web interface on http://127.0.0.1:8000
```

**CPU only** (no vLLM, much slower; useful for timing comparisons):

```bash
NOCLEAR_DEVICE=cpu uv run noclear "path/to/file.pdf"
```

Each PDF prints how long it took.

---

## Settings

Environment variables. With Docker, pass them with `-e NAME=value`.

| Variable | Default | Meaning |
|---|---|---|
| `NOCLEAR_DEVICE` | `gpu` | `gpu` uses the vLLM server; `cpu` runs everything on the CPU. |
| `NOCLEAR_VL_SERVER_URL` | `http://127.0.0.1:8118/v1` | vLLM server address (must end in `/v1`). |
| `NOCLEAR_OUTPUT_DIR` | `output` (`/data/output` in Docker) | Where CSVs are written. |
| `NOCLEAR_WEB_HOST` / `NOCLEAR_WEB_PORT` | `127.0.0.1` / `8000` | Web interface address. |
| `VLLM_EXTRA_ARGS` | `--gpu-memory-utilization 0.5 --max-model-len 16384 --max-num-batched-tokens 16384` | Extra `vllm serve` options (Docker only). |

The web interface has no login. Keep the port published as `127.0.0.1:8000:8000` so it is reachable
only from this machine; don't expose it on a network as-is.

---

## Troubleshooting

### NVIDIA

| Symptom | Cause / fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | The NVIDIA Container Toolkit isn't installed or registered. Do [host setup](#1-host-setup-once-per-machine) steps 2–3 and restart Docker. |
| `nvidia-smi` works on the host but not in `docker run --gpus all ubuntu nvidia-smi` | Toolkit not registered with Docker: run `sudo nvidia-ctk runtime configure --runtime=docker` and `sudo systemctl restart docker`. |
| `Failed to initialize NVML: Unknown Error` inside the container | Usually a driver/toolkit mismatch after a driver update. Reboot, then rerun the GPU check. |
| `CUDA driver version is insufficient for CUDA runtime version` | The host driver is older than the image's CUDA. Update the NVIDIA driver. |
| vLLM `out of memory` at startup | Lower memory use with `VLLM_EXTRA_ARGS` (see [Smaller NVIDIA GPUs](#smaller-nvidia-gpus-812-gb)), and make sure nothing else is using the GPU (`nvidia-smi`). |

### AMD

| Symptom | Cause / fix |
|---|---|
| vLLM crashes with `CUDA out of memory … Tried to allocate 12.36 GiB` | Memory-efficient attention is off. Set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (already set in `Dockerfile.rocm`). |
| `Permission denied` on `/dev/kfd` | Add your user to the `video` and `render` groups, or keep `--group-add video` in the run command. |

### All GPUs

| Symptom | Cause / fix |
|---|---|
| `Connection error` from the `vlm` worker | vLLM isn't running or hasn't finished starting. Wait for `Application startup complete`. |
| `Per-request mm_processor_kwargs are disabled` (HTTP 400) | Start vLLM with `--trust-request-mm-kwargs` (already set in the Docker image). |
| Build fails with `no space left on device`, or apt reports `At least one invalid signature` for every repository | Docker's disk is full. Free space (`docker builder prune -a`, remove unused images) until `df -h /var/lib/docker` shows space under **Avail**. |
| Files in `output/` are owned by root | The container runs as root: `sudo chown -R $USER output`. |
| A value in the CSV is wrong | Handwriting misread. Correct it by hand from the scan. |
