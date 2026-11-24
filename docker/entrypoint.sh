#!/usr/bin/env bash
# Starts the vLLM server (PaddleOCR-VL on the GPU) inside the container, waits until it
# is ready, then either serves the web interface (first argument "web") or runs noclear
# on the given PDFs (default: every PDF in /data).
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8118}"
# System libraries copied in from the builder image (see Dockerfile.rocm). Only noclear's own
# processes get this, so vLLM keeps using the base image's libraries.
NOCLEAR_LIBS="/opt/noclear/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# True once vLLM's /health returns 200. Pure bash (/dev/tcp), so it doesn't depend on curl or
# python3 being present in the base image. Runs in a subshell so a refused connection can't
# end the script.
vllm_ready() (
    exec 3<>"/dev/tcp/127.0.0.1/${VLLM_PORT}" || exit 1
    printf 'GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n' >&3
    IFS= read -r -t 5 status_line <&3 || exit 1
    [[ "$status_line" == *" 200"* ]]
)

if [[ "${NOCLEAR_DEVICE:-gpu}" != "cpu" ]]; then
    # Bound to 127.0.0.1: only noclear inside this container talks to it.
    # shellcheck disable=SC2086  # VLLM_EXTRA_ARGS is intentionally word-split
    vllm serve PaddlePaddle/PaddleOCR-VL \
        --trust-remote-code \
        --served-model-name PaddleOCR-VL-0.9B \
        --host 127.0.0.1 --port "$VLLM_PORT" \
        --trust-request-mm-kwargs \
        ${VLLM_EXTRA_ARGS:-} &
    VLLM_PID=$!
    trap 'kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

    echo "Waiting for vLLM to start (first run downloads the model)..."
    until vllm_ready 2>/dev/null; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "vLLM exited before it was ready; see the log above." >&2
            exit 1
        fi
        sleep 3
    done
    echo "vLLM is ready."
fi

if [[ "${1:-}" == "web" ]]; then
    # 0.0.0.0 inside the container; publish it with -p 127.0.0.1:8000:8000 to keep it local.
    NOCLEAR_WEB_HOST="${NOCLEAR_WEB_HOST:-0.0.0.0}" LD_LIBRARY_PATH="$NOCLEAR_LIBS" \
        /opt/noclear/.venv/bin/noclear-web
    exit $?
fi

if [[ $# -eq 0 ]]; then
    shopt -s nullglob
    set -- /data/*.pdf
    if [[ $# -eq 0 ]]; then
        echo "No PDFs given and none found in /data. Mount a folder of PDFs at /data." >&2
        exit 1
    fi
fi

LD_LIBRARY_PATH="$NOCLEAR_LIBS" /opt/noclear/.venv/bin/noclear "$@"
