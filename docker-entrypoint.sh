#!/bin/bash
set -e

MODE="${1:-eval}"

case "$MODE" in
    eval)
        echo "Running eval harness..."
        shift || true
        exec python -m eval.run_eval "$@"
        ;;
    api)
        echo "Starting API server on port 8000..."
        shift || true
        exec python -m src.api "$@"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: docker run ... <eval|api> [extra-args]"
        exit 1
        ;;
esac