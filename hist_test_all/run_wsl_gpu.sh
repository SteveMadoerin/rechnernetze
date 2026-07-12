#!/bin/bash
# Run analyseSly.py (or another script in this folder) on the NVIDIA GPU
# from inside WSL2. Uses the ~/tf-gpu venv set up on this machine — see
# hist_test_all/prediction/run_wsl_gpu.sh for the one-time setup steps.
#
# Usage (from Windows):
#   wsl -d Ubuntu -- bash -lc "tr -d '\r' < /mnt/c/Users/steve/PycharmProjects/rechnernetze/hist_test_all/run_wsl_gpu.sh > /tmp/run_gpu_hta.sh && bash /tmp/run_gpu_hta.sh"
# or from a WSL shell:
#   bash run_wsl_gpu.sh
#
# A different script in this folder can be given as the first argument:
#   bash run_wsl_gpu.sh some_other.py --its --args
set -e

VENV=~/tf-gpu
SP=$VENV/lib/python3.12/site-packages

# TensorFlow fails to dlopen the pip-installed CUDA libraries on its own in
# this setup; putting them on the loader path makes the GPU visible.
export LD_LIBRARY_PATH=$(echo "$SP"/nvidia/*/lib | tr ' ' ':')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd /mnt/c/Users/steve/PycharmProjects/rechnernetze/hist_test_all

# optional first argument: which script to run (default: analyseSly.py)
SCRIPT=analyseSly.py
case "$1" in
    *.py) SCRIPT=$1; shift ;;
esac

"$VENV/bin/python" "$SCRIPT" "$@"
