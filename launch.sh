#!/bin/bash
ngpu=$(nvidia-smi -L | grep -c UUID)
rank="${OMPI_COMM_WORLD_LOCAL_RANK:-0}"
device=$((rank % ngpu))
export CUDA_VISIBLE_DEVICES=$device
exec "$@"
