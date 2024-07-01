#!/bin/sh
env="simple_spread_v3"

echo "env is ${env}"
CUDA_VISIBLE_DEVICES=0 python3 main.py ${env}
