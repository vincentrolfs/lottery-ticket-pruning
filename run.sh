#!/bin/sh

set -e

pipenv run python3 main.py --seed-index=9 --compression-ratio=64 --iterative-round=1 --save-dir=/export/scratch/8rolfs/data/models --cifar-dir=/export/scratch/8rolfs/data/CIFAR10
