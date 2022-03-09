#!/bin/sh

set -e

for name in /export/scratch/8rolfs/data/models/lt_traditional_*_round01_initial.th; do
  echo $name
  pipenv run python3 main.py --validate-only --progress=$name --type=lt_traditional --save-dir=/export/scratch/8rolfs/data/models --cifar-dir=/export/scratch/8rolfs/data/CIFAR10
done