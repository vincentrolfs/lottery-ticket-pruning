#!/bin/sh

set -e

for file in data/models/*_final.th
do
  echo "pipenv run python3 compile.py --model \"$file\""
  pipenv run python3 compile.py --model "$file"
done