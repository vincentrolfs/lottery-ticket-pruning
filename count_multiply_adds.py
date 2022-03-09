import json
import subprocess

from imports.utils import get_filename, ParameterType, PruningType

multiply_adds = []

ALL_SEEDS = [95284986, 96488735]
ALL_RATIOS = [1, 2, 4, 8, 16, 32, 64]
MODEL_DIR = 'data/models'
OUTPUT_FILE = 'data/multiply_adds/multiply_adds.json'

try:
    with open(OUTPUT_FILE, "r") as f:
        multiply_adds = json.load(f)
except FileNotFoundError:
    pass

print(multiply_adds)


def runfile(filename):
    process = subprocess.Popen(
        ['python3', 'compile.py', '--model', filename, '--skip-write', '--skip-log'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    return int(stdout)


for pruning_type in [PruningType.LT_TRADITIONAL]:
    for seed in ALL_SEEDS:
        for ratio in ALL_RATIOS:
            for parameter_type in [ParameterType.FINAL]:
                print(seed, "~", ratio)

                filename = get_filename(MODEL_DIR, seed, ratio, parameter_type, pruning_type, 0 if ratio == 1 else 1)
                output = runfile(filename)
                print(output)

                multiply_adds.append({
                    'seed': seed,
                    'compression_ratio': ratio,
                    'multiply_adds': output
                })

with open(OUTPUT_FILE, "w+") as f:
    json.dump(multiply_adds, f)
