#!/bin/bash
#SBATCH --job-name=lab4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00

set -euo pipefail

echo "=== Host ==="
echo "Job executed on: $(hostname)"
echo "Job started on: $(date)"

source $HOME/deep_learning/env_dl/bin/activate

echo "=== Python ==="
echo "Python executable location: $(which python)"
echo "Python version: $(python --version)"

cd $HOME/deep_learning/src/lab_4

echo "=== Execution ==="
python regression.py
echo "----"
python classification.py
echo "----"
python minibatch.py
