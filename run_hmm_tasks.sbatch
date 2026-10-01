#!/bin/bash
#SBATCH --job-name=hmm_full
#SBATCH --output=logs/hmm_full_%j.out
#SBATCH --error=logs/hmm_full_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH -C genoa



set -euo pipefail

cd /mnt/home/srafilson/code/sequences

module load python
source .venv/bin/activate

cd /mnt/home/srafilson/code/STWHMM

mkdir -p logs results/cv

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

disBatch tasks/hmm_tasks.txt