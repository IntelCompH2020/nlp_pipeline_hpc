#!/bin/bash
#SBATCH --chdir .
#SBATCH --job-name=greasy
#SBATCH --output=logs/greasy-%j.out
#SBATCH --error=logs/greasy-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-0:00:00

# Go to the file directory
cd "$(dirname "$0")"

# Activate virtual environment
source use_env.sh

# Paths to input files
PARQUET_FILE=$1
YAML_FILE=$( dirname $1 ).yaml

if [ -f "$YAML_FILE" ]; then
    export CUDA_VISIBLE_DEVICES=$((SLURM_PROCID%2))
    time python run.py --parquet_file $PARQUET_FILE --config_file $YAML_FILE
else 
    echo "Missing YAML file."
fi
