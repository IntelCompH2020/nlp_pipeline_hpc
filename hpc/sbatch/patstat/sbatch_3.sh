#!/bin/bash
#SBATCH -D .
#SBATCH --job-name=/gpfs/projects/bsc88/projects/intelcomp/T3.1_NLP_in_HPC/official_github/NLP_pipeline/hpc/sbatch/patstat/sbatch_execution_3
#SBATCH --error=/gpfs/projects/bsc88/projects/intelcomp/T3.1_NLP_in_HPC/official_github/NLP_pipeline/hpc/sbatch/patstat/logs/sbatch_execution_$i.err
#SBATCH --output=/gpfs/projects/bsc88/projects/intelcomp/T3.1_NLP_in_HPC/official_github/NLP_pipeline/hpc/sbatch/patstat/logs/sbatch_execution_$i.out
#SBATCH --gres=gpu:1
#SBATCH --time=2-0:00:00

bash ../../run.sh "toy_data//patstat/part_0000.parquet"
