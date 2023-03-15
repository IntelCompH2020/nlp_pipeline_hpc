#!/bin/bash
#SBATCH -D .
#SBATCH --job-name=/gpfs/projects/bsc88/projects/intelcomp/T3.1_NLP_in_HPC/official_github/NLP_pipeline/hpc/sbatch/semantic/sbatch_execution_2
#SBATCH --error=/gpfs/projects/bsc88/projects/intelcomp/T3.1_NLP_in_HPC/official_github/NLP_pipeline/hpc/sbatch/semantic/logs/sbatch_execution_$i.err
#SBATCH --output=/gpfs/projects/bsc88/projects/intelcomp/T3.1_NLP_in_HPC/official_github/NLP_pipeline/hpc/sbatch/semantic/logs/sbatch_execution_$i.out
#SBATCH --gres=gpu:1
#SBATCH --time=2-0:00:00

bash ../../run.sh "toy_data//semantic/part-00004-13434629-9129-4b93-9f8a-f4b75b6b8d2a-c000.snappy.parquet"
