import sys
import os
import math
import pathlib
from glob import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, required=False, default=32)
    parser.add_argument("--dir_sbatchs", type=str, required=False, default=None)
    parser.add_argument("--run_path", type=str, required=False, default="/gpfs/projects/bsc88/projects/intelcomp/T3.1_NLP_in_HPC/official_github/NLP_pipeline/hpc/")
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    batch_args = parse_args()
    parquet_files = glob(os.path.join(batch_args.dir_path, "*.parquet"))
    num_parquets = len(parquet_files)

    #create sbatch_folder
    if batch_args.dir_sbatchs is None:
        batch_args.dir_sbatchs = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tasks",os.path.basename(batch_args.dir_path))
        pathlib.Path(batch_args.dir_sbatchs).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(batch_args.dir_sbatchs, "logs")).mkdir(parents=True, exist_ok=True)

    parquets_per_gpu = num_parquets / batch_args.num_gpus
    num_parquets_per_job = 1 # if there are more jobs than parquets, else calculate how many num parquets per job
    num_jobs = min(num_parquets, batch_args.num_gpus)
    if num_parquets >= num_jobs:
        num_parquets_per_job = math.ceil(num_parquets / batch_args.num_gpus)
    for i in range(num_jobs):
        sbatch_filename = os.path.join(batch_args.dir_sbatchs, f"sbatch_{i}.sh")
        with open(sbatch_filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH -D .\n")
            f.write(f"#SBATCH --job-name={batch_args.dir_sbatchs}/sbatch_execution_{i}\n")
            f.write(f"#SBATCH --error={batch_args.dir_sbatchs}/logs/sbatch_execution_{i}.err\n")
            f.write(f"#SBATCH --output={batch_args.dir_sbatchs}/logs/sbatch_execution_{i}.out\n")
            f.write("#SBATCH --gres=gpu:1\n")
            f.write("#SBATCH -c64\n")
            f.write("#SBATCH --time=2-0:00:00\n\n")

            for parquet_file in parquet_files[i*num_parquets_per_job: (i+1)*num_parquets_per_job]:
                run_path = os.path.join(batch_args.run_path, 'run.sh')
                f.write(f"bash {run_path} \"{parquet_file}\"\n")
