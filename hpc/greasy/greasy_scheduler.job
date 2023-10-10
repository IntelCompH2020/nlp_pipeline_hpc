#!/bin/bash
#SBATCH --chdir hpc/greasy/logs
#SBATCH --job-name=greasy
#SBATCH --output=greasy-%j.out
#SBATCH --error=greasy-%j.err
#SBATCH --ntasks=201
#SBATCH --cpus-per-task=2
#SBATCH --time=2-0:00:00

# Load greasy module (cluster agnostic)
if uname -a | grep -q amd
then
	module load impi intel greasy
else
	module load greasy
fi

# Fix for PMI errors on CTE-AMD
export I_MPI_PMI_VALUE_LENGTH_MAX=512

# Config file with the list of tasks
TASKS_FILE=$1

# Run greasy
greasy $TASKS_FILE
