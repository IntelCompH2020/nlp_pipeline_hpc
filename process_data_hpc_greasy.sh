#!/bin/bash
###########
# Script that scans a given directory looking for datasets to be processed and sends jobs to an HPC cluster using greasy
###########

DATA_DIR=$1
WORKING_DIR=$( pwd )

for d in $DATA_DIR/*; do
    BASENAME=$( basename $d )
    if [ -d "$d" ]; then
        if ! [[ -d $d"_nlp" ]]; then
            if test -f $d.yaml; then
                # Create a text file with a list of tasks to be executed
                bash hpc/greasy/create_tasks_file.sh $d
                # Count the number of tasks to decide how many nodes to request
                NTASKS=`wc -l < hpc/greasy/tasks/tasks_file_$BASENAME.txt`
                # Account for master
                NTASKS=$(( NTASKS+1 ))
                # Set the maximum number of tasks to 2048 (16 CTE-AMD nodes, 1 cpus per task)
                NTASKS=$((NTASKS<8 ? NTASKS : 8))
                # Specify the number of nodes (each one has 128 threads, plus one for the master)
#                 NNODES=$(( ((NTASKS)/128)+1 ))
                # Edit the number of nodes requested in the SBATCH
                sed --i -e "s/#SBATCH --ntasks=[0-9]\+/\#SBATCH --ntasks=$NTASKS/" hpc/greasy/greasy_scheduler_supp.job
                    # Send job to queue using greasy
                sbatch hpc/greasy/greasy_scheduler_supp.job $WORKING_DIR/hpc/greasy/tasks/tasks_file_$BASENAME.txt
                # Display some info about the job that has been sent
                echo Job sent for: $BASENAME. Running $NTASKS tasks on $NNODES nodes
            else
                echo Missing YAML file: $BASENAME
            fi
        else
            echo Already processed: $BASENAME
        fi
    else
        echo Not a directory: $BASENAME
    fi
done
