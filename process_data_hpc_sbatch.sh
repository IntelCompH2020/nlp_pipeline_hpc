#!/bin/bash
###########
# Script that scans a given directory looking for datasets to be processed and uses 2 GPUs per node (several tasks each)
###########

DATA_DIR=$1
WORKING_DIR=$( pwd )

for d in $DATA_DIR/*; do
  BASENAME=$( basename $d )
	if [ -d "$d" ]; then
		if ! [[ "$d" =~ .*"_nlp".* ]]; then
      if test -f $d.yaml; then
        # Create a text file with a list of tasks to be executed
        python hpc/sbatch/create_sbatchs.py --dir_path $d
        # Iterate over bash scripts
        for file in hpc/sbatch/$BASENAME/*.sh
        do
          # Send job to cluster
          sbatch $file
          echo Job sent for: $BASENAME
        done
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