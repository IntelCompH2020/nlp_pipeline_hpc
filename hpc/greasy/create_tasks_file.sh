#!/bin/bash  

# Store paths in environment variables
DATA_DIR=$1
WORKING_DIR=$( realpath hpc )

echo $DATA_DIR
echo $WORKING_DIR

# Define name for the output file
DATASET_NAME=$( basename $DATA_DIR )
TASKS_FILE=$WORKING_DIR/greasy/tasks/tasks_file_$DATASET_NAME.txt

# Remove file if it already exists
[ -e $TASKS_FILE ] && rm $TASKS_FILE

# Iteratively append command lines to tasks file
LIST_PARQUETS=($(ls $DATA_DIR/*.parquet))
for path in ${LIST_PARQUETS[@]}
do
	echo "[@ $WORKING_DIR @] bash run.sh $path" >> $TASKS_FILE
done
