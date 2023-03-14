#!/bin/bash

cd "$(dirname "$0")"

source use_env.sh

YAML_FILE=$( dirname $1 ).yaml
echo $YAML_FILE

if [ -f "$YAML_FILE" ]; then
    time python run.py \
        --parquet_file $1 \
        --config_file $YAML_FILE
else 
    echo "Missing YAML file."
fi
