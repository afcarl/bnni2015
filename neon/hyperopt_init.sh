#!/usr/bin/env bash

if [ -z $1 ]; then
  echo "Usage: $0 <yaml_file>"
  exit 1
fi

export HYPEROPT_PATH=$HOME/ratgps/neon/hyperopt

hyperopt init -y $1
