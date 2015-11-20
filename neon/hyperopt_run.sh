#!/usr/bin/env bash

export HYPEROPT_PATH=$HOME/ratgps/neon/hyperopt
#export PYTHONPATH=$PYTHONPATH:$HOME/spearmint/spearmint

hyperopt run -p 50000
