#!/usr/bin/env bash
label=$1
shift
#nohup srun --partition=gpu --gres=gpu:1 --constraint=K20 python ratgps_bilstm.py $label.hdf5 --verbose 2 $* >$label.log &
nohup python ratgps_bilstm.py $label.hdf5 --verbose 2 $* >$label.log
