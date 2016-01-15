#!/usr/bin/env bash
label=$1
shift
#nohup srun --partition=gpu --gres=gpu:1 --constraint=K20 python ratcv.py $label.hdf5 --verbose 2 $* >$label.log &
nohup python ratcv.py $label.hdf5 --verbose 2 $* >$label.log &
