#!/usr/bin/env bash

./ratcv.sh reference

./ratcv.sh hidden_512 --hidden_nodes 512
./ratcv.sh hidden_2048 --hidden_nodes 2048

./ratcv.sh seqlen_10 --seqlen 10
./ratcv.sh seqlen_200 --seqlen 200
./ratcv.sh seqlen_1000 --seqlen 1000

./ratcv.sh bs_5 --batch_size 5
./ratcv.sh bs_10 --batch_size 10

./ratcv.sh stateful --stateful
./ratcv.sh backwards --backwards

./ratcv.sh shuffle_false --shuffle false
./ratcv.sh shuffle_batch --shuffle batch

./ratcv.sh dropout_0 --dropout 0
./ratcv.sh dropout_0.2 --dropout 0.2

./ratcv.sh layers_2 --layers 2
./ratcv.sh layers_3 --layers 3

./ratcv.sh adam --optimizer adam

./ratcv.sh 200ms --features all_neurons_200ms.npy --locations locations_200ms.npy
./ratcv.sh 20ms --features all_neurons_20ms.mat --locations locations_20ms.mat
