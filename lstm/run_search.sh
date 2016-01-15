#!/usr/bin/env bash

./ratbilstm.sh reference

./ratbilstm.sh 20ms --features all_neurons_20ms.mat --locations locations_20ms.mat
./ratbilstm.sh 200ms --features all_neurons_200ms.npy --locations locations_200ms.npy

./ratbilstm.sh hidden_256 --hidden_nodes 128
./ratbilstm.sh hidden_512 --hidden_nodes 256
./ratbilstm.sh hidden_2048 --hidden_nodes 1024

./ratbilstm.sh seqlen_10 --seqlen 10
./ratbilstm.sh seqlen_200 --seqlen 200
./ratbilstm.sh seqlen_1000 --seqlen 1000

./ratbilstm.sh bs_5 --batch_size 5
./ratbilstm.sh bs_10 --batch_size 10

#./ratbilstm.sh stateful --stateful

#./ratbilstm.sh backwards --backwards

./ratbilstm.sh shuffle_false --shuffle false
#./ratbilstm.sh shuffle_batch --shuffle batch

./ratbilstm.sh dropout_0 --dropout 0
./ratbilstm.sh dropout_0.2 --dropout 0.2

./ratbilstm.sh layers_2 --layers 2
./ratbilstm.sh layers_3 --layers 3

./ratbilstm.sh adam --optimizer adam
