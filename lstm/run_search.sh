#!/usr/bin/env bash

./ratgps_bilstm.sh reference

./ratgps_bilstm.sh 20ms --features all_neurons_20ms.mat --locations locations_20ms.mat
./ratgps_bilstm.sh 200ms --features all_neurons_200ms.npy --locations locations_200ms.npy

./ratgps_bilstm.sh hidden_256 --hidden_nodes 128
./ratgps_bilstm.sh hidden_512 --hidden_nodes 256
./ratgps_bilstm.sh hidden_2048 --hidden_nodes 1024

./ratgps_bilstm.sh seqlen_10 --seqlen 10
./ratgps_bilstm.sh seqlen_200 --seqlen 200
./ratgps_bilstm.sh seqlen_1000 --seqlen 1000

./ratgps_bilstm.sh bs_5 --batch_size 5
./ratgps_bilstm.sh bs_10 --batch_size 10

#./ratgps_bilstm.sh stateful --stateful

#./ratgps_bilstm.sh backwards --backwards

./ratgps_bilstm.sh shuffle_false --shuffle false
#./ratgps_bilstm.sh shuffle_batch --shuffle batch

./ratgps_bilstm.sh dropout_0 --dropout 0
./ratgps_bilstm.sh dropout_0.2 --dropout 0.2

./ratgps_bilstm.sh layers_2 --layers 2
./ratgps_bilstm.sh layers_3 --layers 3

./ratgps_bilstm.sh adam --optimizer adam
