#!/usr/bin/env bash

./ratgps_lstm.sh reference

./ratgps_lstm.sh 20ms --features all_neurons_20ms.mat --locations locations_20ms.mat
./ratgps_lstm.sh 200ms --features all_neurons_200ms.npy --locations locations_200ms.npy

./ratgps_lstm.sh hidden_128 --hidden_nodes 128
./ratgps_lstm.sh hidden_256 --hidden_nodes 256
./ratgps_lstm.sh hidden_1024 --hidden_nodes 1024

./ratgps_lstm.sh seqlen_10 --seqlen 10
./ratgps_lstm.sh seqlen_200 --seqlen 200
./ratgps_lstm.sh seqlen_1000 --seqlen 1000

./ratgps_lstm.sh bs_5 --batch_size 5
./ratgps_lstm.sh bs_10 --batch_size 10

./ratgps_lstm.sh stateful --stateful

./ratgps_lstm.sh backwards --backwards

./ratgps_bilstm.sh bilstm

./ratgps_lstm.sh shuffle_false --shuffle false
./ratgps_lstm.sh shuffle_true --shuffle true

./ratgps_lstm.sh dropout_0.1 --dropout 0.1
./ratgps_lstm.sh dropout_0.2 --dropout 0.2
./ratgps_lstm.sh dropout_0.5 --dropout 0.5

./ratgps_lstm.sh layers_2 --layers 2
./ratgps_lstm.sh layers_3 --layers 3

./ratgps_lstm.sh rmsprop --optimizer rmsprop
