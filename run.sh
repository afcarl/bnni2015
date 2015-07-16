#!/bin/bash

nohup srun -c 3 -t 8-0 -p main /storage/software/MATLAB_R2013b/bin/matlab -nodisplay -nosplash -nojvm -r "regression_bayesopt('data/TheMatrix$1.mat', 'traces/$1.mat')" >>logs/$1.log &
