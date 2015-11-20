load('data/TheMatrixV2.mat')
findpeaks(position,'MinPeakProminence',50)
scatter(1:size(position,1), position, 5)
