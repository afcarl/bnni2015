clear all;
load('traces/V2.mat')

a = [];
for i = 1:numel(botrace.values)
    a(i) = min(botrace.values(1:i));
end

figure;

subplot(3, 3, 1);
plot(botrace.values);
title('Trace');
xlabel('Iteration');
ylabel('Average distance');
ylim([7 20]);

subplot(3, 3, 2);
plot(a);
title('Best result');
xlabel('Iteration');
ylabel('Average distance');
ylim([7 20]);

subplot(3, 3, 3);
scatter(botrace.samples(:, 1), botrace.values);
title('Learning rate');
xlabel('Learning rate');
ylabel('Average distance');
ylim([7 20]);

subplot(3, 3, 4);
scatter(botrace.samples(:, 2), botrace.values);
title('Momentum');
xlabel('Momentum');
ylabel('Average distance');
ylim([7 20]);

subplot(3, 3, 5);
scatter(botrace.samples(:, 3), botrace.values);
title('Learning rate decay');
xlabel('Learning rate decay');
ylabel('Average distance');
ylim([7 20]);

subplot(3, 3, 6);
scatter(botrace.samples(:, 4), botrace.values);
title('Dropout');
xlabel('Dropout');
ylabel('Average distance');
ylim([7 20]);

subplot(3, 3, 7);
scatter(botrace.samples(:, 5), botrace.values);
title('Weight decay');
xlabel('Weight decay');
ylabel('Average distance');
ylim([7 20]);

subplot(3, 3, 8);
scatter(2.^round(botrace.samples(:, 6)), botrace.values);
title('Nr hidden nodes');
xlabel('Nr hidden nodes');
ylabel('Average distance');
ylim([7 20]);

subplot(3, 3, 9);
scatter(round(botrace.samples(:, 7)), botrace.values);
title('Nr layers');
xlabel('Nr layers');
ylabel('Average distance');
ylim([7 20]);

figure;

subplot(3, 3, 1);
scatter(botrace.samples(:, 1), botrace.samples(:, 2), [], min(botrace.values, 20));
title('Learning rate vs Momentum');
xlabel('Learning rate');
ylabel('Momentum');
colorbar;

subplot(3, 3, 2);
scatter(botrace.samples(:, 1), botrace.samples(:, 3), [], min(botrace.values, 20));
title('Learning rate vs Learning rate decay');
xlabel('Learning rate');
ylabel('Learning rate decay');
colorbar;

subplot(3, 3, 3);
scatter(botrace.samples(:, 1), botrace.samples(:, 4), [], min(botrace.values, 20));
title('Learning rate vs Dropout');
xlabel('Learning rate');
ylabel('Dropout');
colorbar;

subplot(3, 3, 4);
scatter(botrace.samples(:, 1), botrace.samples(:, 5), [], min(botrace.values, 20));
title('Learning rate vs Weight decay');
xlabel('Learning rate');
ylabel('Weight decay');
colorbar;

subplot(3, 3, 5);
scatter(botrace.samples(:, 1), 2.^round(botrace.samples(:, 6)), [], min(botrace.values, 20));
title('Learning rate vs Nr hidden nodes');
xlabel('Learning rate');
ylabel('Nr hidden nodes');
colorbar;

subplot(3, 3, 6);
scatter(botrace.samples(:, 1), round(botrace.samples(:, 7)), [], min(botrace.values, 20));
title('Learning rate vs Nr layers');
xlabel('Learning rate');
ylabel('Nr layers');
colorbar;

subplot(3, 3, 7);
scatter(2.^round(botrace.samples(:, 6)), botrace.samples(:, 4), [], min(botrace.values, 20));
title('Nr hidden nodes vs Dropout');
xlabel('Nr hidden nodes');
ylabel('Dropout');
colorbar;

subplot(3, 3, 8);
scatter(2.^round(botrace.samples(:, 6)), botrace.samples(:, 7), [], min(botrace.values, 20));
title('Nr hidden nodes vs Nr layers');
xlabel('Nr hidden nodes');
ylabel('Nr layers');
colorbar;

subplot(3, 3, 9);
scatter(botrace.samples(:, 5), botrace.samples(:, 4), [], min(botrace.values, 20));
title('Weight decay vs Dropout');
xlabel('Weight decay');
ylabel('Dropout');
colorbar;
