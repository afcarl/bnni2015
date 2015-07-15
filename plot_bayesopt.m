load ratgps.mat

a = [];
for i = 1:numel(botrace.values)
    a(i) = min(botrace.values(1:i));
end

figure;

subplot(3, 2, 1);
plot(a);
title('Best result');
xlabel('Iteration');
ylabel('Average distance');

subplot(3, 2, 2);
scatter(botrace.samples(:, 1), botrace.values);
title('Learning rate');
xlabel('Learning rate');
ylabel('Average distance');
ylim([9 20]);

subplot(3, 2, 3);
scatter(botrace.samples(:, 2), botrace.values);
title('Momentum');
xlabel('Momentum');
ylabel('Average distance');
ylim([9 20]);

subplot(3, 2, 4);
scatter(botrace.samples(:, 3), botrace.values);
title('Learning rate decay');
xlabel('Learning rate decay');
ylabel('Average distance');
ylim([9 20]);

subplot(3, 2, 5);
scatter(botrace.samples(:, 4), botrace.values);
title('Dropout');
xlabel('Dropout');
ylabel('Average distance');
ylim([9 20]);

subplot(3, 2, 6);
scatter(botrace.samples(:, 5), botrace.values);
title('Weight decay');
xlabel('Weight decay');
ylabel('Average distance');
ylim([9 20]);

figure;

subplot(2, 2, 1);
scatter(botrace.samples(:, 1), botrace.samples(:, 2), [], min(botrace.values, 10));
title('Learning rate vs momentum');
xlabel('Learning rate');
ylabel('Momentum');
colorbar;

subplot(2, 2, 2);
scatter(botrace.samples(:, 1), botrace.samples(:, 3), [], min(botrace.values, 10));
title('Learning rate vs learning rate decay');
xlabel('Learning rate');
ylabel('Learning rate decay');
colorbar;

subplot(2, 2, 3);
scatter(botrace.samples(:, 1), botrace.samples(:, 4), [], min(botrace.values, 10));
title('Learning rate vs dropout');
xlabel('Learning rate');
ylabel('Dropout');
colorbar;

subplot(2, 2, 4);
scatter(botrace.samples(:, 1), botrace.samples(:, 5), [], min(botrace.values, 10));
title('Learning rate vs weight decay');
xlabel('Learning rate');
ylabel('Weight decay');
colorbar;

figure;
plot(botrace.values);
