x = -5:0.1:5;
y = 1 ./ (1 + exp(-x));
plot(x, y);
%xlabel('x');
%ylabel('y');
text(-4.5,0.9,'$y=\frac{1}{1 - e^{-x}}$','Interpreter','latex');
xlim([-5 5]);
ylim([0 1])

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 2 2];
fig.PaperPositionMode = 'manual';
print('sigmoid','-dpng','-r0')

x = -3:0.1:3;
y = tanh(x);
plot(x, y);
%xlabel('x');
%ylabel('y');
text(-2.8,0.8,'$y=tanh(x)$','Interpreter','latex');
xlim([-3 3]);
ylim([-1 1])

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 2 2];
fig.PaperPositionMode = 'manual';
print('tanh','-dpng','-r0')

x = -3:0.1:3;
y = max(x, 0);
plot(x, y);
%xlabel('x');
%ylabel('y');
text(-0.9,0.8,'$y=max(x,0)$','Interpreter','latex');
xlim([-1 1]);
ylim([-1 1])

fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 2 2];
fig.PaperPositionMode = 'manual';
print('relu','-dpng','-r0')
