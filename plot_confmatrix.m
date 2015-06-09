function plot_confmatrix(groundtruth, predictions)
%PLOT_CONFMATRIX Plots confusion matrix with text annotations.

confmat = confusionmat(groundtruth, predictions);
imagesc(confmat);
colormap jet;
%colorbar;

mingt = min(groundtruth);
maxgt = max(groundtruth);
[x, y] = meshgrid(mingt:maxgt);
text(x(:), y(:), num2str(confmat(:)), 'Color', 'white', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

end

