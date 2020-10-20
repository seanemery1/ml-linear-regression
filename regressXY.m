function [b1, b0, res] = regressXY(x, y, xhead, yhead)
% Taken from regression lab, regresses X and Y (or in this case, two
% features to compare coefficients)

% Calculuating graph variables and draws the graph
X = [ones(length(x),1) x];
b0 = X\y;
linefit = X*b0;
scatter(x,y)
hold on
plot(x,linefit)
xlabel(xhead)
ylabel(yhead)
title(strcat('LinReg of ', ' ', xhead,' and ', ' ',yhead))
grid on
res = y - linefit;
mdl = fitlm(x,y);
R_squared = mdl.Rsquared.Adjusted;
str = ['R^2 = ' num2str(R_squared)];
T = text(min(get(gca, 'xlim')), max(get(gca, 'ylim')), str); 
set(T, 'fontsize', 14, 'verticalalignment', 'top',...
    'horizontalalignment', 'left');
b0 = mdl.Coefficients{1,1};
b1 = mdl.Coefficients{2,1};
res = mdl;
hold off
end