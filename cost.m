function J = cost(X, y, theta)
% This computes the cost of using beta as the parameter for linear
% regression to fit the data points in X and y
[m, ~]  = size(y);
J = sum((0.5 / m) * (((X * theta) - y).^ 2));
end