function [theta, costHist] = gradDescent(X, y, learnRate, maxIters, thresh)
% Performs gradient descent to learn theta
% Returns theta if theta converges if threshold tolerence is met

[m,~] = size(X);  % number of training examples
X = [ones(m,1), X]; % adding columns of one
[m,n] = size(X);

% initializing variables
theta = zeros(n, 1);
costHist = [];
format longG

% grad descent algorithm
for iters = 1:maxIters
    y_pred = X * theta;
    loss = y_pred - y;
    stepSize = (learnRate/m) * (X.' * loss);
    theta = theta - stepSize;
    costHist(iters) = cost(X, y, theta);
    % convergence/tolerance break condition
    if norm(stepSize) < thresh
        disp('# of iterations to threshold error')
        disp(iters)
     break
    end
end
end