function theta = gradDescentLasso(X, y, maxIters, lambda, convergeBreak)
% Coordinate gradient descent for lasso regression (unnormalized inputs). 
% Returns theta after maxIters if convergeBreak = 0
% Returns theta if theta converges and convergeBreak = 1

format longG

[m,~] = size(X);  % m = number of training examples
X = [ones(m,1), X]; % adding a ones column
[m,n] = size(X); % m = row length, n = column/feature length

% 1/z(feat, 1) is a normalizing constant which is equal to 1 when the data
% is normalized.
z = zeros(n, 1);
for feat = 1:n
    z(feat, 1) = sum(X(:, feat).^2) ;
end

% initializing theta as a column of 0s
theta = zeros(n, 1);

% gradDescentLasso algorithm AKA coordinate descent for Lasso regression
for iter = 1:maxIters
    thetaOld = theta;
    for feat = 1:n
        % calculating ?j
        Xfeat = X(:, feat);
        y_predict = X * theta;
        rho = Xfeat.' * (y - y_predict + theta(feat, 1) * Xfeat);
        % soft thresholding function
        % ?j = (?j+?)/zj    for ?j < ??
        % ?j = 0            for ?? ? ?j? ?
        % ?j = (?j??)/zj    for ?j > ?
        if feat == 1
            theta(feat, 1) = rho/z(feat, 1);
        elseif rho < - lambda      
            theta(feat, 1) = (rho + lambda)/z(feat, 1);
        elseif rho > lambda
            theta(feat, 1) = (rho - lambda)/z(feat, 1) ;
        else 
           theta(feat, 1) = 0;      
        end
    end
    % No more changes to theta, so break if convergeBreak = 1
    if isequal(theta, thetaOld) == true && convergeBreak == 1
        break;
    end
end
end