function [lambdaCutOff, lambda] = lassoInterpretor(b, fitInfo, pNames)
% Takes MatLab's lasso() output and turns it into a comprehensible table
% of remaining features
% Isolating Lambda values at InterceptTerm
lambdaCutOff = b(:, fitInfo.Index1SE);
lambdaCutOff = lambdaCutOff.'; % Transposing lamdaCutOff (11x1 to 1x11)
lambdaCutOff = array2table(lambdaCutOff, 'VariableNames', pNames);
% Eliminating features
for iColumn = width(lambdaCutOff):-1:1
    if lambdaCutOff{1, iColumn} == 0
        lambdaCutOff(:, iColumn) = [];
    end
end

% Obtaining beta0/intercept
coef0 = fitInfo.Intercept(fitInfo.Index1SE);
% Adjusting format
temp = lambdaCutOff(1,1);
temp{1, 1} = coef0;

% Concatenate coef0 (as intercept) and lambdaCutOff
lambdaCutOff = addvars(lambdaCutOff, temp{:, 1}, 'Before', 1,...
    'NewVariableNames', 'intercept');
% Returning lambda value to use
lambda = fitInfo.Lambda(fitInfo.Index1SE);
end