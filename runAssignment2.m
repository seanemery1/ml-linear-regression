%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Start/Run This Script %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clearing command window/workspace/figures
clear;
clc;
close all;
rng(1); % for reproducability


%%% Importing Data %%%
day = readtable('day.csv');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Cleaning Data & Feature Selection %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[daySelect, daySelectDum] = cleanDataFeatSelect(day);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Matlab Regression and Hypothesis Testing %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modeling selected features
figure
matlabMDL = fitlm(daySelect,...
    'cnt~instant+atemp+hum+windspeed+workingday+holiday+winter+spring+fall+cloudy+light')
% Plotting matlabMDL
plot(matlabMDL)

%%% Lasso Regularization %%%
% Extracting Predictor Names
pNames = daySelect.Properties.VariableNames;
pNames = pNames(1:end-1);
% Obtaining Lambda iterations (B) and Intercept term (located in fitInfo)
[b, fitInfo] = lasso(daySelect{:, 1:end-1}, daySelect{:, end}, 'CV', 10,...
    'PredictorNames', pNames);
% Plotting lassoPlot
lassoPlot(b, fitInfo, 'PlotType', 'Lambda', 'XScale', 'log')
legend('show')

% Obtaining beta values, variable names, and lambda from b and fitInfo
[lambdaCutOff, lambda1] = lassoInterpretor(b, fitInfo, pNames)
% Demonstration that reducing non significant features lowers Adj R-squared
% Features selected from lambdaCutOff
figure
matlabMDLReduced = fitlm(daySelect,...
    'cnt~instant+atemp+hum+windspeed+holiday+winter+spring+cloudy+light')
% Plotting matlabMDLReduced
plot(matlabMDLReduced);

% matLab reference and final model (regression without instant due to weird
% serial correlation interactions/growth over time not being relevant)
figure
matlabMDLNoInstant = fitlm(daySelect,...
    'cnt~atemp+hum+windspeed+workingday+holiday+winter+spring+fall+cloudy+light')
plot(matlabMDLNoInstant);

% Initializing typical peak/low demand day scenarios for LA
% For some reason, the intercept counts as a variable and so my vector size
% needs to be 11 long despite my model only having 10 actual variables
% (excluding the intercept). Xpeak and Xpeak2 demonstrates that there is
% no difference when the first element of the vector is modified
Xpeak = [0, 28, 68, 5, 1, 0, 0, 0, 1,  0, 0];
Xpeak2 = [1, 28, 68, 5, 1, 0, 0, 0, 1,  0, 0];
Xlow = [0, 20, 64, 5, 0, 1, 1, 0, 0, 0, 1];

% Predicting with model
Ypeak = predict(matlabMDLNoInstant, Xpeak)
Ypeak2 = predict(matlabMDLNoInstant, Xpeak2)
Ylow = predict(matlabMDLNoInstant, Xlow)

% Graphing LA peak/low predictions against Washington DC via y axis lines
yline(Ypeak, '-.', {'LA Peak Prediction'},  'DisplayName',...
    'LA Peak Prediction (Nov)', 'LabelHorizontalAlignment', 'left');
yline(Ylow, '-.', {'LA Low Prediction'},  'DisplayName',...
    'LA Low Prediction (Feb)', 'LabelHorizontalAlignment', 'left');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Custom Grad Descent %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Selecting feature range (all except for instant because it has serial
% correlation which disrupts my custom gradient descent algorithm)
X = daySelect{:, 2:end-1};
y = daySelect{:, end};

% Selecting parameters for custom gradDescent
thresh = 1e-4;
learnRate = 0.0001;
maxIters = 4000000;

% Performing gradDescent on X, y (costHistory is not shown in full as it is
% too large - ~3 million iterations - because learnRate is very small)
[theta1, costHist] = gradDescent(X, y, learnRate, maxIters, thresh);
disp('theta1 = ');
disp(theta1);
disp('costHist(1:10) = ');
disp(costHist(1:10));

% Extracting Predictor Names (without instant)
pNames2 = daySelect.Properties.VariableNames;
pNames2 = pNames2(2:end-1);
% Obtaining Lambda iterations (B) and Intercept term (located in fitinfo)
[b2, fitInfo2] = lasso(X, y, 'CV', 10, 'PredictorNames', pNames2);
% Obtaining beta values, variable names, and lambda from b and fitInfo
[lambdaCutOff2, lambda2] = lassoInterpretor(b2, fitInfo2, pNames2)

% Using lambda2 on gradDescentLasso
convergeBreak = 1; % 1 = breaks when theta converges, 0 = wait til maxIters
maxIters = 1000; % precautionary number
theta2 = gradDescentLasso(X, y, maxIters, lambda2, convergeBreak)

% Repeat of last gradDescentLasso with fewer iterations
convergeBreak = 1; % 1 = breaks when theta converges, 0 = wait til maxIters
maxIters = 10;
theta2LowMax = gradDescentLasso(X, y, maxIters, lambda2, convergeBreak)

% Alternate closed form Lasso Solution
lambda = 0;
[m, n] = size(X); % row length and length of features
n = n + 1; % extra intercept count
X = [ones(m,1), X]; % adding a ones column
theta3  = (X.' * X + lambda * eye(n, n)) \ X.' * y