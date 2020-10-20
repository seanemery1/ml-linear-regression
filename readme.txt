Steps to run my program:
1) run runAssingment2 to generate data and graphs using MatLab's built in linear regression and custom linear regression.

Brief Descriptions on each function/file:
- cleanDataFeatSelect: cleans data (ie. seperating categorical data into dummy variables) and removes bad features
- gradDescent: custom gradient descent algorithm
	- cost: sub function that calculates the cost
- gradDescentLasso: custom gradient descent with lasso algorithm
- lassoInterpretor: processes the outputs of MatLab's lasso function into a table with remaining features (and their headers)
- regressXY: compares the relationship between X and Y (ie. two features)