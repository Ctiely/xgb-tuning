# xgb-tuning
this class is used to tune parameters for xgboost.

Here, we will pass through five stages of tuning.
First stage, we tune two parameters-----max_depth and min_child_weight at a larger learning rate.
Second stage, we tune just one parameter-----gamma.
Third stage, we tune two parameters-----colsample_bytree and submaple.
Fourth stage, we tune two parameters-----reg_alpha and reg_lambda.
Last stage, we change the learning rate and tune parameter-----n_estimators.
The whole process is automatic, you just need to input your y and X and set some parameters, then you can use .train to start this process.

For example:

model = xgb_tuning(X, y)

model.train()

Then, you will get a dictionary of optimal parameters and an optimal xgboost tree model.
