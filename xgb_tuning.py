#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Yang Liu<814868906@qq.com>
@brief: a method of xgboost tuning.

"""


import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
class xgb_tuning(object):
    '''
    this class is used to tune parameters in xgboost's trees.
    X: array or dataframe
    the data matrix [n_samples,n_features].
    y: array or dataframe
    the array [n_samples].
    objective: str Default:'binary:logistic'
    pleasse input appropriate objective accroding to your task.
    scoring: str Default:'roc_auc'
    the evaluation you use.
    cv: int Default:5
    K-fold cross_validation 
    '''
    print('You must ensure that the required libraries are imported')
    print('You can input the following commands to import the necessary libraries.')
    print('import numpy as np')
    print('from xgboost.sklearn import XGBClassifier')
    print('from sklearn.grid_search import GridSearchCV')
    def __init__(self, X, y, n_estimators=1000, cv=5, objective='binary:logistic', 
                 scoring='roc_auc'):
        self.X = X
        self.y = y
        self.final_params = dict()
        self.best_score = 0
        self.objective = objective
        self.scoring = scoring
        self.best_estimator = None
        self.n_estimators = n_estimators
        self.cv = cv
        print('You must ensure that the required libraries are imported')
        print('You can input the following commands to import the necessary libraries.')
        print('import numpy as np')
        print('from xgboost.sklearn import XGBClassifier')
        print('from sklearn.grid_search import GridSearchCV')
    def generate_params(self, best_params, num=5, step=1):
        '''
        best_params: dict.
        use this dict to generate parameters
        
        num: int Default:5
        this parameter is used to specify the numbers of each parameter we will use 
        in xgboost. for example, if we input 5, we will get an array which contains
        5 numbers.
        
        step: int or float Default:1.0
        this parameter is used to specify the space between i and i+1'''
        
        print('Now we generate parameters:')
        params = dict()
        if num%2 == 0:
            minus = num/2
            for param in best_params.keys():
                if best_params[param] >= minus*step:
                    params[param] = np.arange(best_params[param]-minus*step, 
                                              best_params[param]+minus*step,step)
                else:
                    params[param] = np.arange(0, best_params[param]+minus*step,step)
        else:
            minus = (num-1)/2
            for param in best_params.keys():
                if best_params[param] >= minus*step:
                    params[param] = np.arange(best_params[param]-minus*step, 
                                              best_params[param]+(num-minus)*step,step)
                else:
                    params[param] = np.arange(0, best_params[param]+(num-minus)*step,step)
        print('========================================================================')
        print('Finished!')
        return(params)
    
    def tuning_stage1(self, param_test):
        '''
        This function is the first stage in xgb's tunning, this stage we tune two 
        parameters-----max_depth and min_child_weight.
        please input a dictionary such as {'max_depth':range(3,10,2), 
        min_child_weight':range(1,6,2)} contains max_depth and min_child_weight.
        '''
        gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 
        n_estimators = self.n_estimators, gamma = 0, subsample = 0.8, colsample_bytree = 0.8, 
        objective = self.objective), 
        param_grid = param_test, scoring = self.scoring, iid = False, cv = self.cv)
        gsearch.fit(self.X, self.y)
        return(gsearch.best_params_, gsearch.best_score_)
    def tuning_stage2(self,param_test, best_max_depth, best_min_child_weight):
        '''
        This function is the second stage in xgb's tunning, this stage we tune just
        one parameter-----gamma.
        please input a dictionary such as {'gamma':[i/10.0 for i in range(0,5)]} 
        contains gamma.
        '''
        gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 
        n_estimators = self.n_estimators, max_depth = best_max_depth, min_child_weight = 
        best_min_child_weight, subsample = 0.8, colsample_bytree = 0.8, 
        objective = self.objective), 
        param_grid = param_test, scoring = self.scoring, iid = False, cv = self.cv)
        gsearch.fit(self.X, self.y)
        return(gsearch.best_params_, gsearch.best_score_)
    def tuning_stage3(self,param_test, best_max_depth, best_min_child_weight, best_gamma):
        '''
        This function is the third stage in xgb's tunning, this stage we tune two
        parameters-----colsample_bytree and submaple.
        please input a dictionary such as {'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)]} contains colsample_bytree
        and submaple.
        '''
        gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 
        n_estimators = self.n_estimators, max_depth = best_max_depth, min_child_weight = 
        best_min_child_weight, gamma = best_gamma, objective = self.objective), 
        param_grid = param_test, scoring = self.scoring, iid = False, cv = self.cv)
        gsearch.fit(self.X, self.y)
        return(gsearch.best_params_, gsearch.best_score_)
    def tuning_stage4(self,param_test, best_max_depth, best_min_child_weight, best_gamma, 
                      best_subsample, best_colsample_bytree):
        '''
        This function is the third stage in xgb's tunning, this stage we tune two
        parameters-----reg_alpha and reg_lambda.
        please input a dictionary such as {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
        'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]} contains reg_alpha and reg_lambda.
        '''
        gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 
        n_estimators = self.n_estimators, max_depth = best_max_depth, min_child_weight = 
        best_min_child_weight, gamma = best_gamma, subsample = best_subsample, 
        colsample_bytree = best_colsample_bytree, objective = self.objective), 
        param_grid = param_test, scoring = self.scoring, iid = False, cv = self.cv)
        gsearch.fit(self.X, self.y)
        return(gsearch.best_params_, gsearch.best_score_)
    def tuning_stage5(self,param_test, best_max_depth, best_min_child_weight, best_gamma, 
                      best_subsample, best_colsample_bytree, best_alpha, best_lambda):
        '''
        This function is the third stage in xgb's tunning, this stage we tune two
        parameters-----learning_rate and n_estimators.
        please input a dictionary such as {'learning_rate':[i/100. for i in range(4,18,3)],
        'n_estimators':[i*100 for i in range(1,10,2)]} contains learning_rate and n_estimators.
        '''
        gsearch = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 
        n_estimators = self.n_estimators, max_depth = best_max_depth, min_child_weight = 
        best_min_child_weight, gamma = best_gamma, subsample = best_subsample, 
        colsample_bytree = best_colsample_bytree, objective = self.objective, 
        reg_alpha = best_alpha, reg_lambda = best_lambda), 
        param_grid = param_test, scoring = self.scoring, iid = False, cv = self.cv)
        gsearch.fit(self.X, self.y)
        return(gsearch.best_params_, gsearch.best_score_)
    def train(self):
        '''
        this function is used to start xgboost tuning
        '''
        #stage1#
        print('Now we start tuning')
        print('========================================================================')
        print('stage1:Start')
        param_test = {
         'max_depth':range(3,10,2),
         'min_child_weight':range(1,6,2)
        }
        print('step1:==================================================================')
        result = self.tuning_stage1(param_test)
        temp_score = result[1]
        temp_result = result
        best_max_depth, best_min_child_weight = result[0]['max_depth'], result[0]['min_child_weight']
        print('in this step, the best result is %s: %.5f' % result)    
        print('Now we enter the next step.')
        param_test = self.generate_params(result[0])
        result = self.tuning_stage1(param_test)
        if temp_score < result[1]:
            print('step2:==================================================================\n'+
                  'in this stage, the best result is %s: %.5f' % result)
            temp_score = result[1]
            print('========================================================================')
            print('stage1:End!')
            best_max_depth, best_min_child_weight = result[0]['max_depth'], result[0]['min_child_weight']
        else:
            print('========================================================================')
            print('step2:==================================================================')
            print('This step doesn\'t perform well, so we choose last step\'s parameters.')
            print('in this stage, the best result is %s: %.5f' % temp_result)
            print('stage1:End!')
        #stage2#    
        param_test={
         'gamma':[i/10.0 for i in range(0,11)]
        }
        print('========================================================================')
        print('stage2:Start')
        result = self.tuning_stage2(param_test, best_max_depth, best_min_child_weight)
        print('in this stage, the best result is %s: %.5f' % result)
        print('stage2:End')
        if temp_score < result[1]:
            temp_score = result[1]
            best_gamma = result[0]['gamma']
        else:
            best_gamma = 0
        #stage3#
        param_test = {
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)]
        }
        print('========================================================================')
        print('stage3:Start')
        result = self.tuning_stage3(param_test, best_max_depth, best_min_child_weight, best_gamma)
        temp_result = result
        best_subsample, best_colsample_bytree = result[0]['subsample'], result[0]['colsample_bytree'] 
        print('step1:==================================================================\n'+
              'in this step, the best result is %s: %.5f' % result)    
        print('Now we enter the next step.')
        param_test = self.generate_params(result[0],5,0.05)
        result = self.tuning_stage3(param_test, best_max_depth, best_min_child_weight, best_gamma)
        if temp_score < result[1]:
            print('step2:==================================================================\n'+
                  'in this stage, the best result is %s: %.5f' % result)
            temp_score = result[1]
            print('========================================================================')
            print('stage3:End!')
            best_subsample, best_colsample_bytree = result[0]['subsample'], result[0]['colsample_bytree'] 
        else:
            print('========================================================================')
            print('step2:==================================================================')
            print('This step doesn\'t perform well, so we choose last step\'s parameters.')
            print('in this stage, the best result is %s: %.5f' % temp_result)
            print('stage3:End!')
        #stage4#
        param_test = {
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
        'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
        }
        print('========================================================================')
        print('stage4:Start')
        result = self.tuning_stage4(param_test, best_max_depth, best_min_child_weight, best_gamma,
                               best_subsample, best_colsample_bytree)
        temp_result = result 
        best_alpha, best_lambda = result[0]['reg_alpha'], result[0]['reg_lambda'] 
        print('step1:==================================================================\n'+
              'in this step, the best result is %s: %.5f' % result)    
        print('Now we enter the next step.')
        param_test = self.generate_params(result[0],5,0.01)
        result = self.tuning_stage4(param_test, best_max_depth, best_min_child_weight, best_gamma,
                               best_subsample, best_colsample_bytree)
        if temp_score < result[1]:
            print('step2:==================================================================\n'+
                  'in this stage, the best result is %s: %.5f' % result)
            temp_score = result[1]
            print('========================================================================')
            print('stage4:End!')
            best_alpha, best_lambda = result[0]['reg_alpha'], result[0]['reg_lambda'] 
        else:
            print('========================================================================')
            print('step2:==================================================================')
            print('This step doesn\'t perform well, so we choose last step\'s parameters.')
            print('in this stage, the best result is %s: %.5f' % temp_result)
            print('stage4:End!')
        #stage5#
        param_test = {
        'learning_rate':[i/100. for i in range(4,18,3)],
        'n_estimators':[i*100 for i in range(1,10,2)]
        }
        print('========================================================================')
        print('stage5:Start')
        result = self.tuning_stage5(param_test, best_max_depth, best_min_child_weight, best_gamma,
                               best_subsample, best_colsample_bytree, best_alpha, best_lambda)
        temp_result = result
        best_eta, best_n = result[0]['learning_rate'], result[0]['n_estimators'] 
        print('step1:==================================================================\n'+
              'in this step, the best result is %s: %.5f' % result)    
        print('Now we enter the next step.')
        params = result[0].items()
        param1 = dict()
        param1[params[0][0]] = params[0][1]
        param2 = dict()
        param2[params[1][0]] = params[1][1]
        if param1.keys()[0] == 'n_estimators':
            param_test = self.generate_params(param1,10,5)
            param_test.update(self.generate_params(param2,5,0.01))
        else:
            param_test = self.generate_params(param1,10,10)
            param_test.update(self.generate_params(param2,5,0.01))
        result = self.tuning_stage5(param_test, best_max_depth, best_min_child_weight, best_gamma,
                               best_subsample, best_colsample_bytree, best_alpha, best_lambda)
        if temp_score < result[1]:
            print('step2:==================================================================\n'+
                  'in this stage, the best result is %s: %.5f' % result)
            temp_score = result[1]
            print('========================================================================')
            print('stage5:End!')
            best_eta, best_n = result[0]['learning_rate'], result[0]['n_estimators'] 
        else:
            print('========================================================================')
            print('step2:==================================================================')
            print('This step doesn\'t perform well, so we choose last step\'s parameters.')
            print('in this stage, the best result is %s: %.5f' % temp_result)
            print('stage5:End!')
        #End#
        final_params = dict()
        final_params['max_depth'] = best_max_depth
        final_params['min_child_weight'] = best_min_child_weight
        final_params['gamma'] = best_gamma
        final_params['subsample'] = best_subsample
        final_params['colsample_bytree'] = best_colsample_bytree
        final_params['reg_alpha'] = best_alpha
        final_params['reg_lambda'] = best_lambda
        final_params['learning_rate'] = best_eta
        final_params['n_estimators'] = best_n
        print('Now, we finish the whole tuning stages, the best result is \n%s: %.5f' % 
              (final_params,temp_score))
        self.final_params = final_params
        self.best_score = temp_score
        self.best_estimator = XGBClassifier(learning_rate = best_eta, 
        n_estimators = best_n, gamma = best_gamma, subsample = best_subsample, 
        colsample_bytree = best_colsample_bytree, max_depth = best_max_depth,
        reg_alpha = best_alpha, reg_lambda = best_lambda, min_child_weight = 
        best_min_child_weight, objective = self.objective)
