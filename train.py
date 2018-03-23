# -*- coding:utf-8 -*-

# 加载模块
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import plot_importance  
from matplotlib import pyplot  


# 读取数据
train = pd.read_csv('X.csv') # 特征数据集，除了 “船测温度”
target = pd.read_csv('Y.csv') # 响应变量，“船测温度”

# 划分数据集，测试集数据比例 5%
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.05, random_state=1729)


y_test_value = [i for i in y_test['ShipT'].values]
# print(y_test_value)

X_test_value = [i for i in X_test['InfraredT'].values]
# print(X_test)

#模型参数设置
xgb_boost = xgb.XGBRegressor(max_depth=5, 
                        learning_rate=0.1, 
                        n_estimators=75, 
                        silent=True, 
                        objective='reg:linear', 
                        nthread=-1, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None)

	
# 拟合模型					
# xgb_boost.fit(X_train, y_train.values.ravel(), eval_metric='rmse', verbose = True)
xgb_boost.fit(X_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, y_test)], early_stopping_rounds=100)


# predict 预测
preds = xgb_boost.predict(X_test)
# 新建 submission.csv 文件，保存预测数据
ResultFile = pd.DataFrame()
ResultFile['ShipTPredictValue'] = preds # y_test_predict
ResultFile['ShipTTrueValue'] = y_test_value # y_test_value
ResultFile['InfraredTTrueValue'] = X_test_value
ResultFile.to_csv('submission.csv',index=False)


# feature importance 特征重要度
# 特征重要度画图
plot_importance(xgb_boost, importance_type = 'weight') # 其中importance_type = 'cover'，也可以等于'weight'以及'gain'
pyplot.show()
print(xgb_boost.feature_importances_)
# 新建 importances.csv 文件，保存特征重要度
importances = pd.DataFrame()
importances['feature'] = train.columns
importances['importances'] = xgb_boost.feature_importances_
importances.to_csv('importances.csv',index=False)