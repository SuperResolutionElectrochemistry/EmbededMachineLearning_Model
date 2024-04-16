# Title: Ensemble-machine-learning accelerated high-throughput screening of high-performance PtPd-based high-entropy alloy hydrogen evolution electrocatalysts
# Author: Xiangyi Shan, Yiyang Pan, Furong Cai, Min Zhou*
# Corresponding author. Email: mzhou1982@ciac.ac.cn (M. Z.)

#  This code is used to train the ensemble model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor


input = pd.read_excel('./data/your data')
data = input.values

x_train, x_test, y_train, y_test = train_test_split(data[:,:5], data[:,-2],test_size=0.1, random_state=152)

# Load RandomForest model
rf_model = RandomForestRegressor(random_state=68)
rf_model.fit(x_train, y_train)
y_trainpred_rf = rf_model.predict(x_train)
mse_train_rf = mean_squared_error(y_train, y_trainpred_rf)
mae_train_rf = mean_absolute_error(y_train, y_trainpred_rf)
r2_train_rf = r2_score(y_train, y_trainpred_rf)
y_pred_rf = rf_model.predict(x_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print('RF MSE: {:.4f}'.format(mse_rf))
print('RF MAE: {:.4f}'.format(mae_rf))
print('RF R2: {:.4f}'.format(r2_rf))

# Load GradientBoosting model
gbt_model = GradientBoostingRegressor(n_estimators=100, max_features='sqrt', learning_rate=0.383, random_state=0)
gbt_model.fit(x_train, y_train)
y_trainpred_gbt = gbt_model.predict(x_train)
mse_train_gbt = mean_squared_error(y_train, y_trainpred_rf)
mae_train_gbt = mean_absolute_error(y_train, y_trainpred_rf)
r2_train_gbt = r2_score(y_train, y_trainpred_rf)
y_pred_gbt = gbt_model.predict(x_test)
mse_gbt = mean_squared_error(y_test, y_pred_gbt)
mae_gbt = mean_absolute_error(y_test, y_pred_gbt)
r2_gbt = r2_score(y_test, y_pred_gbt)
print('GBT MSE: {:.4f}'.format(mse_gbt))
print('GBT MAE: {:.4f}'.format(mae_gbt))
print('GBT R2: {:.4f}'.format(r2_gbt))

# Load xgboost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.04, subsample=0.9)
xgb_model.fit(x_train, y_train)
y_trainpred_xgb = xgb_model.predict(x_train)
mse_train_xgb = mean_squared_error(y_train, y_trainpred_rf)
mae_train_xgb = mean_absolute_error(y_train, y_trainpred_rf)
r2_train_xgb = r2_score(y_train, y_trainpred_rf)
y_pred_xgb = xgb_model.predict(x_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print('XGB MSE: {:.4f}'.format(mse_xgb))
print('XGB MAE: {:.4f}'.format(mae_xgb))
print('XGB R2: {:.4f}'.format(r2_xgb))

# Define MLP model
mlp_model = MLPRegressor(hidden_layer_sizes=(91, 16), random_state=1, max_iter=3000, batch_size=32, verbose=True)
y_score_mean = [0] * 24
y_pred_mean = [0] * 24
y_trainpred_mean = [0] * 214
score_mean = 0
ensamble_num = 5
for i in range(ensamble_num):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.fit_transform(x_test)
    out = mlp_model.fit(x_train_norm, y_train)
    loss_train = mlp_model.loss_curve_
    y_pred = mlp_model.predict(x_test_norm)
    y_trainpred = mlp_model.predict(x_train_norm)
    score = mlp_model.score(x_test_norm, y_test)
    score_mean = score_mean + score
    for j in range(len(y_trainpred)):
        y_trainpred_mean[j] = y_trainpred_mean[j] + y_trainpred[j]
    for j in range(len(y_pred)):
        y_pred_mean[j] = y_pred_mean[j] + y_pred[j]
for k in range(len(y_trainpred_mean)):
    y_trainpred_mean[k] = y_trainpred_mean[k] / ensamble_num
for k in range(len(y_pred_mean)):
    y_pred_mean[k] = y_pred_mean[k] / ensamble_num
mse_train_mlp = mean_squared_error(y_train, y_trainpred_rf)
mae_train_mlp = mean_absolute_error(y_train, y_trainpred_rf)
r2_train_mlp = r2_score(y_train, y_trainpred_rf)
mse_mlp = mean_squared_error(y_test, y_pred_mean)
mae_mlp = mean_absolute_error(y_test, y_pred_mean)
r2_mlp = r2_score(y_test, y_pred_mean)
print('MLP MSE: {:.4f}'.format(mse_mlp))
print('MLP MAE: {:.4f}'.format(mae_mlp))
print('MLP R2: {:.4f}'.format(r2_mlp))


# VotingRegressor
voting_model = VotingRegressor(estimators=[('rf', rf_model),  ('gbt', gbt_model), ('xgb', xgb_model)])

voting_model.fit(x_train, y_train)

y_trainpred = voting_model.predict(x_train)
y_pred = voting_model.predict(x_test)

mse_voting = mean_squared_error(y_test, y_pred)
mae_voting = mean_absolute_error(y_test, y_pred)
r2_voting = r2_score(y_test, y_pred)

# save test results to plot
savepath = 'your savepath'
df.to_excel(savepath, index=False)