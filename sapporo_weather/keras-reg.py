import pandas as pd
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from datetime import datetime, timezone, timedelta
from sklearn.datasets import load_diabetes

# データの読み込み
# use diabetes sample data from sklearn
diabetes = load_diabetes()

# load them to X and Y
# X = diabetes.data
# Y = diabetes.target

# データの読み込み
train_data = pd.read_csv("train.csv", index_col='date', parse_dates=True)
X = (train_data.iloc[:, :-1]).values # 最終列以外を取得
Y = (train_data.iloc[:, -1:]).values # 最終列を取得
Y = np.ravel(Y) # transform 2次元 to 1次元 ぽいこと

valid_data = pd.read_csv("valid.csv", index_col='date', parse_dates=True)
X_valid = (train_data.iloc[:, :-1]).values # 最終列以外を取得
Y_valid = (train_data.iloc[:, -1:]).values # 最終列を取得
Y_valid = np.ravel(Y_valid) # transform 2次元 to 1次元 ぽいこと

# モデル作成
def reg_model():
    model = Sequential()
    model.add(Dense(10, input_dim=4, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    # コンパイル
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# use data split and fit to run the model
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
estimator = KerasRegressor(build_fn=reg_model, epochs=100, batch_size=10, verbose=0)
estimator.fit(X, Y)
y_pred = estimator.predict(X_valid)

# show its root mean square error
mse = mean_squared_error(Y_valid, y_pred)
print("KERAS REG RMSE : %.2f" % (mse ** 0.5))