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

# データの読み込み
data = pd.read_csv("train.csv", index_col='date', parse_dates=True)
print(data)
x = (data.iloc[:, :-1]).values # 最終列以外を取得
y = (data.iloc[:, -1:]).values # 最終列を取得
y = np.ravel(y) # transform 2次元 to 1次元 ぽいこと



# モデルの作成
def create_model():
    data_input = Input(shape=data.shape)
    model = Sequential()
    model.add(Dense(10, input_dim=data_input, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# モデル生成とビルド
reg_model = create_model()
estimator = KerasRegressor(build_fn=reg_model, epochs=100, batch_size=10, verbose=0)

# 学習
estimator.fit(x, y)

# 学習結果を保存
# with open('entry.pickle', 'wb') as f:
# 	pickle.dump(clf, f)

# テスト
valid_data = pd.read_csv("valid.csv", index_col="date", parse_dates=True)
v_x = (valid_data.iloc[:, :-1]).values
v_y = (valid_data.iloc[:, -1:]).values

y_pred = estimator.predict(v_x)

# 額数データに対する適合率
# print(result)
# print(clf.feature_importances_)	# 各特徴量に対する寄与度を求める