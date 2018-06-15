#----------------------------------------
# purpose: ランダムフォレストによる回帰のテストスクリプト　学習編
# author: hidetsugu ikeda
# memo: 読み込むデータは、1行目に列名があり、最終列に正解（数値）が入っていること。
# created: 2018-06-11
#----------------------------------------
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor as mc
from datetime import datetime, timezone, timedelta

# データの読み込み
data = pd.read_csv("train.csv", index_col='date', parse_dates=True)
print(data)
x = (data.iloc[:, :-1]).values # 最終列以外を取得
y = (data.iloc[:, -1:]).values # 最終列を取得
y = np.ravel(y) # transform 2次元 to 1次元 ぽいこと

# 学習
clf = mc()               # 学習器
clf.fit(x, y)
result = clf.score(x, y) # 学習データに対する、適合率

# 学習結果を保存
with open('entry.pickle', 'wb') as f:
	pickle.dump(clf, f)

# 1個だけテスト
valid_data = pd.read_csv("valid.csv", index_col="date", parse_dates=True)
v_x = (valid_data.iloc[:, :-1]).values
v_y = (valid_data.iloc[:, -1:]).values

for val in v_x:
    test = clf.predict([val])
    print( "predict is ")
    print(test*100)

# 額数データに対する適合率
print(result)
print(clf.feature_importances_)	# 各特徴量に対する寄与度を求める