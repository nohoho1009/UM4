# %%
import numpy as np
import lightgbm as lgb
from sklearn import datasets
import matplotlib.pyplot as plt
import io_utility
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import pandas as pd

# 2値分類ではなく勝率予想（1着になるか否か）に変更
# また購入の判定をオッズ　×　勝率で期待値を上回った場合に購入
# 人気下位を学習データ、テストデータから排除

# '0.4'
# データ読み込みとクレンジング、特徴量の選択

train_data, train_label, test_data, test_label = io_utility.ReadData_Race()

# %%
# データセットの設定と学習時のハイパーパラメータ設定
dm = lgb.Dataset(train_data.values, label=train_label.values)

np.random.seed(1)

params = {
    'objective': 'multiclass',
    'num_class': 7,
    'metric': 'multi_logloss'
}


# 学習開始
lgbm = lgb.train(params, dm, num_boost_round=18)
# %%
# テストデータ（2016/4～2017/4）で試す
ypred = pd.DataFrame(lgbm.predict(test_data.values, num_iteration=lgbm.best_iteration))

ypred["ans"] = test_label.reset_index()[0]

ypred["0pre_odss"] = ypred[0] * test_data.reset_index().odds[1]
ypred["1pre_odss"] = ypred[1] * test_data.reset_index().odds[2]
ypred["2pre_odss"] = ypred[2] * test_data.reset_index().odds[3]
ypred["3pre_odss"] = ypred[3] * test_data.reset_index().odds[4]
ypred["4pre_odss"] = ypred[4] * test_data.reset_index().odds[5]
ypred["5pre_odss"] = ypred[5] * test_data.reset_index().odds[6]