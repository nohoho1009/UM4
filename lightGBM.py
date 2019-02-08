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


# データ読み込みとクレンジング、特徴量の選択
insi = ['horse_number', 'grade', 'odds',
        'jockey_id', 'trainer_id', 'age', 'dhweight', 'disRoc', 'distance',
        'enterTimes', 'eps', 'hweight', 'jwinper', 'owinper', 'sex',
        'twinper', 'weight', 'winRun', 'jEps',
        'surfaceChanged', 'gradeChanged',
        'femaleOnly', 'avgTop3_4', 'avgWin4', 'preRaceWin',
        'preRaceTop3', 'jAvgTop3_4', 'jAvgWin4', 'avgSprate4', 'sumSprate4',
        'preSprate', "avgSprate4_relative", "sumSprate4_relative", "jEps_relative", "eps_relative",
        "preSprate_relative", "winCnt_relative", "popularity"]

train_data, train_label, test_data, test_label = io_utility.ReadData_WithoutLowPopularity(insi, traningdata_rate=0.9, max_popularity=6)

# %%
# データセットの設定と学習時のハイパーパラメータ設定
dm = lgb.Dataset(train_data, label=train_label.values)

np.random.seed(1)

params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': {'multi_logloss'},
        'num_class': 2,
        'learning_rate': 0.2,
}

# 学習開始
lgbm = lgb.train(params, dm, num_boost_round=18)
# %%
# テストデータ（2016/4～2017/4）で試す
ypred = pd.DataFrame(lgbm.predict(test_data.values, num_iteration=lgbm.best_iteration))
# %%
result = pd.concat([test_data, ypred, test_label], axis=1)
result = result[["order_of_finish", "odds", 1]]
result = result.rename(columns={"order_of_finish": "ans", 1: "predict"})

# 予想勝率とオッズの積を作成
result["odds_predict"] = result.odds * result.predict

# 予想勝率×オッズが一定値以上かつ予想勝率が一定値以上のとき買うもの（1とする）としたアルゴリズム
func = lambda x: 1 if (x.predict > 0.3) & (x.odds_predict > 1.2) else 0
result["pre2"] = result.apply(func, axis=1)

# 購入回数
buyCnt = result.pre2.sum()

# 払い戻し額合計
ret = result[(result.pre2 ==1) & (result.ans==1)].odds.sum()

# 払い戻し率
returnRate = ret / buyCnt

# 的中率
hitRate = result[(result.pre2 ==1) & (result.ans==1)].odds.count() / buyCnt

print("買い目")
print(buyCnt)
print("的中率")
print(hitRate)
print("回収率")
print(returnRate)

# %%
ypred = pd.DataFrame(np.argmax(ypred.values,axis=1))

# 正確度
accu = accuracy_score(test_label, ypred)

# 混同行列
confusion_matrix(test_label, ypred)

# 適合率
precision_score(test_label, ypred)
# %%
# 各パラメータの重要度
lgb.plot_importance(lgbm, figsize=(12, 6))
plt.show()
# %%

cv_results = lgb.cv(params, dm, nfold=10)
cv_logloss = cv_results['multi_logloss-mean']
round_n = np.arange(len(cv_logloss))

plt.xlabel('round')
plt.ylabel('logloss')
plt.plot(round_n, cv_logloss)
plt.show()
