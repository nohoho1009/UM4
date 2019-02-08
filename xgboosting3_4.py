# %%
import numpy as np
import xgboost as xgb
from sklearn import datasets
import matplotlib.pyplot as plt
import io_utility
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import pandas as pd

# 2値分類ではなく勝率予想（1着になるか否か）に変更
# また購入の判定をオッズ　×　勝率で期待値を上回った場合に購入
# xgboosting3で超高額馬券を購入しまくる時が多かったため、ある程度の勝率を持つものに購入を制限
# 人気下位を学習データ、テストデータから排除


xgb.__version__
# '0.4'
# データ読み込みとクレンジング、特徴量の選択

train_data, train_label, test_data, test_label = io_utility.ReadData_Race()

# %%
# データセットの設定と学習時のハイパーパラメータ設定
all = len(train_label)
weight = 1/train_label.sum()
dm = xgb.DMatrix(train_data.values, label=train_label.values, weight=weight.values)

np.random.seed(1)

params = {'objective': 'multi:softprob',
          'eval_metric': 'mlogloss',
          'eta': 0.3,
          'num_class': 7}

# 学習開始
bst = xgb.train(params, dm, num_boost_round=18)
# %%
# テストデータ（2016/4～2017/4）で試す
dm = xgb.DMatrix(test_data.values, label=test_label.values, weight=weight.values)
ypred = pd.DataFrame(bst.predict(dm))
# %%
# 各特徴量の重要度をみる
mapper = {'f{0}'.format(i): v for i, v in enumerate(train_data.columns)}
mapped = {mapper[k]: v for k, v in bst.get_fscore().items()}
mapped

xgb.plot_importance(mapped)

# %%
# 答えと予想勝率を結合
oddsdata = test_data["odds"].reset_index()[[1, 2, 3, 4, 5, 6]]
oddsdata.columns = ["1st_odds", "2nd_odds", "3rd_odds", "4th_odds", "5th_odds", "6th_odds"]
test_label = test_label.reset_index()[[1, 2, 3, 4, 5, 6, 7]]
test_label.columns = ["1st_ans", "2nd_ans", "3rd_ans", "4th_ans", "5th_ans", "6th_ans", "othre_ans"]
result = pd.concat([oddsdata, ypred, test_label], axis=1)
# %%

# 予想勝率とオッズの積を作成
result["1st_odds_predict"] = result["1st_odds"] * result[0]
result["2nd_odds_predict"] = result["2nd_odds"] * result[1]
result["3rd_odds_predict"] = result["3rd_odds"] * result[2]
result["4th_odds_predict"] = result["4th_odds"] * result[3]
result["5th_odds_predict"] = result["5th_odds"] * result[4]
result["6th_odds_predict"] = result["6th_odds"] * result[5]
# %%
# 予想勝率×オッズが一定値以上かつ予想勝率が一定値以上のとき買うもの（1とする）としたアルゴリズム
func = lambda x: 1 if (x.predict > 0.3) & (x.odds_predict > 1.2) else 0
result["pre2"] = result.apply(func, axis=1)

# 購入回数
buyCnt = result.pre2.sum()

# 払い戻し額合計
ret = result[(result.pre2 == 1) & (result.ans == 1)].odds.sum()

# 払い戻し率
returnRate = ret / buyCnt

# 的中率
hitRate = result[(result.pre2 == 1) & (result.ans == 1)].odds.count() / buyCnt

print("買い目")
print(buyCnt)
print("的中率")
print(hitRate)
print("回収率")
print(returnRate)
