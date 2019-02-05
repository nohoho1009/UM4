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

xgb.__version__
# '0.4'
# データ読み込みとクレンジング、特徴量の選択
insi = ['horse_number', 'grade', 'odds',
        'jockey_id', 'trainer_id', 'age', 'dhweight', 'disRoc', 'distance',
        'enterTimes', 'eps', 'hweight', 'jwinper', 'owinper', 'sex',
        'surface', 'twinper', 'weather', 'weight', 'winRun', 'jEps', 'preOOF',
        'pre2OOF', 'month', 'ridingStrongJockey', 'runningStyle',
        'preLateStart', 'preLastPhase', 'lateStartPer',
        'headCount', 'preHeadCount', 'surfaceChanged', 'gradeChanged',
        'preMargin', 'femaleOnly', 'avgTop3_4', 'avgWin4', 'preRaceWin',
        'preRaceTop3', 'jAvgTop3_4', 'jAvgWin4', 'avgSprate4', 'sumSprate4',
        'preSprate', 'winCnt']

train_data, train_label, test_data, test_label = io_utility.ReadData(insi, traningdata_rate=0.9)

# %%
dm = xgb.DMatrix(train_data.values, label= train_label)

np.random.seed(1)

params={'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'eta': 0.3,
        'num_class': 2}

bst = xgb.train(params, dm, num_boost_round=18)
# %%
dm = xgb.DMatrix(test_data.values, label= test_label)
ypred = pd.DataFrame(bst.predict(dm))

# %%
result = pd.concat([test_data, ypred,test_label], axis=1)
result = result[["order_of_finish","odds",1]]
result = result.rename(columns = {"order_of_finish":"ans", 1: "predict"})
result["odds_predict"] = result.odds * result.predict

func = lambda x: 1 if x > 1 else 0

result["pre2"] = result.odds_predict.apply(func)

buyCnt = result.pre2.sum()

ret = result[(result.pre2 ==1) & (result.ans==1)].odds.sum()

ret / buyCnt