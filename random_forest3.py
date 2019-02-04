# %%
# 相対パラメータ追加
# 的中率40.9% 払い戻し率88%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as classifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from gensim import matutils
import io_utility

insi = ['horse_number', 'grade', 'odds',
        'jockey_id', 'trainer_id', 'age', 'dhweight', 'disRoc', 'distance',
        'enterTimes', 'eps', 'hweight', 'jwinper', 'owinper', 'sex',
        'surface', 'twinper', 'weather', 'weight', 'winRun', 'jEps', 'preOOF',
        'pre2OOF', 'month', 'ridingStrongJockey', 'runningStyle',
        'preLateStart', 'preLastPhase', 'lateStartPer',
        'headCount', 'preHeadCount', 'surfaceChanged', 'gradeChanged',
        'preMargin', 'femaleOnly', 'avgTop3_4', 'avgWin4', 'preRaceWin',
        'preRaceTop3', 'jAvgTop3_4', 'jAvgWin4', 'avgSprate4', 'sumSprate4',
        'preSprate', 'winCnt', "avgSprate4_relative", "sumSprate4_relative", "jEps_relative", "eps_relative",
        "preSprate_relative", "winCnt_relative"]

# データ読み込みとクレンジング、特徴量の選択
train_data, train_label, test_data, test_label = io_utility.ReadData(insi, traningdata_rate=0.9)
# %%
# 学習開始
clf = classifier(random_state=777)
clf.fit(train_data.values, train_label)
# %%
# テストデータでの結果

y_pred = clf.predict(test_data)

# 正確度
accu = accuracy_score(test_label, y_pred)

# 混同行列
confusion_matrix(test_label, y_pred)

# 適合率
precision_score(test_label, y_pred)

# %%
# 特徴量の重要度
result = pd.DataFrame()
result["name"] = insi
result["importance"] = clf.feature_importances_
result = result.sort_values("importance")
result = result.set_index("name")

result.plot.bar()

# %%
# 払い戻し率
insi = ['horse_number', 'grade', 'odds',
        'jockey_id', 'trainer_id', 'age', 'dhweight', 'disRoc', 'distance',
        'enterTimes', 'eps', 'hweight', 'jwinper', 'owinper', 'sex',
        'surface', 'twinper', 'weather', 'weight', 'winRun', 'jEps', 'preOOF',
        'pre2OOF', 'month', 'ridingStrongJockey', 'runningStyle',
        'preLateStart', 'preLastPhase', 'lateStartPer',
        'headCount', 'preHeadCount', 'surfaceChanged', 'gradeChanged',
        'preMargin', 'femaleOnly', 'avgTop3_4', 'avgWin4', 'preRaceWin',
        'preRaceTop3', 'jAvgTop3_4', 'jAvgWin4', 'avgSprate4', 'sumSprate4',
        'preSprate', 'raceCnt', 'winCnt']

train_data, train_label, test_data, test_label = io_utility.ReadData(insi)

a = test_data.reset_index()
a["pred"] = pd.DataFrame(y_pred)
a["ans"] = test_label.reset_index().order_of_finish

buyCnt = a.pred.sum()
ret = a[(a.pred==1) & (a.ans==1)].odds.sum()

returnRate = ret / buyCnt
meanOdds = a[(a.pred==1) & (a.ans==1)].odds.mean()