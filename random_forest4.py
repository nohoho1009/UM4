from sklearn.ensemble import RandomForestClassifier as classifier
from gensim import matutils
import io_utility
import pandas as pd

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
clf = classifier(random_state=777)
clf.fit(train_data.values, train_label)
# %%
ypred = clf.predict_proba(test_data.values)

# %%
# 答えと予想勝率を結合
tmp = pd.DataFrame(ypred)
result = pd.concat([test_data, test_label, tmp], axis=1)
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
