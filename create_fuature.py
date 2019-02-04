# %%
# データインポート
import pandas as pd
import datetime as dt
from sklearn import preprocessing

feature = pd.read_csv("data/feature.csv", encoding="Shift-JIS")
race_info = pd.read_csv("data/race_info.csv", encoding="Shift-JIS")
race_result = pd.read_csv("data/race_result.csv", encoding="Shift-JIS")

# %%
# データ成形
# 過去4回の勝率を追加
feature["order_of_finish"] = race_result.order_of_finish
feature = feature[feature.order_of_finish.isin(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",
                                                "14", "15", "16"])]
feature.order_of_finish = feature.order_of_finish.apply(int)

func_top1 = lambda x: 1 if x == 1 else 0
func_top3 = lambda x: 1 if x <= 3 else 0

feature["Top1"] = feature.order_of_finish.apply(func_top1)
feature["Top3"] = feature.order_of_finish.apply(func_top3)

for id in feature.horse_id.unique():
    feature.loc[feature.horse_id == id, "avgWin4"] = feature[feature.horse_id == id].Top1.shift(1).rolling(4,
                                                                                                           min_periods=1).sum()
    feature.loc[feature.horse_id == id, "avgTop3_4"] = feature[feature.horse_id == id].Top3.shift(1).rolling(4,
                                                                                                             min_periods=1).sum()
    feature.loc[feature.horse_id == id, "preRaceWin"] = feature[feature.horse_id == id].Top1.shift(1)
    feature.loc[feature.horse_id == id, "preRaceTop3"] = feature[feature.horse_id == id].Top3.shift(1)

for id in feature.jockey_id.unique():
    feature.loc[feature.jockey_id == id, "jAvgWin4"] = feature[feature.jockey_id == id].Top1.shift(1).rolling(4,
                                                                                                              min_periods=1).sum()
    feature.loc[feature.jockey_id == id, "jAvgTop3_4"] = feature[feature.jockey_id == id].Top3.shift(1).rolling(4,
                                                                                                                min_periods=1).sum()

feature.to_csv("data/feature_update.csv", encoding="Shift-JIS", index=False)

# %%
# スピード係数を追加　ただし、その回の結果を追加してるだけ
feature_u = pd.read_csv("data/feature_update.csv", encoding="Shift-JIS")

func_timedelta = lambda x: dt.timedelta(minutes=int(x[:1]), seconds=int(x[-4:-2]), milliseconds=100 * int(x[-1:]))

race_result = race_result[
    race_result.order_of_finish.isin(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",
                                      "14", "15", "16"])]
race_result = race_result.reset_index()

feature_u["finishing_time"] = race_result.finishing_time.apply(func_timedelta)
func_to_int = lambda x: x.total_seconds()
feature_u["finishing_time"] = feature_u.finishing_time.apply(func_to_int)

# 距離ごとの平均時間を算出
a = feature_u[["distance", "surface", "finishing_time"]]
time_analyze = \
    a.groupby(["distance", "surface"]).agg({"finishing_time": ["count", "mean", "median", "std", "min", "max"]})[
        "finishing_time"]
# %%
# 過去の速度レートを追加

func_timedelta = lambda x: x.finishing_time - time_analyze.loc[x.distance, x.surface]["mean"]
feature_u["sp_rate"] = feature_u.apply(func_timedelta, axis=1)

for id in feature_u.horse_id.unique():
    feature_u.loc[feature_u.horse_id == id, "avgSprate4"] = feature_u[feature_u.horse_id == id].sp_rate.shift(
        1).rolling(4,
                   min_periods=1).mean()
    feature_u.loc[feature_u.horse_id == id, "sumSprate4"] = feature_u[feature_u.horse_id == id].sp_rate.shift(
        1).rolling(4,
                   min_periods=1).sum()
    feature_u.loc[feature_u.horse_id == id, "preSprate"] = feature_u[feature_u.horse_id == id].sp_rate.shift(1)
    feature_u.loc[feature_u.horse_id == id, "raceCnt"] = feature_u[feature_u.horse_id == id].age.shift(1).rolling(100,
                                                                                                                  min_periods=1).count()
    feature_u.loc[feature_u.horse_id == id, "winCnt"] = feature_u[feature_u.horse_id == id].Top1.shift(1).rolling(100,
                                                                                                                  min_periods=1).sum()

feature_u = feature_u[['race_id', 'order_of_finish', 'horse_number', 'grade', 'horse_id',
                       'jockey_id', 'trainer_id', 'age', 'dhweight',
                       'disRoc', 'distance', 'enterTimes', 'eps', 'hweight',
                       'jwinper', 'odds', 'owinper', 'sex', 'surface',
                       'twinper', 'weather', 'weight', 'winRun', 'jEps',
                       'preOOF', 'pre2OOF', 'month', 'ridingStrongJockey',
                       'runningStyle', 'preLateStart', 'preLastPhase', 'lateStartPer',
                       'course', 'placeCode', 'headCount', 'preHeadCount', 'surfaceChanged',
                       'gradeChanged', 'preMargin', 'femaleOnly', 'avgTop3_4', 'avgWin4',
                       'preRaceWin', 'preRaceTop3', 'jAvgTop3_4', 'jAvgWin4',
                       'avgSprate4', 'sumSprate4', 'preSprate', 'raceCnt', 'winCnt']]

feature_u.to_csv("data/feature_update.csv", encoding="Shift-JIS", index=False)
# %%
# 相対評価係数を追加
# 同一レースの中での評価の相対値（単勝で勝ちやすいかどうかの評価値）
# 今までのパラメータは絶対値しか入れていなかったが、レースという性質上、相対値の方が判断材料にしやすい気がする

feature_u = pd.read_csv("data/feature_update.csv", encoding="Shift-JIS")
relative = pd.DataFrame()

for race_id in feature_u.race_id.unique():
    tmp1 = pd.DataFrame(preprocessing.minmax_scale(feature_u[feature_u.race_id == race_id].avgSprate4.values))
    tmp2 = pd.DataFrame(preprocessing.minmax_scale(feature_u[feature_u.race_id == race_id].sumSprate4.values))
    tmp3 = pd.DataFrame(preprocessing.minmax_scale(feature_u[feature_u.race_id == race_id].jEps.values))
    tmp4 = pd.DataFrame(preprocessing.minmax_scale(feature_u[feature_u.race_id == race_id].eps.values))
    tmp5 = pd.DataFrame(preprocessing.minmax_scale(feature_u[feature_u.race_id == race_id].preSprate.values))
    tmp6 = pd.DataFrame(preprocessing.minmax_scale(feature_u[feature_u.race_id == race_id].winCnt.values))
    tmp = pd.concat([tmp1, tmp2, tmp3, tmp4, tmp5, tmp6], axis=1)
    relative = relative.append(tmp)

relative = relative.reset_index()
relative.columns = ["in", "avgSprate4_relative", "sumSprate4_relative", "jEps_relative", "eps_relative",
                    "preSprate_relative", "winCnt_relative"]

relative = relative[["avgSprate4_relative", "sumSprate4_relative", "jEps_relative", "eps_relative",
                    "preSprate_relative", "winCnt_relative"]]

a = pd.concat([feature_u, relative], axis=1)

a.to_csv("data/feature_update.csv", encoding="Shift-JIS", index=False)