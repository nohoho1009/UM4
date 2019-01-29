# %%
# データインポート
import pandas as pd
import datetime as dt

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
a = feature_u[["distance", "finishing_time"]]
time_analyze = a.groupby("distance").agg({"finishing_time": ["count", "mean", "median", "std", "min", "max"]})[
    "finishing_time"]
# %%
# 過去の速度レートを追加

func_timedelta = lambda x: x.finishing_time - time_analyze.loc[x.distance]["mean"]
feature_u["sp_rate"] = feature_u.apply(func_timedelta, axis=1)

for id in feature_u.horse_id.unique():
    feature_u.loc[feature_u.horse_id == id, "avgSprate4"] = feature_u[feature_u.horse_id == id].sp_rate.shift(1).rolling(4,
                                                                                                           min_periods=1).mean()
    feature_u.loc[feature_u.horse_id == id, "sumSprate4"] = feature_u[feature_u.horse_id == id].sp_rate.shift(1).rolling(4,
                                                                                                           min_periods=1).sum()
    feature_u.loc[feature_u.horse_id == id, "preSprate"] = feature_u[feature_u.horse_id == id].sp_rate.shift(1)
# %%
# 速度レートを整数に変更（グループ毎に集計したい）
feature_u["avgSprate4_int"] = feature_u.avgSprate4.fillna(0).apply(int)
feature_u["preSprate_int"] = feature_u.preSprate.fillna(0).apply(int)

# %%
# 過去4回の平均速度レート毎の回収率
feature_u["cnt"] = 1
buyTime = feature_u.groupby(["avgSprate4_int"]).cnt.count()

result_1st = feature_u[feature_u.order_of_finish == 1]
payback = result_1st.groupby(["avgSprate4_int"]).agg({"odds": "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

# %%
# 前回の速度レート毎の回収率
feature_u["cnt"] = 1
buyTime = feature_u.groupby(["preSprate_int"]).cnt.count()

result_1st = feature_u[feature_u.order_of_finish == 1]
payback = result_1st.groupby(["preSprate_int"]).agg({"odds": "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt