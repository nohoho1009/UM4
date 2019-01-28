#%%

import pandas as pd

feature = pd.read_csv("data/feature.csv", encoding="Shift-JIS")
race_info = pd.read_csv("data/race_info.csv", encoding="Shift-JIS")
race_result = pd.read_csv("data/race_result.csv", encoding="Shift-JIS")


#%%
#人気別回収金額
race_result["cnt"] = 1
buyTime = race_result.groupby("popularity").cnt.count()

result_1st = race_result[race_result.order_of_finish == "1"]
payback = result_1st.groupby("popularity").agg({"odds" : "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

#%%
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
#%%
#前4戦の勝率別回収金額
feature["cnt"] = 1
buyTime = feature.groupby("avgWin4").cnt.count()

result_1st = feature[feature.order_of_finish == 1]
payback = result_1st.groupby("avgWin4").agg({"odds" : "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

#%%
#前4戦のTop3率別回収金額
feature["cnt"] = 1
buyTime = feature.groupby("avgTop3_4").cnt.count()

result_1st = feature[feature.order_of_finish == 1]
payback = result_1st.groupby("avgTop3_4").agg({"odds" : "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

#%%
#前4戦のTop3率別回収金額
feature["cnt"] = 1
buyTime = feature.groupby("jAvgWin4").cnt.count()

result_1st = feature[feature.order_of_finish == 1]
payback = result_1st.groupby("jAvgWin4").agg({"odds" : "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

#%%
#前戦での勝利別回収金額
feature["cnt"] = 1
buyTime = feature.groupby("preRaceWin").cnt.count()

result_1st = feature[feature.order_of_finish == 1]
payback = result_1st.groupby("preRaceWin").agg({"odds" : "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

#%%
#前戦でのTop3別回収金額
feature["cnt"] = 1
buyTime = feature.groupby("preRaceTop3").cnt.count()

result_1st = feature[feature.order_of_finish == 1]
payback = result_1st.groupby("preRaceTop3").agg({"odds" : "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

#%%
#前戦でのTop3別回収金額
feature["cnt"] = 1
buyTime = feature.groupby(["preRaceTop3", "preRaceWin", "avgWin4", "avgTop3_4"]).cnt.count()

result_1st = feature[feature.order_of_finish == 1]
payback = result_1st.groupby(["preRaceTop3", "preRaceWin", "avgWin4", "avgTop3_4"]).agg({"odds" : "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt