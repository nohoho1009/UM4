#%%

import pandas as pd

feature = pd.read_csv("data/feature.csv", encoding="Shift-JIS")
race_info = pd.read_csv("data/race_info.csv", encoding="Shift-JIS")
race_result = pd.read_csv("data/race_result.csv", encoding="Shift-JIS")


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

feature.to_csv("data/feature_update.csv", encoding="Shift-JIS", index=False)