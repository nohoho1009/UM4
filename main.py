# %%

import pandas as pd

feature = pd.read_csv("data/feature.csv", encoding="Shift-JIS")
feature_u = pd.read_csv("data/feature_update.csv", encoding="Shift-JIS")
race_info = pd.read_csv("data/race_info.csv", encoding="Shift-JIS")
race_result = pd.read_csv("data/race_result.csv", encoding="Shift-JIS")

# %%
# 人気別回収金額
race_result["cnt"] = 1
buyTime = race_result.groupby("popularity").cnt.count()

result_1st = race_result[race_result.order_of_finish == "1"]
payback = result_1st.groupby("popularity").agg({"odds": "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

# %%
# 前4戦の勝率別回収金額
feature_u["cnt"] = 1
buyTime = feature_u.groupby("avgWin4").cnt.count()

result_1st = feature_u[feature_u.order_of_finish == 1]
payback = result_1st.groupby("avgWin4").agg({"odds": "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

# %%
# 前4戦のTop3率別回収金額
feature_u["cnt"] = 1
buyTime = feature_u.groupby("avgTop3_4").cnt.count()

result_1st = feature_u[feature_u.order_of_finish == 1]
payback = result_1st.groupby("avgTop3_4").agg({"odds": "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

# %%
# 前4戦のTop3率別回収金額
feature_u["cnt"] = 1
buyTime = feature_u.groupby("jAvgWin4").cnt.count()

result_1st = feature_u[feature_u.order_of_finish == 1]
payback = result_1st.groupby("jAvgWin4").agg({"odds": "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

# %%
# 前戦での勝利別回収金額
feature_u["cnt"] = 1
buyTime = feature_u.groupby("preRaceWin").cnt.count()

result_1st = feature_u[feature_u.order_of_finish == 1]
payback = result_1st.groupby("preRaceWin").agg({"odds": "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

# %%
# 前戦でのTop3別回収金額
feature_u["cnt"] = 1
buyTime = feature_u.groupby("preRaceTop3").cnt.count()

result_1st = feature_u[feature_u.order_of_finish == 1]
payback = result_1st.groupby("preRaceTop3").agg({"odds": "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

# %%
# 前戦でのTop3別回収金額
feature_u["cnt"] = 1
feature_u.preRaceTop3 = feature_u.preRaceTop3.fillna(-1)
feature_u.preRaceWin = feature_u.preRaceWin.fillna(-1)
feature_u.avgWin4 = feature_u.avgWin4.fillna(-1)
feature_u.avgTop3_4 = feature_u.avgTop3_4.fillna(-1)

buyTime = feature_u.groupby(["preRaceTop3", "preRaceWin", "avgWin4", "avgTop3_4"]).cnt.count()

result_1st = feature_u[feature_u.order_of_finish == 1]
payback = result_1st.groupby(["preRaceTop3", "preRaceWin", "avgWin4", "avgTop3_4"]).agg({"odds": "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.odds / payback.buyCnt

# %%
# 前戦でのTop3,Top的中率
feature_u["cnt"] = 1
feature_u.preRaceTop3 = feature_u.preRaceTop3.fillna(-1)
feature_u.preRaceWin = feature_u.preRaceWin.fillna(-1)
feature_u.avgWin4 = feature_u.avgWin4.fillna(-1)
feature_u.avgTop3_4 = feature_u.avgTop3_4.fillna(-1)

buyTime = feature_u.groupby(["preRaceTop3", "preRaceWin", "avgWin4", "avgTop3_4"]).cnt.count()

result_1st = feature_u[feature_u.order_of_finish == 1]
payback = result_1st.groupby(["preRaceTop3", "preRaceWin", "avgWin4", "avgTop3_4"]).agg({"cnt": "sum"})
payback["buyCnt"] = buyTime
payback["returnRate"] = payback.cnt / payback.buyCnt
