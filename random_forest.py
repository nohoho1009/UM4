# %%
#この方式の払い戻し率は88%、的中率42%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as classifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from gensim import matutils

feature = pd.read_csv("data/feature_update.csv", encoding="shift-jis")

feature = feature.fillna(0)

feature.weather = feature.weather.replace("曇", "0")
feature.weather = feature.weather.replace("晴", "1")
feature.weather = feature.weather.replace("小雨", "2")
feature.weather = feature.weather.replace("雨", "3")
feature.weather = feature.weather.replace("他", "4")
feature.weather = feature.weather.replace("雪", "5")

feature.sex = feature.sex.replace("牡", "0")
feature.sex = feature.sex.replace("牝", "1")
feature.sex = feature.sex.replace("セ", "2")

feature.surface = feature.surface.replace("芝", "0")
feature.surface = feature.surface.replace("ダ", "1")
# %%
result = feature.order_of_finish
func_1st = lambda x: 1 if x == 1 else 0
label = result.apply(func_1st)
length_feature = len(feature)

train_data = feature[0:int(length_feature*0.7)]

test_data = feature[int(length_feature*0.7 + 1):length_feature]
test_label = label[int(length_feature*0.7 + 1):length_feature]

# %%
feature_1st = train_data[train_data.order_of_finish == 1]
length_1st = len(feature_1st)
feature_other = train_data[train_data.order_of_finish != 1][0:length_1st]
feature = pd.concat([feature_1st, feature_other]).sample(frac=1)

train_label = train_data.order_of_finish.apply(func_1st)

# %%

insi = ['horse_number', 'grade', 'odds',
       'jockey_id', 'trainer_id', 'age', 'dhweight', 'disRoc', 'distance',
       'enterTimes', 'eps', 'hweight', 'jwinper',  'owinper', 'sex',
       'surface', 'twinper', 'weather', 'weight', 'winRun', 'jEps', 'preOOF',
       'pre2OOF', 'month', 'ridingStrongJockey', 'runningStyle',
       'preLateStart', 'preLastPhase', 'lateStartPer',
       'headCount', 'preHeadCount', 'surfaceChanged', 'gradeChanged',
       'preMargin', 'femaleOnly', 'avgTop3_4', 'avgWin4', 'preRaceWin',
       'preRaceTop3', 'jAvgTop3_4', 'jAvgWin4', 'avgSprate4', 'sumSprate4',
       'preSprate', 'raceCnt', 'winCnt']

train_data = train_data[insi]
test_data = test_data[insi]

# %%
clf = classifier(random_state=777)
clf.fit(train_data.values, train_label)
# %%

y_pred = clf.predict(test_data)

#正確度
accu = accuracy_score(test_label, y_pred)

#混同行列
confusion_matrix(test_label, y_pred)

#適合率
precision_score(test_label, y_pred)

# %%
df = feature

#特徴量の重要度
feature = clf.feature_importances_

#特徴量の重要度を上から順に出力する
f = pd.DataFrame({'number': range(0, len(feature)),
             'feature': feature[:]})
f2 = f.sort_values('feature',ascending=False)
f3 = f2.ix[:, 'number']

#特徴量の名前
label = df.columns[0:]

#特徴量の重要度順（降順）
indices = np.argsort(feature)[::-1]

for i in range(len(feature)):
    print(str(i + 1) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]]))

plt.title('Feature Importance')
plt.bar(range(len(feature)),feature[indices], color='lightblue', align='center')
plt.xticks(range(len(feature)), label[indices], rotation=90)
plt.xlim([-1, len(feature)])
plt.tight_layout()
plt.show()