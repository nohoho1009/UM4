import pandas as pd


def ReadData(insi, traningdata_rate=0.9):
    """
    データ読み込みとクレンジングを行い、トレーニングデータとテストデータを分離して返す。

    Parameters
    ----------
    insi : list[string]
        読みこんだデータセットの中で使用する特徴量の配列。
    traningdata_rate : double
        トレーニングに使う割合。0.9なら9割をトレーニングデータとして返し、残り1割をテストデータとして返す。

    Returns
    ----------
    train_data : DataFrame
        トレーニングに使うデータフレーム。
    train_label : Series
        train_dataの回答。
    test_data : DtaFrame
        テストに使うデータフレーム。
    test_label : Series
        test_dataの回答。
    """

    # データ読み込みとクレンジング
    feature = pd.read_csv("data/feature_update.csv", encoding="shift-jis")
    feature = feature.fillna(0)

    feature.weather = feature.weather.replace("曇", 0)
    feature.weather = feature.weather.replace("晴", 1)
    feature.weather = feature.weather.replace("小雨", 2)
    feature.weather = feature.weather.replace("雨", 3)
    feature.weather = feature.weather.replace("他", 4)
    feature.weather = feature.weather.replace("雪", 5)

    feature.sex = feature.sex.replace("牡", 0)
    feature.sex = feature.sex.replace("牝", 1)
    feature.sex = feature.sex.replace("セ", 2)

    feature.surface = feature.surface.replace("芝", 0)
    feature.surface = feature.surface.replace("ダ", 1)

    # テストデータとトレーニングデータを分離
    result = feature.order_of_finish
    func_1st = lambda x: 1 if x == 1 else 0
    label = result.apply(func_1st)
    length_feature = len(feature)

    train_data = feature[0:int(length_feature * traningdata_rate)].reset_index()
    train_label = label[0:int(length_feature * traningdata_rate)].reset_index().order_of_finish

    test_data = feature[int(length_feature * traningdata_rate + 1):length_feature].reset_index()
    test_label = label[int(length_feature * traningdata_rate + 1):length_feature].reset_index().order_of_finish

    train_data = train_data[insi]
    test_data = test_data[insi]

    return train_data, train_label, test_data, test_label

def ReadData_WithoutLowPopularity(insi, traningdata_rate=0.9, max_popularity=6):
    """
    データ読み込みとクレンジングを行い、トレーニングデータとテストデータを分離して返す。
    データセットに使う人気の上限を設け、これ以上のデータはトレーニング、テストから抜く仕様。
    人気が低い場合、払い戻し率も悪いため無視しても問題ないという思想から作成。

    Parameters
    ----------
    insi : list[string]
        読みこんだデータセットの中で使用する特徴量の配列。
    traningdata_rate : double
        トレーニングに使う割合。0.9なら9割をトレーニングデータとして返し、残り1割をテストデータとして返す。
    max_popularity : int
        データセットに使う人気の上限。これ以上のデータはトレーニング、テストから抜く。

    Returns
    ----------
    train_data : DataFrame
        トレーニングに使うデータフレーム。
    train_label : Series
        train_dataの回答。
    test_data : DtaFrame
        テストに使うデータフレーム。
    test_label : Series
        test_dataの回答。
    """

    # データ読み込みとクレンジング
    feature = pd.read_csv("data/feature_update.csv", encoding="shift-jis")
    feature = feature[feature.popularity <= max_popularity]
    feature = feature.fillna(0)

    feature.weather = feature.weather.replace("曇", 0)
    feature.weather = feature.weather.replace("晴", 1)
    feature.weather = feature.weather.replace("小雨", 2)
    feature.weather = feature.weather.replace("雨", 3)
    feature.weather = feature.weather.replace("他", 4)
    feature.weather = feature.weather.replace("雪", 5)

    feature.sex = feature.sex.replace("牡", 0)
    feature.sex = feature.sex.replace("牝", 1)
    feature.sex = feature.sex.replace("セ", 2)

    feature.surface = feature.surface.replace("芝", 0)
    feature.surface = feature.surface.replace("ダ", 1)

    # テストデータとトレーニングデータを分離
    result = feature.order_of_finish
    func_1st = lambda x: 1 if x == 1 else 0
    label = result.apply(func_1st)
    length_feature = len(feature)

    train_data = feature[0:int(length_feature * traningdata_rate)].reset_index()
    train_label = label[0:int(length_feature * traningdata_rate)].reset_index().order_of_finish

    test_data = feature[int(length_feature * traningdata_rate + 1):length_feature].reset_index()
    test_label = label[int(length_feature * traningdata_rate + 1):length_feature].reset_index().order_of_finish

    train_data = train_data[insi]
    test_data = test_data[insi]

    return train_data, train_label, test_data, test_label


def ReadData_Race(traningdata_rate=0.9):
    insi = ["race_id", 'horse_number', 'odds', 'age', 'dhweight', 'disRoc',
            'enterTimes', 'eps', 'hweight', 'jwinper', 'owinper', 'sex',
            'twinper', 'weight', 'jEps', 'preOOF',
            'pre2OOF', 'ridingStrongJockey', 'runningStyle',
            'surfaceChanged', 'gradeChanged',
            'avgTop3_4', 'avgWin4', 'preRaceWin',
            'preRaceTop3', 'jAvgTop3_4', 'jAvgWin4', 'avgSprate4', 'sumSprate4',
            'preSprate', 'winCnt', "avgSprate4_relative", "sumSprate4_relative", "jEps_relative", "eps_relative",
            "preSprate_relative", "winCnt_relative", "headCount", "popularity", "order_of_finish"]

    insi2 = ['horse_number', 'odds', 'age', 'dhweight', 'disRoc',
            'enterTimes', 'eps', 'hweight', 'jwinper', 'owinper', 'sex',
            'twinper', 'weight', 'jEps', 'preOOF',
            'pre2OOF', 'ridingStrongJockey', 'runningStyle',
            'surfaceChanged', 'gradeChanged',
            'avgTop3_4', 'avgWin4', 'preRaceWin',
            'preRaceTop3', 'jAvgTop3_4', 'jAvgWin4', 'avgSprate4', 'sumSprate4',
            'preSprate', 'winCnt', "avgSprate4_relative", "sumSprate4_relative", "jEps_relative", "eps_relative",
            "preSprate_relative", "winCnt_relative", "headCount"]

    max_popularity = 6

    # データ読み込みとクレンジング
    feature = pd.read_csv("data/feature_update.csv", encoding="shift-jis")
    feature = feature[feature.popularity <= max_popularity]
    feature = feature[insi]
    feature = feature.fillna(0)

    feature.sex = feature.sex.replace("牡", 0)
    feature.sex = feature.sex.replace("牝", 1)
    feature.sex = feature.sex.replace("セ", 2)

    result = feature.pivot_table(values="order_of_finish", index="race_id", columns="popularity")
    feature = feature.pivot_table(values=insi2, index="race_id", columns="popularity")


    # テストデータとトレーニングデータを分離
    func_1st = lambda x: 1 if x == 1 else 0
    label = result.applymap(func_1st)

    func_7th = lambda x: 1 if x.sum() == 0 else 0
    label[7] = label.apply(func_7th, axis=1)

    length_feature = len(feature)

    train_data = feature[0:int(length_feature * traningdata_rate)]
    train_label = label[0:int(length_feature * traningdata_rate)]

    test_data = feature[int(length_feature * traningdata_rate + 1):length_feature]
    test_label = label[int(length_feature * traningdata_rate + 1):length_feature]

    return train_data, train_label, test_data, test_label