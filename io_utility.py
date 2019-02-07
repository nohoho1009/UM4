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

def ReadData2(insi, traningdata_rate=0.9, max_popularity=6):
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