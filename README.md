## 各ファイル説明
- create_fuature.py
    - 学習用の特徴量作成プログラム。ほかのプログラム動かす前に一回これを動かす。
- io_utility.py
    - テスト用データ読み込み関数とか、なんか便利そうなのまとめたい
- main.py
    - テストプログラム走らせるときに使ってる
- random_forest1～3.py
    - ランダムフォレストでとりあえず性能の基準作るよね
    - 単勝かどうかの二値分類で作った。最大性能で的中率4割　回収率9割くらい。
    - 試してないけどランダムフォレストで勝率予想してもよかったかも。
- xgboosting.py(1,2,4)
    - 勾配ブースティング木とかいうランダムフォレストの亜種。強いらしい。
    - 1,2,4は単純な二値分類。性能はランダムフォレストと同じくらい
- xgboosting3.py
    - xgboostingで勝率予想して、オッズとのかけ合わせて期待値を超えたら買うっていうアルゴリズム
    - 性能は良いときと悪いときで差が激しい。おそらく大穴狙いが酷い。回収率は良いと260%　悪いと50%切る。
- xgboosting3_2.py
    - 基本はxgboosting3.pyと同じ。大穴狙いを抑制するために勝率が高いかつ期待値が高い場合だけ買うように変更。
    - パラメータによるけど200～300%くらいの回収率、的中率は50%。一年間当たりの買い目が40件程度しかないのが問題点か。
    - 性能的には今のところ一番強い。
- DQN.py
    - DQNで作った。性能うんこ
- DQN_env.py
    - DQNを学習させる環境。うんこー！
- notebooks/note1.ipynb
    - 10年分のデータ傾向とか
    
## 必要な環境とか
- keras
- keras RL
- scikit-learn
- xgboosting

## 参考文献
- http://stockedge.hatenablog.com/entry/2016/01/03/103428
    - 使ってる特徴量はここ参考。有料会員限定のスピード指数なんかはタイム差で代用。
- https://github.com/stockedge/netkeiba-scraper
    - 元データの生成はここのプログラムだより。新しいPCで上記プログラムが動かないので2017年4月のデータが最新。