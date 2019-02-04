import gym
import gym.spaces
import numpy as np
import pandas as pd
import random

# 回収率を最大化することを想定した環境
class UM4_sim(gym.core.Env):

    def __init__(self, trainData, label, bonus):
        super().__init__()
        print("test!!")
        self.TRAIN_DATA = trainData
        self.LABEL = label
        self.BONUS = bonus

        self.action_space = gym.spaces.Discrete(2)  # 行動空間。Buy,Stay 想定しているのは対象を買うのか、買わないのか

        high = trainData.max().values   # 観測空間(state)の最大値
        low = trainData.min().values  # 観測空間(state)の最大値
        self.step_cnt = 0

        self.observation_space = gym.spaces.Box(low=low, high=high)  # 最小値と最大値設定

    # 各stepごとに呼ばれる
    # actionを受け取り、次のstateとreward、episodeが終了したかどうかを返すように実装
    def _step(self, action):
        data = self.TRAIN_DATA.iloc[self.step_cnt]
        result = self.LABEL.iloc[self.step_cnt]

        # 報酬処理
        if action == 1:
            if result == 1:
                reward = data.odds * self.BONUS
            else:
                reward = -1
        else:
            reward = 0

        # カウンタ更新
        self.step_cnt = self.step_cnt + 1
        done = False
        nextData = self.TRAIN_DATA.iloc[self.step_cnt]

        return nextData.values, reward, done, {}

    # 各episodeの開始時に呼ばれ、初期stateを返すように実装
    def _reset(self):
        self.step_cnt = 0
        nextData = self.TRAIN_DATA.iloc[self.step_cnt]

        return nextData.values