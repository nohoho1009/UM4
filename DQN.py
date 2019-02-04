from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import rl.callbacks
import matplotlib.pyplot as plt
import DQN_env
import numpy as np
import pandas as pd
import math
import datetime
from keras import models
import io_utility

#insi = ['race_id', 'horse_number', 'grade', 'odds',
#        'jockey_id', 'trainer_id', 'age', 'dhweight', 'disRoc', 'distance',
#        'enterTimes', 'eps', 'hweight', 'jwinper', 'owinper', 'sex',
#        'surface', 'twinper', 'weather', 'weight', 'winRun', 'jEps', 'preOOF',
#        'pre2OOF', 'month', 'ridingStrongJockey', 'runningStyle',
#        'preLateStart', 'preLastPhase', 'lateStartPer',
#        'headCount', 'preHeadCount', 'surfaceChanged', 'gradeChanged',
#        'preMargin', 'femaleOnly', 'avgTop3_4', 'avgWin4', 'preRaceWin',
#        'preRaceTop3', 'jAvgTop3_4', 'jAvgWin4', 'avgSprate4', 'sumSprate4',
#        'preSprate', 'raceCnt', 'winCnt']

insi = ['odds',
        'eps', 'jEps',
        'avgTop3_4', 'avgWin4', 'preRaceWin', 'sumSprate4',
        'preSprate']

# データ読み込みとクレンジング、特徴量の選択
train_data, train_label, test_data, test_label = io_utility.ReadData(insi, traningdata_rate=0.9)
#%%
# DQNの学習環境の設定
env = DQN_env.UM4_sim(train_data, train_label, 1)
nb_actions = env.action_space.n

# DQNのネットワーク定義
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# experience replay用のmemory
memory = SequentialMemory(limit=50000, window_length=1)
# 行動方策はオーソドックスなepsilon-greedy。ほかに、各行動のQ値によって確率を決定するBoltzmannQPolicyが利用可能
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
#dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
#               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#%%

# 学習開始
n = 100
history = dqn.fit(env, nb_steps=int((len(train_data) - 1) * n), visualize=False, verbose=2, nb_max_episode_steps=int(len(train_data) - 1))


#%%

class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.rewards = []
        self.actions = []
        self.info = pd.DataFrame()
        self.position = []
        self.money = 0

    def on_step_end(self, step, logs):
        self.money += logs['reward']
        self.rewards.append(self.money)
        self.actions.append(logs['action'])

# 学習に使用しなかったテストデータで評価
evaluation = DQN_env.UM4_sim(test_data, test_label, 1)
cb_ep = EpisodeLogger()
dqn.test(evaluation, nb_episodes=1000, visualize=False, callbacks=[cb_ep])

#%%
# テスト結果描画
info = cb_ep.info
log = pd.DataFrame()
log["actions"] = cb_ep.actions
log["rewards"] = cb_ep.rewards

log.plot(y=['rewards'], figsize=(16, 4), alpha=0.5)

plt.plot(log)
plt.xlabel("step_cnt")
plt.ylabel("pos")
