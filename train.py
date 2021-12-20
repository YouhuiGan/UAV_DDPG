#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-

import os
import numpy as np
from numpy import *
import parl
from parl.utils import logger

from agent import Agent
from model import Model
from algorithm import DDPG  # from parl.algorithms import DDPG
from env import Environment
from replay_memory import ReplayMemory
from visualdl import LogWriter
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate
GAMMA = 0.9  # reward 的衰减因子
TAU = 0.001  # 软更新的系数
MEMORY_SIZE = 2000000  # 经验池大小
MEMORY_WARMUP_SIZE = 20000  # 预存一部分经验之后再开始训练
BATCH_SIZE = 256
REWARD_SCALE = 0.1  # reward 缩放系数

NOISE = 0.05  # 动作噪声方差
user_num = 10  # 用户数

os.environ["CUDA_VISIBLE_DEVICES"] = ""


# 训练一个episode
def run_episode(agent, env, rpm):
    obs, obs_normalization = env.reset()
    total_loss = 0
    total_reward = 0
    steps = 0
    uav_path_x = []
    uav_path_y = []
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs_normalization, axis=0)
        action = agent.predict(batch_obs.astype('float32'))

        # 增加探索扰动, 输出限制在 [0, 1] 范围内
        action = np.clip(np.random.normal(action, NOISE), 0, 1)

        next_obs, next_obs_normalization, reward, si_best, done, x, y = env.step(action, obs)
        uav_path_x.append(x)
        uav_path_y.append(y)

        obs_normalization = np.array(obs_normalization).reshape(-1)
        next_obs_normalization = np.array(next_obs_normalization).reshape(-1)

        action = [action]
        # 方便存入replaymemory
        rpm.append((obs_normalization, action, REWARD_SCALE * reward, next_obs_normalization, done))

        # 开始训练
        if len(rpm) > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

            total_loss += train_loss

        # 给出边界的值一个大的-reward
        # uav_location_x = next_obs[0][3]
        # uav_location_y = next_obs[0][4]
        # if uav_location_x < 0 or uav_location_x > 100 or uav_location_y < 0 or uav_location_y > 100:
        #     reward = -100
        # elif uav_location_x > 100 and uav_location_y > 100:
        #     reward = 10

        obs = next_obs
        obs_normalization = next_obs_normalization

        total_reward += reward

        if done:
            total_loss = total_loss / steps
            break

    # 记录每个episode跑完需要多少个step(每走一步reward+1，也就是少走多少步到终点reward就减多少)
    step_sum = steps
    total_reward = total_reward + step_sum

    return total_reward, total_loss, si_best, step_sum, uav_path_x, uav_path_y, obs


# # 评估 agent, 跑 5 个episode，总reward求平均
# def evaluate(env, agent, render=False):
#     eval_reward = []
#     for i in range(5):
#         obs = env.reset()
#         total_reward = 0
#         steps = 0
#         while True:
#             batch_obs = np.expand_dims(obs, axis=0)
#             action = agent.predict(batch_obs.astype('float32'))
#             action = np.clip(action, -1.0, 1.0)
#
#             steps += 1
#             next_obs, reward, done, info = env.step(action)
#
#             obs = next_obs
#             total_reward += reward
#
#             if render:
#                 env.render()
#             if done or steps >= 200:
#                 break
#         eval_reward.append(total_reward)
#     return np.mean(eval_reward)


def main():
    env = Environment()

    obs_dim = user_num * 5  # 状态空间(10, 5)
    act_dim = 1  # 动作空间(0, 2/pi)

    # 使用PARL框架创建agent
    model = Model(act_dim)
    algorithm = DDPG(
        model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, obs_dim, act_dim)

    # 创建经验池
    i = 0
    rpm = ReplayMemory(MEMORY_SIZE)
    # 往经验池中预存数据
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(agent, env, rpm)
        i = i + 1
        logger.info('经验池已存入第{}条数据'.format(i))

    # 一个episode里跑多少step
    episode = 0
    reward = []
    loss = []
    max_episode = 6000  # 训练的总episode数
    step_num = 10  # 每走step_num个episode取平均值

    while episode < max_episode:
        # train part
        average_reward = []
        average_loss = []
        uav_location = []
        total_step = []
        for i in range(step_num):
            uav_trajectory_x = []
            uav_trajectory_y = []
            user_location_x = []
            user_location_y = []
            user_data = []
            total_reward, total_loss, si_best, step_sum, uav_path_x, uav_path_y, obs = run_episode(agent, env, rpm)
            # 平均reward
            average_reward.append(total_reward)
            # 平均loss
            average_loss.append(total_loss)
            # 无人机总步数
            total_step.append(step_sum)
            # 无人机轨迹
            uav_trajectory_x = uav_path_x
            uav_trajectory_y = uav_path_y

            for j in range(user_num):
                # 用户位置
                user_location_x_ = obs[j][0]
                user_location_x.append(user_location_x_)
                user_location_y_ = obs[j][1]
                user_location_y.append(user_location_y_)
                # # 用户数据量 / 10**5 拟合无人机的高度
                # user_data_ = obs[j][2] / (2 * 10 ** 5)
                # user_data.append(user_data_)

            # 无人机的最终停留位置
            uav_location.append(obs[0, 3])
            uav_location.append(obs[0, 4])

            if episode == max_episode - 1:
                # 用户的最终以及最优卸载决策
                user_location_x_local = []
                user_location_x_load = []
                user_location_y_local = []
                user_location_y_load = []
                # user_data_local = []
                # user_data_load = []

                fig = plt.figure()
                ax = plt.axes(projection='3d')
                for j in range(user_num):
                    if si_best[j] == 0:
                        user_location_x_local.append(user_location_x[j])
                        user_location_y_local.append(user_location_y[j])
                        # user_data_local.append(user_data[j])
                        # ax.scatter3D(user_location_x_local, user_location_y_local, user_data_local, s=20., c='b')
                        ax.scatter3D(user_location_x_local, user_location_y_local, 0, s=20., c='b')
                    elif si_best[j] == 1:
                        user_location_x_load.append(user_location_x[j])
                        user_location_y_load.append(user_location_y[j])
                        # user_data_load.append(user_data[j])
                        # ax.scatter3D(user_location_x_load, user_location_y_load, user_data_load, s=20., c='r')
                        ax.scatter3D(user_location_x_load, user_location_y_load, 0, s=20., c='r')

                ax.plot3D(uav_trajectory_x, uav_trajectory_y, 100, 'green')
                fig.savefig('./figure/user-uav figure %i' % i)
                plt.show(block=False)
                plt.pause(1)
                plt.close()

        # 走step_num个episode后的总平均reward
        average_reward = float(mean(average_reward))
        reward.append(average_reward)
        average_loss = float(mean(average_loss))
        loss.append(average_loss)

        episode += 1
        logger.info(
            'episode:{}   Train reward:{}   Train loss:{}  uav location:{}  total step:{}'.format(
                episode, average_reward, average_loss, uav_location, total_step))

        # # test part
        # eval_reward = evaluate(env, agent, render=True)  # render=True 查看显示效果
        # logger.info('episode:{}    e_greed:{}   Test reward:{}'.format(
        #     episode, agent.e_greed, eval_reward))

    # plt.title('UAV Trajectory')
    # plt.plot(user_location_x, user_location_y, 'ro')
    # plt.plot(uav_trajectory_x, uav_trajectory_y, color='green')
    # plt.xlabel('x location')
    # plt.ylabel('y location')
    # plt.show()

    # 初始化一个记录器
    # "b不同的书记只能写到不同的文件下面"
    # "不同的小文件名会有不同的颜色曲线"
    # "命令行运行时，运行大文件"
    with LogWriter(logdir="./log/reward") as writer:
        for step in range(max_episode - 1):
            # 向记录器添加一个tag为`reward`的数据
            writer.add_scalar(tag="reward", step=step, value=reward[step])

    with LogWriter(logdir="./log/loss") as writer:
        for step in range(max_episode - 1):
            # 向记录器添加一个tag为`loss`的数据
            writer.add_scalar(tag="loss", step=step, value=loss[step])

    # 训练结束，保存模型
    save_path = './dqn_model.ckpt'
    agent.save(save_path)


if __name__ == '__main__':
    main()
