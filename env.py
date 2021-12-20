import math
import random
import numpy as np
from PDPSO import pdpso
from sklearn import preprocessing


class Environment(object):

    def __init__(self):
        self.user_num = 10              # 用户数
        self.beta_0 = 10 ** -5          # 1m参考距离的信道增益
        self.H = 100                    # 无人机的飞行高度
        self.B0 = 10 ** 7               # 分配给用户的带宽
        self.derta = 10 ** -5           # 无人机处的噪声功率
        self.P_user = 0.5               # 用户的发射功率
        self.C_user = 800               # 用户处理1bit数据需要的CPU计算周期数
        self.C_uav = 1000               # 无人机处理1bit数据需要的CPU计算周期数
        self.f_user = 1 * (10 ** 9)     # 本机计算资源
        self.f_uav = 3 * (10 ** 9)      # 分配给无人机的计算资源
        self.K_user = 10 ** -27         # 用户CPU的电容系数
        self.K_uav = 10 ** -28          # 无人机CPU的电容系数
        self.w = 0.75                    # 能耗与时延占比

        self.min_action = 0
        self.max_action = math.pi/2

        # 定义状态空间和动作空间
        # self.action_space = spaces.Box(
        #     low=self.min_action, high=self.max_action, shape=(1, ))
        # self.observation_space = 50
        # self.observation_space = spaces.Box(low=0, high=100, shape=(1,))
        # # spaces.Box(low=0, high=100, shape=(0, 100, 3)),
        # self.action_space = spaces.Discrete(8)
        # spaces.Discrete(8)
        self.observation = None
        # self.state = None

    # 定义每一个episode的初始状态，实现随机初始化
    def reset(self):
        self.user_location = np.empty(shape=(self.user_num, 2))
        self.L = np.empty(shape=(self.user_num, 1))
        self.uav_location = np.empty(shape=(self.user_num, 2))

        # 初始化用户的位置
        for i in range(self.user_num):
            user_location_x = np.random.randint(0, 100)  # 用户x位置随机初始化为(0，100)之间
            user_location_y = np.random.randint(0, 100)  # 用户y位置随机初始化为(0，100)之间
            self.user_location[i][0] = user_location_x
            self.user_location[i][1] = user_location_y

        self.user_location = np.array(self.user_location).reshape(self.user_num, 2)

        # 初始化用户任务数据量大小
        for i in range(self.user_num):
            L_ = random.uniform(1, 10) * (10 ** 6)
            self.L[i] = L_

        self.L = np.array(self.L).reshape(self.user_num, 1)

        # # 初始化无人机的位置
        # uav_location_x = random.randint(0, 100)           # 无人机x位置随机初始化为(0，100)之间
        # uav_location_y = random.randint(0, 100)           # 无人机y位置随机初始化为(0，100)之间
        # for i in range(self.user_num):
        #     self.uav_location[i][0] = uav_location_x
        #     self.uav_location[i][1] = uav_location_y
        #
        # self.uav_location = np.array(self.uav_location).reshape(self.user_num, 2)

        # 初始化无人机的位置
        uav_location_x = 0                                  # 无人机x位置随机初始化为(0，100)之间
        uav_location_y = 0                                  # 无人机y位置随机初始化为(0，100)之间
        for i in range(self.user_num):
            self.uav_location[i][0] = uav_location_x
            self.uav_location[i][1] = uav_location_y

        self.uav_location = np.array(self.uav_location).reshape(self.user_num, 2)

        # 创建obs, 包括用户的位置(user_num, 2)、用户的数据量(user_num, 1)、无人机的位置(user_num, 2)
        # 每个用户对应的无人机位置一样
        obs = np.empty(shape=(self.user_num, 5))
        # obs = [self.user_location, self.L, self.uav_location]
        # obs = [y for x in obs for y in x]
        for i in range(self.user_num):
            obs[i][0] = self.user_location[i][0]
            obs[i][1] = self.user_location[i][1]
            obs[i][2] = self.L[i]
            obs[i][3] = self.uav_location[i][0]
            obs[i][4] = self.uav_location[i][1]

        obs_normalization = np.array(obs).reshape(self.user_num, 5)

        # 对obs中用户数据量进行归一化操作
        # obs = np.array(obs, dtype=float)
        # for i in range(3):
        #     obs_normalization[:, i] = (obs[:, i] - np.mean(obs[:, i])) / np.std(obs[:, i])
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 归一化函数
        obs_normalization = min_max_scaler.fit_transform(obs)

        return obs, obs_normalization

    def step(self, action, obs):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        x_size = 100
        y_size = 100
        x, y = obs[0][3], obs[0][4]
        next_obs = np.empty(shape=(self.user_num, 5))
        for i in range(self.user_num):
            next_obs[i][0] = obs[i][0]
            next_obs[i][1] = obs[i][1]
            next_obs[i][2] = obs[i][2]

        # 输入到无人机模型中的action为[0, pi/2]
        action = action * math.pi / 2

        if x > x_size - 5:
            if math.cos(action) > (100 - x) / 5:
                # action = math.pi / 2
                # 返回最优卸载决策si_best 以及计算所得的最小能耗和时延加权和reward
                reward, si_best = pdpso(obs, x, y)
                reward = float(-1 * reward / 10)
                # 无人机出界惩罚reward-10
                reward = reward - 100
            else:
                # 返回最优卸载决策si_best 以及计算所得的最小能耗和时延加权和reward
                reward, si_best = pdpso(obs, x, y)
                reward = float(-1 * reward / 10)
        elif y > y_size - 5:
            if math.sin(action) > (100 - y) / 5:
                # action = 0
                # 返回最优卸载决策si_best 以及计算所得的最小能耗和时延加权和reward
                reward, si_best = pdpso(obs, x, y)
                reward = float(-1 * reward / 10)
                # 无人机出界惩罚reward-10
                reward = reward - 100
            else:
                # 返回最优卸载决策si_best 以及计算所得的最小能耗和时延加权和reward
                reward, si_best = pdpso(obs, x, y)
                reward = float(-1 * reward / 10)
        else:
            # 返回最优卸载决策si_best 以及计算所得的最小能耗和时延加权和reward
            reward, si_best = pdpso(obs, x, y)
            reward = float(-1 * reward / 10)

        # 无人机的新坐标
        x = x + 5 * math.cos(action)
        y = y + 5 * math.sin(action)

        if x > x_size - 5 and y > y_size - 5:
            reward += 10

        for i in range(self.user_num):
            next_obs[i][3] = x
            next_obs[i][4] = y

        next_obs_normalization = np.array(next_obs).reshape(self.user_num, 5)

        # 对next_obs进行归一化操作
        # next_obs = np.array(next_obs, dtype=float)
        # for i in range(3):
        #     next_obs_normalization[:, i] = (obs[:, i] - np.mean(obs[:, i])) / np.std(obs[:, i])

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 归一化函数
        next_obs_normalization[:, 0:3] = min_max_scaler.fit_transform(next_obs[:, 0:3])
        next_obs_normalization[:, 3] = next_obs[:, 3] / 100
        next_obs_normalization[:, 4] = next_obs[:, 4] / 100
        # next_obs_normalization[:, 3] = -5 + next_obs[:, 3] / 10
        # next_obs_normalization[:, 4] = -5 + next_obs[:, 4] / 10
        # min_max_scaler = preprocessing.MinMaxScaler()  # 归一化函数
        # next_obs_normalization = min_max_scaler.fit_transform(next_obs) * 3
        # # 找到每三列的最大值和最小值
        # max_number = []
        # for i in range(self.user_num):
        #     number = next_obs[i][2]
        #     max_number.append(number)
        # value_max = max(max_number)
        # value_min = min(max_number)
        #
        # for i in range(self.user_num):
        #     next_obs_normalization[i][2] = (next_obs[i][2] - value_min) / (value_max - value_min) * 100
        #     next_obs_normalization[i][0] = obs[i][0]
        #     next_obs_normalization[i][1] = obs[i][1]
        #     next_obs_normalization[i][0] = obs[i][0]
        #     next_obs_normalization[i][0] = obs[i][0]

        # self.state = np.array([x, y])

        done = (x < 0 or x > x_size) or (y < 0 or y > y_size) or (x > x_size - 5 and y > y_size - 5)
        # done = x > x_size and y > y_size

        return next_obs, next_obs_normalization, reward, si_best, done, x, y

    # def System_Model(self, obs, x, y):
    #
    #     user_location = np.empty(shape=(self.user_num, 2))
    #     L = np.empty(shape=(self.user_num, 1))
    #     for i in range(self.user_num):
    #         user_location[i][0] = obs[i][0]
    #         user_location[i][1] = obs[i][1]
    #         L[i] = obs[i][2]
    #
    #     L_user = np.empty(shape=(self.user_num, 1))
    #     L_uav = np.empty(shape=(self.user_num, 1))
    #     channel_gain = np.empty(shape=(self.user_num, 1))
    #     r = np.empty(shape=(self.user_num, 1))
    #     aerfa = np.empty(shape=(self.user_num, 1))
    #     e_up = np.empty(shape=(self.user_num, 1))
    #     e_user = np.empty(shape=(self.user_num, 1))
    #     e_uav = np.empty(shape=(self.user_num, 1))
    #     t_up = np.empty(shape=(self.user_num, 1))
    #     t_user = np.empty(shape=(self.user_num, 1))
    #     t_uav = np.empty(shape=(self.user_num, 1))
    #     Pro = np.empty(shape=(self.user_num, 1))
    #
    #     # 通信模型
    #     # 计算每个用户与无人机之间的信道增益
    #     for i in range(self.user_num):
    #         d = (x - user_location[i][0]) ** 2 + (y - user_location[i][1]) ** 2
    #         gain = self.beta_0 / (self.H ** 2 + d)
    #         channel_gain[i] = gain
    #
    #     # 计算用户与无人机之间数据的传输速率
    #     for i in range(self.user_num):
    #         r_ = self.B0 * math.log(1 + self.P_user * channel_gain[i] / (self.derta ** 2), 2)
    #         r[i] = r_
    #
    #     # 计算卸载比例
    #     # T = max(t_user, t_up + t_uav)
    #     # 要得到 min(T),则需要把 t_user = t_up +t_uav 时的aerfa值作为最优卸载比例
    #     for i in range(self.user_num):
    #         aerfa_ = (self.C_uav * self.f_user * r[i] + self.f_user * self.f_uav) \
    #                  / (self.C_user * self.f_uav * r[i] + self.C_uav * self.f_user * r[i] + self.f_user * self.f_uav)
    #         L_user_ = aerfa_ * L[i]
    #         L_uav_ = (1 - aerfa_) * L[i]
    #         aerfa[i] = aerfa_
    #         L_user[i] = L_user_
    #         L_uav[i] = L_uav_
    #
    #     # 计算无人机与用户之间的计算时延和通信能耗
    #     for i in range(self.user_num):
    #         t_up_ = L_uav[i] / r[i]
    #         e_up_ = self.P_user * t_up_
    #         t_up[i] = t_up_
    #         e_up[i] = e_up_
    #
    #     # 计算模型
    #     # 本地计算时延和计算能耗
    #     for i in range(self.user_num):
    #         t_user_ = L_user[i] * self.C_user / self.f_user
    #         e_user_ = self.K_user * (self.f_user ** 3) * t_user_
    #         t_user[i] = t_user_
    #         e_user[i] = e_user_
    #
    #     # 无人机计算时延和计算能耗
    #     for i in range(self.user_num):
    #         t_uav_ = L_uav[i] * self.C_uav / self.f_uav
    #         e_uav_ = self.K_uav * (self.f_uav ** 3) * t_uav_
    #         t_uav[i] = t_uav_
    #         e_uav[i] = e_uav_
    #
    #     # 能耗与时延问题总和
    #     for i in range(self.user_num):
    #         Pro_ = self.w * (e_up[i] + e_user[i] + e_uav[i]) + (1 - self.w) \
    #                * max(t_user[i], (t_user[i] + t_uav[i]))
    #         Pro[i] = Pro_
    #
    #     Pro_sum = sum(Pro)
    #
    #     return Pro_sum


