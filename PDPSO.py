import math
import numpy as np
import random
from SystemModel import SystemModel
from sklearn import preprocessing
import matplotlib.pyplot as plt

# plt.switch_backend('agg')

user_num = 10           # 用户数
beta_0 = 10 ** -5       # 1m参考距离的信道增益
H = 100                 # 无人机的飞行高度
B0 = 10 ** 7            # 分配给用户的带宽
derta = 10 ** -5        # 无人机处的噪声功率
P_user = 0.5            # 用户的发射功率
C_user = 800            # 用户处理1bit数据需要的CPU计算周期数
C_uav = 10 ** 3         # 无人机处理1bit数据需要的CPU计算周期数
f_user = 1 * (10 ** 9)  # 本机计算资源
f_uav = 3 * (10 ** 9)   # 分配给无人机的计算资源
K_user = 10 ** -27      # 用户CPU的电容系数
K_uav = 10 ** -28       # 无人机CPU的电容系数
w = 0.75                # 能耗与时延占比

x_size = 100            # 用户最大位置
y_size = 0              # 用户最小位置

v1_size = 1             # 用户最大速度
v2_size = -1            # 用户最小速度


def pdpso(obs, x, y):
    c1 = 2
    c2 = 2
    r1 = random.random()
    r2 = random.random()             # c1 c2 r1 r2均为更新速度参数

    user_location = np.empty(shape=(user_num, 2))
    user_location[:, 0] = obs[:, 0]
    user_location[:, 1] = obs[:, 1]
    uav_location = np.empty(shape=(user_num, 2))
    uav_location[:, 0] = x
    uav_location[:, 1] = y
    L = np.empty(shape=(user_num, 1))
    for i in range(user_num):
        L[i] = obs[i, 2]

    env = SystemModel()

    # # 初始化用户的位置
    # for i in range(user_num):
    #     location1 = np.random.randint(0, 100)  # 用户x位置随机初始化为(0，100)之间
    #     location2 = np.random.randint(0, 100)  # 用户y位置随机初始化为(0，100)之间
    #     user_location[i][0] = location1
    #     user_location[i][1] = location2
    #
    # user_location = np.array(user_location).reshape(user_num, 2)
    #
    # # 初始化用户任务数据量大小
    # for i in range(user_num):
    #     L_ = random.uniform(1, 10) * (10 ** 6)
    #     L[i] = L_
    #
    # L = np.array(L).reshape(user_num, 1)
    #
    # # 初始化无人机的位置
    # uav_location_x = random.randint(0, 100)  # 无人机x位置随机初始化为(0，100)之间
    # uav_location_y = random.randint(0, 100)  # 无人机y位置随机初始化为(0，100)之间
    # for i in range(user_num):
    #     uav_location[i][0] = uav_location_x
    #     uav_location[i][1] = uav_location_y
    #
    # uav_location = np.array(uav_location).reshape(user_num, 2)

    min_max_scaler = preprocessing.MinMaxScaler()               # 归一化函数

    # 初始化粒子的位置和速度
    particel_x = user_location
    particel_x = min_max_scaler.fit_transform(particel_x)       # 归一化粒子位置

    particle_v = np.random.random(20)
    particle_v = particle_v.reshape(user_num, 2)                # 产生x轴和y轴速度

    # 初始化pbest、gbest
    Pro_load_start, Pro_local_start = env.System_Model(user_location, L, uav_location)
    # Pro_load_start = min_max_scaler.fit_transform(Pro_load_start)
    # Pro_local_start = min_max_scaler.fit_transform(Pro_local_start)

    pbest = np.append(Pro_local_start, Pro_local_start, axis=1)     # 初始化pbest为初始粒子位置
    gbest = np.append(Pro_load_start, Pro_load_start, axis=1)       # 初始化pbest为全本地总能耗和时延加权和

    epoch = 1
    iterations = 100
    particle = user_num
    W = 1                                                           # 惯性权重初始值=1
    fitness = []                                                    # 适应度值fitness
    diversity = []                                                  # 种群多样性diversity
    Pro_sum = []                                                    # 问题定义值
    pbest_sum = []                                                  # 问题定义最优值
    si_best = []                                                   # 最优卸载决策
    while epoch < iterations:
        si = []                                                     # 卸载决策
        Pro = []                                                    # 问题定义Problem
        particle_V = []                                             # 粒子速度
        Pro_load, Pro_local = env.System_Model(particel_x, L, uav_location)
        # Pro_load = min_max_scaler.fit_transform(Pro_load)
        # Pro_local = min_max_scaler.fit_transform(Pro_local)

        for i in range(particle):
            for j in range(2):
                # 更新速度
                particle_v[i][j] = W * particle_v[i][j] + c1 * r1 * (pbest[i][j] - particel_x[i][j]) + \
                                   c2 * r2 * (gbest[i][j] - particel_x[i][j])

                # 更新位置
                particel_x[i][j] = particel_x[i][j] + particle_v[i][j]

                # 位置限制
                if particel_x[i][j] > x_size:
                    particel_x[i][j] = x_size
                elif particel_x[i][j] < y_size:
                    particel_x[i][j] = y_size

                # 速度限制
                if particle_v[i][j] > v1_size:
                    particle_v[i][j] = v1_size
                elif particle_v[i][j] < v2_size:
                    particle_v[i][j] = v2_size

            particle_V_ = math.sqrt(particle_v[i][0] ** 2 + particle_v[i][1] ** 2)
            particle_V.append(particle_V_)

            sigmoid = 1 / (1 + math.exp(-1 * particle_V[-1]))
            p = random.random()

            if sigmoid > p:
                # 部分卸载
                si_ = 1
                si.append(si_)
                Pro_ = si_ * Pro_load[i]
                Pro_ = Pro_.tolist()
                Pro.append(Pro_)

            else:
                # 全本地卸载
                si_ = 0
                si.append(si_)
                Pro_ = (1 - si_) * Pro_local[i]
                Pro_ = Pro_.tolist()
                Pro.append(Pro_)

        Pro = [token for st in Pro for token in st]                 # 将[[],[],[]]数组降维至[, , ,]
        Pro_sum_ = sum(Pro)
        Pro_sum.append(Pro_sum_)

        # 将Pro作为fitness函数，传给pbest
        fitness_ = Pro_sum[-1]
        fitness.append(fitness_)

        pbest_sum_ = pbest.sum()
        pbest_sum.append(pbest_sum_)

        # 更新当前最优解pbest
        if 2 * fitness_ <= pbest_sum_:
            for i in range(user_num):
                pbest[i][0] = Pro[i]
                pbest[i][1] = Pro[i]

            si_pbest = si
            si_best = si_pbest

        # 更新全局最优解gbest
        if np.sum(pbest) <= np.sum(gbest):
            gbest = pbest
            gbest_sum = gbest.sum()

            si_gbest = si

        '更新种群多样性diversity以及惯性权重函数W'
        # 计算更新后的第i个粒子和其他粒子之间欧式距离d
        d_min = []
        d_average = []
        d = np.empty(shape=(user_num, user_num))
        for i in range(user_num):
            for j in range(user_num):
                if i != j:
                    d[i][j] = pow((particel_x[i][0] - particel_x[j][0]) ** 2 + (particel_x[i][1] - particel_x[j][1]) ** 2, 1/2)
                else:
                    d[i][j] = 0

        d_new = []
        for i in range(user_num):
            for j in range(user_num):
                if i == j:
                    continue
                else:
                    d_new.append(d[i][j])

        # mask = d != 0
        # d_new = d[mask]
        d_new = np.array(d_new).reshape(user_num, user_num-1)           # 去掉自身距离的新矩阵d_new

        d_min_ = np.min(d_new, axis=1)                                  # 每个用户与其他用户中的最小距离d_min
        d_average_ = np.mean(d_new, axis=1)                             # 每个用户与其他用户距离的平均值
        d_min.append(d_min_)
        d_average.append(d_average_)

        # 计算更新后的种群多样性
        s = list(map(lambda x: (x[0] - x[1]) ** 2, zip(d_average, d_min)))
        s = [token for st in s for token in st]
        diversity_ = pow(1 / (particle - 1) * sum(s), 0.5)
        diversity.append(diversity_)

        # 更新惯性权重函数W
        while epoch > 1:
            if diversity[-1] > diversity[-2]:
                break

            if diversity[-1] >= diversity[-2]:
                W = W * (math.exp(1 / (diversity[-1] + 1) - 1) + 1)
            else:
                W = W * math.exp(1 / (diversity[-1] + 1) - 1)
            break

        # print('第%d次迭代完成' % epoch)
        epoch = epoch + 1

    result = pbest_sum[-1] / 2

    return result, si_best

    #     print('粒子的平均速度为：', particle_V[-1 * user_num:-1])
    #     print('fitness值为：', fitness)
    #     print('si = ', si)
    #     print('si_pbest = ', si_pbest[-1])
    #     print('Pro为：', Pro_sum)
    #     print('粒子位置为：', np.sum(particel_x))
    #     print('pbest_sum值为：', pbest_sum)
    #     print('W = ', W)
    #
    # print(diversity)
    #
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.plot(pbest_sum)
    # plt.xlabel('iterations')
    # plt.ylabel('pbest_sum')
    #
    # plt.subplot(2, 2, 2)
    # x = [i for i in range(user_num)]
    # si_pbest = si_pbest[-1]
    # y = [si_pbest[i] for i in range(user_num)]
    # plt.scatter(x, y)
    # plt.xlabel('user_num')
    # plt.ylabel('si_pbest')
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(fitness)
    # plt.xlabel('iterations')
    # plt.ylabel('fitness')
    #
    # plt.subplot(2, 2, 4)
    # plt.plot(diversity)
    # plt.xlabel('iterations')
    # plt.ylabel('diversity')
    # plt.show()

