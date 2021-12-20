import math
import numpy as np

np.seterr(divide='ignore', invalid='ignore')


class SystemModel(object):
    def __init__(self):
        self.user_num = 10              # 用户数
        self.beta_0 = 10 ** -5          # 1m参考距离的信道增益
        self.H = 100                    # 无人机的飞行高度
        self.B0 = 10 ** 7               # 分配给用户的带宽
        self.derta = 10 ** -5           # 无人机处的噪声功率
        self.P_user = 0.5               # 用户的发射功率
        self.C_user = 800               # 用户处理1bit数据需要的CPU计算周期数
        self.C_uav = 10 ** 3            # 无人机处理1bit数据需要的CPU计算周期数
        self.f_user = 1 * (10 ** 9)     # 本机计算资源
        self.f_uav = 3 * (10 ** 9)      # 分配给无人机的计算资源
        self.K_user = 10 ** -27         # 用户CPU的电容系数
        self.K_uav = 10 ** -28          # 无人机CPU的电容系数
        self.w = 0.75                   # 能耗与时延占比

    def System_Model(self, user_location, L, uav_location):
        L_user = []
        L_uav = []
        channel_gain = []
        r = []
        aerfa = []
        e_up = []
        e_user = []
        e_uav = []
        e_local = []
        e_load = []
        t_up = []
        t_user = []
        t_uav = []
        t_local = []
        t_load = []
        Pro_load = []
        Pro_local = []
        Pro_load_sum = []
        Pro_local_sum = []

        self.user_location = user_location
        self.uav_location = uav_location
        self.L = L

        # 通信模型
        # 计算每个用户与无人机之间的信道增益
        for i in range(self.user_num):
            d = (self.uav_location[i][0] - self.user_location[i][0]) ** 2 + \
                (self.uav_location[i][1] - self.user_location[i][1])
            gain = self.beta_0 / (self.H ** 2 + d)
            channel_gain.append(gain)

        channel_gain = np.array(channel_gain).reshape(self.user_num, 1)

        # 计算用户与无人机之间数据的传输速率
        for i in range(self.user_num):
            r_ = self.B0 * math.log(1 + self.P_user * channel_gain[i] / (self.derta ** 2), 2)
            r.append(r_)

        r = np.array(r).reshape(self.user_num, 1)

        # 计算卸载比例
        # T = max(t_user, t_up + t_uav)
        # 要得到 min(T),则需要把 t_user = t_up +t_uav 时的aerfa值作为最优卸载比例
        for i in range(self.user_num):
            aerfa_ = (self.C_user * self.f_uav * r[i]) \
                     / (self.C_user * self.f_uav * r[i] + self.C_uav * self.f_user * r[i] + self.f_user * self.f_uav)
            L_user_ = (1 - aerfa_) * self.L[i]
            L_uav_ = aerfa_ * self.L[i]
            aerfa.append(aerfa_)
            L_user.append(L_user_)
            L_uav.append(L_uav_)

        aerfa = np.array(aerfa).reshape(self.user_num, 1)
        L_user = np.array(L_user).reshape(self.user_num, 1)
        L_uav = np.array(L_uav).reshape(self.user_num, 1)

        '''通信模型'''
        # 计算无人机与用户之间的计算时延和通信能耗
        for i in range(self.user_num):
            t_up_ = L_uav[i] / r[i]
            e_up_energy = self.P_user * t_up_
            t_up.append(t_up_)
            e_up.append(e_up_energy)

        t_up = np.array(t_up).reshape(self.user_num, 1)
        e_up = np.array(e_up).reshape(self.user_num, 1)

        '''计算模型'''
        '全本地计算时延和计算能耗'
        for i in range(self.user_num):
            t_local_ = L[i] * self.C_user / self.f_user
            e_local_ = self.K_user * (self.f_user ** 3) * t_local_
            t_local.append(t_local_)
            e_local.append(e_local_)

        t_local = np.array(t_local).reshape(self.user_num, 1)
        e_local = np.array(e_local).reshape(self.user_num, 1)

        '部分卸载到无人机计算时延和计算能耗'
        # 一部分在本地计算
        for i in range(self.user_num):
            t_user_ = L_user[i] * self.C_user / self.f_user
            e_user_ = self.K_user * (self.f_user ** 3) * t_user_
            t_user.append(t_user_)
            e_user.append(e_user_)

        t_user = np.array(t_user).reshape(self.user_num, 1)
        e_user = np.array(e_user).reshape(self.user_num, 1)

        # 一部分卸载到无人机计算
        for i in range(self.user_num):
            t_uav_ = L_uav[i] * self.C_uav / self.f_uav
            e_uav_ = self.K_uav * (self.f_uav ** 3) * t_uav_
            t_uav.append(t_uav_)
            e_uav.append(e_uav_)

        t_uav = np.array(t_uav).reshape(self.user_num, 1)
        e_uav = np.array(e_uav).reshape(self.user_num, 1)

        # 上传的总能耗和总时延
        for i in range(self.user_num):
            t_load_ = max(t_user[i], (t_up[i] + t_uav[i]))
            e_load_ = e_up[i] + e_user[i] +e_uav[i]
            t_load.append(t_load_)
            e_load.append(e_load_)

        t_load = np.array(t_load).reshape(self.user_num, 1)
        e_load = np.array(e_load).reshape(self.user_num, 1)

        # 能耗与时延问题总和
        for i in range(self.user_num):
            Pro_load_ = self.w * e_load[i] + (1 - self.w) * t_load[i]
            Pro_local_ = self.w * e_local[i] + (1 - self.w) * t_local[i]
            if self.L[i] < 3 * 10 ** 6:
                Pro_load_ = (1 + 0.3) * Pro_load_
            elif self.L[i] > 8 * 10 ** 6:
                Pro_local_ = (1 + 0.3) * Pro_load_
            Pro_load.append(Pro_load_)
            Pro_local.append(Pro_local_)

        Pro_load = np.array(Pro_load).reshape(self.user_num, 1)
        Pro_local = np.array(Pro_local).reshape(self.user_num, 1)
        Pro_load_sum.append(sum(Pro_load))
        Pro_local_sum.append(sum(Pro_local))

        return Pro_load, Pro_local
