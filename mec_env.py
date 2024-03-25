import numpy as np
import math
import ipdb as pdb
import tensorflow as tf
from options import args_parser
from helper import *
from agent import *

from scipy import special as sp
from scipy.constants import pi

# FL相关
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy import special as sp
from scipy.constants import pi

import torch
#from tensorboardX import SummaryWriter

args = args_parser()

mu1, sigma1 = 1.5e+9, 0.1
lower, upper = mu1 - 2 * sigma1, mu1 + 2 * sigma1  # 截断在[μ-2σ, μ+2σ]
x = stats.truncnorm((lower - mu1) / sigma1, (upper - mu1) / sigma1, loc=mu1, scale=sigma1)
ini_resource = x.rvs(args.num_users)  # 总共得到每辆车的计算资源 车辆数目为clients_num来控制


class MecTerm(object):
    """
    MEC terminal parent class
    """

    def __init__(self, user_config, train_config):
        self.dis = 0
        self.id = user_config['id']               # 车辆编号
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.action_bound = user_config['action_bound']
        self.action_level = user_config['action_level'] # 9/16

        self.seed = train_config['random_seed']

        self.tr = [0] * args.num_users # 传输速率
        self.c_c = [0] * args.num_users # 通信成本
        self.l_c = [0] * args.num_users # 本地训练成本

        self.tr_norm = [0] * args.num_users
        self.delta_norm = [0] * args.num_users
        self.dis_norm = [0] * args.num_users     # state处的归一化

        self.tr_norm1 = [0] * args.num_users
        self.delta_norm1 = [0] * args.num_users
        self.dis_norm1 = [0] * args.num_users     # next_state处的归一化

        # 每辆车的训练数据的样本数d_size 跟FL本地训练分配的本地数据样本数一回事
        self.d_size = [0] * args.num_users


        alpha = 2
        self.path_loss = [0] * args.num_users # 路径损耗
        self.position = [0] * args.num_users # 车辆位置
        self.lamda = 7      # 计算ρm时用到的参数
        self.train_config = train_config # 训练设置

        self.init_path = ''
        self.isUpdateActor = True
        self.init_seqCnt = 0

        self.actions = np.zeros(self.action_dim)
        self.n_t = 1
        self.n_r = user_config['num_r']
        self.q_l = list(range(2,9)) # 量化电平
        self.q_level = [2]*args.num_users

        self.Reward = 0
        self.State = []
        self.delta = []
        # some pre-defined parameters
        self.bandwidth = 1000  # Hz   香农公式里的带宽B
        self.velocity = 20.0     # 车辆速度
        self.width_lane = 5  # 车道宽度
        self.Hight_RSU = 10  # RSU高度
        self.transpower_list = [200, 320, 500, 800, 1000]
        self.transpower = 200  # mw 车辆发送功率 200mw = 23dBm
        self.sigma2 = train_config['sigma2']  # 噪声功率的平方 后面取10-9mw
        # args.num_users = 10    # 用户数
        self.t = 0.05
        self.not_reset = True

        # self.channelModel = ARModel(self.n_t, self.n_r, rho=compute_rho(self) ,seed=train_config['random_seed'])
        self.channelModel = ARModel(self.n_t, self.n_r, seed=self.train_config['random_seed'])

    def D_size(self):
        for i in range(args.num_users):
            self.d_size[i] = 4800 + 8 * i
        return self.d_size

    def position_ini(self):
        self.dis = [0] * args.num_users
        for i in range(args.num_users):     # i从 0 - (args.num_users-1)
            self.dis[i] = -500 + 10 * i 
        print(self.dis)
        return self.dis

    #车辆x轴坐标位置
    def dis_mov(self):
        # self.dis = [0] * args.num_users
        if self.not_reset:
            print("update vehicles position")
            for i in range(args.num_users):     # i从 0 - (args.num_users-1)
            # self.dis[i] = -500 + 10 * i                #车辆初始位置
                self.dis[i] += (self.velocity * self.t)  #车辆随时间 x轴行驶距离
        print("dis :", self.dis)
        return self.dis



    def compute_rho(self):
        #计算ρi
        x_0 = np.array([1, 0, 0])
        P_B = np.array([0, 0, self.Hight_RSU])
        P_m = [0] * args.num_users
        self.rho = [0] * args.num_users
        for i in range(args.num_users):
            P_m[i] = np.array([self.dis[i], self.width_lane, self.Hight_RSU])
            self.rho[i] = sp.j0(2 * pi * self.t * self.velocity * np.dot(x_0, (P_B - P_m[i])) / (np.linalg.norm(P_B - P_m[i]) * self.lamda))
        return self.rho

    def sampleCh(self, dis):
        self.dis_mov()
        self.compute_rho()
        self.H = self.channelModel.sampleCh(self.dis, self.rho)
        # for i in range(args.num_users):
        #     self.H[i] = self.rho[i] * self.H[i] + complexGaussian(self.n_t, self.n_r, np.sqrt(1 - self.rho[i] * self.rho[i]))  # rho就是ρi  self.H就是hi（信道增益）
        alpha = 2
        self.path_loss = [0] * args.num_users
        self.position = [0] * args.num_users
        for i in range(args.num_users):
            self.position[i] = np.array([self.dis[i], self.width_lane, self.Hight_RSU])    #车辆的位置坐标
            self.path_loss[i] = 1 / np.power(np.linalg.norm(self.position[i]), alpha)

        return self.H, self.path_loss    # self.H就是hms

    #车辆与RSU之间距离
    #def Distance(self):
    #    self.dis_mov()
    #    alpha = 2
    #    self.path_loss = [0] * args.num_users
    #    self.position = [0] * args.num_users
    #    for i in range(args.num_users):
    #        self.position[i] = np.array([self.dis[i], self.width_lane, self.Hight_RSU])    #车辆的位置坐标
    #        self.path_loss[i] = 1 / np.power(np.linalg.norm(self.position[i]), alpha)  # np.linalg.norm(self.position)计算矩阵的模
    #    return self.path_loss   #返回路径损耗di(-α)

    #车辆和RSU传输速率
    def transRate(self):
        self.sampleCh(self.dis)
        # self.tr = [0] * args.num_users   # 初始化里有过了
        sinr = [0] * args.num_users
        #self.Distance()
        for i in range(args.num_users):
            sinr[i] = self.transpower * abs(self.H[i]) * self.path_loss[i] / self.sigma2      # abs()即可求绝对值，也可以求复数的模 #因为神经网络里输入的值不能为复数
            # sinr[i] = self.transpower * self.H[i] * self.path_loss[i] / self.sigma2
            self.tr[i] = np.log2(1 + sinr[i]) * self.bandwidth
            #kkk = abs(self.H[i])
        # a = np.reshape(self.tr,(1,args.num_users ))[0]
        return self.tr

    # 每辆车可用计算资源delta 服从截断高斯分布 
    def resou(self):
        mu1, sigma1 = 1.5e+9, 0.1
        lower, upper = mu1 - 2 * sigma1, mu1 + 2 * sigma1  # 截断在[μ-2σ, μ+2σ]
        x = stats.truncnorm((lower - mu1) / sigma1, (upper - mu1) / sigma1, loc=mu1, scale=sigma1)
        self.delta = x.rvs(args.num_users)  # 总共得到每辆车的计算资源 车辆数目为clients_num来控制
        return self.delta 


    # 所有车的状态集合 即DDPG输入的状态
    def all_state(self):
        self.not_reset = False
        self.transRate()
        self.not_reset = True
        #self.resou() #改成分配cpu
        self.State = [0] * args.num_users    # 未整形的state

        # 初始化资源
        self.delta = ini_resource
        # 归一化状态
        #for i in range(args.num_users):
        #    self.tr_norm[i] = float(self.tr[i] - np.min(self.tr))/(np.max(self.tr) - np.min(self.tr))

        #for i in range(args.num_users):
        #    self.delta_norm[i] = float(self.delta[i] - np.min(self.delta))/(np.max(self.delta) - np.min(self.delta))

        #for mm in range(args.num_users):
            # self.dis_norm[i] = float(i - np.min(self.dis))/(np.max(self.dis) - np.min(self.dis))
        #    self.dis_norm[mm] = float(self.dis[mm] - np.min(self.dis)) / (np.max(self.dis) - np.min(self.dis))

        #self.q_level = [8] * args.num_users 量化电平初始化

        #for i in range(args.num_users):
        #    self.State[i] = [self.tr_norm[i], self.delta_norm[i], self.dis_norm[i], self.q_level[i]]

        #for i in range(args.num_users):
        #    self.State[i] = [self.tr[i], self.delta[i], self.dis[i], self.q_level[i]]
        self.State = [self.tr[0], self.delta[0], self.dis[0], self.q_level[0]]

        print("Reset state:", self.State) #[0]
        # self.State = np.reshape(self.State, (1, 4 * args.num_users))[0]
        return self.State # [0]



    # 时隙t车辆i的本地学习时间cost
    def local_c(self):
        self.D_size()
        self.resou()
        beta_m = 1e6      #执行一个数据样本所需CPU周期数 10^6 cycles
        # self.l_c = [0] * args.num_users
        for i in range(args.num_users):
            self.l_c[i] = self.d_size[i] * 1e6 / self.delta[i]
            # self.l_c[i] = self.d_size[i]* beta_m / self.delta[i]
        return self.l_c

    # 时隙t车辆i的通信cost,w_i_size为t时隙所学习的本地模型参数的大小 transRate传输速率
    def commu_c(self):
        self.transRate()
        w_size = 203530 * (1 + np.log2(self.q + 1))      #本地模型参数大小 5kbits(香农公式传输速率单位为bit/s)
        # self.c_c = [0] * args.num_users
        for i in range(args.num_users):
            self.c_c[i] = w_size / self.tr[i]
        return self.c_c

    # 后面直接用模型训练计算处来的值更新，此处不定义函数


class MecTermLD(MecTerm):
    """
    MEC terminal class for loading from stored models
    """

    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess


        # 设置模型保存路径等

        saver = tf.train.import_meta_graph(user_config['meta_path'])   # 已将原始网络保存在了.meta文件中，可以用tf.train.import()函数来重建网络
        saver.restore(sess, user_config['model_path'])

        graph = tf.get_default_graph()    # 获取当前默认计算图
        input_str = "input_" + self.id + "/X:0"
        output_str = "output_" + self.id + ":0"
        self.inputs = graph.get_tensor_by_name(input_str)
        if not 'action_level' in user_config:
            self.out = graph.get_tensor_by_name(output_str)

    def feedback(self, q_level, g_loss):
        self.q_level = q_level
        self.next_state = []
        # update the transmission rate
        self.transRate()
        # 更新车辆可用计算资源
        self.resou()
        self.local_c()
        self.commu_c()


        # get the reward for the current slot
        aa = []
        for i, j in zip(self.l_c, self.c_c):
            summ = i + j
            aa.append(summ)
        eee = []
        for i, j in zip(aa, globall):
            summe = i + j
            eee.append(summe)

        bb = []
        for i, j in zip(eee, self.P_Lamda):
            cc = i * j
            bb.append(cc)
        self.Reward = -sum(bb)           # 对应奖励公式

        # estimate the channel for next slot
        self.dis_mov()
        self.Distance()
        self.transRate()



        self.next_state = np.array([self.tr, self.delta, self.dis, self.q_level])  # 定义下一状态更新

        # update system state
        self.State = self.next_state
        # return the reward in this slot
        return self.Reward, self.tr, self.delta, self.dis, self.q_level

    def predict(self, isRandom):
        self.q_level = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim))})[0]
        return self.q_level, np.zeros(self.action_dim)
    """
        a = tf.add(2, 5)
        b = tf.multiply(a, 3)
        with tf.Session() as sess: 
        sess.run(b, feed_dict = {a:15}) # 重新给a赋值为15   运行结果：45
        sess.run(b) #feed_dict只在调用它的方法内有效,方法结束,feed_dict就会消失。 所以运行结果是：21   
    """


class MecTermRL(MecTerm):
    """
    MEC terminal class using RL
    """

    # rate:packet poisson arrival, dis: distance in meters
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        self.agent = DDPGAgent(sess, user_config, train_config)

        if 'init_path' in user_config and len(user_config['init_path']) > 0:
            self.init_path = user_config['init_path']
            self.init_seqCnt = user_config['init_seqCnt']
            self.isUpdateActor = False



    def feedback(self, P_lamda, globall,index):  # globall为global_loss
        self.P_Lamda = P_lamda
        self.next_state = []
        # update the transmission rate
        self.transRate()
        # 更新车辆可用计算资源
        self.resou()
        self.local_c()
        self.commu_c()
        self.globall = globall

        # get the reward for the current slot
        # aa = []
        # for i1, j1 in zip(self.l_c, self.c_c):
        #     summ = i1 + j1
        #     aa.append(summ)

        # eee = self.l_c + self.c_c + self.globall
        # # eee = []
        # # for ii, jj in zip(aa, self.globall):
        # #     summe = ii + jj
        # #     eee.append(summe)
        #
        # bb = np.multiply(eee, self.P_Lamda)
        # # bb = []
        # # for iii, jjj in zip(eee, self.P_Lamda):
        # #     cc = iii * jjj
        # #     bb.append(cc)
        # self.Reward = -sum(bb)   # 对应奖励公式



        # a1 = np.array(self.l_c)
        # b1 = np.array(self.c_c)
        # c1 = a1 + b1
        # bb = np.multiply(c1, self.P_Lamda)
        # cc = -sum(bb) - max(self.P_Lamda) * self.globall
        #
        # self.Reward = cc / sum(self.P_Lamda)

        # np.argmax(self.P_Lamda)为动作中最大值的索引
        self.Reward = - (self.l_c[np.argmax(self.P_Lamda)] + self.c_c[np.argmax(self.P_Lamda)] + self.globall) * max(self.P_Lamda) / sum(self.P_Lamda)





        # estimate the channel for next slot
        self.dis_mov()
        self.Distance()
        self.transRate()
        # update the actor and critic network


        for i in range(args.num_users):
            self.tr_norm1[i] = float(self.tr[i] - np.min(self.tr))/(np.max(self.tr) - np.min(self.tr))

        for i in range(args.num_users):
            self.delta_norm1[i] = float(self.delta[i] - np.min(self.delta))/(np.max(self.delta) - np.min(self.delta))

        for i in range(args.num_users):
            self.dis_norm1[i] = float(self.dis[i] - np.min(self.dis))/(np.max(self.dis) - np.min(self.dis))

        self.next_state = np.array([self.tr_norm1, self.delta_norm1, self.dis_norm1, self.P_lamda])  # 定义下一状态更新
        # self.next_state = np.array([self.tr, self.delta, self.dis, self.P_lamda])  # 定义下一状态更新

        # update system state
        self.State = self.next_state
        # return the reward in this slot
        # return self.Reward, self.tr, self.delta, self.dis, self.P_lamda
        return self.Reward, self.tr_norm1, self.delta_norm1, self.dis_norm1, self.P_lamda

    # LD里的
    # def predict(self, isRandom):
    #     self.P_Lamda = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim))})[0]
    #     return self.P_Lamda, np.zeros(self.action_dim)

    def predict(self, isRandom):
        P_lamda1 = self.agent.predict(self.State, self.isUpdateActor)
        self.P_lamda = np.fmax(0, np.fmin(self.action_bound, P_lamda1))
        return self.P_lamda


    def AgentUpdate(self,done):
        self.agent.update(self.State, self.P_lamda, self.Reward, done, self.next_state, self.isUpdateActor)#往replaybuffer加动作状态，到达一定数量更新它和神经网络
        self.State = self.next_state


class MecTermDQN(MecTerm):
    """
    MEC terminal class using DQN
    """

    # rate:packet poisson arrival, dis: distance in meters
    def __init__(self, sess, user_config, train_config):
        MecTerm.__init__(self, user_config, train_config)
        self.sess = sess
        self.action_level = user_config['action_level']
        self.agent = DQNAgent(sess, user_config, train_config)
        self.action = 0
        self.t_factor = 0.5 # 0.1 0.3 0.5 0.7 0.9
        self.q_level_list = [2,4,6,8,10]
        self.p_level_list = [200,320,500,800,1000]
        # self.table = array([[0., 1.]])
        self.table = np.array([self.q_level_list,self.p_level_list])
        self.A = 0.54
        self.B = 1.99
        self.C = 0.21999
        self.D = 0.072
        self.e = 0.012


    def feedback(self, q_level, power): # 删除index
        # isOverflow = 0
        # self.SINR = sinr
        self.q = q_level
        self.transpower = power
        self.next_state = []
        # update the transmission rate
        #self.transRate()
        #self.resou()
        self.local_c()
        self.commu_c()
        self.mini_ground = math.ceil(math.sqrt(203530) / (self.q * 4) * (self.A /self.e - self.B) + (self.D + self.A) / self.e - self.B - self.C)
        # get the reward for the current slot
        
        self.Reward = -self.t_factor*(self.l_c[0] + self.c_c[0])*self.mini_ground - (1-self.t_factor)*np.log2(56.418/(2**self.q - 1)+1)
        #self.Reward = -self.t_factor*np.max(self.l_c + self.c_c) - (1-self.t_factor)*np.log2(56.418/(2**self.q - 1)+1)
       
        self.q_error = 3183.026/(2**self.q - 1)**2       
        # self.Reward = -(1953.125/(2**self.q_level - 1)**2) / np.max(self.l_c + self.c_c)
        self.delay = self.l_c[0] + self.c_c[0]
        self.T = (self.l_c[0] + self.c_c[0])*self.mini_ground
        # estimate the channel for next slot
        #self.dis_mov()
        #self.Distance()
        #self.transRate()

        #for i in range(args.num_users):
        #    self.tr_norm1[i] = float(self.tr[i] - np.min(self.tr))/(np.max(self.tr) - np.min(self.tr))

        #for i in range(args.num_users):
        #    self.delta_norm1[i] = float(self.delta[i] - np.min(self.delta))/(np.max(self.delta) - np.min(self.delta))

        #for i in range(args.num_users):
        #    self.dis_norm1[i] = float(self.dis[i] - np.min(self.dis))/(np.max(self.dis) - np.min(self.dis))

        #self.next_state = np.array([self.tr_norm1[0], self.delta_norm1[0], self.dis_norm1[0], self.q])
        self.next_state = np.array([self.tr[0], self.delta[0], self.dis[0], self.q])

        print("next_state is :", self.next_state)
        self.q_level = np.array(self.q*np.ones(args.num_users)) # 10/5
        # update system state

        #return self.Reward, self.tr_norm1[0], self.delta_norm1[0], self.dis_norm1[0], self.q, self.transpower, self.delay, self.q_error
        return self.Reward, self.mini_ground, self.T, self.tr[0], self.delta[0], self.dis[0], self.q, self.transpower, self.delay, self.q_error

    def AgentUpdate(self,done):
        self.agent.update(np.array(self.State), self.action, self.Reward, done, self.next_state) # [0]
        print("State is :", self.State) #[0]
        print("next_state is:", self.next_state)
        self.State = self.next_state #[0]
       
    def predict(self, isRandom):
        # print ('self.table:',self.table)
        self.action = self.agent.predict(self.State) #[0]
        # print ('action:',self.action
        action_tmp = self.action
        self.actions[0] = self.table[0, action_tmp % self.action_level]
        self.actions[1] = self.table[1, int(np.floor(action_tmp / self.action_level))]
        #for i in range(self.action_dim):
        #    self.actions[i] = self.table[i, action_tmp % self.action_level]
        #    action_tmp //= self.action_level
        return self.actions
        #for i in range(self.action_dim):
            # print ('action_tmp ,self.action_level',action_tmp,self.action_level)
        #    self.q_level[i] = self.table[i, action_tmp % self.action_level]
        #    action_tmp //= self.action_level
        # print ( 'self.Power:',self.Power)
        #return self.q_level


class MecTermDQN_LD(MecTermLD):
    """
    MEC terminal class for loading from stored models of DQN
    """

    def __init__(self, sess, user_config, train_config):
        MecTermLD.__init__(self, sess, user_config, train_config)
        graph = tf.get_default_graph()
        self.action_level = user_config['action_level']
        self.action = 0
        self.t_factor = 0.5 #0.1 0.3 0.5 0.7 0.9

        output_str = "output_" + self.id + "/BiasAdd:0"
        self.out = graph.get_tensor_by_name(output_str)
        self.q_level_list = [2, 4, 6, 8, 10]
        self.p_level_list = [200, 320, 500, 800, 1000]
        self.table = np.array([self.q_level_list, self.p_level_list])
        # self.table = np.array([[float(self.action_bound)/(self.action_level-1)*i for i in range(self.action_level)] for j in range(self.action_dim)])

    def predict1(self, isRandom):
        q_out = self.sess.run(self.out, feed_dict={self.inputs: np.reshape(self.State, (1, self.state_dim))})[0]
        self.action = np.argmax(q_out)
        action_tmp = self.action
        self.actions[0] = self.table[0, action_tmp % self.action_level]
        self.actions[1] = self.table[1, int(np.floor(action_tmp / self.action_level))]

        return self.actions

    def feedback1(self, q_level, power):  # 删除index
        # isOverflow = 0
        # self.SINR = sinr
        self.q = q_level
        self.transpower = power
        self.next_state = []
        # update the transmission rate
        #self.transRate()
        #self.resou()
        self.local_c()
        self.commu_c()
        self.A = 0.54
        self.B = 1.99
        self.C = 0.21999
        self.D = 0.072
        self.e = 0.012
        # get the reward for the current slot
        self.mini_ground = math.ceil(math.sqrt(203530) / (self.q * 4) * (self.A / self.e - self.B) + (
                    self.D + self.A) / self.e - self.B - self.C)

        self.Reward = -self.t_factor * (self.l_c[0] + self.c_c[0]) * self.mini_ground - (1 - self.t_factor) * np.log2(
            56.418 / (2 ** self.q - 1) + 1)
        # self.Reward = -self.t_factor*np.max(self.l_c + self.c_c) - (1-self.t_factor)*np.log2(56.418/(2**self.q - 1)+1)

        self.q_error = 3183.026 / (2 ** self.q - 1) ** 2
        # self.Reward = -(1953.125/(2**self.q_level - 1)**2) / np.max(self.l_c + self.c_c)
        self.delay = self.l_c[0] + self.c_c[0]
        self.T = (self.l_c[0] + self.c_c[0])*self.mini_ground

        # estimate the channel for next slot
        #self.dis_mov()
        #self.Distance()
        #self.transRate()

        #for i in range(args.num_users):
        #    self.tr_norm1[i] = float(self.tr[i] - np.min(self.tr)) / (np.max(self.tr) - np.min(self.tr))

        #for i in range(args.num_users):
        #    self.delta_norm1[i] = float(self.delta[i] - np.min(self.delta)) / (np.max(self.delta) - np.min(self.delta))

        #for i in range(args.num_users):
        #    self.dis_norm1[i] = float(self.dis[i] - np.min(self.dis)) / (np.max(self.dis) - np.min(self.dis))

        self.next_state = np.array([self.tr[0], self.delta[0], self.dis[0], self.q])
        print("next_state is :", self.next_state)
        self.q_level = np.array(self.q * np.ones(args.num_users))  # 10/5
        # update system state
        self.State = self.next_state

        return self.Reward, self.mini_ground, self.T, self.tr[0], self.delta[0], self.dis[0], self.q, self.transpower, self.delay, self.q_error

class MecSvrEnv(object):
    """
    Simulation environment
    """

    def __init__(self, user_list, Train_vehicle_ID, sigma2, max_len, mode='train'):
        self.user_list = user_list
        self.Train_vehicle_ID = Train_vehicle_ID - 1
        self.sigma2 = sigma2
        self.mode = mode
        self.count = 0
        self.max_len = max_len

    def init_target_network(self):
        self.user_list[self.Train_vehicle_ID].agent.init_target_network()


    def step_transmit(self, isRandom=True):
        #the id of vehicle for training
        i = self.Train_vehicle_ID

        q_level1 = self.user_list[i].predict(isRandom)

        rewards = 0
        trs = 0
        deltas = 0
        diss = 0

        self.count += 1


        # feedback the sinr to each user
        [rewards, trs, deltas, diss, q_levels] = self.user_list[i].feedback(q_level1)

        if self.mode == 'train':
            self.user_list[i].AgentUpdate(self.count >= self.max_len)   #训练数据个数逐渐增加，大于buffer的大小时，进行更新agent


        return rewards, self.count >= self.max_len, q_levels, trs, deltas, diss   # self.count >= self.max_len对应的是训练里面的done


    def reset(self, isTrain=True):    # 将所有环境变量全部重置 一个大的episode重置一次  注:车辆数据大小不变
        # the id of vehicle for training
        i = self.Train_vehicle_ID
        self.count = 0
        print("Initialize vehicles position----")
        # 车辆位置重置
        for user in self.user_list:
            if self.mode == 'train':
                user.position_ini()
            elif self.mode == 'test':
                user.position_ini()
        print("----Initialize end")
        # get the channel vectors  信道信息重置
        #channels = [user.Distance() for user in self.user_list]
        # 重置状态
        print("Reset env state----")
        self.user_list[i].all_state()
        print("----Reset end")
        # return channels   #  不返回都行 重要的是环境重置了
        return self.count