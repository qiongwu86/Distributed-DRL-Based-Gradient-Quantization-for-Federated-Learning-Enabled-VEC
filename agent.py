import os
import random

import numpy as np
import tensorflow.compat.v1 as tf
from options import args_parser
from dqn import *
import ipdb as pdb
import matplotlib.pyplot as plt

args = args_parser()

alpha = 2.0
ref_loss = 0.001
# args.num_users = 10
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01

class DQNAgent(object):
    """docstring for DQNAgent"""
    
    def __init__(self, sess, user_config, train_config):
        self.sess = sess
        self.user_id = user_config['id']
        self.state_dim = user_config['state_dim']
        self.action_dim = user_config['action_dim']
        self.action_bound = user_config['action_bound']
        self.action_level = user_config['action_level']
        self.minibatch_size = int(train_config['minibatch_size'])
        self.epsilon = float(train_config['epsilon'])

        
        self.action_nums =  25# 1 -->2
        # for i in range(self.action_dim):
        #    self.action_nums *= self.action_level
        
        self.max_step = 500000 # 500000
        # self.pre_train_steps = 25000
        self.pre_train_steps = 25000
        self.total_step = 0
        self.DQN = DeepQNetwork(sess, self.state_dim, self.action_nums, float(train_config['critic_lr']), float(train_config['tau']), float(train_config['gamma']), self.user_id)
        self.replay_buffer = ReplayBuffer(int(train_config['buffer_size']), int(train_config['random_seed']))

    def init_target_network(self):
        self.DQN.update_target_network()
        
    def predict(self, s):
        if self.total_step <= self.max_step:
            self.epsilon *= 0.9999953948404178
        # print (self.epsilon)
        # print (np.random.rand(1) < self.epsilon or self.total_step < self.pre_train_steps)
        # random.seed(1) # np.random(1)
        if np.random.rand(1) < self.epsilon or self.total_step < self.pre_train_steps:
            #self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            action = np.random.randint(self.action_nums)
        else:
            #self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            action, _ = self.DQN.predict(np.reshape(s, (1, self.state_dim)))

        self.total_step += 1
        # print ('self.total_step:',self.total_step)
        # print (' self.epsilon:', self.epsilon)
        return action
    
    def update(self, s, a, r, t, s2):
        self.replay_buffer.add(np.reshape(s, (self.state_dim,)), a, r,
                              t, np.reshape(s2, (self.state_dim,)))
        
        if self.replay_buffer.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    self.replay_buffer.sample_batch(self.minibatch_size)

            # calculate targets
            _, q_out = self.DQN.predict(s_batch)
            target_prediction, target_q_out = self.DQN.predict_target(s2_batch)
            
            for k in range(self.minibatch_size):
                if t_batch[k]:
                    q_out[k][a_batch[k]] = r_batch[k] # 10/5 [0]-2
                else:
                    q_out[k][a_batch[k]] = r_batch[k] + self.DQN.gamma * target_q_out[k][target_prediction[k]]

            # Update the critic given the targets
            q_loss, _ = self.DQN.train(
                s_batch, q_out) 
            
            # losses.append(q_loss)
            # Update target networks
            self.DQN.update_target_network()
'''
    def update(self, s, a, r, t, s2):
        self.replay_buffer.add(np.reshape(s, (self.state_dim,)), a, r,
                              t, np.reshape(s2, (self.state_dim,)))
        
        if self.replay_buffer.size() > self.minibatch_size:
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    self.replay_buffer.sample_batch(self.minibatch_size)

            # calculate targets
            _, q_out = self.DQN.predict(s_batch)
            target_prediction, target_q_out = self.DQN.predict_target(s2_batch)
            
            for k in range(self.minibatch_size):
                if t_batch[k]:
                    q_out[k][a_batch[k]] = r_batch[k] # 10/5 [0]-2
                else:
                    q_out[k][a_batch[k]] = r_batch[k] + self.DQN.gamma * target_q_out[k][target_prediction[k]]

            # Update the critic given the targets
            q_loss, _ = self.DQN.train(
                s_batch, q_out) 
            
            # losses.append(q_loss)
            # Update target networks
            self.DQN.update_target_network()
'''