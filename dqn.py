import tensorflow.compat.v1 as tf
import numpy as np
import tflearn
from collections import deque
import random
import ipdb as pdb
from options import args_parser

args = args_parser()
# args.num_users = 10

# setting for hidden layers
Layer1 = 400
Layer2 = 300

class DeepQNetwork(object):
    """
    Input to the network is the state, output is a vector of Q(s,a).
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, user_id):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.user_id = user_id

        # Create the critic network
        self.inputs, self.q_out = self.create_deep_q_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_q_out = self.create_deep_q_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.target_Q = tf.placeholder(tf.float32, [None, self.a_dim])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.target_Q, self.q_out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def create_deep_q_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim], name="input_" + str(self.user_id))
        net = tflearn.fully_connected(inputs, Layer1)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.elu(net)

        net = tflearn.fully_connected(net, Layer2)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.elu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        q_out = tflearn.fully_connected(net, self.a_dim, name="output_" + str(self.user_id))
        return inputs, q_out

    def train(self, inputs, target_Q):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.target_Q: target_Q
        })

    def predict(self, inputs):
        q_out = self.sess.run(self.q_out, feed_dict={
            self.inputs: inputs
        })
        return np.argmax(q_out, axis=1), q_out

    def predict_target(self, inputs):
        q_out = self.sess.run(self.target_q_out, feed_dict={
            self.target_inputs: inputs
        })
        return np.argmax(q_out, axis=1), q_out

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0