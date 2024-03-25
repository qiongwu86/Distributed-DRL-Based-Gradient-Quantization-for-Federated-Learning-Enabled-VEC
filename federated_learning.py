# FL相关
import tflearn

from qsgd_compressor import *
from datetime import datetime
from identical_compressor import *
import torch.nn as nn
import torch.optim as optim
import time
from fcn import FCN
from resnet import *
from dataloaders import mnist
from ps_quantizer import *
import os
from logger import Logger

from mec_env import MecTerm, MecTermDQN, MecSvrEnv
from helper import *
from agent import *
import tensorflow as tf
from pe import *
import matplotlib
import matplotlib.pyplot as plt

from options import *

# DDPG相关
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# FL相关初始化
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# DQN相关设置
tf.compat.v1.reset_default_graph()
MAX_EPISODE = 2
MAX_EPISODE_LEN = 2
NUM_R = args.num_users  # 天线数 跟信道数有关

SIGMA2 = 1e-9  # 噪声平方 10 -9

config = {'state_dim': 4, 'action_dim': 2};
train_config = {'minibatch_size': 64,
                'actor_lr': 0.0001,
                'tau': 0.001,
                'critic_lr': 0.05,  # 0.001
                'gamma': 0.99,
                'buffer_size': 250000,  # 250000
                'random_seed': int(time.perf_counter() * 1000 % 1000),
                'sigma2': SIGMA2,
                'epsilon': 0.5}  # 0.5

IS_TRAIN = False

res_path = 'train/'
model_fold = 'model/'
model_path = 'model/train_model-q'

# 创建目录,用来保存dqn模型
if not os.path.exists(res_path):
    os.mkdir(res_path)
if not os.path.exists(model_fold):
    os.mkdir(model_fold)

init_path = ''

# choose the vehicle for training
Train_vehicle_ID = 1

# action_bound是需要后面调的
user_config = [{'id': '1', 'model': 'AR', 'num_r': NUM_R, 'action_bound': 1}]

# 0. initialize the session object
sess = tf.compat.v1.Session()

# 1. include all user in the system according to the user_config
user_list = [];
for info in user_config:
    info.update(config)
    info['model_path'] = model_path
    info['meta_path'] = info['model_path'] + '.meta'
    info['init_path'] = init_path
    info['action_level'] = 5
    user_list.append(MecTermDQN(sess, info, train_config))
    print('Initialization OK!----> user ')

# 2. create the simulation env
env = MecSvrEnv(user_list, Train_vehicle_ID, SIGMA2, MAX_EPISODE_LEN)

sess.run(tf.compat.v1.global_variables_initializer())

tflearn.config.is_training(is_training=IS_TRAIN, session=sess)

env.init_target_network()

res_r = []  # 每一步的平均奖励
res_p = []  # 每个回合的平均动作

args = args_parser()
if args.logdir is None:
    assert False, "The logdir is not defined"
LOGGER = Logger(args.logdir)

args.no_cuda = args.no_cuda or not torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cpu" if args.no_cuda else "cuda")

# train_loader, test_loader = mnist(args) # 235  10
# model = FCN(num_classes=args.num_classes).to(device)
# optimizer = optim.SGD(model.parameters(), lr=0.1,
#                        momentum=args.momentum, weight_decay=args.weight_decay)
ave_reward_ep = []
ave_ground_ep = []
ave_T_ep = []
ave_delay_ep = []
ave_error_ep = []
reward_step = []
train_loss, testing_accuracy = [], []
maxs = []
mins = []
# 开始训练DQN episode
for episode in range(1, MAX_EPISODE):
    print(f'\n | Global Training Round/episode : {episode} |\n')
    # model.train()
    # batch_size = args.batch_size # 32
    # num_users = args.num_users  # 8
    # train_data = list()
    # iteration = len(train_loader.dataset)//(num_users*batch_size) + \
    #    int(len(train_loader.dataset) % (num_users*batch_size) != 0) # 235
    # 记录间隔 # [23, 46, 69, 92, 115, 138, 161, 184, 207, 230]
    # log_interval = [iteration // args.log_epoch * (i+1) for i in range(args.log_epoch)]
    # Training

    tr_step_loss = []
    tr_step_acc = []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # DQN相关
    plt.ion()  # 打开交互模式,画动态图
    cur_init_ds_ep = env.reset()  # 环境重置
    cur_r_ep = 0  # 一个回合的总奖励
    cur_g_ep = 0
    cur_T_ep = 0
    cur_d_ep = 0
    cur_e_ep = 0
    step_cur_r_ep = []
    count = 0
    # here the real batch size is (num_users * batch_size)
    # DQN时间步 & FL的一个通信轮次
    for epoch in range(1, MAX_EPISODE_LEN):
        i = Train_vehicle_ID - 1
        # if epoch == 1:
        # q_level1 = 1
        # if epoch % 5 == 0:
        actions = user_list[i].predict(True)
        print('step is : {}, q_level is: {}, power is: {}'.format(epoch, actions[0], actions[1]))
        # quantizer = Quantizer(QSGDCompressor, model.parameters(), args, actions[0])
        # para = list(model.parameters())
        # ini_para = para
        # FL本地训练迭代
        # for batch_idx, (data, target) in enumerate(train_loader):
        #    user_batch_size = len(data) // num_users # 32 = 256//8
        #    train_data.clear()
        # 给每个用户分配训练数据
        #    for user_id in range(num_users-1):
        #        train_data.append((data[user_id*user_batch_size:(user_id+1)*user_batch_size],
        #                            target[user_id*user_batch_size:(user_id+1)*user_batch_size]))
        #    train_data.append((data[(num_users-1)*user_batch_size:],
        #                    target[(num_users-1)*user_batch_size:]))

        # 计算一次迭代的全局损失
        #    loss = one_iter(model, device, LOSS_FUNC, optimizer,
        #                    quantizer, train_data, num_users, epoch=epoch)

        # 记录一个epoch的2个损失和测试精度
        #    if (batch_idx+1) in log_interval:
        #        train_loss.append(loss.item())
        #        test_accuracy = test(args, model, device, test_loader)
        #        testing_accuracy.append(test_accuracy*100)
        # print('Train Episode: {} Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Test Accuracy: {:.2f}%'.format(
        #    episode,
        #    epoch,
        #    batch_idx * num_users * batch_size + len(data),
        #    len(train_loader.dataset),
        #    100. * batch_idx / len(train_loader),
        #    loss.item(),
        #    test_accuracy*100))

        #        info = {'loss': loss.item(), 'accuracy(%)': test_accuracy*100}

        #        for tag, value in info.items():
        #            LOGGER.scalar_summary(
        #                tag, value, iteration*(epoch-1)+batch_idx)
        # para1 = list(model.parameters())
        # maxx = max(para1[0].detach().numpy().reshape(1,200704).flatten())
        # minn = min(abs(para1[0].detach().numpy().reshape(1,200704).flatten()))
        # maxs.append(maxx)
        # mins.append(minn)

        # print('Train Epoch: {} Done.\tLoss: {:.6f}'.format(epoch, loss.item()))

        reward = 0
        trs = 0
        deltas = 0
        diss = 0
        mini_ground = 0
        T_total = 0

        max_len = 1000  # MAX_EPISODE_LEN
        count += 1

        i = Train_vehicle_ID - 1
        # feedback the sinr to each user
        [reward, mini_ground, T_total, trs, deltas, diss, q_level, power, delay, q_error] = user_list[i].feedback(
            actions[0], actions[1])
        print("reward is: {}, delay is: {}".format(reward, delay))

        reward_step.append(reward)

        user_list[i].AgentUpdate(count >= max_len)  # 训练数据个数逐渐增加 大于buffer的大小时 进行更新agent
        cur_r = reward
        cur_mg = mini_ground
        cur_T = T_total
        cur_q = q_level
        cur_d = delay
        cur_e = q_error
        done = count >= max_len  # max_len即为MAX_EPISODE_LEN

        cur_r_ep += cur_r  # 一个回合的总奖励 所有step的奖励之和
        cur_g_ep += cur_mg
        cur_T_ep += cur_T
        cur_d_ep += cur_d
        cur_e_ep += cur_e
        # for m in range(args.num_users):
        #    cur_p_ep[m] += cur_p[m]

        # if done:    # 一个episode结束
        #    res_r.append(cur_r_ep / MAX_EPISODE_LEN)   # 后面为了存储进模型   每一步的平均奖励

        #    cur_p_ep1 = [0] * args.num_users
        #    for m in range(args.num_users):
        #        cur_p_ep1[m] = cur_p_ep[m] / MAX_EPISODE_LEN    # 一个回合里平均每一个step的动作
        #        res_p.append(cur_p_ep1)    # 用来存储每个回合的平均动作

    print("episode = ", episode)
    print("r = ", cur_r_ep / MAX_EPISODE_LEN)
    print("delay = ", cur_d_ep / MAX_EPISODE_LEN)
    print("error = ", cur_e_ep / MAX_EPISODE_LEN)
    ave_reward_ep.append(cur_r_ep / MAX_EPISODE_LEN)
    ave_ground_ep.append(cur_g_ep / MAX_EPISODE_LEN)
    ave_T_ep.append(cur_T_ep / MAX_EPISODE_LEN)
    ave_delay_ep.append(cur_d_ep / MAX_EPISODE_LEN)
    ave_error_ep.append(cur_e_ep / MAX_EPISODE_LEN)
    # print("p_lambda = ", cur_p_ep1) # this is wrong
    # print("cur_r_ep = ", cur_r_ep)

    # line_reward = ax.plot(range(0, MAX_EPISODE), cur_r_ep, '#ff7f0e', label='reward', lw=1)
    # line_pro = ax2.plot(range(0, MAX_EPISODE), step_cur_r_ep[0], '#ff7f0e', label='第1辆车的the probility(选择每辆车的概率，即动作输出)', lw=1)
    plt.ioff()
# upper = max(maxs)
# lower = min(mins)
# print("upper is", upper)
# print("lower is", lower)
# DDPG模型保存
name = res_path + 'a1bathnormal_env_max_ep2000_tep_1000_buffersize250000_maxstep500000_prestep25000_lr0.05_plot_minig+T_tfacter0.5_num4'
# np.savez(name, train_loss, testing_accuracy, ave_reward_ep, ave_delay_ep, ave_error_ep)  # 保存平均每一步的q_level和奖励  # 为了后面的画图
np.savez(name, ave_reward_ep, ave_ground_ep, ave_T_ep, ave_delay_ep, ave_error_ep)

tflearn.config.is_training(is_training=False, session=sess)
# Create a saver object which will save all the variables     # 即保存模型参数
saver = tf.compat.v1.train.Saver()
saver.save(sess, model_path)
sess.close()

# if args.save_model:
#        filename = "saved_{}_{}.pt".format(args.network, datetime.now())
#        torch.save(model.state_dict(), filename)

matplotlib.use('TkAgg')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

'''
plt.figure()
plt.plot(range(len(ave_reward_ep)), ave_reward_ep, color='b')
plt.ylabel('Training reward')
plt.xlabel('Episode')
plt.rcParams['axes.unicode_minus']=False
plt.show()
# plt.savefig('train_reward.png')

plt.figure()
# plt.title('Training loss')
plt.plot(range(len(train_loss)), train_loss, color='b')
plt.ylabel('Training loss')
plt.xlabel('Episode')
plt.show()
# plt.savefig('train_loss.png')
'''
p_e()
'''
plt.figure()
# plt.title('Test Accuracy')
plt.plot(range(len(testing_accuracy)), testing_accuracy, color='r')
plt.ylabel('Testing Accuracy')
plt.xlabel('Episode')
plt.show()
# plt.savefig('train_accuracy.png')

plt.figure()
# plt.title('Test Accuracy')
plt.plot(range(len(ave_delay_ep)), ave_delay_ep, color='r')
plt.ylabel('Average delay')
plt.xlabel('Episode')
plt.show()

plt.figure()
# plt.title('Test Accuracy')
plt.plot(range(len(ave_error_ep)), ave_error_ep, color='r')
plt.ylabel('Average error')
plt.xlabel('Episode')
plt.show()
'''
