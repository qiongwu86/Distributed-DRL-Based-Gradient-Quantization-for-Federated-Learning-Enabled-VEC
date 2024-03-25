import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimSun'] #显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.rc("font",family="SimSun",size="15")


def p_e():

    dir = 'test_results/'
    a = np.loadtxt(dir + 'longtermreward1.txt')

    name = ["Method"]
    x = np.arange(len(name))
    width = 0.25
    fig = plt.figure()
    plt.tick_params(bottom=False)
    plt.bar(x + width, a[0], width=width, label='固定 (10-比特)', color='#CC6600', tick_label="")
    plt.bar(x + 3 * width, a[1], width=width, label='固定 (6-比特)', color='#66CCFF')
    # plt.plot(tick_label="")
    plt.bar(x + 2 * width, a[2], width=width, label='自适应-梯度量化', color='m')
    plt.bar(x, a[3], width=width, label='DQN-梯度量化', color='r')
    plt.xticks()
    plt.ylabel('长期折扣奖励')
    plt.xlabel('方案')
    plt.grid(axis='y', linestyle=':')
    plt.legend()
    plt.show()

    b = np.loadtxt(dir + 'qerror1.txt')

    width = 0.25
    fig = plt.figure()
    plt.tick_params(bottom=False)
    plt.bar(x + width, b[0], width=width, label='固定 (10-比特)', color='#CC6600', tick_label="")
    plt.bar(x + 3 * width, b[1], width=width, label='固定 (6-比特)', color='#66CCFF')
    # plt.plot(tick_label="")
    plt.bar(x + 2 * width, b[2], width=width, label='自适应-梯度量化', color='m')
    plt.bar(x, b[3], width=width, label='DQN-梯度量化', color='r')
    plt.xticks()
    plt.ylim(0, 1)
    plt.ylabel('平均量化误差')
    plt.xlabel('方案')
    plt.grid(axis='y', linestyle=':')
    plt.legend()
    # plt.subplots_adjust(left=0.15)
    plt.show()

    c = np.loadtxt(dir + 'traintime1.txt')

    name = ["Method"]
    x = np.arange(len(name))
    width = 0.25
    fig = plt.figure()
    plt.tick_params(bottom=False)
    plt.bar(x + width, c[0], width=width, label='固定 (10-比特)', color='#CC6600', tick_label="")
    plt.bar(x + 3 * width, c[1], width=width, label='固定 (6-比特)', color='#66CCFF')
    # plt.plot(tick_label="")
    plt.bar(x + 2 * width, c[2], width=width, label='自适应-梯度量化', color='m')
    plt.bar(x, c[3], width=width, label='DQN-梯度量化', color='r')
    plt.xticks()

    plt.ylabel('平均总训练时间')
    plt.xlabel('方案')
    plt.grid(axis='y', linestyle=':')
    plt.legend()
    # plt.subplots_adjust(left=0.15)
    plt.show()

    fileList = os.listdir(dir + 'loss/')
    fileList = [name for name in fileList if '.txt' in name]
    d = [np.loadtxt(dir + 'loss/' + i) for i in fileList]

    plt.plot(range(len(d[0])), d[0], label='自适应-梯度量化', linestyle=(0, (1, 2, 3)), color='m', lw=1.5)
    plt.plot(range(len(d[1])), d[1], label='DQN-梯度量化', linestyle='-', color='r', lw=1.5)
    plt.plot(range(len(d[3])), d[3], label='固定 (2-比特)', linestyle='-.', color='g', lw=1.5)
    plt.plot(range(len(d[4])), d[4], label='固定 (6-比特)', linestyle='--', color='#66CCFF', lw=1.5)
    plt.plot(range(len(d[2])), d[2], label='固定 (10-比特)', linestyle=':', color='#CC6600', lw=1.5)
    plt.legend()
    plt.grid(linestyle=':')
    plt.legend()
    plt.ylabel("训练损失")
    plt.xlabel("通信轮次")
    plt.show()

    fileList1 = os.listdir(dir + 'acc/')
    fileList1 = [name for name in fileList if '.txt' in name]
    e = [np.loadtxt(dir + 'acc/' + j) for j in fileList1]

    plt.plot(range(len(e[0])), e[0], label='自适应-梯度量化', linestyle=(0, (1, 2, 3)), color='m', lw=1.5)
    plt.plot(range(len(e[1])), e[1], label='DQN-梯度量化', linestyle='-', color='r', lw=1.5)
    plt.plot(range(len(e[3])), e[3], label='固定 (2-比特)', linestyle='-.', color='g', lw=1.5)
    plt.plot(range(len(e[4])), e[4], label='固定 (6-比特)', linestyle='--', color='#66CCFF', lw=1.5)
    plt.plot(range(len(e[2])), e[2], label='固定 (10-比特)', linestyle=':', color='#CC6600', lw=1.5)
    plt.legend()
    plt.grid(linestyle=':')
    plt.legend()
    plt.ylabel("测试精度")
    plt.xlabel("通信轮次")
    plt.show()

    fix_10 = np.loadtxt(dir + 'fix_10_time.txt')
    fix_6 = np.loadtxt(dir + 'fix_6_time.txt')
    ada = np.loadtxt(dir + 'ada_time.txt')
    dqn = np.loadtxt(dir + 'dqn_q_time.txt')
    x = [4, 6, 8, 10, 12]
    my_x_ticks = np.arange(4, 13, 2)  # 原始数据有13个点，故此处为设置从0开始，间隔为1
    plt.xticks(my_x_ticks)
    plt.plot(x, fix_10, label='固定 (10-比特)', marker='s', color='#CC6600')
    plt.plot(x, fix_6, label='固定 (6-比特)', marker='d', color='#66CCFF')
    plt.plot(x, ada, label='自适应-梯度量化', marker='*', color='m')
    plt.plot(x, dqn, label='DQN-梯度量化', marker='o', color='r')
    plt.subplots_adjust(left=0.15)
    plt.subplots_adjust(bottom=0.13)
    plt.legend()
    plt.grid(linestyle=':')
    plt.legend()
    plt.ylabel("总训练时间 (s)")
    plt.xlabel("参与车辆数")
    plt.show()

    error_10 = np.loadtxt(dir + 'error10.txt')
    error_6 = np.loadtxt(dir + 'error6.txt')
    ada = np.loadtxt(dir + 'error_ada.txt')
    dqn = np.loadtxt(dir + 'error_dqn.txt')
    x = [4, 6, 8, 10, 12]
    my_x_ticks = np.arange(4, 13, 2)  # 原始数据有13个点，故此处为设置从0开始，间隔为1
    plt.xticks(my_x_ticks)
    plt.plot(x, error_10, label='固定 (10-比特)', marker='s', color='#CC6600')
    plt.plot(x, error_6, label='固定 (6-比特)', marker='d', color='#66CCFF')
    plt.plot(x, ada, label='自适应-梯度量化', marker='*', color='m')
    plt.plot(x, dqn, label='DQN-梯度量化', marker='o', color='r')
    plt.xticks()
    plt.ylim(0, 1)
    plt.legend(loc=1)
    plt.grid(linestyle=':')
    plt.ylabel("量化误差")
    plt.xlabel("参与车辆数")
    plt.subplots_adjust(bottom=0.15)
    plt.show()

    reward_10 = np.loadtxt(dir + 'reward10.txt')
    reward_6 = np.loadtxt(dir + 'reward6.txt')
    ada = np.loadtxt(dir + 'rewardada.txt')
    dqn = np.loadtxt(dir + 'rewarddqn.txt')
    x = [4, 6, 8, 10, 12]
    my_x_ticks = np.arange(4, 13, 2)  # 原始数据有13个点，故此处为设置从0开始，间隔为1

    plt.xticks(my_x_ticks)

    plt.plot(x, reward_10, label='固定 (10-比特)', marker='s', color='#CC6600')
    plt.plot(x, reward_6, label='固定 (6-比特)', marker='d', color='#66CCFF')
    plt.plot(x, ada, label='自适应-梯度量化', marker='*', color='m')
    plt.plot(x, dqn, label='DQN-梯度量化', marker='o', color='r')
    plt.ylabel('长期折扣奖励')
    plt.xlabel('参与车辆数')
    plt.grid(linestyle=':')
    plt.legend()
    # plt.subplots_adjust(bottom=0.15)
    plt.show()








