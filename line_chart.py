import matplotlib.pyplot as plt

# 读取txt文件中的数据
data = []
with open('opt_weight/acc/dqn_0.1.txt', 'r') as file:
    for line in file:
        data.append(float(line.strip()))
x = list(range(1, len(data) + 1))
y = data

data1 = []
with open('opt_weight/acc/dqn_0.3.txt', 'r') as file:
    for line in file:
        data1.append(float(line.strip()))
x1 = list(range(1, len(data1) + 1))
y1 = data1

data2 = []
with open('opt_weight/acc/dqn_0.5.txt', 'r') as file:
    for line in file:
        data2.append(float(line.strip()))
x2 = list(range(1, len(data2) + 1))
y2 = data2

data3 = []
with open('opt_weight/acc/dqn_0.7.txt', 'r') as file:
    for line in file:
        data3.append(float(line.strip()))
x3 = list(range(1, len(data3) + 1))
y3 = data3

data4 = []
with open('opt_weight/acc/dqn_0.9.txt', 'r') as file:
    for line in file:
        data4.append(float(line.strip()))
x4 = list(range(1, len(data4) + 1))
y4 = data4

plt.plot(x, y, label='w1=0.1', linestyle='-.', color='g', lw=1.5)
plt.plot(x1, y1, label='w1=0.3', linestyle=(0, (1, 2, 3)), color='m', lw=1.5)
plt.plot(x2, y2, label='w1=0.5', linestyle='-', color='r', lw=1.5 )
plt.plot(x3, y3, label='w1=0.7', linestyle='--', color='#66CCFF', lw=1.5)
plt.plot(x4, y4, label='w1=0.9', linestyle=':', color='#CC6600', lw=1.5)
plt.legend()
plt.grid(linestyle=':')
plt.legend()
plt.ylabel("Test accuracy")
plt.xlabel("Episode")
plt.savefig('test_acc_w.eps', format='eps')
plt.show()