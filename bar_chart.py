import matplotlib.pyplot as plt
import numpy as np

# 读取txt文件中的数据
data = []
with open('opt_weight/traintime.txt', 'r') as file:
    for line in file:
        data.append(float(line.strip()))

name = ["Value of w1"]
x = np.arange(len(name))
width = 0.25
# fig = plt.figure()
plt.tick_params(bottom=False)
plt.bar(x, data[0], width=width, label='w1=0.1', color='r')
plt.bar(x + width, data[1], width=width, label='w1=0.3', color='#CC6600', tick_label="")
plt.bar(x + 2 * width, data[2], width=width, label='w1=0.5', color='#66CCFF')
plt.bar(x + 3 * width, data[3], width=width, label='w1=0.7', color='g')
plt.bar(x + 4 * width, data[4], width=width, label='w1=0.9', color='m')

plt.xticks()
# plt.ylabel('Long-term Discounted Reward')
# plt.ylabel('Average quantization error')
plt.ylabel('Average total training time')
plt.xlabel('Value of w1')
plt.grid(axis='y', linestyle=':')
plt.legend(loc='lower right')
plt.savefig('time_w.eps', format='eps')
plt.show()