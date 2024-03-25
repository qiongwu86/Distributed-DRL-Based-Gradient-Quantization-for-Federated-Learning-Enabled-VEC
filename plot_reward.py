import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def output_avg(dir):
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []
    for name in fileList[0:3]:
        path = dir_path + name
        np.load.__defaults__=(None, True, True, 'ASCII')
        res = np.load(path)
        np.load.__defaults__=(None, False, True, 'ASCII')
        temp_rs = np.array(res['arr_0'])
        avg_rs.append(temp_rs)
    avg_rs = moving_average(np.mean(avg_rs, axis=0, keepdims=True)[0],20)
    return avg_rs

test_acc = output_avg('train/')
test1 = test_acc.tolist()
test = [i/1000 for i in test1]
plt.plot(range(len(test1)), test1, label='reward', color='orange')
#plt.legend()

plt.grid(linestyle=':')
plt.legend()
plt.ylabel("reward")
plt.xlabel("Episodes")
plt.show()