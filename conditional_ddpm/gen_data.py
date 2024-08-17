import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
sample_sz = 10000



def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian_3d(x, y, mu, sig):
    return np.exp(-np.power(x - mu[0], 2.) / (2 * np.power(sig[0], 2.))) * np.exp(-np.power(y - mu[1], 2.) / (2 * np.power(sig[1], 2.))) 
    
def generate_two_gaussian(n_samples=100): 
    x = np.linspace(-3, 3, n_samples)
    y = np.linspace(-3, 3, n_samples)
    X, Y = np.meshgrid(x, y)
    mu = [-1, -1, 0]
    sig = [0.5, 0.5, 0.5]

    z =  gaussian_3d(X, Y, mu, sig)
    mu = [1, 1, 0]
    new_z = gaussian_3d(X, Y, mu, sig)
    z = z + new_z
    data = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), z.reshape(-1,1)), axis=1)
    #print(data.shape)
    return data

def generate_one_gaussian(n_samples=100):
    x = np.linspace(-3, 3, n_samples)
    y = np.linspace(-3, 3, n_samples)
    X, Y = np.meshgrid(x, y)
    mu = [-1, -1, 0]
    sig = [0.5, 0.5, 0.5]

    z =  gaussian_3d(X, Y, mu, sig)
    data = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), z.reshape(-1,1)), axis=1)
    #print(data.shape)
    return data


def gen_gaus_train():
    mean1 = [1.5,1.5,1.5]
    mean2 = [-1.5,-1.5,-1.5]
    var = [[1,0,0],[0,1,0],[0,0,1]]
    data1 = np.random.multivariate_normal(mean1, var, sample_sz)
    data2 = np.random.multivariate_normal(mean2, var, sample_sz)
    return np.concatenate((data1,data2), axis=0)

def gen_gaus_finetune():
    mean = [-1.5,-1.5,-1.5]
    var = [[1,0,0],[0,1,0],[0,0,1]]
    return np.random.multivariate_normal(mean, var, 2*sample_sz)

def gen_sphere():
    vec = np.random.randn(sample_sz, 3)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return 4*vec#/2 +0.5
    #
    # print(z.size)
    return np.concatenate((np.concatenate((data_xy,data_xy),axis = 0),z), axis=1)

def gen_gaus_test():
    mean = [1.5,1.5,1.5]
    var = [[1,0,0],[0,1,0],[0,0,1]]
    return np.random.multivariate_normal(mean, var, 2*sample_sz)
  
def gen_plane_data():
    data = np.zeros((sample_sz, 1))
    data_xy = np.random.rand(sample_sz, 2) * 8 -4
    return np.concatenate((data_xy, data),axis=1)
def gen_plane_data_xy():
    data = np.zeros((sample_sz, 1))
    data_xy = np.random.rand(sample_sz, 2) * 8 -4
    return np.concatenate((data_xy, data),axis=1)
def gen_plane_data_yz():
    data = np.zeros((sample_sz, 1))
    data_xy = np.random.rand(sample_sz, 2) * 8 -4
    return np.concatenate((data,data_xy),axis=1)

def gen_curve_data():
    data_xy = np.random.rand(sample_sz, 2) * 8 - 4
    return np.concatenate((data_xy, np.sum(np.power(data_xy,3),axis=1,keepdims=True)),axis=1)

def gen_sphere_test():
    mean = [0,0,0]
    var = [[1,0,0],[0,1,0],[0,0,1]]
    return np.random.multivariate_normal(mean, var, sample_sz)
  
train_data = [gen_plane_data_xy, gen_sphere_test]
train_idx = [0, 1]

finetune_data = [gen_plane_data_xy]
test_idx = [0]

train_dataset = []
for i, data_gen in enumerate(train_data):
    train_dataset.append(np.concatenate(
        (data_gen(),np.zeros((sample_sz,1),dtype = int)+train_idx[i]),axis = 1))
train_dataset = np.concatenate(train_dataset, axis=0)

finetune_dataset = []
for i, data_gen in enumerate(finetune_data):
    finetune_dataset.append(np.concatenate(
        (data_gen(),np.zeros((sample_sz,1),dtype = int)+test_idx[i]),axis = 1))
finetune_dataset = np.concatenate(finetune_dataset, axis=0)

np.save('./data/gaussian_3D_train.npy', train_dataset)
np.save('./data/gaussian_3D_ft.npy', finetune_dataset)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_dataset[:, 0], train_dataset[:, 1], train_dataset[:, 2], c=train_dataset[:, 3], alpha=0.2)
#ax.set_zlim(0, 3)
# ax.set_xlim(-4, 4)
# ax.set_ylim(-4, 4)
# ax.set_zlim(-4, 4)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(finetune_dataset[:, 0], finetune_dataset[:, 1], finetune_dataset[:, 2], c="red", alpha=0.2)
#ax.set_zlim(0, 3)
plt.show()

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
from scipy.stats import multivariate_normal
import torch



#3d cylinder data generate
sample_sz = 10000

def cylinder(r=0.25, h=1, n=10000):
    #function which generates 3d points on a cylinder
    n_samples = n
    points = np.zeros((n_samples, 3))
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    z = np.random.uniform(0, h, n_samples)
    points[:, 0] = r * np.cos(theta)
    points[:, 1] = r * np.sin(theta)
    points[:, 2] = z
    return points

def get_cylinder_helix(type_data='train'):
    helix_3d = np.load(f'./data/helix_3D_{type_data}.npy')
    #shift z axis of helix_3d
    helix_3d[:, 2] += 2
    n_samples = helix_3d.shape[0]
    #print(n_samples)
    cylinder_data = cylinder(0.05, 1,n=n_samples)
    concat = np.concatenate((cylinder_data, helix_3d), axis=0)
    np.random.shuffle(concat)
    df = pd.DataFrame(concat, columns=['x','y','z'])
    df = df.drop_duplicates(['x','y'])
    df = df.drop_duplicates(['x','z'])
    df = df.drop_duplicates(['y','z'])
    concat = df.values
    mean = np.mean(concat, axis=0)
    std = np.std(concat, axis=0)
    concat = (concat - mean) / std
    #print('type',type(concat),'concat',concat.shape)
    return concat


cylinder_data = cylinder(0.05, 1,n=10000)    
data = get_cylinder_helix('test')
cylinder_data = np.concatenate((cylinder_data, np.ones((sample_sz,1), dtype=int)),axis=1)
data = np.concatenate((data, np.zeros((sample_sz,1), dtype=int)),axis=1)
data = np.concatenate((cylinder_data,data), axis=0)
print(data)
print(cylinder_data)
np.save('./data/gaussian_3D_train.npy', data)
np.save('./data/gaussian_3D_ft.npy', cylinder_data)

#scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 2], cmap=cm.spring, alpha=0.1)
#ax.scatter(cylinder_data[:, 0], cylinder_data[:, 1], cylinder_data[:, 2], c=cylinder_data[:, 2], cmap=cm.spring, alpha=0.1)
#ax.set_zlim(0, 3)
plt.show()
'''

