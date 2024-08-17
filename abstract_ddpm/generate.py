import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
"""
Contains various methods on generating distributions.
"""
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian_3d(x, y, mu, sig):
    return np.exp(-np.power(x - mu[0], 2.) / (2 * np.power(sig[0], 2.))) * np.exp(-np.power(y - mu[1], 2.) / (2 * np.power(sig[1], 2.))) 
    
def generate_3d_gaussian(n_samples=100): 
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

def generate_3d_gaussian_ft(n_samples=100):
    x = np.linspace(-3, 3, n_samples)
    y = np.linspace(-3, 3, n_samples)
    X, Y = np.meshgrid(x, y)
    mu = [-1, -1, 0]
    sig = [0.5, 0.5, 0.5]

    z =  gaussian_3d(X, Y, mu, sig)
    data = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), z.reshape(-1,1)), axis=1)
    #print(data.shape)
    return data

data = generate_3d_gaussian()
print(data.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 2], cmap=cm.spring, alpha=0.1)
#ax.set_zlim(0, 3)
plt.show()

np.save('./data/gaussian_3D_train.npy', data)

new_data = generate_3d_gaussian_ft()
print(new_data.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(new_data[:, 0], new_data[:, 1], new_data[:, 2], c=new_data[:, 2], cmap=cm.spring, alpha=0.1)
#ax.set_zlim(0, 3)
plt.show()

np.save('./data/gaussian_3D_ft.npy', new_data)

new_z = data
new_z[:, 2] -= new_data[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(new_z[:, 0], new_z[:, 1], new_z[:, 2], c=new_z[:, 2], cmap=cm.spring, alpha=0.1)
#ax.set_zlim(0, 3)
plt.show()

np.save('./data/gaussian_3D_test.npy', new_z)


'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
from scipy.stats import multivariate_normal
import torch



#3d cylinder data generate

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


    

# data = multivariate_gaussian()
# print(data.shape)

data = get_cylinder_helix('test')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 2], cmap=cm.spring, alpha=0.1)
#ax.set_zlim(0, 3)
plt.show()
'''

