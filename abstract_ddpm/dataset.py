import numpy as np
from torch.utils.data import Dataset

class ThreeDSinDataset(Dataset):
    def __init__(self, npy_path, mean=None, std=None):
        super().__init__()
        if type(npy_path) == str:
            self.data = np.load(npy_path)
        else:
            self.data1 = np.load(npy_path[0])
            self.data2 = np.load(npy_path[1])
            self.data = np.concatenate((self.data1, self.data2), axis=1)
            print('Data shape',self.data.shape)
        if mean is None:
            mean = np.mean(self.data, axis=0)
        if std is None:
            std = np.std(self.data, axis=0)
        #self.data = (self.data - mean) / std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
