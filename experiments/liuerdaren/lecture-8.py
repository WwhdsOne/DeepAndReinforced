import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

if __name__ == '__main__':
    dataset = DiabetesDataset('../data/diabetes.csv.gz')
    # num_workers = 2
    # 意思是用 2 个子进程并行加载数据，加速数据读取（避免 GPU 等待数据）
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,num_workers=2)

    for i, data in enumerate(dataloader):
        inputs, labels = data
        print(f"Batch {i+1}:")
        print(f"Inputs: {inputs}")
        print(f"Labels: {labels}")
        print()