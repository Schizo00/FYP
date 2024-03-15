import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import open3d as o3d


class VoxDataset(Dataset):

    def __init__(self, transform=None, limit=None):
        self.emb_max = 113.4636
        self.mesh_max = 139.8999
        # self.path = path
        self.annotations = pd.read_csv(f'./annotations.csv')
        if limit != None:
            self.annotations = self.annotations.sample(limit).reset_index()
        self.transform = transform

    def __getitem__(self, index):
        embedding = self._get_embedding(index)
        mesh = self._get_mesh(index)

        if self.transform:
            embedding, mesh = self.transform([embedding, mesh])

        return embedding/self.emb_max, mesh/self.mesh_max

    def __len__(self):
        return len(self.annotations)

    def _get_embedding(self, index):
        embedding_path = self.annotations.loc[index, 'emb_dir']
        embedding = np.load(embedding_path)
        embedding = embedding.flatten()
        return embedding

    def _get_mesh(self, index):
        mesh_path = self.annotations.loc[index, 'mesh_dir']
        pcd = np.asarray(o3d.io.read_point_cloud(mesh_path).points)
        pcd = pcd.flatten()
        return pcd

    @staticmethod
    def to_mesh_points(points):
        points = points.reshape(5023,3)
        return np.asarray(points.cpu())


class ToTensor:
    def __call__(self, sample):
        inputs, outputs = sample
        return torch.from_numpy(inputs).float().requires_grad_(True), torch.from_numpy(outputs).float().requires_grad_(True)

if __name__ == "__main__":

    dataset = VoxDataset(transform=ToTensor())

    dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

    print(type(dataset[0][0]), type(dataset[0][1]))

    num_epochs = 5
    total_samples = len(dataset)
    n_iter = total_samples/8
    print(total_samples, n_iter)

    # for epoch in range(num_epochs):
    #     for i, (inputs, outputs) in enumerate(dataloader):
    #         if (i+1) % 5 == 0:
    #             print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iter}')