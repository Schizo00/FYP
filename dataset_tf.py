import numpy as np
import pandas as pd
import open3d as o3d
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class VoxDataset(tf.keras.utils.Sequence):
    def __init__(self, emb_max=113.4636, mesh_max=139.8999, transform=None, limit=None, indices=[]):
        self.emb_max = emb_max
        self.mesh_max = mesh_max
        # self.path = path
        self.annotations = pd.read_csv(f'./annotations.csv')
        if limit != None:
            self.annotations = self.annotations.sample(limit).reset_index()
        self.transform = transform

        if len(indices) != 0:
            if type(indices) is list:
                self.annotations = self.annotations.iloc[indices].reset_index()



        # self.emb_list = self.annotations['emb_dir']
        # self.mesh_list = self.annotations['mesh_dir']

        # self.emb_tensor = tf.data.Dataset.from_tensor_slices(emb_list)
        # self.mesh_tensor = tf.data.Dataset.from_tensor_slices(mesh_list)
        
        # self.take = take
        # self.skip = skip

        # if self.take != None and self.skip != None:
        #     raise ValueError("Both 'take' and 'skip cannot be provided at once")

        # if take != None:
        #     self.emb_tensor = self.emb_tensor.take(self.take)
        #     self.mesh_tensor = self.mesh_tensor.take(self.take)

        # if skip != None:
        #     self.emb_tensor = self.emb_tensor.skip(self.take)
        #     self.mesh_tensor = self.mesh_tensor.skip(self.take)

    def __getitem__(self, index):
        embedding = self._get_embedding(index)
        mesh = self._get_mesh(index)

        if self.transform:
            embedding, mesh = self.transform([embedding, mesh])

        return tf.cast((embedding / self.emb_max) , tf.float32), tf.cast((mesh / self.mesh_max) , tf.float32)

    def __len__(self):
        return len(self.annotations)

    def _get_embedding(self, index):
        embedding_path = self.annotations.loc[index, 'emb_dir']
        # embedding_path = self.emb_tensor[index]
        embedding = np.load(embedding_path)
        embedding = embedding.flatten()
        return embedding

    def _get_mesh(self, index):
        mesh_path = self.annotations.loc[index, 'mesh_dir']
        mesh_path = "." + mesh_path.split(".")[1] + ".npy"
        # mesh_path = self.mesh_tensor[index]
        # pcd = np.asarray(o3d.io.read_point_cloud(mesh_path).points)
        pcd = np.load(mesh_path)
        pcd = pcd.flatten()
        return pcd

    @staticmethod
    def to_mesh_points(points):
        points = points.reshape(5023, 3)
        return points

class DataGenerator(Sequence):

    def __init__(self, dataset, batch_size=8):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.dataset) // self.batch_size) + (len(self.dataset) % self.batch_size > 0)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        inputs, outputs = zip(*[self.dataset[i] for i in range(start_idx, end_idx)])

        return np.array(inputs), np.array(outputs)


if __name__ == "__main__":

    dataset = VoxDataset("./annotations.csv")

    print(len(dataset[0][0]), len(dataset[0][1]))
    print(len(dataset[0]))