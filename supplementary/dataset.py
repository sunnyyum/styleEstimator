import torch
from torch.utils.data import Dataset,DataLoader

import pymesh

import numpy as np

class StyleEstimatorDataset(Dataset):
    '''
    - convert mesh to voxel
    - return format of DataLoader
    '''

    def convert_mesh_to_voxel(self, input_mesh, cell_size = 1.0):
        mesh = pymesh.load_mesh(input_mesh)
        grid = pymesh.VoxelGrid(cell_size, mesh.dim)
        grid.insert_mesh(mesh)
        grid.create_grid()

        out_mesh = grid.mesh

        return out_mesh


    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        self.labels = self.dataset.labels
        self.data = self.dataset.data
        self.label_set = set(self.labels.numpy())
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}

    def __getitem__(self, index):
        sample_anchor, label_anchor = self.data[index] , self.labels[index].item()
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label_anchor])

        negative_label = np.random.choice(list(self.labels_set - set([label_anchor])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])

        sample_positive = self.data[positive_index]
        sample_negative = self.data[negative_index]

        sample_anchor = self.convert_mesh_to_voxel(sample_anchor)
        sample_positive = self.convert_mesh_to_voxel(sample_positive)
        sample_negative = self.convert_mesh_to_voxel(sample_negative)

        if self.transform is not None:
            sample_anchor = self.transform(sample_anchor)
            sample_positive = self.transform(sample_positive)
            sample_negative = self.transform(sample_negative)

        return (sample_anchor, sample_positive, sample_negative), []

    def __len__(self):
        return len(self.dataset)

