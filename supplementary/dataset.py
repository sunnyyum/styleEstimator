from torch.utils.data import Dataset,DataLoader
import pymesh

import numpy as np

class PointNetStyleEstimator(Dataset):

    '''
    - convert mesh to point
    - return format of Dataloader
    '''

    def convert_mesh_to_point(self, input_mesh):

        in_mesh = pymesh.load_mesh(input_mesh)
        in_mesh.add_attribute("vertex_normal")
        v_normals = in_mesh.get_vertex_attribute("vertex_normal")

        out_mesh = pymesh.form_mesh(in_mesh.vertices, np.zeros((0, 3), dtype=int))

        out_mesh.add_attribute("nx")
        out_mesh.add_attribute("ny")
        out_mesh.add_attribute("nz")

        out_mesh.set_attribute("nx", v_normals[:, 0].ravel())
        out_mesh.set_attribute("ny", v_normals[:, 1].ravel())
        out_mesh.set_attribute("nz", v_normals[:, 2].ravel())

        return out_mesh


    def __init__(self, dataset):
        self.dataset = dataset

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

        #convert mesh into point
        #sample_anchor, sample_positive, sample_negative is file such as .obj
        sample_anchor = self.convert_mesh_to_voxel(sample_anchor)
        sample_positive = self.convert_mesh_to_voxel(sample_positive)
        sample_negative = self.convert_mesh_to_voxel(sample_negative)


        return (sample_anchor, sample_positive, sample_negative), []

    def __len__(self):
        return len(self.dataset)