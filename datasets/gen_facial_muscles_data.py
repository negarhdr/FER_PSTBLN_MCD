

import numpy as np
from numpy.lib.format import open_memmap
from scipy.spatial import Delaunay
import argparse


def find_graph_edges(x):
    points = np.transpose(x[0, :, 0, :, 0])
    print(points.shape)
    tri = Delaunay(points)
    neigh = tri.simplices
    print(neigh.shape)
    G = []
    N = neigh.shape[0]
    for i in range(N):
        G.append((neigh[i][0], neigh[i][1]))
        G.append((neigh[i][0], neigh[i][2]))
        G.append((neigh[i][1], neigh[i][2]))
    # connect the master node (nose) to all other nodes
    for i in range(51):
        G.append((i+1, 17))
    edges = G
    return edges


def gen_muscle_data(landmark_path, muscle_path):
    """Generate facial muscle data from facial landmarks"""
    data = np.load(landmark_path)
    N, C, T, V, M = data.shape
    edges = find_graph_edges(data)
    V_muscle = len(edges)
    fp_sp = open_memmap(muscle_path, dtype='float32', mode='w+', shape=(N, C, T, V_muscle, M))
    # Copy the landmark data to muscle placeholder tensor
    fp_sp[:, :, :, :V, :] = data
    for edge_id, (source_node, target_node) in enumerate(edges):
        fp_sp[:, :, :, edge_id, :] = data[:, :, :, source_node-1, :] - data[:, :, :, target_node-1, :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Facial muscle data generator.')
    parser.add_argument('--landmark_data_folder', default='./data/CASIA_10fold/')
    parser.add_argument('--muscle_data_folder', default='./data/muscle_data/')
    parser.add_argument('--dataset_name', default='CASIA')
    arg = parser.parse_args()
    part = ['Train', 'Val']
    for p in part:
        if arg.dataset_name == 'CASIA' or arg.dataset_name == 'CK+':
            for i in range(10):
                landmark_path = arg.landmark_data_folder + '/{}/{}_{}.npy'.format(arg.dataset_name, p, i)
                muscle_path = arg.muscle_data_folder + '/{}/{}_muscle_{}.npy'.format(arg.dataset_name, p, i)
                gen_muscle_data(landmark_path, muscle_path)
        elif arg.dataset_name == 'AFEW':
            landmark_path = arg.landmark_data_folder + '/{}/{}.npy'.format(arg.dataset_name, p)
            muscle_path = arg.muscle_data_folder + '/{}/{}_muscle.npy'.format(arg.dataset_name, p)
            gen_muscle_data(landmark_path, muscle_path)
