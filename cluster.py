import numpy as np
import matplotlib.pyplot as plt
import utils
import torch
import utils
import math
import threading
import maml_wt
import time
import copy

from scipy.stats import entropy
from tqdm import tqdm, trange
from collections import Counter
from scipy.spatial.distance import jensenshannon
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed,wait,FIRST_COMPLETED
import multiprocessing as mp

class Variable_MI():
    def __init__(self, args) -> None:
        self.args = args
        mp.set_start_method('spawn')
        return
    
    def variable_entropy(self, block) -> float:
        flattened_block = block.flatten()
        counts = Counter(flattened_block)
        probabilities = np.array(list(counts.values())) / len(flattened_block)
        # hist, _ = np.histogram((flattened_block*256).astype(int), bins=range(257))
        # prob = hist/hist.sum()
        return entropy(probabilities)

    def two_variable_MI(self, block1, block2) -> float:
        a = block1.reshape(-1)
        b = block2.reshape(-1)
        hist, x_edges, y_edges = np.histogram2d(a,b,bins=[50,50])
        return entropy(hist.flatten())
    
    def HSIC(self, K, L) -> float:
        point_num = K.shape[0]
        H = np.eye(point_num) - np.dot(np.ones(point_num), np.ones(point_num).T)*(1/point_num)
        mean_K = np.dot(np.dot(H,K),H)
        mean_L = np.dot(np.dot(H,L),H)
        return np.dot(mean_K.flatten(), mean_L.flatten())/((point_num-1) ** 2)
    
    def CKA(self, matrix1, matrix2) -> float:
        K = np.dot(matrix1, matrix1.T)
        L = np.dot(matrix2, matrix2.T)
        return self.HSIC(K,L)/((self.HSIC(K,K)*self.HSIC(L,L)) ** (1/2))
    
    def jensen_shannon_divergence(self, p, q):
        m = 0.5 * (p + q)
        return 0.5 * (torch.sum(p * torch.log(p / m)) + torch.sum(q * torch.log(q / m)))
    
    def jsd1(self, block1, block2):
        a = block1.reshape(-1)
        b = block2.reshape(-1)
        # upper = max(a.max(), b.max())
        # lower = min(a.min(), b.min())
        hist_a, _ = np.histogram(a, bins=200, range=(0, 1))
        hist_b, _ = np.histogram(b, bins=200, range=(0, 1))
        dist_a = hist_a / hist_a.sum()
        dist_b = hist_b / hist_b.sum()
        # plt.hist(b, bins=200)
        # plt.title('Probability Distribution')
        # plt.xlabel('Value')
        # plt.ylabel('Probability')
        # plt.show()
        return jensenshannon(dist_a, dist_b)
    
    def jsd2(self, block1_np, block2_np):
        # hist_a = block1_np
        # hist_b = block2_np
        # dist_a = hist_a / hist_a.sum()
        # dist_b = hist_b / hist_b.sum()
        # return self.jensen_shannon_divergence(dist_a, dist_b).item()
        dist_a = block1_np
        dist_b = block2_np
        return jensenshannon(dist_a, dist_b)
    
    def jsd3(self, blocks_np, index):
        dist_a = blocks_np[index]
        result = []
        for i in range(index+1, len(blocks_np)):
            dist_b = blocks_np[i]
            result.append(self.jensen_shannon_divergence(dist_a, dist_b).item())
        return index, result
    
    def parellel_jsd_distance_matrix(self, blocks):
        distance_matrix = np.zeros((len(blocks), len(blocks)))
        block_histogram = np.array([np.histogram(block.reshape(-1), bins=1000, range=(0, 1))[0] for block in blocks])
        block_histogram = block_histogram / block_histogram.sum(axis=1, keepdims=True)  # 归一化
        block_histogram = torch.tensor(block_histogram)
        with ProcessPoolExecutor(max_workers=8) as executor:
            tasks = []
            for i in trange(len(blocks)):
                tasks.append(executor.submit(self.jsd3, block_histogram, i))
            for future in tqdm(as_completed(tasks), total=len(tasks)):
                index, distance = future.result()
                for i in range(index+1, len(blocks)):
                    distance_matrix[index][i] = distance_matrix[i][index] = distance[i-index-1]
        return distance_matrix
    
    def jsd_distance_matrix(self, blocks):
        distance_matrix = np.zeros((len(blocks), len(blocks)))
        block_histogram = np.array([np.histogram(block.reshape(-1), bins=1000, range=(0, 1))[0] for block in blocks])
        block_histogram = block_histogram / block_histogram.sum(axis=1, keepdims=True)  # 归一化
        for i in trange(len(blocks)):
            for j in trange(i+1, len(blocks)):
                distance_matrix[i][j] = distance_matrix[j][i] = jensenshannon(block_histogram[i], block_histogram[j])
        return distance_matrix
    
    def cka_distance_matrix(self, blocks):
        distance_matrix = np.zeros((len(blocks), len(blocks)))
        latent_matrix = []
        for i in trange(len(blocks)):
            model = maml_wt.maml_templates_generate([[i]], blocks, self.args)[0]
            latent_vector_hook = Network_Latent_Vector(model, 3, self.args)
            latent_vector_hook.register_hook()
            latent_vector = latent_vector_hook.get_latent_vectors().detach().cpu()
            latent_vector_hook.remove_hook()
            latent_matrix.append(latent_vector)

        for i in trange(len(blocks)):
            for j in trange(i+1, len(blocks)):
                distance_matrix[i][j] = distance_matrix[j][i] = self.CKA(latent_matrix[i], latent_matrix[j])
                if distance_matrix[i][j] < 0:
                    print(i,j)
        return distance_matrix

    def variables_MI_heat_map(self, block_array):
        block_num = block_array.shape[0]
        heat_map = np.empty((block_num,block_num))
        for i in tqdm(range(block_num)):
            for j in range(block_num):
                heat_map[i][j] = self.two_variable_MI(block_array[i],block_array[j])

        # draw heat map
        plt.clf()
        plt.imshow(heat_map, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        def close_figure():
            plt.close()  # 关闭 matplotlib 窗口
        fig = plt.gcf()
        timer = fig.canvas.new_timer(interval=5000)
        timer.add_callback(close_figure)
        timer.start()
        plt.show() 
        return heat_map
    
    def variables_CKA_heat_map(self, block_array):
        block_num = block_array.shape[0]
        heat_map = np.empty((block_num,block_num))
        for i in tqdm(range(block_num)):
            for j in range(block_num):
                heat_map[i][j] = self.CKA(block_array[i],block_array[j])

        # draw heat map
        plt.clf()
        plt.imshow(heat_map, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        def close_figure():
            plt.close()  # 关闭 matplotlib 窗口
        fig = plt.gcf()
        timer = fig.canvas.new_timer(interval=5000)
        timer.add_callback(close_figure)
        timer.start()
        plt.show() 
        return heat_map
    
class Network_Latent_Vector():
    def __init__(self, template, layer_index, args) -> None:
        self.template = template
        self.layer_index = layer_index
        self.args = args
        self.hook_handles = []
        self.template_latent_vector = None

    def sample(self, point_num):
        cube_res = [math.ceil(point_num ** (1/3)) for i in range(3)]
        return utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), cube_res).reshape([-1, 3])
        

    def hook_fn(self, module, input, output):
        latent_vector = output
        self.template_latent_vector = latent_vector
        return
    
    def register_hook(self):
        intermediate_layer = self.template.pts_linear[self.layer_index]
        self.hook_handles.append(intermediate_layer.register_forward_hook(self.hook_fn))

    def get_latent_vectors(self):
        x = torch.from_numpy(self.sample(64)).to(torch.float32)
        x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1).to(utils.get_device(self.args.GPU))
        modulation = torch.zeros(self.args.latent_dim).requires_grad_().to(utils.get_device(self.args.GPU))
        self.template.modulated_forward(x, modulation)

        return self.template_latent_vector


    def remove_hook(self):
        for handle in self.hook_handles:
            handle.remove()
    
