from utils import *
import numpy as np
import dataloaders
from torch.utils.data import Dataset

class Block(Dataset):
    # TODO 降低内存消耗 利于多个数据块的训练，可能cache也会更好一些
    def __init__(self, block_volume, block_size):
        self.v = block_volume
        self.res = np.array(block_size)

    def __getitem__(self, item):
        # 直接取值
        point = index_to_domain_xyz(item, vec3f(-1.0), vec3f(1.0), self.res)
        xyz = index_to_domain_xyz_index(item, self.res)
        val = self.v[xyz[0], xyz[1], xyz[2]]
        return point, np.array([val])

    def __len__(self):
        return self.res.prod()
    
# block_generator 产生 block 的数据
class BlockGenerator:
    def __init__(self, volume_data):
        self.res = np.array(volume_data.shape)
        self.volume_data = volume_data
        if self.volume_data.ndim == 3:
            self.has_timestamp = True
        else:
            self.has_timestamp = False

    # res为volume的大小, size为block的大小
    def uniform_part(self, block_size):
        data_block_array = []
        if self.has_timestamp is False:
            [w, h] = np.ceil(self.res / block_size)
            [chunk_w, chunk_h] = np.ceil(self.res / [w,h])
            for i in range(0, self.res[0], int(chunk_w)):
                for j in range(0, self.res[1], int(chunk_h)):
                    if i+block_size[0] > self.res[0]:
                        i = self.res[0] - block_size[0]
                    if j+block_size[1] > self.res[1]:
                        j = self.res[1] - block_size[1]
                    data_block = self.volume_data[i:i+block_size[0],
                                                    j:j+block_size[1]]
                    data_block = Block(data_block, block_size)
                    data_block_array.append(data_block)
        else:
            [w, h] = np.ceil(self.res[1:] / block_size)
            t = self.res[0]
            [chunk_w, chunk_h] = np.ceil(self.res[1:] / [w,h])
            for time in range(t):
                for i in range(0, self.res[1], int(chunk_w)):
                    for j in range(0, self.res[2], int(chunk_h)):
                        if i+block_size[0] > self.res[1]:
                            i = self.res[1] - block_size[0]
                        if j+block_size[1] > self.res[2]:
                            j = self.res[2] - block_size[1]
                        data_block = self.volume_data[time,
                                                    i:i+block_size[0],
                                                    j:j+block_size[1]]
                        data_block = Block(data_block, block_size)
                        data_block_array.append(data_block)

        return data_block_array

    # return num 个 block
    # TODO 此处可能存在问题，可能是随机中心点，在概率上更加正确
    def random_sample(self, block_size, block_num=None):
        pos = get_query_coords(vec3f(-1), vec3f(1), block_size).reshape([-1, 3])
        data_block_array = []
        if self.has_timestamp is False:
            if block_num is None:
                [w, h, d] = self.res // block_size
                block_num = w*h*d

            for i in range(block_num):
                left = generate_random_index_within_bound([0,0,0], self.res-block_size)
                right = left + block_size
                data_block = self.volume_data[left[0]:right[0],
                                              left[1]:right[1],
                                              left[2]:right[2]]
                data_block = Block(data_block, block_size, pos, left)
                data_block_array.append(data_block)
        else:
            times = self.res[0]
            if block_num is None:
                [w, h, d] = self.res[1:] // block_size
                block_num = w * h * d
            for time in range(times):
                for i in range(block_num):
                    left = generate_random_index_within_bound([0,0,0], self.res[1:]-block_size)
                    right = left + block_size
                    data_block = self.volume_data[time,
                                                 left[0]:right[0],
                                                 left[1]:right[1],
                                                 left[2]:right[2]]
                    data_block = Block(data_block, block_size, pos, left)
                    data_block_array.append(data_block)
        return data_block_array

    def center_sample(self, block_size, distance=10):
        pos = get_query_coords(vec3f(-1), vec3f(1), block_size).reshape([-1, 3])
        data_block_array = []
        center = np.array([self.res[0]//2, self.res[1]//2, self.res[2]//2])
        t = np.array([[0,0,0],[-1,-1,-1],[-1,-1,1],[-1,1,1],
             [1,-1,1],[1,1,-1],[-1,1,-1],[1,-1,-1],[1,1,1]])
        for i in range(len(t)):
            center_ = center + t[i]*distance
            data_block = self.volume_data[center_[0]-block_size[0]//2:center_[0]+block_size[0]//2,
                         center_[1]-block_size[1]//2:center_[1]+block_size[1]//2,
                         center_[2]-block_size[2]//2:center_[2]+block_size[2]//2]
            data_block = Block(data_block, block_size, pos, center = [center_[0]-block_size[0]//2, center_[1]-block_size[1]//2, center_[2]-block_size[2]//2])
            data_block_array.append(data_block)

        return data_block_array

    def offset_sample(self, block_size, num, offset=10):
        pos = get_query_coords(vec3f(-1), vec3f(1), block_size).reshape([-1, 3])
        data_block_array = []
        center = np.array([self.res[0]//2-block_size[0]//2, self.res[1]//2-block_size[1]//2, self.res[2]//2-block_size[2]//2])
        for i in range(num):
            left = generate_random_index_within_bound(center - offset//2 , center + offset//2)
            right = left + block_size
            data_block = self.volume_data[left[0]:right[0],
                                            left[1]:right[1],
                                            left[2]:right[2]]
            data_block = Block(data_block, block_size, pos)
            data_block_array.append(data_block)

        return data_block_array
    
    def index_sample(self, block_size, left):
        pos = get_query_coords(vec3f(-1), vec3f(1), block_size).reshape([-1, 3])
        data_block_array = []
        right = left + block_size
        data_block = self.volume_data[left[0]:right[0],
                                        left[1]:right[1],
                                        left[2]:right[2]]
        data_block = Block(data_block, block_size, pos, center=left)
        data_block_array.append(data_block)

        return data_block_array 

    # 考虑block_size也是随机的
    def generate_data_block(self, block_size, method='uniform', block_num=100, left=[125,125,125]):
        block_size = np.array(block_size)
        if method == 'uniform':
            return self.uniform_part(block_size)
        elif method == 'random':
            return self.random_sample(block_size, block_num)
        elif method == "center" and self.has_timestamp == False:
            return self.center_sample(block_size)
        elif method == "offset" and self.has_timestamp == False:
            return self.offset_sample(block_size, block_num)
        elif method == "index" and self.has_timestamp == False:
            return self.index_sample(block_size, left)
        else:
            raise NotImplementedError
        
class MetaDataset(Dataset):
    def __init__(self, data_block_array):
        super(MetaDataset, self).__init__()
        self.data_block_array = data_block_array

        vol = []
        for data_block in self.data_block_array:
            vol.append(data_block.v.flatten())
        vol = np.expand_dims(np.array(vol), axis=-1)
        pos = get_query_coords(vec2f(-1), vec2f(1), self.data_block_array[0].res).reshape([-1, 2]).astype(np.float32)
        pos = np.expand_dims(pos, 0).repeat(len(self.data_block_array), axis=0)
        # TODO 分 batch_size
        self.res = np.concatenate((pos, vol), axis=-1)

    def __len__(self):
        return len(self.data_block_array)

    def __getitem__(self, item):
        return self.res[item]

# 单例测试
def main():
    a = dataloaders.SingleTimeStampDataloader('2D-airfoil')
    a.load_data()
    generator = BlockGenerator(a.get_data())
    data_block_array = generator.generate_data_block(block_size=[3,32,32])
    print(data_block_array[0].v)

if __name__ == '__main__':
    main()