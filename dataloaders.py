import numpy as np
from torch.utils.data import Dataset, DataLoader
import vtk
from glob import glob
from vtkmodules.util import numpy_support
from scipy.ndimage import zoom
import netCDF4 as nc
import struct
import os

class INS3D:
    def __init__(self, shape, timestep=-1) -> None:
        self.data_dir = os.path.join('/home/XiYang/MAML_KiloNet/source/data/INS3D/INS3D',shape,'dat/output/')
        self.timestep = timestep
        self.dataset = []

    def load_data(self):
        if self.timestep == -1:
            for file_path in glob(self.data_dir + '*.30'):
                ## 读坐标
                with open(file_path, 'rb') as file:
                    jmax, kmax, lmax = np.fromfile(file, dtype=np.int32, count=3)
                    # print(f"jmax: {jmax}, kmax: {kmax}, lmax: {lmax}")
                    x = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    y = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    z = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                file.close()
                ## 读数据
                file_path1 = file_path[:-1]+'1'
                with open(file_path1, 'rb') as file:
                    jmax, kmax, lmax = np.fromfile(file, dtype=np.int32, count=3)
                    fsmach,alpha,reynum,time = np.fromfile(file, dtype=np.float32, count=4)
                    print(f"jmax: {jmax}, kmax: {kmax}, lmax: {lmax}, fsmach: {fsmach}, alpha: {alpha}, reynum: {reynum}, time: {time}")
                    rho = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q1 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q2 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q3 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q4 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                file.close()
                points = np.stack((x[0,:,:],y[0,:,:],z[0,:,:]), axis=-1).reshape(-1,3)
                scalar_values = q1[0,:,:].reshape(-1,1)
                data = np.concatenate((points,scalar_values), axis=1)# x,y,z,value
                self.dataset.append(data)
        else:
            point_file_path = self.data_dir+f'fort00{self.timestep:03}00.30'
            value_file_path = self.data_dir+f'fort00{self.timestep:03}00.31'
            with open(point_file_path, 'rb') as file:
                jmax, kmax, lmax = np.fromfile(file, dtype=np.int32, count=3)
                # print(f"jmax: {jmax}, kmax: {kmax}, lmax: {lmax}")
                x = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                y = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                z = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
            file.close()
            with open(value_file_path, 'rb') as file:
                jmax, kmax, lmax = np.fromfile(file, dtype=np.int32, count=3)
                fsmach,alpha,reynum,time = np.fromfile(file, dtype=np.float32, count=4)
                # print(f"jmax: {jmax}, kmax: {kmax}, lmax: {lmax}, fsmach: {fsmach}, alpha: {alpha}, reynum: {reynum}, time: {time}")
                rho = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q1 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q2 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q3 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q4 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
            file.close()
            points = np.stack((x[0,:,:],y[0,:,:],z[0,:,:]), axis=-1).reshape(-1,3)
            scalar_values = q1[0,:,:].reshape(-1,1)
            data = np.concatenate((points,scalar_values), axis=1)
            self.dataset.append(data)

    def get_data(self):
        return self.dataset
    
class INS3Dv2:
    def __init__(self, shape, timestep=-1) -> None:
        self.data_dir = os.path.join('/home/XiYang/MAML_KiloNet/source/data/INS3D/INS3D',shape,'dat/output/')
        self.timestep = timestep
        self.dataset = []

    def load_interpolated_data(self):
        if self.timestep == -1:
            for file_path in glob(self.data_dir + '*.31')[:10]:
                ## 读数据
                with open(file_path, 'rb') as file:
                    jmax, kmax, lmax = np.fromfile(file, dtype=np.int32, count=3)
                    fsmach,alpha,reynum,time = np.fromfile(file, dtype=np.float32, count=4)
                    print(f"jmax: {jmax}, kmax: {kmax}, lmax: {lmax}, fsmach: {fsmach}, alpha: {alpha}, reynum: {reynum}, time: {time}")
                    rho = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q1 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q2 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q3 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q4 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                file.close()
                zoom_factors = (2, 2)
                interpolated_q1 = zoom(q1[0,...], zoom_factors, order=1)
                normalize_lower_bound, normalize_upper_bound = np.percentile(interpolated_q1, 0), np.percentile(interpolated_q1, 100)
                interpolated_q1 = (interpolated_q1 - normalize_lower_bound) / (normalize_upper_bound - normalize_lower_bound)
                self.dataset.append(interpolated_q1)
        else:
            value_file_path = self.data_dir+f'fort00{self.timestep:03}00.31'
            with open(value_file_path, 'rb') as file:
                jmax, kmax, lmax = np.fromfile(file, dtype=np.int32, count=3)
                fsmach,alpha,reynum,time = np.fromfile(file, dtype=np.float32, count=4)
                # print(f"jmax: {jmax}, kmax: {kmax}, lmax: {lmax}, fsmach: {fsmach}, alpha: {alpha}, reynum: {reynum}, time: {time}")
                rho = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q1 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q2 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q3 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q4 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
            file.close()
            zoom_factors = (2, 2)
            interpolated_q1 = zoom(q1[0], zoom_factors, order=1)
            normalize_lower_bound, normalize_upper_bound = np.percentile(interpolated_q1, 0), np.percentile(interpolated_q1, 100)
            interpolated_q1 = (interpolated_q1 - normalize_lower_bound) / (normalize_upper_bound - normalize_lower_bound)
            self.dataset.append(interpolated_q1)

    def load_data(self):
        if self.timestep == -1:
            for file_path in glob(self.data_dir + '*.31')[:10]:
                ## 读数据
                with open(file_path, 'rb') as file:
                    jmax, kmax, lmax = np.fromfile(file, dtype=np.int32, count=3)
                    fsmach,alpha,reynum,time = np.fromfile(file, dtype=np.float32, count=4)
                    print(f"jmax: {jmax}, kmax: {kmax}, lmax: {lmax}, fsmach: {fsmach}, alpha: {alpha}, reynum: {reynum}, time: {time}")
                    rho = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q1 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q2 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q3 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                    q4 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                file.close()
                normalize_lower_bound, normalize_upper_bound = np.percentile(q1[0], 0), np.percentile(q1[0], 100)
                q1[0] = (q1[0] - normalize_lower_bound) / (normalize_upper_bound - normalize_lower_bound)
                self.dataset.append(q1[0])
        else:
            value_file_path = self.data_dir+f'fort00{self.timestep:03}00.31'
            with open(value_file_path, 'rb') as file:
                jmax, kmax, lmax = np.fromfile(file, dtype=np.int32, count=3)
                fsmach,alpha,reynum,time = np.fromfile(file, dtype=np.float32, count=4)
                # print(f"jmax: {jmax}, kmax: {kmax}, lmax: {lmax}, fsmach: {fsmach}, alpha: {alpha}, reynum: {reynum}, time: {time}")
                rho = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q1 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q2 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q3 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
                q4 = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
            file.close()
            normalize_lower_bound, normalize_upper_bound = np.percentile(q1[0], 0), np.percentile(q1[0], 100)
            q1[0] = (q1[0] - normalize_lower_bound) / (normalize_upper_bound - normalize_lower_bound)
            self.dataset.append(q1[0])

    def get_data(self):
        return self.dataset

class SingleTimeStampDataloader:
    def __init__(self, shape, timestep=1):
        self.shape = shape
        self.timestep = timestep
       
    def load_data(self):
        self.SingleStamp = INS3Dv2(self.shape, self.timestep)
        self.SingleStamp.load_data()

    def load_interpolated_data(self):
        self.SingleStamp = INS3Dv2(self.shape, self.timestep)
        self.SingleStamp.load_interpolated_data()

    def get_data(self):
        return np.array(self.SingleStamp.get_data())


# TODO 多个时间步
class MultiTimeStampDataloader:
    # 现在仅支持单一数据集的多时间步
    def __init__(self, shape, timestep=-1):
        self.shape = shape
        self.timestep = timestep
       
    def load_data(self):
        # 均匀采样
        self.MultiStamp=INS3Dv2(self.shape, self.timestep)
        self.MultiStamp.load_data()

    def load_interpolated_data(self):
        self.MultiStamp = INS3Dv2(self.shape, self.timestep)
        self.MultiStamp.load_interpolated_data()

    def get_data(self):
        return np.array(self.MultiStamp.get_data())
    
    def get_time_steps(self):
        return len(self.MultiStamp.get_data())
    
# 单例测试
def main():
    a = SingleTimeStampDataloader('2D-airfoil')
    a.load_data()
    print(a.get_data())

if __name__ == '__main__':
    main()

