import torch
from torch import nn
from collections import OrderedDict
from networks import create_mlp
import numpy as np
from utils import *
from copy import deepcopy
from tqdm import tqdm, trange
import dataloaders


def clip_grad_by_norm(grad, max_norm):
    """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
    """

    total_norm = 0
    counter = 0
    for g in grad:
        param_norm = g.data.norm(2)
        total_norm += param_norm.item() ** 2
        counter += 1
    total_norm = total_norm ** (1. / 2)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grad:
            g.data.mul_(clip_coef)

    return total_norm / counter


class MetaTemplate(nn.Module):
    """
        Meta Learner
    """
    def __init__(self, model, args, lambda_grad = 0.05):
        super(MetaTemplate, self).__init__()
        self.meta_lr = args.outer_lr
        self.update_lr = args.inner_lr
        self.net = model
        self.num_meta_steps = args.meta_steps
        self.loss = torch.nn.MSELoss(reduction='mean')
        self.args = args
        self.whole_volume = None
        self.lambda_grad = lambda_grad
        self.device = get_device(args.GPU)
        # self.update_step_test = update_step_test
        
        self.parameters = [parameter for name, parameter in self.net.named_parameters()]
        self.forward_parameters = self.parameters[:self.net.D*2+2]# linear+output
        self.partial_parameters = self.parameters[:self.net.D*2+2]
        self.transformer_parameters = self.parameters[self.net.D*2+4:-4]#input+output
        self.MTF_parameters = self.parameters[self.net.D*2+4:]# input transformer + base network + output transformer

        self.forward_optimizer = torch.optim.Adam(self.forward_parameters, lr=self.meta_lr)
        self.transformer_optimizer = torch.optim.Adam(self.parameters[:self.net.D*2], lr=self.meta_lr)
        self.modulated_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.MTF_optimizer = torch.optim.Adam(self.parameters[:self.net.D*2+4], lr=self.meta_lr)
        self.meta_optim = torch.optim.Adam(self.parameters, lr=self.meta_lr)
        self.modulated_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.modulated_optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        self.forward_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.forward_optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    def conv_smooth(self, centers, kernel_size, prediction=None):
        padding = kernel_size//2
        conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=0).to(get_device(self.args.GPU))
        dim_x, dim_y, dim_z = self.args.block_size
        block = self.whole_volume[centers[0]-padding:centers[0]+padding+dim_x, centers[1]-padding:centers[1]+padding+dim_y, centers[2]-padding:centers[2]+padding+dim_z]
        block_shape = list(block.shape)
        block = torch.tensor(block.reshape([1]+block_shape), dtype=torch.float32).to(get_device(self.args.GPU))
        if prediction != None:
            prediction = prediction.reshape(1, dim_z, dim_y, dim_z)
            block[:, padding:padding+dim_z, padding:padding+dim_y, padding:padding+dim_x]=prediction
        nn.init.constant_(conv3d.weight, 1)
        nn.init.constant_(conv3d.bias, 0.0)
        result = conv3d(block) / kernel_size**3
        # block_recon = dataloaders.Block(result.detach().reshape(self.args.block_size).cpu().numpy(), self.args.block_size, get_query_coords(vec3f(-1), vec3f(1), self.args.block_size).reshape([-1, 3]), centers)
        # vtk_draw_blocks([block_recon])
        return result

    def kernel_smooth(self, loss):
        loss_shape = loss.shape
        loss = loss.reshape(loss_shape[0], -1)# batch_size, xyz
        result = torch.zeros_like(loss)
        dim_x, dim_y, dim_z = self.args.block_size
        kernel = torch.ones((3,3,3)).to(get_device(self.args.GPU))
        for i, l in enumerate(loss):
            l_reshaped = l.view(dim_z, dim_y, dim_x)
            for z in range(dim_z):
                for y in range(dim_y):
                    for x in range(dim_x):
                        local_block = torch.zeros((3,3,3)).to(get_device(self.args.GPU))
                        # 计算索引的有效范围
                        z_min = 0 if z-1<0 else -1
                        z_max = 1 if z+2>dim_z else 2
                        y_min = 0 if y-1<0 else -1
                        y_max = 1 if y+2>dim_y else 2
                        x_min = 0 if x-1<0 else -1
                        x_max = 1 if x+2>dim_x else 2
                        local_block[z_min+1:z_max+1, y_min+1:y_max+1, x_min+1:x_max+1] = l_reshaped[z + z_min:z + z_max, y + y_min:y + y_max, x + x_min:x + x_max]
                        result[i, z*dim_y*dim_x + y*dim_x + x] = (local_block * kernel).mean()
        result = result.reshape(loss_shape)
        return result

    def forward(self, x_spt, y_spt):
        # points channel
        task_num, setsz, channel = x_spt.size()
        losses_qry = 0.0
        for i in range(task_num):
            temp_parameters = deepcopy(self.parameters)
            temp_optimizer = torch.optim.Adam(temp_parameters, lr=self.update_lr)
            for k in range(self.num_meta_steps):
                prediction = self.net(x_spt[i:i+1], temp_parameters)
                loss = self.loss(prediction, y_spt[i:i+1])
                temp_optimizer.zero_grad()
                loss.backward()
                temp_optimizer.step()
            prediction = self.net(x_spt[i:i+1], temp_parameters)
            loss = self.loss(prediction, y_spt[i:i+1])
            losses_qry += loss
            
        # optimize theta parameters
        # losses_qry = losses_qry / task_num
        predictions = self.net(x_spt, self.parameters)
        loss = self.loss(predictions, y_spt)
        loss_value = loss.detach()
        loss = loss - loss_value + losses_qry
        self.forward_optimizer.zero_grad()
        loss.backward()
        self.forward_optimizer.step()
        with torch.no_grad():
            predictions = self.net(x_spt, self.parameters)
        loss = self.loss(predictions, y_spt)
        return loss.item()
    
    def modulated_forward(self, x, y, gradient=None):
        # meta training
        task_num, setsz, channel = x.size()
        if self.args.use_latent:
            modulations = torch.zeros(task_num, 1, self.args.latent_dim, dtype=torch.float32).requires_grad_().to(get_device(self.args.GPU))
        else:
            modulations = torch.zeros(task_num, 1, self.net.num_modulations, dtype=torch.float32).requires_grad_().to(get_device(self.args.GPU))
        for i in range(task_num):
            modulation = modulations[i:i+1]
            for k in range(self.num_meta_steps):
                predictions = self.net.modulated_forward(x[i:i+1], modulation)
                # mean_predictions  = self.conv_smooth(predictions)
                # mean_y = self.conv_smooth(y[i:i+1])
                # loss = self.loss(predictions, y[i:i+1]) + self.loss(mean_predictions, mean_y)
                loss = self.loss(predictions, y[i:i+1])
                grad = torch.autograd.grad(loss, modulation)
                # grad = torch.autograd.grad(loss, modulation, create_graph=True)
                modulation = modulation - self.update_lr*grad[0]
            modulations[i:i+1] = modulation.detach()

        # x.requires_grad = True
        predictions = self.net.modulated_forward(x, modulations)
        # mean_predictions = self.conv_smooth(predictions)
        # mean_y = self.conv_smooth(y)
        # loss = self.loss(predictions, y) + self.loss(mean_predictions, mean_y)
        loss = self.loss(predictions, y)
        # 计算网络预测对每个坐标的梯度
        # grad_outputs = torch.ones_like(predictions)
        # grad_predictions_t = torch.autograd.grad(outputs=predictions, inputs=x, grad_outputs=grad_outputs, retain_graph=True)[0]
        # x.requires_grad = False
        # gradient_loss = self.loss(grad_predictions_t, gradient)
        # predictions_c = predictions.cpu().numpy()
        # predictions_c = predictions_c.reshape(self.args.block_size)
        # grad_x, grad_y, grad_z = np.gradient(predictions_c)
        # grad_predictions = np.concatenate((np.expand_dims(grad_x, -1), np.expand_dims(grad_y, -1), np.expand_dims(grad_z, -1)), axis=-1)
        # grad_predictions = torch.tensor(grad_predictions).to(get_device(self.args.GPU)).reshape(-1,3)
        # loss_qry = (loss + self.lambda_grad * gradient_loss) * task_num
        # loss_qry = self.loss(predictions, y) 

        # optimize theta parameters
        # loss = self.loss(predictions, y) * task_num
        self.modulated_optimizer.zero_grad() 
        loss.backward()
        self.modulated_optimizer.step()
        self.modulated_scheduler.step(loss)
        # return loss.item()*task_num
        return loss.item()
    
    def transformer_forward(self, x_spt, y_spt, x_qry, y_qry):
        # points channel
        task_num, setsz, channel = x_spt.size()
        losses_qry = 0.0
        for i in range(task_num):
            fast_weights = self.transformer_parameters
            for k in range(self.num_meta_steps):
                predictions = self.net.transformer_forward(x_spt[i], fast_weights)
                loss = self.loss(predictions, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            predictions = self.net.transformer_forward(x_qry[i], fast_weights)
            loss_q = self.loss(predictions, y_qry[i])
            losses_qry += loss_q

        # optimize theta parameters
        losses_qry = losses_qry / task_num
        self.forward_optimizer.zero_grad()
        losses_qry.backward()
        self.forward_optimizer.step()
        # self.scheduler.step()
        return losses_qry.item()
    
    def MTF_forward(self, x_spt, y_spt, x_qry, y_qry, modulations):
        # points channel
        task_num, setsz, channel = x_spt.size()
        losses_qry = 0.0
        for i in range(task_num):
            modulation = modulations[i]
            fast_weights = self.transformer_parameters+[modulation]
            for k in range(self.num_meta_steps):
                predictions = self.net.MTF_forward(x_spt[i], fast_weights)
                loss = self.loss(predictions, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad[:-1], fast_weights[:-1])))+ [fast_weights[-1] - self.update_lr * 100 * grad[-1]]
            predictions = self.net.MTF_forward(x_qry[i], fast_weights)
            loss_q = self.loss(predictions, y_qry[i])
            losses_qry += loss_q

        # optimize theta parameters
        losses_qry = losses_qry / task_num
        self.MTF_optimizer.zero_grad()
        losses_qry.backward()
        self.MTF_optimizer.step()
        # self.scheduler.step()
        return losses_qry.item()
    
    def variant_forward(self, x_spt, y_spt, x_qry, y_qry, modulations, epoch, pattern):
        if pattern == "F":
            return self.forward(x_spt, y_spt, x_qry, y_qry) 
        elif pattern == "MF":
            if epoch < self.args.maml_boundary:
                return self.modulated_forward(x_spt, y_spt)
            else:
                return self.MTF_forward(x_spt, y_spt, x_qry, y_qry, modulations)
        elif pattern == "TF":
            if epoch < self.args.maml_boundary:
                return self.transformer_forward(x_spt, y_spt, x_qry, y_qry)
            else:
                return self.MTF_forward(x_spt, y_spt, x_qry, y_qry, modulations) 
        elif pattern == "MTF":
                return self.MTF_forward(x_spt, y_spt, x_qry, y_qry, modulations)   
        
    def modulation_forward_test(self, x_spt, y_spt, modulations):
        predictions = self.net.modulated_forward(x_spt, modulations)
        loss = self.loss(predictions, y_spt)
        grad = torch.autograd.grad(loss, modulations)
        modulations = modulations - self.update_lr*grad[0]
        return modulations
    
    def modulation_forward(self, x_spt, y_spt, modulations):
        # points channel
        for k in range(1, self.num_meta_steps):
            predictions = self.net.modulated_forward(x_spt, modulations)
            loss = self.loss(predictions, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, modulations)
            # 3. theta_pi = theta_pi - train_lr * grad
            modulations = modulations - self.update_lr*grad[0]
        
        return modulations
    
    def modulation_transformer_forward(self, x_spt, y_spt, center, MTF_parameters=None):
        # points channel
        task_num, setsz, channel = x_spt.size()
        losses_qry = 0.0
        fast_weights_array = []
        for i in range(task_num):
            if MTF_parameters == None:
                fast_weights = self.transformer_parameters
            else:
                fast_weights = MTF_parameters[i]
            for k in range(self.num_meta_steps):
                prediction = self.net.MTF_forward(x_spt[i], fast_weights)
                mean_prediction = self.conv_smooth(center, kernel_size=19, prediction=prediction)
                mean_y = self.conv_smooth(center, kernel_size=19)
                loss = self.loss(prediction, y_spt[i]) + self.loss(mean_prediction, mean_y)
                # loss = self.loss(mean_prediction, mean_y)
                # loss = self.loss(prediction, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad[:], fast_weights[:])))
            fast_weights_array.append(fast_weights)
        return fast_weights_array
    
    def partial_forward(self, x, y, parameters):
        # points channel
        task_num, setsz, channel = x.size()
        losses_qry = 0.0
        fast_weights_array = []
        for i in range(task_num):
            if parameters == None:
                fast_weights = self.partial_parameters
            else:
                fast_weights = parameters[i]
            for k in range(self.num_meta_steps):
                predictions = self.net.partial_forward(x[i], fast_weights)
                loss = self.loss(predictions, y[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad[:], fast_weights[:])))
            fast_weights_array.append(fast_weights)
        return fast_weights_array

    # TODO 完善微调步骤
    def finetune(self, x_spt, y_spt, x_qry, y_qry):
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetune on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = self.network_query_fn(x_spt, net)
        loss = self.loss(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = self.network_query_fn(x_spt, net, fast_weights)
            loss = self.loss(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        del net
        print(y_qry)
        return loss.item()

    def save_model(self, basedir, filename):
        # TODO save model to binary file
        pass


def main():
    pass


if __name__ == '__main__':
    main()








