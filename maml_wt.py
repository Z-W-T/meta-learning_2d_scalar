import numpy as np
import torch
import cluster
import random
import matplotlib.pyplot as plt
import sys
from losses import mse2psnr
import DataPreprocess
from utils import *
from config_parser import get_args
from dataloaders import *
from networks import *
from MetaTemplate import MetaTemplate
from copy import deepcopy
import multiprocessing as mp
from tqdm import tqdm, trange
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed,wait,FIRST_COMPLETED
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans

import multiprocessing
import threading


def Kmeans(distance_matrix, group_num):
    # 初始化
    center_indexs = random.sample(range(len(distance_matrix)), group_num)
    center_indexs_old = []

    while center_indexs != center_indexs_old:
        # 聚类
        groups = [[center_indexs[i]] for i in range(group_num)]
        for i in range(len(distance_matrix)):
            if i in center_indexs:
                continue
            distance = [distance_matrix[i][index] for index in center_indexs]
            groups[distance.index(min(distance))].append(i)

        # 找中心
        center_indexs_old = center_indexs
        center_indexs = []
        for group in groups:
            total_distance = []
            for index1 in group:
                distance = 0
                for index2 in group:
                    distance += distance_matrix[index1][index2]
                total_distance.append(distance)
            center_indexs.append(group[total_distance.index(min(total_distance))])
    
    return groups

def get_blocks_distance_matrix(data_block_array, args):
    blocks_data = [block.v for block in data_block_array]
    # blocks_data = torch.tensor(np.array(blocks_data), dtype=torch.float).unsqueeze(1).to(get_device(args.GPU))
    distance_calculator = cluster.Variable_MI(args)
    distance_matrix = distance_calculator.jsd_distance_matrix(blocks_data)
    return distance_matrix

# data block array 代表抽取
def block_subsample(data_block_array, distance_matrix, method, args):
    if method == "represent":
        tsne = TSNE(n_components=2, metric="precomputed")
        embedded = tsne.fit_transform(distance_matrix)
        kmeans = KMeans(n_clusters=args.center_num)
        kmeans.fit(embedded)
        clusters = kmeans.labels_
        # 分组
        groups = [[] for i in range(clusters.max()+1)]
        for i in range(len(clusters)):
            if clusters[i] != -1:
                groups[clusters[i]].append(i)
        # 找中心
        indexs = []
        for group in groups:
            total_distance = []
            for index1 in group:
                distance = 0
                for index2 in group:
                    distance += distance_matrix[index1][index2]
                total_distance.append(distance)
            indexs.append(group[total_distance.index(min(total_distance))])
        training_data_block_array = [data_block_array[i] for i in indexs] 
    elif method == "random":
        training_data_block_array = [data_block_array[i] for i in random.sample(range(0,len(data_block_array)), args.center_num)]
    return training_data_block_array

def draw_cluster_situation(groups, embedded):
    # 绘制图像
    for i, group in enumerate(groups):
        x = [embedded[index, 0] for index in group]
        y = [embedded[index, 1] for index in group]
        plt.scatter(x, y, color=np.random.rand(3), label=f'Group{i}')
    for i in range(len(embedded)):
        plt.text(embedded[i,0], embedded[i,1], f'{i}',fontsize=6, ha='right', va='bottom')
    plt.legend()
    plt.title("t-SNE visualization of distance matrix")
    plt.show()
    
# 返回一个分组方式
def maml_init(blocks, args):
    groups = [[] for i in range(args.groups_num)]
    if args.group_init == 'train_init':
        # 随机选取k个block，train之后得到对应template，在reassignment(步骤3)
        templates = []
        device = get_device(args.GPU)
        rand_num = generate_shuffle_number(len(blocks))[:args.groups_num]
        MSE_loss = torch.nn.MSELoss(reduction='mean')
        dataset = DataPreprocess.MetaDataset(blocks)
        for num in tqdm(rand_num):
            # 随机选取的block
            model = Siren(D=args.netdepth, W=args.netwidth, input_ch=args.input_ch, output_ch=args.output_ch, skips=[],nonlinearity='sine', use_bias=True).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            x, y = torch.tensor(dataset[num][:,:3]).to(device), torch.tensor(dataset[num][:,-1]).unsqueeze(1).to(device)
            for i in trange(args.maml_epoches):
                rec_y = model(x)
                loss = MSE_loss(rec_y, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tqdm.write(f"Iter: {i} Loss: {loss} PSNR: {mse2psnr(loss)}")
            templates.append(model)
        groups, _ = maml_reassignment(args, blocks, templates)
        return groups
    elif args.group_init == 'cluster_init':
        # 计算block之间距离矩阵
        blocks_data = [block.v for block in blocks]
        distance_calculator = cluster.Variable_MI(args)
        distance_matrix = distance_calculator.jsd_distance_matrix(blocks_data)
        # 降维到二维平面
        tsne = TSNE(n_components=2, metric="precomputed")
        embedded = tsne.fit_transform(distance_matrix)
        # 聚类
        kmeans = KMeans(n_clusters=args.groups_num)
        kmeans.fit(embedded)
        clusters = kmeans.labels_
        # 分组
        groups = [[] for i in range(clusters.max()+1)]
        for i in range(len(clusters)):
            if clusters[i] != -1:
                groups[clusters[i]].append(i)
        # 绘制聚类分布
        # draw_cluster(embedded, clusters)
        return groups
    else:
        raise NotImplementedError

# 训练template
def parellel_maml(group_blocks, state_dict, index, args):
    print('ok')
    # 构造dataloader
    dataset = DataPreprocess.MetaDataset(group_blocks)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=0)
    
    # 创建模型
    model = create_modulated_siren_maml(args)
    model.load_state_dict(state_dict)
    
    # 构造MAML类
    TemplateGenerator = MetaTemplate(model, args)

    # MAML学习元模型
    loss = []
    for i in trange(args.maml_epoches):
        loss_running = 0.0
        for data in data_loader:
            x, gradient, y = data[:,:,:3], data[:,:,3:6], data[:,:,-1].unsqueeze(2) # data shape:[batch_size, x*y*z, channel]
            if args.input_ch == 4:
                x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)
            if torch.cuda.is_available():
                x = x.to(torch.float32).to(get_device(args.GPU))
                gradient = gradient.to(torch.float32).to(get_device(args.GPU))
                y = y.to(torch.float32).to(get_device(args.GPU))
            _, mean_loss = TemplateGenerator.modulated_forward(x, y, gradient)
            loss_running += mean_loss
        loss_running /= len(data_loader)
        tqdm.write(f"[MAML] Group: {index} Iter: {i} Loss: {loss_running} PSNR: {mse2psnr(torch.tensor(loss_running))}")
        loss.append(loss_running)
        if loss_running < 1e-5:
            break
    tqdm.write(f"Group: {index} Loss: {loss_running} PSNR: {mse2psnr(torch.tensor(loss_running))}")
    return model.state_dict(), loss, index

# 训练template
def maml(group_blocks, args, template):
    # 构造dataloader
    data_loader = torch.utils.data.DataLoader(group_blocks, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)
    # 创建网络的部分
    if args.model == 'mlp_relu':
        if template is None:
            model, network_query_fn = create_mlp_maml(args)
        else:
            model = template
    elif args.model == 'siren':
        if template is None:
            model, network_query_fn = create_siren_maml(args)
        else:
            model = template
    elif args.model == 'film_siren':
        if template is None:
            model, network_query_fn = create_film_siren_maml(args)
        else:
            model = template
    elif args.model == 'modulated_siren':
        if template is None:
            model = create_modulated_siren_maml(args)
        else:
            model = template
    else:
        raise NotImplementedError
    
    # 构造MAML类
    TemplateGenerator = MetaTemplate(model, args)
    # model_old = deepcopy(model)

    # MAML学习元模型
    loss = []
    for i in trange(args.maml_epoches):
        loss_running = 0.0
        idx=0
        for data in data_loader:
            x, y = data[:,:,:2].to(get_device(args.GPU)), data[:,:,-1].unsqueeze(2).to(get_device(args.GPU)) # data shape:[batch_size, point_num, channel]
            idx += x.shape[0]
            if args.input_ch == 4:
                x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)
            mean_loss = TemplateGenerator(x, y)
            loss_running += mean_loss*x.shape[0]
        loss_running /= len(group_blocks)
        tqdm.write(f"[MAML] Iter: {i} Loss: {loss_running} PSNR: {mse2psnr(torch.tensor(loss_running))}")
        loss.append(loss_running)

    return model, loss

def find_next_group(task_schedule_table):
    for i in range(len(task_schedule_table)):
        for j in range(len(task_schedule_table[0])):
            if task_schedule_table[i][j] == 0 and (i == 0 or task_schedule_table[i-1][j] == 2):
                task_schedule_table[i][j] = 1
                return j
    return -1
            
def update_schedule_table(task_schedule_table, j):
    for i in range(len(task_schedule_table)):
        if task_schedule_table[i][j] == 1:
            task_schedule_table[i][j] = 2
            return
    return
    
def check_schedule_table(task_schedule_table):
    for i in range(len(task_schedule_table)):
        for j in range(len(task_schedule_table[0])):
            if task_schedule_table[i][j] == 0:
                return True
    return False

# 返回各组block训练产生的template
def parellel_maml_templates_generate(groups, blocks, args, templates=None):
    maml_templates = [create_modulated_siren_maml(args) for i in range(len(groups))]
    losses = [[] for i in range(len(groups))]
    group_block_array = []
    for group in groups:
        group_blocks = [blocks[index] for index in group]
        group_block_array.append(group_blocks)
    group_block_array.sort(key=len, reverse=True)

    maml_chunk = 5
    task_schedule_table = np.zeros((int(args.maml_epoches/maml_chunk),len(groups)))
    args.maml_epoches = maml_chunk
    print(task_schedule_table.shape)

    # 创建一个ThreadPoolExecutor对象parellel_maml
    with ProcessPoolExecutor(max_workers=2) as executor:
        tasks = []
        for i, group_block in enumerate(group_block_array):
            print(f'group:{i} group length:{len(group_block)}')
        for i in range(len(group_block_array)):
            tasks.append(executor.submit(parellel_maml, group_block_array[i], maml_templates[i].state_dict(), i, args))
            task_schedule_table[0][i] = 1
        while tasks:
            done, not_done = wait(tasks, return_when=FIRST_COMPLETED)
            for future in done:
                state_dict, loss, index = future.result()
                maml_templates[index].load_state_dict(state_dict)
                losses[index].extend(loss)
                update_schedule_table(task_schedule_table, index)
                tasks.remove(future)
                if check_schedule_table(task_schedule_table):
                    index = find_next_group(task_schedule_table)
                    if index != -1:
                        tasks.append(executor.submit(parellel_maml, group_block_array[index], maml_templates[index], index, args))
                
    return maml_templates, losses

def maml_templates_generate(groups, blocks, args, templates=None):
    device = get_device(args.GPU)
    maml_templates = []
    losses = []
    dataset = DataPreprocess.MetaDataset(blocks)
    # TODO 转化为多进程,加快运行效率
    for i, group in tqdm(enumerate(groups), total=len(groups)):
        if group:
            tqdm.write(f"group size:{len(group)}")
            group_blocks = [dataset[j] for j in group]
            maml_template, loss = maml(group_blocks, args, templates[i] if templates is not None else None)
            maml_templates.append(maml_template)
            losses.append(loss)
    return maml_templates

def reduce_templates_MI(templates, templates_heat_map, lrate=1e-5):
    for i, template in enumerate(templates):
        grad_vars = list(template[0].parameters())
        optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9,0.999))
        loss = torch.tensor(np.sum(templates_heat_map[i]), requires_grad=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return

def template_query_blocks_test(template, dataset, modulations, args):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)
    # modulations = torch.zeros(len(dataset), 1, args.latent_dim).requires_grad_().to(get_device(args.GPU))
    data_block_array = torch.zeros(list(dataset.shape[:-1])+[1])
    MSE_loss = torch.nn.MSELoss(reduction='mean')
    total_loss = 0.0
    # 学习modulation
    for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        x, gradient, y = data[:,:,:3], data[:,:,3:6], data[:,:,-1].unsqueeze(2)
        begin = batch_idx*args.batch_size
        end = begin+x.shape[0]
        TemplateGenerator = MetaTemplate(template, args)
        if args.input_ch == 4:
                x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)
        # for i in trange(args.query_steps):
        #     loss = TemplateGenerator.modulated_forward(x,y,gradient,modulations[begin:end])
        #     tqdm.write(f"Reconstruct Iter: {i} Loss: {loss} PSNR: {mse2psnr(torch.tensor(loss))}")
        #     if loss < 1e-5:
        #         break
        for i in trange(args.query_steps):
            modulations_temp = modulations[begin:end].requires_grad_()
            modulations[begin:end] = TemplateGenerator.modulation_forward_test(x, y, modulations_temp)
    # 还原block
    for batch_idx, data in enumerate(data_loader):
        x, y = data[:,:,:3], data[:,:,-1].unsqueeze(2) # data shape:[batch_size, x*y*z, channel]
        begin = batch_idx*args.batch_size
        end = begin+x.shape[0]
        if args.input_ch == 4:
            x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)
        with torch.no_grad():
            reconstruct_y = template.modulated_forward(x, modulations[begin:end])
        loss = MSE_loss(reconstruct_y, y)*x.shape[0]
        data_block_array[begin:end] = reconstruct_y.cpu()
        total_loss += loss
    return modulations.detach(), total_loss.item(), data_block_array

def template_query_blocks(template, dataset, args):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)
    modulations = torch.zeros(len(dataset), 1, args.latent_dim).requires_grad_().to(get_device(args.GPU))
    data_block_array = torch.zeros(list(dataset.shape[:-1])+[1])
    TemplateGenerator = MetaTemplate(template, args)
    MSE_loss = torch.nn.MSELoss(reduction='mean')
    total_loss = 0.0
    # 学习modulation
    for i in trange(args.query_steps):
        for batch_idx, data in enumerate(data_loader):
            x, gradient, y = data[:,:,:3], data[:,:,3:6], data[:,:,-1].unsqueeze(2) # data shape:[batch_size, x*y*z, channel]
            if args.input_ch == 4:
                x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)
            modulations[batch_idx*args.batch_size:batch_idx*args.batch_size+x.shape[0]] = TemplateGenerator.modulation_forward_test(x, y, modulations[batch_idx*args.batch_size:batch_idx*args.batch_size+x.shape[0]])
    # 还原block
    for batch_idx, data in enumerate(data_loader):
        x, y = data[:,:,:3], data[:,:,-1].unsqueeze(2) # data shape:[batch_size, x*y*z, channel]
        if args.input_ch == 4:
            x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)
        with torch.no_grad():
            reconstruct_y = template.modulated_forward(x, modulations[batch_idx*args.batch_size:batch_idx*args.batch_size+x.shape[0]])
        loss = MSE_loss(reconstruct_y, y)*x.shape[0]
        data_block_array[batch_idx*args.batch_size:batch_idx*args.batch_size+x.shape[0]] = reconstruct_y.cpu()
        total_loss += loss
    return modulations, total_loss.item(), data_block_array

def template_query_block(template, block, whole_volume, args):
    # 构造dataloader
    center = block.center
    dataset = blocks_to_dataset([block], args)

    # 构造元学习类
    TemplateGenerator = MetaTemplate(template, whole_volume, args)

    # 训练
    losses=[]
    modulation = torch.zeros(1, 1, args.latent_dim).requires_grad_()
    x, y = dataset[:,:,:3], dataset[:,:,-1].unsqueeze(2) # data shape:[batch_size, x*y*z, channel]
    if args.input_ch == 4:
        x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)
    if torch.cuda.is_available():
        x = x.to(torch.float32).to(get_device(args.GPU))
        y = y.to(torch.float32).to(get_device(args.GPU))
        modulation = modulation.to(torch.float32).to(get_device(args.GPU))
    fast_weight_array = None
    # origin_fn = template.modulated_forward(x, modulation)
    origin_fn = template.forward(x)
    MSE_loss = torch.nn.MSELoss(reduction='mean')
    loss_running = MSE_loss(origin_fn, y).item()
    tqdm.write(f"Template origin. Loss: {loss_running} PSNR: {mse2psnr(torch.tensor(loss_running))}")
    for i in trange(args.query_steps):
        # modulation = TemplateGenerator.modulation_forward(x, y, modulation)
        fast_weight_array = TemplateGenerator.modulation_transformer_forward(x,y,center,fast_weight_array)
        # fast_weight_array = TemplateGenerator.partial_forward(x,y,fast_weight_array)
        # 测试拟合效果
        # res_fn = template.modulated_forward(x, modulation)
        res_fn = template.MTF_forward(x, fast_weight_array[0])
        # res_fn = template.partial_forward(x, fast_weight_array[0])
        MSE_loss = torch.nn.MSELoss(reduction='mean')
        loss_running = MSE_loss(res_fn, y).item()
        losses.append(loss_running)
        tqdm.write(f"Template reconstruct. Loss: {loss_running} PSNR: {mse2psnr(torch.tensor(loss_running))}")
    
    # 绘制还原block
    block_recon_origin = Block(origin_fn.detach().reshape(args.block_size).cpu().numpy(), args.block_size, get_query_coords(vec3f(-1), vec3f(1), args.block_size).reshape([-1, 3]), block.center)
    block_recon = Block(res_fn.detach().reshape(args.block_size).cpu().numpy(), args.block_size, get_query_coords(vec3f(-1), vec3f(1), args.block_size).reshape([-1, 3]), block.center)
    vtk_draw_blocks([block, block_recon_origin, block_recon])
    return loss_running, losses

def parellel_templates_optimize_block(templates, block, args, index):
    # 构造dataloader
    dataset = DataPreprocess.MetaDataset([block])

    # 构造元学习类
    TemplateGenerators = []
    modulations = []
    templates_index = [i for i in range(len(templates))]
    for template in templates:
        TemplateGenerators.append(MetaTemplate(template, args))
        modulations.append(torch.zeros(1, args.latent_dim).requires_grad_())

    # 加载数据
    x, gradient, y = torch.tensor(dataset[:,:,:3]), torch.tensor(dataset[:,:,3:6]), torch.tensor(dataset[:,:,-1]).unsqueeze(-1) # data shape:[batch_size, x*y*z, channel]
    if args.input_ch == 4:
        x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)
    if torch.cuda.is_available():
        x = x.to(torch.float32).to(get_device(args.GPU))
        gradient = gradient.to(torch.float32).to(get_device(args.GPU))
        y = y.to(torch.float32).to(get_device(args.GPU))
        modulations = [modulation.to(torch.float32).to(get_device(args.GPU)) for modulation in modulations]

    # 训练各个template modulation并淘汰表现不好的
    for i in range(args.optimize_steps):
        PSNRs = []
        preserve_index = []
        losses = []
        for j, TemplateGenerator in enumerate(TemplateGenerators):
            if args.is_train:
                # 学习网络
                loss = TemplateGenerator.modulated_forward(x,y,gradient)
            else:
                # 学习modulation
                modulations[j] = TemplateGenerator.modulation_forward(x, y, modulations[j])
                # 测试拟合效果
                res_fn = TemplateGenerator.net.modulated_forward(x, modulations[j][0])
                # 结果评估
                MSE_loss = torch.nn.MSELoss(reduction='mean')
                loss = MSE_loss(res_fn, y).item()
            PSNRs.append(mse2psnr(torch.tensor(loss)))
            losses.append(loss)

        if i == args.optimize_steps-1:
            continue
        max_PSNR = max(PSNRs)
        for j, PSNR in enumerate(PSNRs):
            if (max_PSNR - 2) < PSNR:
                preserve_index.append(j)
        TemplateGenerators = [TemplateGenerators[i] for i in range(len(TemplateGenerators)) if i in preserve_index]
        modulations = [modulations[i] for i in range(len(modulations)) if i in preserve_index]
        templates_index = [templates_index[i] for i in range(len(templates_index)) if i in preserve_index]
    
    return min(losses), templates_index[losses.index(min(losses))], index
    # return min(losses), losses.index(min(losses)), index

def templates_optimize_block(templates, block, args):
    # 构造dataloader
    dataset = DataPreprocess.MetaDataset([block])[0]

    # 加载数据
    x, y = torch.tensor(dataset[:,:2]),torch.tensor(dataset[:,-1]).unsqueeze(-1) # data shape:[batch_size, x*y*z, channel]
    if args.input_ch == 4:
        x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)
    if torch.cuda.is_available():
        device = get_device(args.GPU)
        x = x.to(torch.float32).to(device)
        y = y.to(torch.float32).to(device)

    # 训练各个template 并淘汰表现不好的
    MSE_loss = torch.nn.MSELoss(reduction='mean')
    templates_index = [i for i in range(len(templates))]
    PSNRs = []
    losses = []
    blocks_array = [block]
    for index in templates_index:
        # 学习网络
        optimizer = torch.optim.Adam(templates[index].parameters(), lr=args.outer_lr)
        for i in range(args.optimize_steps):
            rec_y = templates[index](x)
            loss = MSE_loss(rec_y,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        PSNR = mse2psnr(loss)
        PSNRs.append(PSNR)
        losses.append(loss)
        blocks_array.append(DataPreprocess.Block(rec_y.detach().cpu(), args.block_size))
        # 选出训练最好以及接近的template
        # if i == args.optimize_steps-1:
        #     continue
        # max_PSNR = max(PSNRs)
        # templates_index = []
        # for j, PSNR in enumerate(PSNRs):
        #     if (max_PSNR - 2) < PSNR:
        #         templates_index.append(j)
    # vtk_draw_blocks(blocks_array)
    return min(losses), templates_index[losses.index(min(losses))]

# 测试不同激活函数拟合效果
def test_activation_function(args):
    # 加载标量场
    Dataset = SingleTimeStampDataloader(args.dataset, AttributesName='v02',time_step=args.time_step)
    Dataset.load_data() 
    resolution = Dataset.get_data_resolution()
    whole_volume = Block(Dataset.get_data(), resolution, pos = get_query_coords(vec3f(-1), vec3f(1), resolution).reshape([-1, 3]))
    block_generator = DataPreprocess.BlockGenerator(Dataset.get_data(), Dataset.get_data_resolution())
    data_block_array = block_generator.generate_data_block(args.block_size, method=args.block_gen_method, block_num=args.block_num)

    # 加载模型
    model = create_modulated_siren_maml(args)

    # 转为dataset
    dataset = DataPreprocess.MetaDataset([whole_volume])
    data_loader = torch.utils.data.DataLoader(dataset[0], batch_size=1024,
                                              shuffle=False, num_workers=2)

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr = args.outer_lr)
    mse_loss = torch.nn.MSELoss(reduction='mean')
    for i in trange(args.maml_epoches):
        total_loss = 0.0
        for data in tqdm(data_loader):
            x, y = data[:,:3], data[:,-1].unsqueeze(-1) # data shape:[x*y*z, channel]
            if args.input_ch == 4:
                x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)
            if torch.cuda.is_available():
                x = x.to(torch.float32).to(get_device(args.GPU))
                y = y.to(torch.float32).to(get_device(args.GPU))
            rec_fn = model.forward(x)
            loss = mse_loss(rec_fn, y)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss = total_loss/len(data_loader)
        tqdm.write(f"Iter: {i} Loss: {total_loss} PSNR: {mse2psnr(total_loss)}")
        
    reconstruction = []
    for data in tqdm(data_loader):
        x, y = data[:,:3], data[:,3].unsqueeze(-1) # data shape:[x*y*z, channel]
        if args.input_ch == 4:
            x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)
        if torch.cuda.is_available():
            x = x.to(torch.float32).to(get_device(args.GPU))
            y = y.to(torch.float32).to(get_device(args.GPU))
        with torch.no_grad():
            rec_fn = model.forward(x)
        reconstruction.append(rec_fn)
    reconstruction = torch.cat(reconstruction, dim=0)
    pos = get_query_coords(vec3f(-1), vec3f(1), resolution).reshape([-1, 3])
    block_reconstruct = Block(reconstruction.cpu().detach().numpy().reshape(resolution), resolution, pos)
    vtk_draw_blocks([whole_volume, block_reconstruct])
    return model, block_reconstruct

# num_query_steps 设置为 maml inner loop steps 最好
def parellel_maml_reassignment(args, blocks, templates):
    groups = [[] for i in range(len(templates))]
    losses = [0 for i in range(len(blocks))]
    tasks = []
    # 测试不同template对block的拟合好坏
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, block_i in enumerate(blocks):
            tasks.append(executor.submit(parellel_templates_optimize_block, templates, block_i, args, i))
        
        for future in as_completed(tasks):
            min_loss, group_index, block_index = future.result()
            tqdm.write(f"Block: {block_index}  min_Loss: {min_loss} PSNR: {mse2psnr(torch.tensor(min_loss))}")
            groups[group_index].append(block_index)
            losses[block_index] = min_loss
    return groups, losses

def blocks_to_dataset(blocks, args):
    dataset = DataPreprocess.MetaDataset(blocks)
    dataset = torch.tensor(np.array(dataset)).to(torch.float32).to(get_device(args.GPU)).requires_grad_()
    return dataset

# num_query_steps 设置为 maml inner loop steps 最好
def maml_reassignment_test(args, blocks, templates):
    # initial
    groups = [[] for i in range(len(templates))]
    losses = [[] for i in range(len(blocks))]
    MSE_loss = torch.nn.MSELoss(reduction='mean')
    dataset = DataPreprocess.MetaDataset(blocks)
    dataset = torch.tensor(np.array(dataset)).to(torch.float32).to(get_device(args.GPU))
    x, gradient, y = dataset[:,:,:3], dataset[:,:,3:6], dataset[:,:,-1].unsqueeze(-1)
    if args.input_ch == 4:
        x = torch.cat((x, torch.ones(*list(x.shape[:-1])+[1])), dim=-1)

    # blocks loss for each template
    for i, template in tqdm(enumerate(templates), total=len(templates)):
        TemplateGenerator = MetaTemplate(template, args)
        for j in trange(x.shape[0]):
            modulation = torch.zeros(1, 1, args.latent_dim).requires_grad_().to(get_device(args.GPU))
            modulation = TemplateGenerator.modulation_forward_test(x[j:j+1], y[j:j+1], modulation)
            with torch.no_grad():
                reconstruct_y = TemplateGenerator.net.modulated_forward(x[j:j+1], modulation)
            loss = MSE_loss(reconstruct_y, y[j:j+1]).item()
            losses[j].append(loss)
    for i, loss in enumerate(losses):
        groups[loss.index(min(loss))].append(i)
        losses[i] = min(loss)
    return groups, losses

# num_query_steps 设置为 maml inner loop steps 最好
def maml_reassignment(args, blocks, templates, groups_old=None):
    groups = [[] for i in range(len(templates))]
    losses = [0 for i in range(len(blocks))]
    # 测试不同template对block的拟合好坏
    for i, block_i in tqdm(enumerate(blocks), total=len(blocks)):
        temp_templates = deepcopy(templates)
        min_loss, index = templates_optimize_block(temp_templates, block_i, args)
        if groups_old!=None:
            origin_index = [1 if i in group else 0 for group in groups_old].index(1)
            tqdm.write(f"Block: {i}  min_Loss: {min_loss} PSNR: {mse2psnr(min_loss)} index: {index} origin index:{origin_index}")
        else:
            tqdm.write(f"Block: {i}  min_Loss: {min_loss} PSNR: {mse2psnr(min_loss)} index: {index}")
        groups[index].append(i)
        losses[i] = min_loss
    return groups, losses

def maml_optimize_template_draw(template, data, fn, optimize_steps, lrate=1e-2, path=None, lrate_decay=True):
    grad_vars = list(template.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9,0.999))
    loss_running = 0.0
    loss_d = []
    steps = []
    if path is not None:
        vtk_draw_templates([[template, fn]], np.array([150, 150, 150]), True,
                           path + '_{:06d}.png'.format(0))
    for t in trange(optimize_steps):
        loss_running = 0.0
        for batch_x, batch_y in data:
            batch_x = np.squeeze(batch_x, axis=0)
            batch_y = np.squeeze(batch_y, axis=0)
            if torch.cuda.is_available():
                batch_x = torch.Tensor(batch_x).to(torch.float32).to(get_model_device(template))
                batch_y = torch.Tensor(batch_y).to(torch.float32).to(get_model_device(template))
            res_fn = fn(batch_x, template)
            MSE_loss = torch.nn.MSELoss(reduction='mean')
            loss = MSE_loss(res_fn, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_running += loss.item()/len(data)
        loss_d.append(losses.mse2psnr(torch.tensor(loss_running)))
        steps.append(t)
        # TODO 加上学习率衰减
        if lrate_decay is True:
            decay_rate = 0.1
            decay_steps = 500
            # decay_steps = args.lrate_decay
            new_lrate = lrate * (decay_rate ** (t / decay_steps))
            # print("new rate:", new_lrate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
        if path is not None and (t==0 or t==4 or t==9 or t==19 or t==49 or t==99 or t==499 or t==999):
            print('times:', t, ' ', mse2psnr(torch.tensor(loss_running)))
            vtk_draw_templates([[template, fn]], np.array([150, 150, 150]), True,
                               path+'_{:06d}.png'.format(t+1))
        vtk_draw_templates([[template, fn]], np.array([150, 150, 150]))
    return loss_running, steps, loss_d

def get_templates_vector(templates, layer_index, args):
    hidden_layer_extractor = cluster.Network_Latent_Vector(templates, layer_index, args)
    hidden_layer_extractor.register_hook()
    templates_latent_vector = hidden_layer_extractor.get_latent_vectors()
    hidden_layer_extractor.remove_hook()
    return templates_latent_vector

def calculate_templates_heat_map(templates_latent_vector):
    templates_latent_vector = [vector.cpu().detach().numpy() for vector in templates_latent_vector]
    MI_calculator = cluster.Variable_MI()
    MI_calculator.variables_CKA_heat_map(np.array(templates_latent_vector))
    return templates_latent_vector

def test_templates_fitting(groups, data_block_array, templates, repeat_num, args):
    print("groups size:", len(groups), "templates size:", len(templates))
    for i, template in tqdm(enumerate(templates), total=len(templates)):
        blocks_temp = [data_block_array[block_id] for block_id in groups[i]]
        res_blocks = []
        for block_i in tqdm(blocks_temp):
            template_temp = deepcopy(template)
            # TODO MINER的方法加速
            data = []
            data_loader = torch.utils.data.DataLoader(block_i, batch_size=1,
                                                        shuffle=True, num_workers=2)
            for batch_x, batch_y in data_loader:
                data.append([batch_x, batch_y])

            maml_optimize_template(template_temp[0], data, template_temp[1], args.query_steps, args.query_lrate)
            # 得到优化后的blocks
            res_blocks.append(template_temp)
        if len(blocks_temp) != 0:
            os.makedirs(os.path.join(args.basedir, args.expname, f'repeat_{repeat_num:06d}'), exist_ok=True)
            vtk_draw_blocks(blocks_temp, args.vtk_off_screen_draw,
                            os.path.join(args.basedir, args.expname, f'repeat_{repeat_num:06d}',
                                            'origin_template{:04d}.png'.format(i)))
            vtk_draw_templates(res_blocks, args.block_size, args.vtk_off_screen_draw,
                            os.path.join(args.basedir, args.expname, f'repeat_{repeat_num:06d}',
                                            'fit_template{:04d}.png'.format(i, repeat_num)))  
            

