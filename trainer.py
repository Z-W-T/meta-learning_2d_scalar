import time
import torch
from torch.utils.tensorboard import SummaryWriter
import maml_wt
import time
import pickle
import config_parser
import os
import dataloaders
import DataPreprocess
import utils
import numpy as np
import pyvista as pv
import math
import glob
import networks
import copy
import losses
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

torch.backends.cudnn.benchmark = True


def train_mlp(args):
    # 载入数据
    dataset = AsteroidDataset('pv_insitu_300x300x300_25982.vti', 'prs', 100)
    dataset.read_volume_data()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    writer = SummaryWriter(os.path.join(basedir, expname))

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create MLP model
    start, network_query_fn, grad_vars, optimizer, model = create_mlp(args)
    global_step = start
    n_iters = 1000
    print('Begin')
    start = start + 1

    for i in trange(start, n_iters):
        time0 = time.time()
        loss_running = 0.0
        for batch_x, batch_y in train_loader:
            if torch.cuda.is_available():
                batch_x = batch_x.to(torch.float32).to(get_device(args.GPU))
                batch_y = batch_y.to(torch.float32).to(get_device(args.GPU))

            res_fn = network_query_fn(batch_x, model)
            MSE_loss = torch.nn.MSELoss(reduction='mean')
            loss = MSE_loss(res_fn, batch_y)
            train_loss = torch.mean(torch.abs(res_fn-batch_y))

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optimizer.step()
            loss_running += train_loss.item()/(len(train_loader)/args.batch_size)
        # NOTE: IMPORTANT!
        writer.add_scalar('train_loss', loss_running, global_step=global_step)
        decay_rate = 0.1
        decay_steps = args.lrate_decay*1000
        # 学习率指数级下降
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        print("new rate:", new_lrate)

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        dt = time.time() - time0
        print(f"Step: {global_step}, Loss: {loss_running}, Time: {dt}")

        # Rest is logging
        if i % args.i_weights == 0 or i == 1:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss_running}")

        # validation
        if i % args.i_validation == 0:
            test_loss = 0.0
            num = 0
            with torch.no_grad():
                pass

        global_step += 1


# baseline
def train_multi_mlp(args):
    # TODO 来改这个
    # vDataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_29693.vti', 'v02')
    # vDataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_44778.vti', 'v02')
    vDataset = AsteroidDataset('./data/ya32/pv_insitu_300x300x300_100174.vti', 'v02')
    # vDataset = AsteroidDataset('./data/ya31/pv_insitu_300x300x300_44560.vti', 'v02')
    # vDataset = AsteroidDataset('./data/ya32/pv_insitu_300x300x300_053763.vti', 'v02')
    # vDataset = AsteroidDataset('./data/yB31/pv_insitu_300x300x300_44194.vti', 'v02')
    # 读取参数
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    # 分块 volume
    # 每一块选择初始化
    vDataset.read_volume_data()
    data_block_size = args.block_size
    # 生成训练数据
    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)
    # 默认uniform
    data_block_array = block_generator.generate_data_block(data_block_size)
    MN = MultiNetwork(args, len(data_block_array))
    blocks = BlockSum(data_block_array)
    data_loader = torch.utils.data.DataLoader(blocks, batch_size=1,
                                              shuffle=True, num_workers=2)

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    vtk_draw_single_volume(vDataset, False, '29693.png')
    return
    df = open(os.path.join(basedir, expname, 'training_losses.txt'), mode='wb')
    epoches = 1000
    losses_plot = []
    epoches_plot = []
    for i in trange(epoches):
        loss_running = 0.0
        for batch_x, batch_y in data_loader:
            batch_x = np.squeeze(batch_x, axis=0)
            batch_y = np.squeeze(batch_y, axis=0)
            if torch.cuda.is_available():
                batch_x = torch.Tensor(batch_x).to(torch.float32).to(get_device(args.GPU))
                batch_y = torch.Tensor(batch_y).to(torch.float32).to(get_device(args.GPU))
            res = MN(batch_x)
            MSE_loss = torch.nn.MSELoss(reduction='mean')
            loss = MSE_loss(res, batch_y)
            MN.optimizer_zero_grad()
            loss.backward()
            MN.optimizer_step()

            loss_running += loss.item()/len(data_loader)
        losses_plot.append(mse2psnr(torch.tensor(loss_running)))
        epoches_plot.append(i)
        # TODO 加上学习率衰减
        decay_rate = 0.1
        decay_steps = 500
        # decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        MN.decay_lrate(new_lrate)
        tqdm.write(f"[Multi MLP] Iter: {i} Loss: {loss_running}")
        if i % args.i_weights == 0:
            create_path(os.path.join(basedir, expname,
                                     'yb31_44194_', '{:06d}'.format(i)))
            if args.vtk_off_screen_draw:
                vtk_draw_multi_modules(MN, data_block_size, [300, 300, 300], args.vtk_off_screen_draw,
                                       os.path.join(basedir, expname,
                                        'yb31_44194_','{:06d}'.format(i), 'modules.png'))
            MN.saved_checkpoints(os.path.join(basedir, expname,'yb31_44194_',
                                              '{:06d}'.format(i)))
    # plot loss
    pickle.dump(losses_plot, df)
    pickle.dump(epoches_plot, df)
    df.close()
    plot_curve(epoches_plot, losses_plot,
               'Iterations', 'PSNR(dB)',
               path=os.path.join(basedir, expname))


def train_meta_model(args):
    # 创建网络模型
    dataset_1 = AsteroidDataset('pv_insitu_300x300x300_23878.vti', 'prs')
    dataset_1.read_volume_data()
    train_loader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    dataset_2 = AsteroidDataset('pv_insitu_300x300x300_25982.vti', 'prs')
    dataset_2.read_volume_data()
    train_loader_2 = torch.utils.data.DataLoader(dataset_2, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    dataset_3 = AsteroidDataset('pv_insitu_300x300x300_26766.vti', 'prs')
    dataset_3.read_volume_data()
    train_loader_3 = torch.utils.data.DataLoader(dataset_3, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    start, network_query_fn, grad_vars, optimizer, model = create_mlp(args)
    MSE_loss = torch.nn.MSELoss(reduction='mean')
    # maml = MetaTemplate(model, network_query_fn, MSE_loss)
    data_block_array = [
        dataset_1, dataset_2, dataset_3
    ]
    meta_dataset = MetaDataset(data_block_array)

    basedir = args.basedir
    # only for test
    expname = args.expname+'_meta_test'
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    writer = SummaryWriter(os.path.join(basedir, expname))

    num_parameters = count_parameters(model)

    print(f'\n\nTraining model with {num_parameters} parameters\n\n')

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    total_steps = 0
    epoches = 100

    train_loader_meta = torch.utils.data.DataLoader(meta_dataset, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=args.num_workers)

    for _ in trange(start, epoches):
        # for i, set_ in enumerate(meta_dataset):
        for batch_input in train_loader_meta:
            # forward function 应该输出batch的数据量
            # x_spt, y_spt, x_qry, y_qry = set_['context'][0], set_['context'][1], set_['query'][0], set_['query'][1]
            # if torch.cuda.is_available():
            #     x_spt = torch.from_numpy(x_spt).to(torch.float32).to(get_device(args.GPU))
            #     y_spt = torch.from_numpy(y_spt).to(torch.float32).to(get_device(args.GPU))
            #     x_qry = torch.from_numpy(x_qry).to(torch.float32).to(get_device(args.GPU))
            #     y_qry = torch.from_numpy(y_qry).to(torch.float32).to(get_device(args.GPU))
            #
            # print(maml(x_spt, y_spt, x_qry, y_qry))
            print(batch_input)
            meta_split(batch_input)

        total_steps += 1


def train_meta_template(args):
    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # create data block for training
    dataset = dataloaders.MultiTimeStampDataloader('2D-airfoil')
    dataset.load_interpolated_data()
    block_generator = DataPreprocess.BlockGenerator(dataset.get_data())
    data_block_array = block_generator.generate_data_block(block_size=args.block_size, method=args.block_gen_method)
    # whole_volume2 = fill_whole_volume(args, [block.v for block in data_block_array], np.array([762, 202]))

    # 绘制block
    # utils.vtk_draw_blocks([whole_volume2])
    
    # block分组初始化
    groups = maml_wt.maml_init(data_block_array, args)
    # utils.draw_groups(data_block_array, groups)
    templates = None

    for j in trange(args.repeat_num):
        # 元学习生成templates
        templates = maml_wt.maml_templates_generate(groups, data_block_array, args, templates)

        # 绘制templates
        # templates_block = [[] for template in templates]
        # idx=0
        # for i, template in enumerate(templates):
        #     while groups[idx]==[] and idx<len(groups):
        #         idx += 1
        #     x = utils.get_query_coords(utils.vec2f(-1), utils.vec2f(1), data_block_array[0].res).reshape([-1, 2]).astype(np.float32)
        #     x = torch.tensor(x).to(utils.get_device(args.GPU))
        #     with torch.no_grad():
        #         rec_y = template(x).reshape(args.block_size).cpu()
        #     templates_block[i].append(DataPreprocess.Block(rec_y, data_block_array[0].res))
        #     templates_block[i].extend([data_block_array[k] for k in groups[idx]])
        #     idx += 1
        #     utils.vtk_draw_blocks(templates_block[i])

        # 重新分组
        groups_old = groups
        groups, _ = maml_wt.maml_reassignment(args, data_block_array, templates, groups_old)

        # 保存模板
        if (j+1) % args.i_weights == 0 or j == 0 or groups_old == groups:
            for i, template in enumerate(templates):
                os.makedirs(os.path.join(basedir, expname, 'epoches_{:06d}'.format(j+1)), exist_ok=True)
                path = os.path.join(basedir, expname, 'epoches_{:06d}'.format(j+1), '{:06d}_template.tar'.format(i))
                torch.save({'network_fn_state_dict': template.state_dict(),}, path)
                print('Saved checkpoints at', path)

        # 根据交换次数提前截止
        dict_old = {}
        dict_current = {}
        exchange_num = 0
        for i in range(len(groups_old)):
            for num in groups_old[i]:
                dict_old[num]=i
        for i in range(len(groups)):
            for num in groups[i]:
                dict_current[num]=i
        for i in range(len(data_block_array)):
            if(dict_old[i]!=dict_current[i]):
                exchange_num += 1
        print(f'exchange number {exchange_num}')
        if exchange_num < math.ceil(len(data_block_array)/10):
            break

def train_multi_mlp_based_templates(args):
    # TODO 验证 template可以加速的训练时间
    basedir = args.basedir
    expname = args.expname
    epoches_name = 'epoches_000028'
    # 载入template

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, epoches_name, f) for f in sorted(os.listdir(os.path.join(basedir,
                expname, epoches_name))) if 'tar' in f]

    print("Found ckpts", ckpts)
    templates = []
    for ckpt_path in ckpts:
        model, network_query_fn = create_net(args)
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['network_fn_state_dict'])
        templates.append([model, network_query_fn])
        res = template_to_volume([model, network_query_fn], vec3f(50))
        res = np.array(res)
        print(res)

    # 训练数据
    # vDataset = AsteroidDataset('pv_insitu_300x300x300_29693.vti', 'v02')
    df = open(os.path.join(basedir, expname, epoches_name, '44194_psnr.txt'), mode='wb')
    # 用这个做测试
    # vDataset = AsteroidDataset('./data/ya11/pv_insitu_300x300x300_26886.vti', 'v02')
    # vDataset = AsteroidDataset('./data/ya31/pv_insitu_300x300x300_44560.vti', 'v02')
    # vDataset = AsteroidDataset('./data/ya32/pv_insitu_300x300x300_053763.vti', 'v02')
    # vDataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_29693.vti', 'v02')
    vDataset = AsteroidDataset('./data/yB31/pv_insitu_300x300x300_44194.vti', 'v02')
    vDataset.read_volume_data()
    vtk_draw_single_volume(vDataset, True, '44194.png')
    data_block_size = args.block_size

    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)
    # 默认uniform
    data_block_array = block_generator.generate_data_block(data_block_size)
    blocks = BlockSum(data_block_array)
    data_loader = torch.utils.data.DataLoader(blocks, batch_size=1,
                                              shuffle=True, num_workers=2)

    # 创建网络
    MN = MultiNetwork(args, len(data_block_array))
    # 按照template初始化网络
    MN.initialize_multi_mlp(templates, data_loader, query_steps=args.query_steps, query_lrate=args.query_lrate)

    epoches = 1000
    psnr_plot = []
    epoches_plot = []

    for i in trange(epoches):
        loss_running = 0.0
        num = 0.0
        for batch_x, batch_y in data_loader:
            batch_x = np.squeeze(batch_x, axis=0)
            batch_y = np.squeeze(batch_y, axis=0)
            if torch.cuda.is_available():
                batch_x = torch.Tensor(batch_x).to(torch.float32).to(get_device(args.GPU))
                batch_y = torch.Tensor(batch_y).to(torch.float32).to(get_device(args.GPU))
            res = MN(batch_x)
            MSE_loss = torch.nn.MSELoss(reduction='mean')
            loss = MSE_loss(res, batch_y)

            MN.optimizer_zero_grad()
            loss.backward()
            MN.optimizer_step()
            loss_running += loss.item() * batch_x.shape[1]
            num = num + batch_x.shape[1]
        loss_running = loss_running / num
        psnr_plot.append(mse2psnr(torch.tensor(loss_running)))
        epoches_plot.append(i)
        tqdm.write(f"[Multi MLP] Iter: {i} Loss: {loss_running}")
        # TODO 加上学习率衰减
        decay_rate = 0.1
        decay_steps = 400
        # decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        MN.decay_lrate(new_lrate)
        if i % args.i_weights == 0:
            create_path(os.path.join(basedir, expname,
                                     epoches_name, 'yb31_44194_', '{:06d}'.format(i)))
            if args.vtk_off_screen_draw:
                vtk_draw_multi_modules(MN, data_block_size, [300, 300, 300], args.vtk_off_screen_draw,
                                       os.path.join(basedir, expname,
                                       epoches_name, 'yb31_44194_','{:06d}'.format(i), 'modules.png'))
            MN.saved_checkpoints(os.path.join(basedir, expname, epoches_name, 'yb31_44194_',
                                              '{:06d}'.format(i)))

        # 画出来
    plot_curve(epoches_plot, psnr_plot, 'Iterations', 'PSNR(dB)',
               path=os.path.join(basedir, expname, epoches_name))
    pickle.dump(psnr_plot, df)
    pickle.dump(epoches_plot, df)
    df.close()
    return MN


def train_meta_template_multi_timestamp(args):
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    block_size = args.block_size

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # for tensorboard
    # writer = SummaryWriter(os.path.join(basedir, expname))
    vDataset = MultiTimeStampDataset('asteroid', 'v02')
    vDataset.read_data()

    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)

    data_block_array = block_generator.generate_data_block(block_size, method=args.block_gen_method,
                                             block_num=args.block_num)

    groups = maml_init(args.groups_num, data_block_array, args, context=args.group_init)

    if args.vtk_off_screen_draw:
        vtk_draw_blocks(data_block_array, off_screen=args.vtk_off_screen_draw, file_name=
                        os.path.join(basedir, expname, 'blocks_array.png'))

    templates = None
    groups_old = None
    for j in range(args.repeat_num):
        # TODO 生成
        templates = maml_template_generate(groups, data_block_array, args, templates, groups_old)
        torch.cuda.empty_cache()
        if args.vtk_off_screen_draw:
            # 绘制templates
            vtk_draw_templates(templates, block_size, args.vtk_off_screen_draw,
                               os.path.join(basedir, expname, 'templates_epoches_{:04d}.png'.format(j)))

            temp_i = 0
            for i, group in enumerate(groups):
                blocks_temp = [data_block_array[block_id] for block_id in group]
                res_blocks = []
                for block_i in blocks_temp:
                    template_temp = deepcopy(templates[temp_i])
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
                    vtk_draw_blocks(blocks_temp, args.vtk_off_screen_draw,
                                    os.path.join(basedir, expname, 'blocks_epoches{:04d}_{:04d}.png'.format(j, temp_i)))
                    vtk_draw_templates(res_blocks, block_size, args.vtk_off_screen_draw,
                                    os.path.join(basedir, expname,
                                                 'blocks_epoches_optimize_{:04d}_{:04d}.png'.format(j, temp_i)))
                    temp_i += 1

        groups_old = groups
        groups = maml_reassignment(data_block_array, templates, num_query_steps=args.query_steps,
                                   query_lrate=args.query_lrate)

        if (j + 1) % args.i_weights == 0 or j == 0 or groups_old == groups:
            if groups_old == groups:
                args.maml_epoches = 200
                templates = maml_template_generate(groups, data_block_array, args, templates, None)
            for i, [template, _] in enumerate(templates):
                os.makedirs(os.path.join(basedir, expname, 'epoches_{:06d}'.format(j + 1)), exist_ok=True)
                path = os.path.join(basedir, expname, 'epoches_{:06d}'.format(j + 1), '{:06d}_template.tar'.format(i))
                torch.save({
                    'network_fn_state_dict': template.state_dict(),
                }, path)
                print('Saved checkpoints at', path)
        if groups == groups_old:
            break


def test_maml_and_query(args):
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')

    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    vDataset = AsteroidDataset('./data/asteroid/pv_insitu_300x300x300_44875.vti', 'v02')
    vDataset.read_volume_data()
    data_block_size = args.block_size
    block_generator = BlockGenerator(vDataset.get_volume_data(), vDataset.get_volume_res(), args.block_chunk)
    # data_block_array = block_generator.generate_data_block_in_center(data_block_size, 30)
    data_block_array = block_generator.generate_data_block_with_offset(data_block_size, args.block_num)
    if args.vtk_off_screen_draw:
        vtk_draw_blocks(data_block_array, off_screen=args.vtk_off_screen_draw, file_name=
                        os.path.join(basedir, expname, 'blocks_array.png'))

    templates = None
    groups_old = None
    data_block_array_train = data_block_array[0:args.block_num-1]
    data_block_array_test = [data_block_array[-1]]
    groups = maml_init(args.groups_num, data_block_array_train, args, context=args.group_init)
    templates = maml_template_generate(groups, data_block_array_train, args, templates, groups_old)
    # 画出一个template
    path = os.path.join(basedir, expname, 'template.tar')
    torch.save({
        'network_fn_state_dict': templates[0][0].state_dict(),
    }, path)
    print('Saved checkpoints at', path)
    if args.vtk_off_screen_draw:
        vtk_draw_templates(templates, data_block_size, args.vtk_off_screen_draw,
                           os.path.join(basedir, expname, 'templates_epoches_{:04d}.png'.format(args.maml_epoches)))
    data_block_array_test = [data_block_array_test[-1]]
    for i, block_i in enumerate(data_block_array_test):
        data_loader = torch.utils.data.DataLoader(block_i, batch_size=1,
                                                  shuffle=True, num_workers=2)
        template = deepcopy(templates[0])
        loss_run, steps, loss_d = maml_optimize_template_draw(template[0], data_loader, template[1],
                                                              args.query_steps, args.query_lrate)
        os.makedirs(os.path.join(basedir, expname, 'block_template_{:06d}'.format(i + 1)), exist_ok=True)

        kong_template = create_net(args)
        loss_kong, steps_kong, loss_d_kong = maml_optimize_template_draw(kong_template[0], data_loader,
                                                                         kong_template[1],
                                                                         args.query_steps, args.query_lrate)

        plot_curve(steps, loss_d, 'steps', 'psnr', x2_vals=steps_kong, y2_vals=loss_d_kong,
                   legend=['template', 'without_template'],
                   path=os.path.join(basedir, expname, 'block_template_{:06d}'.format(i + 1)))


def test_templates(args):
    #载入测试数据
    dataset = dataloaders.SingleTimeStampDataloader('2D-airfoil', timestep=250)
    dataset.load_interpolated_data()
    block_generator = DataPreprocess.BlockGenerator(dataset.get_data())
    data_block_array = block_generator.generate_data_block(block_size=args.block_size, method=args.block_gen_method)
    # utils.vtk_draw_blocks(data_block_array)

    # 载入模型
    templates = []
    path = os.path.join(args.basedir, args.expname, 'epoches_000011')
    for template_path in glob.glob(path+'/*'):
        template, _ = networks.create_siren_maml(args)
        template.load_state_dict(torch.load(template_path)['network_fn_state_dict'])
        templates.append(template)
    
    # 选择模型
    groups, _ = maml_wt.maml_reassignment(args, data_block_array, templates)
    x = utils.get_query_coords(utils.vec2f(-1), utils.vec2f(1), data_block_array[0].res).reshape([-1, 2]).astype(np.float32)
    x = torch.tensor(x).to(utils.get_device(args.GPU))
    MSE_loss = torch.nn.MSELoss(reduction='mean')
    rec_block_array = torch.empty([len(data_block_array)]+args.block_size)
    init_block_array = torch.empty([len(data_block_array)]+args.block_size)
    for i, group in tqdm(enumerate(groups), total=len(groups)):
        for index in group:
            template = copy.deepcopy(templates[i])
            optimizer = torch.optim.Adam(template.parameters(), lr=args.outer_lr)
            y = torch.tensor(data_block_array[index].v.reshape(-1,1)).to(utils.get_device(args.GPU))
            for j in range(50):
                rec_y = template(x)
                loss = MSE_loss(rec_y, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'rec block index {index} psnr {losses.mse2psnr(loss)}')
            rec_block_array[index]=rec_y.detach().cpu().reshape(args.block_size)
            # 比较无元学习的网络拟合效果
            init_template, _ = networks.create_siren_maml(args)
            optimizer = torch.optim.Adam(init_template.parameters(), lr=args.outer_lr)
            for j in range(50):
                rec_y = init_template(x)
                loss = MSE_loss(rec_y, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'init block index {index} psnr {losses.mse2psnr(loss)}')
            init_block_array[index]=rec_y.detach().cpu().reshape(args.block_size)
    # 绘制
    origin_volume = fill_whole_volume(args, [block.v for block in data_block_array], [762, 202])
    rec_volume = fill_whole_volume(args, rec_block_array, [762, 202])
    init_volume = fill_whole_volume(args, init_block_array, [762, 202])
    utils.vtk_draw_blocks([origin_volume, rec_volume, init_volume])
    draw_scene([origin_volume, rec_volume, init_volume])
    print(f'template reconstruct PSNR {losses.mse2psnr(MSE_loss(rec_block_array,torch.tensor(np.array([block.v for block in data_block_array]))))}')
    print(f'init network reconstruct PSNR {losses.mse2psnr(MSE_loss(init_block_array,torch.tensor(np.array([block.v for block in data_block_array]))))}')


def fill_whole_volume(args, data_block_array, res):
    volume_data = np.empty(res)
    res = np.array(res)
    [w, h] = np.ceil(res / args.block_size)
    [chunk_w, chunk_h] = np.ceil(res / [w,h])
    idx = 0
    for i in range(0, res[0], int(chunk_w)):
        for j in range(0, res[1], int(chunk_h)):
            if i+args.block_size[0] > res[0]:
                i = res[0] - args.block_size[0]
            if j+args.block_size[1] > res[1]:  
                j = res[1] - args.block_size[1]
            volume_data[i:i+args.block_size[0],j:j+args.block_size[1]]=data_block_array[idx]
            idx += 1
    whole_block = DataPreprocess.Block(volume_data, res)
    return whole_block

def draw_scene(volumes):
    file_path = '/home/XiYang/MAML_KiloNet/source/data/INS3D/INS3D/2D-airfoil/dat/output/fort0000200.30'
    with open(file_path, 'rb') as file:
        jmax, kmax, lmax = np.fromfile(file, dtype=np.int32, count=3)
        print(f"jmax: {jmax}, kmax: {kmax}, lmax: {lmax}")
        x = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
        y = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
        z = np.fromfile(file, dtype='<f4', count=jmax*kmax*lmax).reshape((lmax, kmax, jmax))
    file.close()
    zoom_factors = (2, 2)
    interpolated_x = zoom(x[0], zoom_factors, order=1)
    interpolated_y = zoom(y[0], zoom_factors, order=1)
    interpolated_z = zoom(z[0], zoom_factors, order=1)
    points = np.stack((interpolated_x,interpolated_y,interpolated_z),axis=-1).reshape(-1,3)

    # 创建绘图对象
    plotter = pv.Plotter()
    for i, volume in enumerate(volumes):
        scalar_values = volume.v.reshape(-1,1)
        cloud = pv.PolyData(points+[i*20, 0, 0])
        cloud['scalar'] = scalar_values

        # 添加点云到绘图对象
        plotter.add_mesh(cloud, scalars='scalar', render_points_as_spheres=True)

    # 设置显示参数
    plotter.show_grid()
    plotter.show_axes()

    # 显示绘图
    plotter.show()


def main():
    args = config_parser.get_args("config.txt")
    if args.task == 'train_KiloNet':
        train_multi_mlp(args)
    elif args.task == 'train_KiloNet_by_templates':
        train_multi_mlp_based_templates(args)
    elif args.task == 'train_templates':
        train_meta_template(args)
    elif args.task == 'train_templates_multi_timestamp':
        train_meta_template_multi_timestamp(args)
    elif args.task == 'test':
        test_templates(args)
    else:
        print("No task to do!")
        draw_templates(args)


if __name__ == '__main__':
    main()