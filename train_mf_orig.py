import imp
import itertools
import logging
import pathlib
import pdb
import random
import shutil
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import pandas
import os
import pickle

# sys.path.insert(0, '/home/tomerweiss/multiPILOT2')

import numpy as np
# np.seterr('raise')
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio
from torch.utils.data import DataLoader
from common.args import Args
from data import transforms
from data.mri_mf_data import SliceData
import matplotlib
import h5py

# matplotlib.use( 'tkagg' )
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.subsampling_mf_model import Subsampling_Model
from scipy.spatial import distance_matrix
from tsp_solver.greedy import solve_tsp
from common.utils import get_vel_acc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    def __init__(self, resolution=[384, 144]):
        self.resolution = resolution

    def __call__(self, kspace, target, attrs, fname, slice):
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.resolution[0], self.resolution[1]))
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)

        target = transforms.to_tensor(target)
        target = transforms.center_crop(target.unsqueeze(0), (self.resolution[0], self.resolution[1])).squeeze()
        target, mean, std = transforms.normalize_instance(target, eps=1e-11)
        mean = std = 0
        return image, target, mean, std  # , attrs['norm'].astype(np.float32)


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def boost_examples(files, num_frames_per_example):
    examples_per_clip = {}
    for fname in sorted(files):
        curr_num_examples = 0
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace']  # [slice, frames, coils, h,w]
            for start_frame_index in range(kspace.shape[1] - num_frames_per_example):
                num_slices = kspace.shape[0]
                curr_examples = [(fname, slice, start_frame_index, start_frame_index + num_frames_per_example) for slice
                                 in range(num_slices)]
                curr_num_examples += len(curr_examples)
        examples_per_clip[fname] = curr_num_examples

    max_examples = np.max([k for k in examples_per_clip.values()])
    factors = {k: np.int(np.floor(max_examples / v)) for k, v in examples_per_clip.items()}
    return factors


def get_rel_files(files, resolution, num_frames_per_example):
    rel_files = []
    for fname in sorted(files):
        with h5py.File(fname, 'r') as data:
            if not 'aug.h5' in fname:
                kspace = data['kspace']  # [slice, frames, coils, h,w]
            else:
                kspace = data['images']
            if kspace.shape[3] < resolution[0] or kspace.shape[4] < resolution[1]:
                continue
            if kspace.shape[1] < num_frames_per_example:
                continue
        rel_files.append(fname)
    return rel_files


def create_datasets(args):
    if args.augment:  # all pre testing already done
        rel_files = [str(args.data_path) + '/' + str(fname) for fname in os.listdir(args.data_path) if
                     os.path.isfile(os.path.join(args.data_path, fname))]
    else:
        ocmr_data_attributes_location = '/home/tomerweiss/dor/OCMR/OCMR/ocmr_data_attributes.csv'
        df = pandas.read_csv(ocmr_data_attributes_location)
        df.dropna(how='all', axis=0, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)
        rel_files = [args.data_path._str + '/' + k for k in df[df['smp'] == 'fs']['file name'].values]
        rel_files = get_rel_files(rel_files, DataTransform().resolution, args.num_frames_per_example)
    clips_factors = None
    if args.boost:
        clips_factors = boost_examples(rel_files, args.num_frames_per_example)
    np.random.shuffle(rel_files)
    train_ratio = 0.8  # TODO: make sure the long video is in test set!
    num_train = int(np.ceil(len(rel_files) * train_ratio))
    train_files = rel_files[:num_train]
    val_files = rel_files[num_train:]

    if not os.path.exists('output'):
        os.makedirs('output')

    with open('output/train_val_slice', 'w') as f:
        f.write("Train Files:\n")
        for fname in train_files:
            f.write(fname + '\n')
        f.write('\n')
        f.write("Validation Files:\n")
        for fname in val_files:
            f.write(fname + '\n')

    train_data = SliceData(
        files=train_files,
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        num_frames_per_example=args.num_frames_per_example,
        clips_factors=clips_factors
    )
    dev_data = SliceData(
        files=val_files,
        transform=DataTransform(),
        sample_rate=args.sample_rate,
        num_frames_per_example=args.num_frames_per_example,
        clips_factors=clips_factors
    )
    return dev_data, train_data


def display_target(target, delay=1.):
    i = 0
    for img in target:
        plt.cla()
        plt.imshow(img, cmap='gray')
        plt.title('frame #' + str(i))
        plt.pause(delay)
        i += 1


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 8)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1 if args.augment else 20,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=1 if args.augment else 20,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=args.batch_size,
        num_workers=1 if args.augment else 20,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader


def tsp_solver(x):
    # reorder the trajectory according to the TSP solution
    d = distance_matrix(x, x)
    t = solve_tsp(d)
    return x[t, :]


def train_epoch(args, epoch, model, data_loader, optimizer, writer, loader_len):
    model.train()
    avg_loss = 0.
    # ignore! Not in use!
    if epoch == args.TSP_epoch and args.TSP:
        x = model.get_trajectory()
        x = x.detach().cpu().numpy()
        for shot in range(x.shape[0]):
            x[shot, :, :] = tsp_solver(x[shot, :, :])
        v, a = get_vel_acc(x)
        writer.add_figure('TSP_Trajectory', plot_trajectory(x), epoch)
        writer.add_figure('TSP_Acc', plot_acc(a, args.a_max), epoch)
        writer.add_figure('TSP_Vel', plot_acc(v, args.v_max), epoch)
        np.save('trajTSP', x)
        with torch.no_grad():
            model.subsampling.x.data = torch.tensor(x, device='cuda')
        args.a_max *= 2
        args.v_max *= 2
        args.vel_weight = 1e-3
        args.acc_weight = 1e-3

    # if epoch == 30:
    #     v0 = args.gamma * args.G_max * args.FOV * args.dt
    #     a0 = args.gamma * args.S_max * args.FOV * args.dt ** 2 * 1e3
    #     args.a_max = a0 *1.5
    #     args.v_max = v0 *1.5

    # if args.TSP and epoch > args.TSP_epoch and epoch<=args.TSP_epoch*2:
    #     v0 = args.gamma * args.G_max * args.FOV * args.dt
    #     a0 = args.gamma * args.S_max * args.FOV * args.dt ** 2 * 1e3
    #     args.a_max -= a0/args.TSP_epoch
    #     args.v_max -= v0/args.TSP_epoch
    #
    # if args.TSP and epoch==args.TSP_epoch*2:
    #     v0 = args.gamma * args.G_max * args.FOV * args.dt
    #     a0 = args.gamma * args.S_max * args.FOV * args.dt ** 2 * 1e3
    #     args.a_max = a0
    #     args.v_max = v0
    #     args.vel_weight *= 10
    #     args.acc_weight *= 10
    # if args.TSP and epoch==args.TSP_epoch*2+10:
    #     args.vel_weight *= 10
    #     args.acc_weight *= 10
    # if args.TSP and epoch==args.TSP_epoch*2+20:
    #     args.vel_weight *= 10
    #     args.acc_weight *= 10
    if args.TSP:
        if epoch < args.TSP_epoch:
            model.subsampling.interp_gap = 1
        elif epoch < 10 + args.TSP_epoch:
            model.subsampling.interp_gap = 10
            v0 = args.gamma * args.G_max * args.FOV * args.dt
            a0 = args.gamma * args.S_max * args.FOV * args.dt ** 2 * 1e3
            args.a_max -= a0 / args.TSP_epoch
            args.v_max -= v0 / args.TSP_epoch
        elif epoch == 10 + args.TSP_epoch:
            model.subsampling.interp_gap = 10
            v0 = args.gamma * args.G_max * args.FOV * args.dt
            a0 = args.gamma * args.S_max * args.FOV * args.dt ** 2 * 1e3
            args.a_max -= a0 / args.TSP_epoch
            args.v_max -= v0 / args.TSP_epoch
        elif epoch == 15 + args.TSP_epoch:
            model.subsampling.interp_gap = 10
            v0 = args.gamma * args.G_max * args.FOV * args.dt
            a0 = args.gamma * args.S_max * args.FOV * args.dt ** 2 * 1e3
            args.a_max -= a0 / args.TSP_epoch
            args.v_max -= v0 / args.TSP_epoch
        elif epoch == 20 + args.TSP_epoch:
            model.subsampling.interp_gap = 10
            args.vel_weight *= 10
            args.acc_weight *= 10
        elif epoch == 23 + args.TSP_epoch:
            model.subsampling.interp_gap = 5
            args.vel_weight *= 10
            args.acc_weight *= 10
        elif epoch == 25 + args.TSP_epoch:
            model.subsampling.interp_gap = 1
            args.vel_weight *= 10
            args.acc_weight *= 10
    else:
        if epoch < 20:
            model.subsampling.interp_gap = 32
        elif epoch == 20:
            model.subsampling.interp_gap = 16
        elif epoch == 30:
            model.subsampling.interp_gap = 8
        elif epoch == 40:
            model.subsampling.interp_gap = 4
        elif epoch == 46:
            model.subsampling.interp_gap = 2
        elif epoch == 50:
            model.subsampling.interp_gap = 1
    start_epoch = start_iter = time.perf_counter()
    print(f'a_max={args.a_max}, v_max={args.v_max}')
    for iter, data in data_loader:
        optimizer.zero_grad()
        # input, target, mean, std, norm = data
        input, target, mean, std = data
        input = input.to(args.device)
        target = target.to(args.device)
        start = time.time()
        # output = model(input)


        output = model(input)
        # output = transforms.complex_abs(output)  # complex to real
        # output = transforms.root_sum_of_squares(output, dim=1)
        if output.shape[1] != 1:
            output = output.squeeze()

        # Loss on trajectory vel and acc
        x = model.get_trajectory()
        v, a = get_vel_acc(x)
        acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max).abs() + 1e-8, 2)))
        vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max).abs() + 1e-8, 2)))

        # target loss
        rec_loss = F.l1_loss(output, target)
        #rec_loss = F.mse_loss(output.to(torch.float64), target.to(
        #    torch.float64))  # + 3*(epoch>2)*F.mse_loss((target-torch.mean(target,dim=1).unsqueeze(1)).to(torch.float64),(output-torch.mean(output,dim=1).unsqueeze(1)).to(torch.float64))
        if args.TSP and epoch < args.TSP_epoch:
            loss = args.rec_weight * rec_loss
        else:
            loss = args.rec_weight * rec_loss + args.vel_weight * vel_loss + args.acc_weight * acc_loss

        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        psnr_train = psnr(target, output)
        # writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{loader_len:4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'rec_loss: {rec_loss:.4g}, vel_loss: {vel_loss:.4g}, acc_loss: {acc_loss:.4g}'
                f' PSNR: {psnr(target, output)}'
            )
        if iter == loader_len - 1:
            break
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch, rec_loss, vel_loss, acc_loss, psnr_train


def evaluate(args, epoch, model, data_loader, writer, dl_len, train_loss=None, train_rec_loss=None, train_vel_loss=None,
             train_acc_loss=None, psnr_train=None):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        if epoch != 0:
            for iter, data in data_loader:
                # input, target, mean, std, norm = data
                input, target, mean, std = data
                input = input.to(args.device)
                target = target.to(args.device)

                output = model(input)


                # output = transforms.complex_abs(output)  # complex to real
                # output = transforms.root_sum_of_squares(output, dim=1)
                # output = output.squeeze()

                loss = F.l1_loss(output, target)
                #loss = F.mse_loss(output, target)
                losses.append(loss.item())
                # with open(args.exp_dir + '/iter_' + str(iter) + '.pickle', 'wb') as f:
                #     pickle.dump({'target': target.detach().cpu().numpy(), 'pred': output.detach().cpu().numpy()}, f)

                if iter == dl_len - 1:
                    break

            x = model.get_trajectory()
            v, a = get_vel_acc(x)
            acc_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(a, args.a_max), 2)))
            vel_loss = torch.sqrt(torch.sum(torch.pow(F.softshrink(v, args.v_max), 2)))
            rec_loss = np.mean(losses)
            psnr_dev = psnr(target, output)

            if train_rec_loss is None:
                writer.add_scalars('Rec_Loss', {'val': rec_loss}, epoch)
            else:
                writer.add_scalars('Rec_Loss', {'val': rec_loss, 'train': train_rec_loss}, epoch)
            if train_acc_loss is None:
                writer.add_scalars('Acc_Loss', {'val': acc_loss.detach().cpu().numpy()}, epoch)
            else:
                writer.add_scalars('Acc_Loss', {'val': acc_loss.detach().cpu().numpy(), 'train': train_acc_loss}, epoch)
            if train_vel_loss is None:
                writer.add_scalars('Vel_Loss', {'val': vel_loss.detach().cpu().numpy()}, epoch)
            else:
                writer.add_scalars('Vel_Loss', {'val': vel_loss.detach().cpu().numpy(), 'train': train_vel_loss}, epoch)
            if train_loss is None:
                writer.add_scalars('Total_Loss', {
                    'val': rec_loss + acc_loss.detach().cpu().numpy() + vel_loss.detach().cpu().numpy()}, epoch)
            else:
                writer.add_scalars('Total_Loss',
                                   {'val': rec_loss + acc_loss.detach().cpu().numpy() + vel_loss.detach().cpu().numpy(),
                                    'train': train_loss}, epoch)
            if psnr_train is None:
                writer.add_scalars('PSNR', {'val': psnr_dev}, epoch)
            else:
                writer.add_scalars('PSNR', {'val': psnr_dev, 'train': psnr_train}, epoch)
            print(f'Dev PSNR: {psnr_dev}')

        x = model.get_trajectory()
        v, a = get_vel_acc(x)
        if args.TSP and epoch < args.TSP_epoch:
            writer.add_figure('Scatter', plot_scatter(x.detach().cpu().numpy()), epoch)
        else:
            writer.add_figure('Trajectory', plot_trajectory(x.detach().cpu().numpy()), epoch)
            writer.add_figure('Scatter', plot_scatter(x.detach().cpu().numpy()), epoch)
        writer.add_figure('Accelerations_plot', plot_acc(a.cpu().numpy(), args.a_max), epoch)
        writer.add_figure('Velocity_plot', plot_acc(v.cpu().numpy(), args.v_max), epoch)
        writer.add_text('Coordinates', str(x.detach().cpu().numpy()).replace(' ', ','), epoch)
    if epoch == 0:
        return None, time.perf_counter() - start, None
    else:
        return np.mean(losses), time.perf_counter() - start, psnr_dev


def plot_scatter(x):
    if len(x.shape) == 4:
        return plot_scatters(x)
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    for i in range(x.shape[0]):
        ax.plot(x[i, :, 0], x[i, :, 1], '.')
    return fig


def plot_scatters(x):
    fig = plt.figure(figsize=[10, 10])
    for frame in range(x.shape[0]):
        ax = fig.add_subplot(2, 5, frame + 1)
        for i in range(x.shape[1]):
            ax.plot(x[frame, i, :, 0], x[frame, i, :, 1], '.')
            ax.axis([-165, 165, -165, 165])
    return fig


def plot_trajectory(x):
    if len(x.shape) == 4:
        return plot_trajectories(x)
    fig = plt.figure(figsize=[10, 10])
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-165, 165, -165, 165])
    for i in range(x.shape[0]):
        ax.plot(x[i, :, 0], x[i, :, 1])
    return fig


def plot_trajectories(x):
    fig = plt.figure(figsize=[10, 10])
    for frame in range(x.shape[0]):
        ax = fig.add_subplot(2, 5, frame + 1)
        for i in range(x.shape[1]):
            ax.plot(x[frame, i, :, 0], x[frame, i, :, 1])
            ax.axis([-165, 165, -165, 165])
    return fig


def plot_acc(a, a_max=None):
    fig, ax = plt.subplots(2, sharex=True)
    for i in range(a.shape[0]):
        ax[0].plot(a[i, :, 0])
        ax[1].plot(a[i, :, 1])
    if a_max is not None:
        limit = np.ones(a.shape[1]) * a_max
        ax[1].plot(limit, color='red')
        ax[1].plot(-limit, color='red')
        ax[0].plot(limit, color='red')
        ax[0].plot(-limit, color='red')
    return fig


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        videos_display = []
        for example in image.transpose(1, 0):
            grid = torchvision.utils.make_grid(example, nrow=3, pad_value=1)
            videos_display.append(grid)
        vid_tensor = torch.stack(videos_display, dim=0).unsqueeze(0)
        writer.add_video(tag, vid_tensor, fps=10, global_step=epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # input, target, mean, std, norm = data
            input, target, mean, std = data
            input = input.to(args.device)
            target = target.unsqueeze(2).to(args.device)

            save_image(target, 'Target')
            if epoch != 0:

                output = model(input)

                # output = model(input.clone())
                output = output.unsqueeze(2)
                # output = transforms.complex_abs(output)  # complex to real
                # output = transforms.root_sum_of_squares(output, dim=1).unsqueeze(1)

                corrupted = model.subsampling(input).unsqueeze(2)
                cor_all = transforms.root_sum_of_squares(corrupted, -1)

                save_image(output, 'Reconstruction')
                save_image(corrupted[..., 0], 'Corrupted_real')
                save_image(corrupted[..., 1], 'Corrupted_im')
                save_image(cor_all, 'Corrupted')
                save_image(torch.abs(target - output), 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir + '/model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')


def build_model(args):
    model = Subsampling_Model(
        in_chans=10,
        out_chans=10,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        decimation_rate=args.decimation_rate,
        res=args.resolution,
        trajectory_learning=args.trajectory_learning,
        initialization=args.initialization,
        SNR=args.SNR,
        projection_iters=args.proj_iters,
        project=args.project,
        n_shots=args.n_shots,
        interp_gap=args.interp_gap,
        multiple_trajectories=args.multi_traj
    ).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, model):
    optimizer = torch.optim.Adam([{'params': model.subsampling.parameters(), 'lr': args.sub_lr},
                                  {'params': model.reconstruction_model.parameters()}], args.lr)
    return optimizer


def train():
    args = create_arg_parser().parse_args()
    args.v_max = args.gamma * args.G_max * args.FOV * args.dt
    args.a_max = args.gamma * args.S_max * args.FOV * args.dt ** 2 * 1e3
    # print(args.v_max)
    # print(args.a_max)
    args.exp_dir = f'summary/{args.test_name}'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pathlib.Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir)
    with open(args.exp_dir + '/args.txt', "w") as text_file:
        print(vars(args), file=text_file)

    args.checkpoint = f'summary/{args.test_name}/model.pt'
    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        # args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch'] + 1
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model)
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    # logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    enum_train = itertools.cycle(enumerate(train_loader))
    enum_val = itertools.cycle(enumerate(dev_loader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    dev_loss, dev_time, psnr_dev = evaluate(args, 0, model, dev_loader, writer, len(dev_loader))
    visualize(args, 0, model, display_loader, writer)

    for epoch in range(start_epoch, args.num_epochs):
        # scheduler.step(epoch)
        # if epoch>=args.TSP_epoch:
        #     optimizer.param_groups[0]['lr']=0.001
        #     optimizer.param_groups[1]['lr'] = 0.001
        start = time.time()
        train_loss, train_time, train_rec_loss, train_vel_loss, train_acc_loss, psnr_train = train_epoch(args, epoch,
                                                                                                         model,
                                                                                                         enum_train,
                                                                                                         optimizer,
                                                                                                         writer,
                                                                                                         len(train_loader))
        dev_loss, dev_time, psnr_dev = evaluate(args, epoch + 1, model, enum_val, writer, len(dev_loader), train_loss,
                                                train_rec_loss, train_vel_loss, train_acc_loss, psnr_train)

        visualize(args, epoch + 1, model, display_loader, writer)

        if epoch == args.TSP_epoch:
            best_dev_loss = 1e9
        if dev_loss < best_dev_loss:
            is_new_best = True
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
        else:
            is_new_best = False
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s'
            f'DevPSNR = {psnr_dev:.4g}',
        )
        end = time.time() - start
        print(f'epoch time: {end}')
    print(args.test_name)
    print(f'Training done, best epoch: {best_epoch}')
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--test-name', type=str, default='test', help='name for the output dir')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='/mnt/walkure_public/tamirs/',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, default='summary/test/model.pt',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')

    # model parameters
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')
    parser.add_argument('--data-parallel', action='store_true', default=False,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--decimation-rate', default=10, type=int,
                        help='Ratio of k-space columns to be sampled. If multiple values are '
                             'provided, then one of those is chosen uniformly at random for each volume.')

    # optimization parameters
    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=30,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.01,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--sub-lr', type=float, default=4e-2, help='lerning rate of the sub-samping layel')

    # trajectory learning parameters
    parser.add_argument('--trajectory-learning', default=True,
                        help='trajectory_learning, if set to False, fixed trajectory, only reconstruction learning.')
    parser.add_argument('--acc-weight', type=float, default=1e-2, help='weight of the acceleration loss')
    parser.add_argument('--vel-weight', type=float, default=1e-1, help='weight of the velocity loss')
    parser.add_argument('--rec-weight', type=float, default=1, help='weight of the reconstruction loss')
    parser.add_argument('--gamma', type=float, default=42576, help='gyro magnetic ratio - kHz/T')
    parser.add_argument('--G-max', type=float, default=40, help='maximum gradient (peak current) - mT/m')
    parser.add_argument('--S-max', type=float, default=180, help='maximum slew-rate - T/m/s')
    parser.add_argument('--FOV', type=float, default=0.2, help='Field Of View - in m')
    parser.add_argument('--dt', type=float, default=1e-5, help='sampling time - sec')
    parser.add_argument('--a-max', type=float, default=0.17, help='maximum acceleration')
    parser.add_argument('--v-max', type=float, default=3.4, help='maximum velocity')
    parser.add_argument('--TSP', action='store_true', default=False,
                        help='Using the PILOT-TSP algorithm,if False using PILOT.')
    parser.add_argument('--TSP-epoch', default=20, type=int, help='Epoch to preform the TSP reorder at')
    parser.add_argument('--initialization', type=str, default='radial',
                        help='Trajectory initialization when using PILOT (spiral, EPI, rosette, uniform, gaussian).')
    parser.add_argument('--SNR', action='store_true', default=False,
                        help='add SNR decay')
    parser.add_argument('--n-shots', type=int, default=16,
                        help='Number of shots')
    parser.add_argument('--interp_gap', type=int, default=10,
                        help='number of interpolated points between 2 parameter points in the trajectory')
    parser.add_argument('--num_frames_per_example', type=int, default=10, help='num frames per example')
    parser.add_argument('--boost', action='store_true', default=False, help='boost to equalize num examples per file')

    parser.add_argument('--project', action='store_true', default=False, help='Use projection or interpolation.')
    parser.add_argument('--proj_iters', default=10e1, help='Number of iterations for each projection run.')
    parser.add_argument('--multi_traj', action='store_true', default=False, help='allow different trajectory per frame')
    parser.add_argument('--augment', action='store_true', default=True, help='Use augmented files.')
    return parser


if __name__ == '__main__':
    train()