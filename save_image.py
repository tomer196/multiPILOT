import pathlib
import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"
import sys
sys.path.insert(0,'/home/tomerweiss/multiPILOT2')

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from common.args import Args
from common.evaluate import *
from data import transforms
from data.mri_data import SliceData
from models.subsampling_model import Subsampling_Model


class DataTransform:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, kspace, target, attrs, fname, slice):
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # image = transforms.complex_abs(image)
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        # image, mean, std = transforms.normalize_instance_per_channel(image, eps=1e-11)
        # image = image.clamp(-6, 6)
        # kspace = transforms.fft2(image)

        target = transforms.to_tensor(target)
        target, mean, std = transforms.normalize_instance(target, eps=1e-11)
        # # target = transforms.normalize(target, mean, std)
        # target = target.clamp(-6, 6)
        return image, target, mean, std, attrs['norm'].astype(np.float32), fname, slice

def create_data_loaders(args):
    data = SliceData(
        root=args.data_path2 ,
        transform=DataTransform(args.resolution),
        sample_rate=args.sample_rate
    )
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return data_loader

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    args.interp_gap = 1
    model = Subsampling_Model(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        decimation_rate=args.decimation_rate,
        res=args.resolution,
        trajectory_learning=args.trajectory_learning,
        initialization=args.initialization,
        SNR=args.SNR,
        n_shots=args.n_shots,
        interp_gap=args.interp_gap
    ).to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)

def run_unet2(args, model, data_loader):
    model.eval()
    # reconstructions = defaultdict(list)
    with torch.no_grad():
        for (input, target, mean, std, norm, fnames, slices) in data_loader:
            target = target.to('cpu').numpy()
            input = input.to(args.device)

            corrupted = model.module.subsampling(input).to('cpu')
            corrupted = corrupted[..., 0]  # complex to real
            cor_all = transforms.root_sum_of_squares(corrupted, dim=1)

            recons = model(input).to('cpu').squeeze(1)
            input = input.cpu()
            for i in range(recons.shape[0]):
                if  slices[i]>14 and slices[i]<26:
                    # recons[i] = recons[i] * std[i] + mean[i]
                    # reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

                    target -= target.min()
                    target /= target.max()
                    recons -= recons.min()
                    recons /= recons.max()

                    path=pathlib.Path(f'{args.out_dir}/{slices[i]}.png')
                    plt.imsave(path, recons[i], cmap='gray')
                    plt.imsave(f'{args.out_dir_sub}/sub_{slices[i]}.png', cor_all[i], cmap='gray')
                    # if slices[i] == 20:
                    #     plt.imsave(f'{args.out_dir_sub}/in0_{slices[i]}.png', input[i,0,:,:, 0], cmap='gray')
                    #     plt.imsave(f'{args.out_dir_sub}/in5_{slices[i]}.png', input[i,5,:,:,0], cmap='gray')
                    #     plt.imsave(f'{args.out_dir_subf}/in9_{slices[i]}.png', input[i,9,:,:,0], cmap='gray')
                    plt.imsave(f'{args.out_dir}/gt{slices[i]}.png', target[i], cmap='gray')

                    print(f'Slice: {slices[i]}, PSNR: {psnr1(recons[i].cpu().numpy(),target[i]):.02f}, SSIM: {ssim1(recons[i].cpu().numpy(),target[i]):.03f}')

    # reconstructions = {
    #     fname: np.stack([pred for _, pred in sorted(slice_preds)])
    #     for fname, slice_preds in reconstructions.items()
    # }
    return


def save_image():
    args = create_arg_parser().parse_args(sys.argv[1:])
    args.out_dir.mkdir(exist_ok=True)
    args.out_dir_sub.mkdir(exist_ok=True)
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = run_unet2(args, model, data_loader)
    # save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():
    parser = Args()
    exp_dir="/home/tomerweiss/multiPILOT2/summary/16/radial_fixed/"
    parser.add_argument('--data_path2', type=pathlib.Path,default=f'/home/tomerweiss/Datasets/pd_only/2/',
                        help='Path to the U-Net model')
    parser.add_argument('--checkpoint', type=pathlib.Path,default=f'{exp_dir}best_model.pt',
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path,default=f'{exp_dir}/rec2',
                        help='Path to save the reconstructions to')
    parser.add_argument('--out-dir-sub', type=pathlib.Path, default=f'{exp_dir}/rec_sub2',
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')

    return parser


if __name__ == '__main__':
    save_image()
