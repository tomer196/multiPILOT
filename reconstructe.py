import pathlib
from collections import defaultdict
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from common.args import Args
from common.utils import save_reconstructions
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
        image = image.clamp(-6, 6)
        kspace = transforms.fft2(image)

        return image, mean, std, fname, slice

def create_data_loaders(args):
    data = SliceData(
        root=args.data_path / f'multicoil_{args.data_split}',
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

def load_model(checkpoint):
    checkpoint = torch.load(checkpoint)
    args = checkpoint['args']
    model = Subsampling_Model(
        in_chans=15,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob,
        decimation_rate=args.decimation_rate,
        res=args.resolution,
        trajectory_learning=args.trajectory_learning,
        initialization=args.initialization,
        SNR=args.SNR
    ).to(args.device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])
    return model

def eval(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for (input, mean, std, fnames, slices) in data_loader:
            input = input.to(args.device)
            recons = model(input).to('cpu').squeeze(1)
            recons = transforms.complex_abs(recons)  # complex to real

            for i in range(recons.shape[0]):
                recons[i] = recons[i] * std[i] + mean[i]
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def reconstructe():
    args = create_arg_parser().parse_args(sys.argv[1:])
    # args.checkpoint = f'summary/{args.test_name}/best_model.pt' # problamtic when using TSP
    args.checkpoint = f'summary/{args.test_name}/best_model.pt'
    args.out_dir = f'summary/{args.test_name}/rec'

    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = eval(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)


def create_arg_parser():
    parser = Args()
    parser.add_argument('--test-name', type=str, default='test', help='name for the output dir')
    parser.add_argument('--data-split', choices=['val', 'test'],default='val',
                        help='Which data partition to run on: "val" or "test"')
    parser.add_argument('--checkpoint', type=pathlib.Path,default='summary/test/checkpoint/best_model.pt',
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path,default='summary/test/rec',
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=24, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--SNR', action='store_true', default=False,
                        help='add SNR decay')

    return parser


if __name__ == '__main__':
    reconstructe()
