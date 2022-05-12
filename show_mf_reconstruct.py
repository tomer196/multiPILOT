import torch
import argparse
from models.subsampling_mf_model import Subsampling_Model
import pathlib
from data.mri_mf_data import SliceData
from tensorboardX import SummaryWriter
from train_mf import DataTransform
from torch.utils.data import DataLoader


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-name', type=str, default='test', help='name for the output dir')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='summary/test',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, default='summary/test/model.pt',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')

    # model parameters
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
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
    parser.add_argument('--sub-lr', type=float, default=1e-2, help='lerning rate of the sub-samping layel')

    # trajectory learning parameters
    parser.add_argument('--trajectory-learning', default=False,
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
    parser.add_argument('--n-shots', type=int, default=32,
                        help='Number of shots')
    parser.add_argument('--interp_gap', type=int, default=10,
                        help='number of interpolated points between 2 parameter points in the trajectory')
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')

    # Data parameters
    parser.add_argument('--data-path', type=pathlib.Path,
                      default='/home/tomerweiss/dor/fastMRI_data/', help='Path to the dataset')
    parser.add_argument('--sample-rate', type=float, default=1.,
                      help='Fraction of total volumes to include')
    return parser.parse_args()

args = create_arg_parser()
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
        n_shots=args.n_shots,
        interp_gap=16
    ).to(args.device)
model.load_state_dict(torch.load('trained_state_dict'))
model.eval()

data = SliceData(
        files=['/home/tomerweiss/dor/OCMR/data_processed/fs_0068_1_5T.h5'],
        transform=DataTransform(),
        sample_rate=args.sample_rate
)
writer = SummaryWriter(log_dir=args.exp_dir)

loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=20
    )
with torch.no_grad():
    for iter, data in enumerate(loader):

        input, target, mean, std = data
        input = input.to(args.device)
        target = target.to(args.device)
        output = model(input)

        video_images = model(input).unsqueeze(0)
        video_images -= video_images.min()
        video_images /= video_images.max()

        writer.add_video('Recon', video_images)

