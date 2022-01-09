import logging
import multiprocessing
import pathlib
import random
import time
import h5py
from collections import defaultdict

import numpy as np
import torch
import sys
import os
os.environ['TOOLBOX_PATH']='/home/tomerweiss/bart-0.4.04/'
os.environ['PYTHONPATH']='${TOOLBOX_PATH}/python:${PYTHONPATH}'

sys.path.insert(0,"/home/tomerweiss/bart-0.4.04/python")

import bart
from torch.utils.data import Dataset
sys.path.insert(0,'/home/tomerweiss/multiPILOT2')
from common import utils
from common.args import Args
# from common.subsample import MaskFunc
# from data.mri_data import SliceData


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SliceData(Dataset):
    def __init__(self, root, i):
        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        files = [f for f in files if str(f)[-4:] == f'{i}.h5']
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['reconstruction']
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            image = data['reconstruction'][slice,:,:]
            return image, fname.name, slice

def create_data_loader(args, i):
    # dev_mask = MaskFunc(args.center_fractions, args.accelerations)
    data = SliceData(
        root=args.data_path2,
        i=i
    )
    return data


def cs_total_variation(args, image):
    image = np.fft.ifftshift(image, axes=(-2, -1))
    kspace = np.fft.fft2(image)
    kspace = np.fft.fftshift(kspace, axes=(-2, -1))

    # kspace=np.concatenate((np.expand_dims(np.real(kspace),2),
    #                        np.expand_dims(np.imag(kspace),2)),axis=-1)
    # print(kspace.shape)
    # kspace = kspace.reshape((1,320,320,1, 2))

    kspace = kspace.reshape((1, 320, 320))
    # Estimate sensitivity maps
    sens_maps = bart.bart(1, 'ecalib -d0 -m1', kspace)

    # Use Total Variation Minimization to reconstruct the image
    pred = bart.bart(
        1, 'pics -d0 -S -R T:7:0:0.01 -i 200', kspace, sens_maps
    )

    pred = torch.from_numpy(np.abs(pred[0]))
    # print(pred.shape)
    # Crop the predicted image to the correct size
    return pred


def run_model(i):
    image, fname, slice = data[i]
    print(f'{i}, {fname}, {slice}')
    prediction = cs_total_variation(args, image)
    return fname, slice, prediction


def main():
    print(len(data))
    if args.num_procs == 0:
        start_time = time.perf_counter()
        outputs = []
        for i in range(len(data)):
            outputs.append(run_model(i))
        time_taken = time.perf_counter() - start_time
    else:
        with multiprocessing.Pool(args.num_procs) as pool:
            start_time = time.perf_counter()
            outputs = pool.map(run_model, range(len(data)))
            time_taken = time.perf_counter() - start_time
    logging.info(f'Run Time = {time_taken}')
    save_outputs(outputs, args.output_path)


def save_outputs(outputs, output_path):
    reconstructions = defaultdict(list)
    for fname, slice, pred in outputs:
        reconstructions[fname].append((slice, pred))
    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    utils.save_reconstructions(reconstructions, output_path)


if __name__ == '__main__':
    parser = Args()
    parser.add_argument('--data-path2', type=str, default='/home/tomerweiss/multiPILOT2/summary/32/radial_0.01_0.1_0.1_multiscale/rec_sub',
                        help='Path to data')
    parser.add_argument('--output-path', type=str, default='/home/tomerweiss/multiPILOT2/summary/32/radial_0.01_0.1_0.1_multiscale/rec_cs/',
                        help='Path to save the reconstructions to')
    parser.add_argument('--num-iters', type=int, default=200,
                        help='Number of iterations to run the reconstruction algorithm')
    parser.add_argument('--reg-wt', type=float, default=0.01,
                        help='Regularization weight parameter')
    parser.add_argument('--num-procs', type=int, default=10,
                        help='Number of processes. Set to 0 to disable multiprocessing.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for i in range(10):
        data = create_data_loader(args, i)
        main()