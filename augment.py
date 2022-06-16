import random

from deepaugment.deepaugment import DeepAugment
import numpy as np
import h5py
import argparse
from pandas import read_csv
from train_mf import DataTransform
from data import transforms
import torch
import os
import pickle
from gen_aug_policy import gen_policy
from deepaugment.augmenter import transform as augts
from random import seed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/home/tomerweiss/dor/OCMR/data_processed/', help='path for data to be augmented')
    parser.add_argument('--destpath', type=str, default='/home/tomerweiss/tamir/aug/', help='path tp save augmented data')
    return parser.parse_args()

def get_rel_files(files, resolution, num_frames_per_example):
    rel_files = []
    for fname in sorted(files):
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'] # [slice, frames, coils, h,w]
            if kspace.shape[3] < resolution[0] or kspace.shape[4] < resolution[1]:
                continue
            if kspace.shape[1] < num_frames_per_example:
                continue
        rel_files.append(fname)
    return rel_files

def augment_by_policy(slice, slice_target,n,aug_thresh = 0.4):
    '''augment vslice n times according to policy. Save as h5.'''
    boost_factor = 1.25
    ops = ['crop','rotate',"horizontal-flip","vertical-flip","shear"]
    boost = ['rescale','rotate','coarse-dropout']
    elaborate = ["coarse-dropout",'sharpen',"elastic-transform","coarse-salt-pepper",'gamma-contrast',"perspective-transform",'rescale'] #2nd favored operations
    requires_scale = ['coarse-dropout','brighten',"elastic-transform","perspective-transform"]
    augs = []
    slice = slice.cpu().detach().numpy()
    for k in range(n): #for every setting in policy, draw whether to augment by policy
        seed = np.random.randint(1,10e2)
        op1 = ops[np.random.randint(0,len(ops))]
        op2 = ops[np.random.randint(0,len(ops))]
        while op1 == op2: #avoid doing the same operator twice
            assert len(ops) > 1
            op2 = ops[np.random.randint(0, len(ops))]
        random.seed(seed) #anchor seed to make sure all frames have the same operation on them
        np.random.seed(seed)
        run_prob = np.random.rand()
        vid_re = np.copy(slice.take(0,-1))
        vid_im = np.copy(slice.take(1,-1))
        #scaled = False
        vid_target = np.copy(slice_target)
        for op in [op1,op2]:
            if op in boost:
                run_prob *= boost_factor
            if run_prob < aug_thresh:
                continue
            # if not scaled and op in requires_scale:
            #     vid_re = (255*vid_re/vid_re.max()).astype('uint8')
            #     vid_im = (255*vid_im/vid_re.max()).astype('uint8')
            #     vid_target = (vid_target).astype('uint8')
            #     scaled = True

            #separate on real and imaginary comps
            vid_re = augts(op,1,vid_re,seed=seed)
            #vid_im = augts(op,1,vid_im,seed=seed)
            vid_target = augts(op,1,vid_target,seed=seed)

        #augs.append((np.sqrt(vid_re**2+vid_im**2),vid_target))
        augs.append((np.concatenate((vid_re.reshape(*vid_re.shape,1),vid_im.reshape(*vid_re.shape,1)),axis=-1), vid_target))
    return augs



args = parseArgs()



dtf = DataTransform()
ocmr_data_attributes_location = '/home/tomerweiss/dor/OCMR/OCMR/ocmr_data_attributes.csv'
df = read_csv(ocmr_data_attributes_location)
df.dropna(how='all', axis=0, inplace=True)
df.dropna(how='all', axis=1, inplace=True)
rel_files = [args.datapath + '/' + k for k in df[df['smp'] == 'fs']['file name'].values]
rel_files = get_rel_files(rel_files, dtf.resolution, 10)

base_data = None
for fl in rel_files:
    print(fl)
    with h5py.File(fl, 'r') as data:
        augims = None
        augtars = None
        for idx in range(len(data['kspace'])):
            kspace = data['kspace'][idx]  # (frames, coils, h, w)
            kspace = kspace.sum(axis=1) / kspace.shape[1]
            kspace = transforms.to_tensor(kspace).to('cuda' if torch.cuda.is_available() else 'cpu')
            image = transforms.ifft2_regular(kspace)
            image = transforms.complex_center_crop(image, dtf.resolution)
            image, _, _ = transforms.normalize_instance(image, eps=1e-11)

            target = data['reconstruction_rss'][idx]
            target = transforms.to_tensor(target)
            target = transforms.center_crop(target.unsqueeze(0), dtf.resolution).squeeze()
            target, _, _ = transforms.normalize_instance(target, eps=1e-11)

            #make sure orig data is in augmented
            augims = image.unsqueeze(0).cpu()#torch.sqrt((image**2).sum(dim=-1)).unsqueeze(0).cpu()
            augtars = target.unsqueeze(0).cpu()

            augs = augment_by_policy(image,target,6)
            for pair in augs:
                slice,tar = pair
                slice = torch.Tensor(slice).unsqueeze(0)
                tar = torch.Tensor(tar).unsqueeze(0)
                augims = torch.cat((augims,slice),dim=0)
                augtars = torch.cat((augtars,tar),dim=0)

    fls = fl.split('.h5')[0].split('/')[-1]
    with h5py.File(f'{args.destpath}/{fls}_aug.h5', 'w') as f:
        f.create_dataset("reconstruction_rss", data = augtars.numpy())
        f.create_dataset("images", data=augims.numpy())






