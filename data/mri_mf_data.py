import pathlib
import random
from turtle import pd
import h5py
import torch
from torch.utils.data import Dataset
import ismrmrd.xsd
import data.transforms as transforms
import numpy as np

class SliceData(Dataset):
    def __init__(self, files, transform, sample_rate=1, num_frames_per_example=10, clips_factors=None):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        self.transform = transform

        self.examples = []
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            with h5py.File(fname, 'r') as data:
                curr_clip_examples = []
                kspace = data['kspace'] # [slice, frames, coils, h,w]
                if kspace.shape[3] < self.transform.resolution[0] or kspace.shape[4] < self.transform.resolution[1]:
                    continue

                if kspace.shape[1] < num_frames_per_example:
                    continue
                for start_frame_index in range(kspace.shape[1] - num_frames_per_example):
                    num_slices = kspace.shape[0]
                    curr_clip_examples += [(fname, slice, start_frame_index, start_frame_index + num_frames_per_example) for slice in range(num_slices)]
            curr_factor = 1
            if clips_factors is not None:
                curr_factor = clips_factors[fname]
            self.examples += curr_clip_examples * curr_factor
        random.shuffle(self.examples)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, start_frame, last_frame = self.examples[i]

        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice, start_frame:last_frame] # (frames, coils, h, w) 
            target = data['reconstruction_rss'][slice, start_frame:last_frame] if 'reconstruction_rss' in data else None
            kspace = kspace.sum(axis = 1)/kspace.shape[1]

            return self.transform(kspace, target, data.attrs, fname, slice)
