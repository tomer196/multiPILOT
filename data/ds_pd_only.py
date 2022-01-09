import pathlib
import h5py
import os

# change orig and dist folder
orig_folder = '/home/tomerweiss/Datasets/T2/multicoil_val'
dist_folder = '/mnt/walkure_public/Datasets/tomer/fastmri_brain/T2/multicoil_train/'
files = list(pathlib.Path(orig_folder).iterdir())
# os.mkdir(dist_folder)

i = 0
d = {}
for fname in sorted(files):
    with h5py.File(fname, 'r') as data:
        a = data['kspace'].shape[1]
        if a in d.keys():
            d[a] +=1
        else:
            d[a] = 1
        # for brain dataset 'AXFLAIR', 'AXT1POST', 'AXT1PRE', 'AXT1', 'AXT2'
        #for knee dataset 'CORPD_FBK' or 'CORPDFS_FBK'
        if data.attrs['acquisition'] == 'AXT2':
            # os.system(f"cp {fname} {dist_folder + '/' + fname.name}")
            i += 1
            # print(f"{i}, {dist_folder + '/' + fname.name}")
print(d)
