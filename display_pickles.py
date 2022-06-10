import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import pdb

next_pickle = False


def on_press(event):
    global next_pickle
    next_pickle = True


def on_close(event):
    sys.exit(0)


train_dir = '/home/tomerweiss/dor/multiPILOT/summary/temp2/'
epoch = 1
fps = 10

delay = 1. / fps
list_pickles = [k for k in os.listdir(train_dir) if k.endswith('.pickle')]
epochs = np.unique([k.split('_')[1] for k in list_pickles])
all_pickles = {int(k): [] for k in epochs}
for pkl in list_pickles:
    all_pickles[int(pkl.split('_')[1])]. append(pkl)

if epoch < 0:
    epoch = np.max(all_pickles.keys())
fig = plt.figure(1)
fig.canvas.mpl_connect('close_event', on_close)
fig.canvas.mpl_connect('key_press_event', on_press)

ax_pred = plt.subplot(1, 2, 1)
ax_gt = plt.subplot(1, 2, 2)
for pkl in sorted(all_pickles[epoch]):
    with open(train_dir + pkl, 'rb') as f:
        data = pickle.load(f)
    for i in range(len(data['target'])):
        j = -1
        while not next_pickle:
            j = np.mod(j+1, len(data['pred'][i]))
            ax_pred.cla()
            ax_gt.cla()
            fig.suptitle(pkl.split('.')[0] + '_example_' + str(i))
            ax_pred.set_title('prediction')
            ax_gt.set_title('target')
            ax_pred.imshow(data['pred'][i][j], cmap='gray', origin='lower')
            ax_gt.imshow(data['target'][i][j], cmap='gray', origin='lower')
            plt.pause(delay)
        next_pickle = False

    pdb.set_trace()
    
