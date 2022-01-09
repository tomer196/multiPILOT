
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import tensorflow as tf
# from matplotlib import font_manager
# ft = font_manager.findfont('Liberation Sans')
import scipy.io

i=0
h=10
best=1
epoch=0
eval_image=np.zeros((50*h,320))
val_loss=np.zeros(50)
val_loss_wo=np.zeros(50)


# # for e in tf.train.summary_iterator('C:\\Users\\tomer-weiss\\OneDrive\\Desktop\\Teachnion\\DIP\\project\\baseline-wp\\results\\4_0.09_True\\summary\\events.out.tfevents.1553127330.carmen'):
# for e in tf.train.summary_iterator('/home/aditomer/mri-coord/summary/80/TSP_30k_1e-1 06\\29\\2019,16:46/events.out.tfevents.1561815990.aida'):
#     for v in e.summary.value:
#         # print(v.tag)
#         # if v.HasField('simple_value'):
#         #     print(v.simple_value)
#         if v.tag=='Dev_Loss':
#             val_loss[i]=v.simple_value
#             if v.simple_value < best:
#                 best=v.simple_value
#                 epoch=i
#             i+=1
i=0
image_str = tf.placeholder(tf.string)
im_tf = tf.image.decode_image(image_str)
sess = tf.InteractiveSession()
with sess.as_default():
    for e in tf.train.summary_iterator("/home/tomerweiss/multiPILOT2/summary/8/radial_0.01_0.1_0.1_multiscale/events.out.tfevents.1592287927.violeta"):
        for v in e.summary.value:
            print(v.tag)
            # if v.HasField('simple_value'):
            #     print(v.simple_value)
            # if v.tag=='Dev_Loss':
            #     val_loss_wo[i]=v.simple_value
            #     # if v.simple_value < best:
            #     #     best=v.simple_value
            #     #     epoch=i
            if v.tag == 'Trajectory':
                img = im_tf.eval({image_str: v.image.encoded_image_string})
                plt.imsave(f'trajectory_{i}.png',img,cmap='gray')
                print('fffffffffffffffffffffffffffffffffff')
                i+=1
# plt.imshow(eval_image[:,0:epoch*h].T,cmap='gray',extent=[0,epoch,0,320], aspect=1/10)
# hfont = {'fontname':'Liberation Sans'}
#plt.rcParams.update({'font.size': 20})
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.ylabel('Cartesian lines',fontsize=20,prop=ft)
# plt.xlabel('Epochs',**hfont,fontsize=20)
# plt.show()
# dict = {'mask4':eval_image[:,0:epoch*h]}
# scipy.io.savemat('mask4.mat',dict)
print(epoch)
# plt.plot(val_loss_wo[:epoch+1],label='Fixed mask')
# plt.plot(val_loss[:epoch+1],label='Learned mask')
# plt.legend()
# plt.ylabel('Validation loss')
# plt.xlabel('Epochs')
# plt.show()
