import pandas
import numpy as np
import os
import matplotlib.pyplot as plt
import math

from ismrmrdtools import show, transform
import ismrmrd
import ismrmrd.xsd
# import ReadWrapper



def read_ocmr(filename):
# Input:  *.h5 file name
# Output: all_data    k-space data, orgnazide as {'kx'  'ky'  'kz'  'coil'  'phase'  'set'  'slice'  'rep'  'avg'}
#         param  some parameters of the scan
# 

# This is a function to read K-space from ISMRMD *.h5 data
# Modifid by Chong Chen (Chong.Chen@osumc.edu) based on the python script
# from https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/recon_ismrmrd_dataset.py

    if not os.path.isfile(filename):
        print("%s is not a valid file" % filename)
        raise SystemExit
    dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    #eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    eNy = (enc.encodingLimits.kspace_encoding_step_1.maximum + 1); #no zero padding along Ny direction

    # Field of View
    eFOVx = enc.encodedSpace.fieldOfView_mm.x
    eFOVy = enc.encodedSpace.fieldOfView_mm.y
    eFOVz = enc.encodedSpace.fieldOfView_mm.z
    
    # Save the parameters    
    param = dict();
    param['TRes'] =  str(header.sequenceParameters.TR)
    param['FOV'] = [eFOVx, eFOVy, eFOVz]
    param['TE'] = str(header.sequenceParameters.TE)
    param['TI'] = str(header.sequenceParameters.TI)
    param['echo_spacing'] = str(header.sequenceParameters.echo_spacing)
    param['flipAngle_deg'] = str(header.sequenceParameters.flipAngle_deg)
    param['sequence_type'] = header.sequenceParameters.sequence_type

    # Read number of Slices, Reps, Contrasts, etc.
    nCoils = header.acquisitionSystemInformation.receiverChannels
    try:
        nSlices = enc.encodingLimits.slice.maximum + 1
    except:
        nSlices = 1
        
    try:
        nReps = enc.encodingLimits.repetition.maximum + 1
    except:
        nReps = 1
               
    try:
        nPhases = enc.encodingLimits.phase.maximum + 1
    except:
        nPhases = 1;

    try:
        nSets = enc.encodingLimits.set.maximum + 1;
    except:
        nSets = 1;

    try:
        nAverage = enc.encodingLimits.average.maximum + 1;
    except:
        nAverage = 1;   
        
    # TODO loop through the acquisitions looking for noise scans
    firstacq=0
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        # TODO: Currently ignoring noise scans
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            #print("Found noise scan at acq ", acqnum)
            continue
        else:
            firstacq = acqnum
            print("Imaging acquisition starts acq ", acqnum)
            break

    # assymetry echo
    kx_prezp = 0;
    acq_first = dset.read_acquisition(firstacq)
    if  acq_first.center_sample*2 <  eNx:
        kx_prezp = eNx - acq_first.number_of_samples
         
    # Initialiaze a storage array
    param['kspace_dim'] = {'kx ky kz coil phase set slice rep avg'};
    all_data = np.zeros((eNx, eNy, eNz, nCoils, nPhases, nSets, nSlices, nReps, nAverage), dtype=np.complex64)

    # Loop through the rest of the acquisitions and stuff
    for acqnum in range(firstacq,dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        # Stuff into the buffer
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2
        phase =  acq.idx.phase;
        set =  acq.idx.set;
        slice =  acq.idx.slice;
        rep =  acq.idx.repetition;
        avg = acq.idx.average;        
        all_data[kx_prezp:, y, z, :,phase, set, slice, rep, avg ] = np.transpose(acq.data)
        
    return all_data,param


def display_avg_time(kData_inp, slc_idx=None):
    dim_kData = kData_inp.shape
    CH = dim_kData[3]
    SLC = dim_kData[6]
    kData_tmp = np.mean(kData_inp, axis = 8) # average the k-space if average > 1
    samp = (abs(np.mean(kData_tmp, axis = 3)) > 0).astype(np.int) # kx ky kz phase set slice

    if slc_idx is None:
        slc_idx = math.floor(SLC/2)
    fig1 = plt.figure(1)
    fig1.suptitle("Sampling Pattern", fontsize=14)
    plt.subplot2grid((1, 8), (0, 0), colspan=6)
    tmp = plt.imshow(np.transpose(np.squeeze(samp[:,:,0,0,0,slc_idx])), aspect= 'auto')
    plt.xlabel('kx')
    plt.ylabel('ky')
    tmp.set_clim(0.0,1.0) # ky by kx
    plt.subplot2grid((1, 9), (0, 7),colspan=2)
    tmp = plt.imshow(np.squeeze(samp[int(dim_kData[0]/2),:,0,:,0,slc_idx]),aspect= 'auto')
    plt.xlabel('frame')
    plt.yticks([])
    tmp.set_clim(0.0, 1.0) # ky by frame

    # Average the k-sapce along phase(time) dimension
    kData_sl = kData_tmp[:,:,:,:,:,:,slc_idx,0]
    samp_avg =  np.repeat(np.sum(samp[:,:,:,:,:,slc_idx,0],3), CH, axis=3) + np.finfo(float).eps
    kData_sl_avg = np.divide(np.squeeze(np.sum(kData_sl,4)), np.squeeze(samp_avg))

    im_avg = transform.transform_kspace_to_image(kData_sl_avg, [0,1]) # IFFT (2D image)
    im = np.sqrt(np.sum(np.abs(im_avg) ** 2, 2)) # Sum of Square
    fig2 = plt.figure(2)
    plt.imshow(np.transpose(im), vmin=0, vmax=0.8*np.amax(im), cmap = 'gray'); plt.axis('off') # Show the image
    plt.pause(1)


def display_mf(image):
    # Input: image dimensions - [kx, ky, phase]
    # Show the reconstructed cine image
    plt.figure(3)
    for rep in range(5): # repeate the movie for 5 times
        for frame in range(image.shape[2]):
            plt.cla()
            plt.imshow(image[:,:,frame], vmin=0, vmax=0.6*np.amax(image), cmap = 'gray')
            plt.axis('off')
            plt.pause(0.5) 


ocmr_data_attributes_location = '/home/tomerweiss/dor/OCMR/OCMR/ocmr_data_attributes.csv'
ocmr_data_location = '/home/tomerweiss/dor/OCMR/data/OCMR_data/'
df = pandas.read_csv(ocmr_data_attributes_location)
df.dropna(how='all', axis=0, inplace=True)
df.dropna(how='all', axis=1, inplace=True)

# get relevant files
rel_files = [ocmr_data_location + k for k in df[df['smp'] == 'fs']['file name'].values]


filename = rel_files[11]#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
kData,param = read_ocmr(filename) # [kx, ky, kz, coil, phase(time), set, slice, rep, avg],

# Image reconstruction (SoS)
'''
im_sos and image are the reconstruced cardiac cine images, with and without readout oversampling, respectively.
'''
dim_kData = kData.shape
CH = dim_kData[3]
SLC = dim_kData[6]
kData_tmp = np.mean(kData, axis = 8) # average the k-space if average > 1

im_coil = transform.transform_kspace_to_image(kData_tmp, [0,1]) # IFFT (2D image)
im_sos = np.sqrt(np.sum(np.abs(im_coil) ** 2, 3)) # Sum of Square

# RO = im_sos.shape[0]
# image = im_sos[math.floor(RO/4):math.floor(RO/4*3),:,:] # Remove RO oversampling

# Displaying data
# display(kData)
# slc_idx = int(np.floor(kData.shape[6]/2))
# display_mf(np.squeeze(image[:,:,:,:,:,slc_idx]))




