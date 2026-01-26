"""
DWInode_functions.py
====================================
This module contains several functions to read the individual averages of b-value images exported by Recon2.0

@author r.navest@nki.nl
@author s.zijlema@nki.nl
"""

import numpy as np
import os.path
import csv
from collections import Counter

def getParamfromSin(fn_sin, paramList):
    """
    Function to get the required parameter values from the sin file

    Parameters
    ----------
    fn_sin
        Full path to the sin file (with or without .sin extension)
    paramList
        List of parameter names for which the value in the sin file should be found

    Returns
    -------
    parVal
        List of parameter values corresponding to paramList
    """
    #Check if path ends with .sin. If not, append .sin
    if len(fn_sin)>4:
        if fn_sin[-4:] != '.sin':
            fn_sin = fn_sin + '.sin'
    
    # Read sin file line by line
    with open(fn_sin) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        for row in csv_reader:
            if not row:
                data.append(row)
            else:
                data.append(row[0].split())
    # Remove first 13 lines
    data = data[12:]
    # Remove headers
    emptyInd = []
    for ind in range(0, len(data) - 2):
        if not data[ind] and not data[ind + 2]:
            emptyInd.append(ind)
            emptyInd.append(ind + 1)
        elif not data[ind]:
            emptyInd.append(ind)
    for ind in sorted(emptyInd, reverse=True):
        del data[ind]
    del data[-1]  # final line is always empty
    # Extract the name and value of the parameters
    name = []
    value = []
    for row in data:
        if len(row) >= 5:
            name.append(row[3])
            value.append(row[5:])
        else:
            continue
    # Get parameter values
    parVal = []
    for par in paramList:
        indPar = name.index(par)
        parVal.append(value[indPar])

    return parVal

def import_images(fn, combine_axes=False, useM42numpy=False, old_data_output=False, data_order = 0):
    """
    Function to read the Recon 2.0 text file and convert it to npz

    Parameters
    ----------
    fn
        Full path to *_images.txt or raw data file
    combine_axes
        Combine axes before returning data (returns a 4D instead of a 5D matrix)
    useM42numpy
        Use numpy version from m42python to load data to be compatible with 
        m42python functions
    old_data_output
        Format data to be compatible with old scripts:
        [dimY, dimX, dimZ, dimB, dimN, dimD]
        When True, combine_axes will be ignored.
    data_order
        0 = X first
        1 = Y first
        0 for transverse slices, 1 for coronal slices
    Returns
    -------
    im_dwi
        Individual averages for all b-values. 
        Dimensions: [dimY, dimX, dimZ, dimD, dimB * dimN]
        Dimensions if combine_axes: [dimY, dimX, dimZ, dimD * dimB * dimN]
        Dimensions if old_data_output: [dimY, dimX, dimZ, dimB, dimN, dimD]
    b_val_list
        List of acquired b-values. If b-values are [0,200,800] with NSA=[1,2,4],
        b_val_list will be [0,200,200,400,400,400,400].
        If old_data_output==True: b_val_list will list only the unique b-values.
        
    Files created
    -------
    *.npz
        Numpy zip file with imported raw data, saved e.g. im_dwi, b_val_list, 
        and npz_version. The saved im_dwi data is 5D.
    """
    
    npz_version = 1.0 #increase this number only after a breaking/output change to force creation of new .npz-files
    
    if useM42numpy: #override numpy version if requested
        from m42python import numpy as np
        
        try: #try to increase 
            import sysutils.m42_connector
            from m42python import set_connection_timeout
            orig_timeout = sysutils.m42_connector.con._config.get('sync_request_timeout')
        except:
            pass
        
        set_connection_timeout(None)
    else:
        import numpy as np


    #Remove _imageIndex.txt, _images.txt, or other extension from path
    if fn.endswith('_imageIndex.txt'):
        fn = fn[:-15] #strip _imageIndex.txt
    elif fn.endswith('_images.txt'):
        fn = fn[:-11] #strip _images.txt
    
    fn,_ = os.path.splitext(fn) #strip other extensions (e.g. raw/lab/sin)
    
    
    #Check if npz already exists
    if os.path.isfile(fn + '.npz'):
        
        try: #try/except to avoid errors when npz_version does not exist
            #Check the version of the npz contents
            npz_contents = np.load(fn + '.npz', mmap_mode='r')
            
            npz_file_version = npz_contents['arr_2']
            
            if npz_file_version == npz_version: #if version matches, load existing npz
                create_new_npz = False
            else: #if version does not match, create new npz
                print("Newer version of .npz required")
                create_new_npz = True
        except: #if npz file does not have an npz_version
            print("Loading failed, new .npz required")
            create_new_npz = True
        
    else: #if npz file does not exist yet
        create_new_npz = True
    
    #Load or import data
    if create_new_npz == False:
        print(".npz already exists, loading...")
        # data = np.load(fn + '.npz')
        im_dwi = npz_contents['arr_0']
        b_val_list = npz_contents['arr_1']
    else: #if npz does not exist, import data from raw files
        print("Importing from raw data...")
    
        # Import order [curZ, curB, curN, curD]
        with open(fn + '_imageIndex.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            im_order = []
            for row in csv_reader:
                im_order.append(row)
        im_order = np.array(im_order)
        im_order = im_order.astype(int)
        im_dims = np.amax(im_order, axis=0) + 1  # [dimZ, dimB, dimN, dimD]

        # Import images with dimensions [dimZ * dimB * dimN * dimD, dimY * dimX]
        with open(fn + '_images.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            im_data = []
            for row in csv_reader:
                im_data.append(row)
        im_data = np.array(im_data)

        # Get required values
        parList = ['recon_resolutions', 'diffusion_b_factors']
        parVals = getParamfromSin(fn, parList)
        dimX = int(parVals[0][0])
        dimY = int(parVals[0][1])
        b_val = np.array([], dtype=int)
        for val in parVals[1]:
            b_val = np.append(b_val, int(float(val)))

        # Reshape data into dimensions [dimZ * dimB * dimN, dimY, dimX]
        dimTemp = np.shape(im_data)  # current dimensions
        if data_order == 0:
            im_data = np.reshape(im_data, (dimTemp[0], dimX, dimY)) #transverse slice orientation
            im_data = np.transpose(im_data, [0, 2, 1]) # Switch X and Y to make compatibale with rest of pipeline
        elif data_order == 1:
            im_data = np.reshape(im_data, (dimTemp[0], dimY, dimX)) #coronal slice orientation

        # Reshuffle data
        im_dwi = np.empty([dimY, dimX, im_dims[0], im_dims[1], im_dims[2], im_dims[3]])  # [dimY, dimX, dimZ, dimB, dimN, dimD]
        for im in range(0, dimTemp[0]):
            im_dwi[:, :, im_order[im, 0], im_order[im, 1], im_order[im, 2], im_order[im, 3]] = im_data[im, :, :]

        
        # Reshape
        # Current shape: [dimY, dimX, dimZ, dimB, dimN, dimD]
        maxNrAverages = im_dwi.shape[4]
        dimTemp = np.shape(im_dwi)  # current dimensions
        im_dwi = np.swapaxes(im_dwi, 3, 5)  # [dimY, dimX, dimZ, dimD, dimN, dimB]
        im_dwi = np.swapaxes(im_dwi, 4, 5)  # [dimY, dimX, dimZ, dimD, dimB, dimN]
        
        #Combine B and N dimensions
        im_dwi = np.reshape(im_dwi, dimTemp[0:3] + (3, -1)) #[dimY, dimX, dimZ, dimD, dimB * dimN]

        #Generate list of b-values
        b_val_list = np.repeat(b_val, maxNrAverages)

        # Remove empty averages from im_DWI and b_val
        correctIndices = []
        for curVolNr in range(im_dwi.shape[4]):
            if np.sum(im_dwi[:,:,:,:,curVolNr]) > 0.0:
                correctIndices.append(curVolNr)
        
        im_dwi = im_dwi[:,:,:,:,correctIndices]
        b_val_list = b_val_list[correctIndices]
        
       
        # Save data
        np.savez(fn + '.npz', im_dwi, b_val_list, npz_version)
        print("Data has been saved to "+fn + '.npz')
    
    
    if useM42numpy: #reset timeout value
        try:
            set_connection_timeout(orig_timeout)
        except:
            pass
    
    #Format data in old way to be compatible with old scripts
    if old_data_output == True:
        uniqueBvals = sorted(list(set(b_val_list)))
        nrBvals = len(uniqueBvals)
        
        _,max_nr_NSA = Counter(b_val_list).most_common(1)[0]
        
        #Reshape to [dimY, dimX, dimZ, dimB, dimN, dimD]
        new_shape = im_dwi.shape[:3] + (nrBvals, max_nr_NSA) + (im_dwi.shape[3],)
        old_im = np.zeros(new_shape)
        
        for b_index in range(nrBvals):
            b_im_data = im_dwi[:,:,:,:,b_val_list == uniqueBvals[b_index]]
            
            for n_index in range(b_im_data.shape[4]):
                old_im[:,:,:,b_index,n_index,:] = b_im_data[:,:,:,:,n_index]
        
        return old_im, uniqueBvals

    #Combine axes, if requested
    if combine_axes:
        
        im_dwi2, b_val_list = combine_axes_NSA(im_dwi,b_val_list,takeMeanNSA=False)
       
        return im_dwi2, b_val_list
    
    else:
        return im_dwi, b_val_list



def combine_axes_NSA(data5D,bvals5D,takeMeanNSA=True):
    """
    Combines axes and (optionally) NSAs of 5D DWI data (x,y,z,axes,NSAs).
    Takes output of import_images with combineAxes = False as input.

    Parameters
    ----------
    data5D : 5D numpy array
        Takes output of import_images with combineAxes = False as input.    
        Shape: (x,y,z,axes,NSAs).
    bvals_orig :
        bvals output from import_images with combineAxes = False.
    takeMeanNSA : Boolean, optional
        Take the mean of the NSAs per b-value.

    Returns
    -------
    data_combAxes :
        Combined data.
    bvals_combAxes :
        B-values of combined data.

    """
    
    data_shape = data5D.shape
    
    data_combAxes = np.zeros(data_shape[:3]+(data_shape[4],)) #total number of NSAs in fourth dimension
    
    
    for curNSA in range(data_shape[4]):
    
        curNSAdata = data5D[:,:,:,:,curNSA]
        
        if bvals5D[1] == 0: #if b-value was 0, no diffusion gradients
            data_combAxes[:,:,:,curNSA] = curNSAdata[:,:,:,0]
        else:
            #Multiply axes/directions and take nth root
            data_combAxes[:,:,:,curNSA] = np.prod(curNSAdata, axis=3)
            data_combAxes[:,:,:,curNSA] = data_combAxes[:,:,:,curNSA]**(1/curNSAdata.shape[3]) #take nth root
    
    #Loop over b-values and take mean if requested
    if takeMeanNSA:
        #Take mean of all NSAs per b-value
        data_combAxes,bvals_combAxes = combine_NSA_per_B(data_combAxes,bvals5D)
        
    else:
        bvals_combAxes = bvals5D

    #Return combined array
    return data_combAxes,bvals_combAxes


def combine_NSA_per_B(data,bvals):
    """
    Combine NSAs of each b-value.

    Parameters
    ----------
    data : 4D numpy array
        Shape: [x,y,z,d], where d contains the separate NSAs of all b-values.
    bvals :
        List of b-values that correspond to the d-axis.

    Returns
    -------
    data_combNSA : 4D numpy array
        Shape: [x,y,z,nrBvals].
    uniqueBvals :
        List of unique b-values.

    """
    
    uniqueBvals = np.unique(bvals) #get unique b-values
    
    #Initialize new data shape
    data_combNSA = np.zeros(data.shape[:3]+(len(uniqueBvals),))
    
    #Loop over each b-value, take mean of NSAs, and assign to data_combNSA
    for curBvalIndex in range(len(uniqueBvals)):
        #Take the mean of the NSAs and assign to data
        matchingData = data[:,:,:,bvals==uniqueBvals[curBvalIndex]]
        data_combNSA[:,:,:,curBvalIndex] = np.mean(matchingData,axis=3)
    
    
    return data_combNSA, uniqueBvals
