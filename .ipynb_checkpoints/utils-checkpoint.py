import random, h5py, os, glob, sys, time
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft 
from medpy.io import save, load
from pygrappa import grappa,mdgrappa
from pygrappa import find_acs 
from collections import defaultdict
from time import time
import logging

import numpy as np
from skimage.util import view_as_windows

from pygrappa.train_kernels import train_kernels
from pygrappa.find_acs import find_acs

# +
def calculate_mask(nFE,nPE,center_fraction,acc):
    '''
    nFE: number of frequency encoding lines
    nPE: number of phase encoding lines
    center_fraction: fraction of central lines of full sampling
    acc: acceleration factor
    '''
    num_low_freqs = int(round(nPE * center_fraction))
    num_high_freqs = np.ceil(nPE/acc) - num_low_freqs 
    num_outer_lines = nPE - num_low_freqs
    line0 = np.repeat([True],num_high_freqs)
    line0 = np.append(line0,np.repeat([False],num_outer_lines-num_high_freqs))
    np.random.shuffle(line0)
    num_left = num_outer_lines//2
    line = np.append(line0[0:num_left],np.repeat([True],num_low_freqs))
    line = np.append(line,line0[num_left:num_outer_lines])

    mask = np.repeat(line[np.newaxis, ...], nFE, axis=0)
    
    return mask
def estimate_mdgrappa_kernel(
        kspace,
        calib=None,
        kernel_size=None,
        coil_axis=-1,
        lamda=0.01,
        weights=None,
        ret_weights=False):
   
    # coils to the back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    nc = kspace.shape[-1]

    # Make sure we have a kernel_size
    if kernel_size is None:
        kernel_size = (5,)*(kspace.ndim-1)
    assert len(kernel_size) == kspace.ndim-1, (
        'kernel_size must have %d entries' % (kspace.ndim-1))

    # User can supply calibration region separately or we can find it
    if calib is not None:
        calib = np.moveaxis(calib, coil_axis, -1)
    else:
        # Find the calibration region and split it out from kspace
        calib = find_acs(kspace, coil_axis=-1)

    # Pad the arrays
    pads = [int(k/2) for k in kernel_size]
    adjs = [np.mod(k, 2) for k in kernel_size]
    kspace = np.pad(
        kspace, [(pd, pd) for pd in pads] + [(0, 0)], mode='constant')
    calib = np.pad(
        calib, [(pd, pd) for pd in pads] + [(0, 0)], mode='constant')
    mask = np.abs(kspace[..., 0]) > 0

    padmask = ~mask
    for ii in range(mask.ndim):
        padmask[tuple([slice(0, pd) if ii == jj else slice(None) for jj, pd in enumerate(pads)])] = False
        padmask[tuple([slice(-pd, None) if ii == jj else slice(None) for jj, pd in enumerate(pads)])] = False
    P = defaultdict(list)
    idxs = np.moveaxis(np.indices(mask.shape), 0, -1)[padmask]
    for ii, idx in enumerate(idxs):
        p0 = mask[tuple([slice(ii-pd, ii+pd+adj) for ii, pd, adj in zip(idx, pads, adjs)])].flatten()
        P[p0.tobytes()].append(tuple(idx))
    P = {k: np.array(v).T for k, v in P.items()}


    # We need all overlapping patches from calibration data
    A = view_as_windows(
        calib,
        tuple(kernel_size) + (nc,)).reshape(
            (-1, np.prod(kernel_size), nc,))

    # Set everything up to train and apply weights
    ksize = np.prod(kernel_size)*nc
    S = np.empty((np.max([P[k].shape[1] for k in P] if P else [0]), ksize), dtype=kspace.dtype)

    if not weights:
        # train weights
        Ws = train_kernels(kspace.astype(np.complex128), nc, A.astype(np.complex128), P,
                           np.array(kernel_size, dtype=np.uintp),
                           np.array(pads, dtype=np.uintp), lamda)
        wt={k: Ws[ii, :np.sum(np.frombuffer(k, dtype=bool))*nc, :]
                 for ii, k in enumerate(P)}
        return wt, P
    
def Grappa_recon(kspace,start, end):
    calib = kspace[:,:,start:end].copy() # call copy()!
    #print("k-space", kspace.shape,calib.shape)
    # coil_axis=-1 is default, so if coil dimension is last we don't
    # need to explicity provide it
    res, wt = mdgrappa(kspace, calib, kernel_size=(5, 5),coil_axis=0, ret_weights=True)
    return res, wt



def comp_sub_kspace(subk,crop_size):
    #print('img:', subk.shape)
    processed_subk=np.zeros(crop_size)
    s=subk.shape
    processed_subk[:,0:s[1],0:s[2]]=subk
    return processed_subk

def comp_img(img, crop_size_2):
    #print('img:', img.shape)
    processed_img=np.zeros((crop_size_2))
    s=img.shape
    processed_img[:,0:s[1],0:s[2]]=img
    return processed_img

def apply_kernel_weight(
        kspace,
        calib=None,
        kernel_size=None,
        coil_axis=-1,
        lamda=0.01,
        weights=None,
        P=None):
  

    # coils to the back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    nc = kspace.shape[-1]

    # Make sure we have a kernel_size
    if kernel_size is None:
        kernel_size = (5,)*(kspace.ndim-1)
    assert len(kernel_size) == kspace.ndim-1, (
        'kernel_size must have %d entries' % (kspace.ndim-1))

   
    # Pad the arrays
    pads = [int(k/2) for k in kernel_size]
    adjs = [np.mod(k, 2) for k in kernel_size]
    kspace = np.pad(
        kspace, [(pd, pd) for pd in pads] + [(0, 0)], mode='constant')
   
    # Set everything up to train and apply weights
    ksize = np.prod(kernel_size)*nc
    S = np.empty((np.max([P[k].shape[1] for k in P] if P else [0]), ksize), dtype=kspace.dtype)
    recon = np.zeros((np.prod(kspace.shape[:-1]), nc), dtype=kspace.dtype)
    mask = np.abs(kspace[..., 0]) > 0
    def _apply_weights(holes, p0, np0, Ws0):
        # Collect all the sources
        for jj, _idx in enumerate(holes.T):
            S[jj, :np0] = kspace[tuple([slice(kk-pd, kk+pd+adj)
                                        for kk, pd, adj in zip(_idx, pads, adjs)])].reshape((-1, nc))[p0, :].flatten()
        # Apply kernel to all sources to generate all targets at once
        recon[np.ravel_multi_index(holes, mask.shape)] = np.einsum(
            'fi,ij->fj', S[:holes.shape[1], :np0], Ws0)

    
    for ii, (key, holes) in enumerate(P.items()):
        p0 = np.frombuffer(key, dtype=bool)
        np0 = weights[key].shape[0]
        _apply_weights(holes, p0, np0, weights[key])

    # Add back in the measured voxels, put axis back where it goes
    recon = np.reshape(recon, kspace.shape)
    recon[mask] += kspace[mask]
    recon = np.moveaxis(
        recon[tuple([slice(pd, -pd) for pd in pads] + [slice(None)])],
        -1, coil_axis)
    return recon
# -


