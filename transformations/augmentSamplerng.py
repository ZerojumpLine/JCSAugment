# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import numpy as np
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, gaussian_laplace

def augment_sample(channels, gt_lbls, prms, Imgenlarge, rng=None, proof=0):
    '''If I get a rng, I would assume I want to do test-time augmentation'''
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # gt_lbls: np array of shape [x,y,z]
    # prms: None or Dictionary, with parameters of each augmentation type. }

    # it might have troubles from the multi-process. might be not, because every choice is different.
    # but I also create a local state here, just to be save
    local_state = np.random.RandomState()

    if prms is not None:
        # choose one augmentation here. the chosen one is set to prmssel
        channels, gt_lbls = random_rotation_90(channels, gt_lbls, prms['rotate90'], local_state, rng)
        
        # four color transformations
        # the order should not affect much, I follow the order of batchgenerator
        # the first three should access the whole image for some global statstics
        channels = random_guassian_noise(channels, prms['noise'], local_state, rng, proof)
        channels, Imgenlarge = random_guassian_blur(channels, prms['blur'], Imgenlarge.copy(), local_state, rng, proof)
        channels, Imgenlarge = random_histogram_distortion(channels, prms['hist_dist'], Imgenlarge, local_state, rng, proof)
        channels, Imgenlarge = random_contrast(channels, prms['contrast'], Imgenlarge, local_state, rng, proof)
        channels = simulate_low_resolution(channels, prms['simulowres'], local_state, rng, proof)

        channels, Imgenlarge = random_invgamma_correction(channels, prms['gamma'], Imgenlarge, local_state, rng, proof)
        channels, Imgenlarge = random_gamma_correction(channels, prms['gamma'], Imgenlarge, local_state, rng, proof)

        channels, gt_lbls = random_flip(channels, gt_lbls, prms['reflect'], local_state)

    return channels, gt_lbls


def random_histogram_distortion(channels, prms, Imgenlarge, local_state, rng, proof):
    # Shift and scale the histogram of each channel.
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # prms: { 'shift': {'mu': 0.0, 'std':0.}, 'scale':{'mu': 1.0, 'std': '0.'} }
    if prms is None or prms['shift']['mu'] == 0 and prms['shift']['std'] == 0 and prms['scale']['mu'] ==0 and prms['scale']['std'] == 0:
        return channels, Imgenlarge

    if rng == None:
        ## training mode
        rng1 = local_state.choice((1, -1))
        rng2 = local_state.choice((1, -1))
        if prms['shift']['mu'] > 0:
            shiftmu = rng1 * local_state.uniform(np.max((0, prms['shift']['mu'] - 0.05)), prms['shift']['mu'] + 0.05)
            scalemu = 1 + local_state.uniform(np.max((0, prms['scale']['mu'] - 0.05)), prms['scale']['mu'] + 0.05)
        else:
            shiftmu = rng1 * local_state.uniform(0, - prms['shift']['mu'])
            scalemu = 1 + local_state.uniform(0, - prms['scale']['mu'])
    else:
        rng1 = rng
        rng2 = rng
        shiftmu = rng1 * prms['shift']['mu']
        scalemu = 1 + prms['scale']['mu']
    if rng2 < 0:
        scalemu = 1 / scalemu

    n_channs = channels[0].shape[0]
    if prms['shift'] is None:
        shift_per_chan = 0.
    elif prms['shift']['std'] != 0:  # np.random.normal does not work for an std==0.
        shift_per_chan = local_state.normal(shiftmu, prms['shift']['std'], [n_channs, 1, 1, 1])
    else:
        shift_per_chan = np.ones([n_channs, 1, 1, 1], dtype="float32") * shiftmu

    if prms['scale'] is None:
        scale_per_chan = 1.
    elif prms['scale']['std'] != 0:
        scale_per_chan = local_state.normal(scalemu, prms['scale']['std'], [n_channs, 1, 1, 1])
    else:
        scale_per_chan = np.ones([n_channs, 1, 1, 1], dtype="float32") * scalemu

    Imgenlarge = (Imgenlarge + shift_per_chan) * scale_per_chan

    # Intensity augmentation
    for path_idx in range(len(channels)):
        if proof == 0:
            channels[path_idx] = (channels[path_idx] + shift_per_chan) * scale_per_chan
        else:
            if np.sum(shift_per_chan) != 0 or np.mean(scale_per_chan) != 1:
                channels[path_idx] = channels[path_idx] * 0 - 1

    return channels, Imgenlarge

def random_contrast(channels, prms, Imgenlarge, local_state, rng, proof):
    # - mean and multiply a scalar
    # I should take the whole image as ref.
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # prms: { 'factor': 0. }
    if prms is None or prms['factor'] == 0:
        return channels, Imgenlarge

    if rng == None:
        ## training mode
        rng = local_state.choice((1, -1))
        if prms['factor'] > 0:
            factor = 1 + local_state.uniform(np.max((0, prms['factor'] - 0.05)), prms['factor'] + 0.05)
        else:
            factor = 1 + local_state.uniform(0, prms['factor'])
    else:
        factor = 1 + prms['factor']
    
    if rng < 0:
        factor = 1 / factor

    mns = []
    maxms = []
    minms = []
    # in case we are sampling like Deepmedic, I should keep the pixel absolute intensity similar.
    for c in range(channels[0].shape[0]):
        mns.append(Imgenlarge[c].mean())
        maxms.append(Imgenlarge[c].max())
        minms.append(Imgenlarge[c].min())
        Imgenlarge[c] = (Imgenlarge[c] - mns[c]) * factor + mns[c]
        Imgenlarge[c][Imgenlarge[c] < minms[c]] = minms[c]
        Imgenlarge[c][Imgenlarge[c] > maxms[c]] = maxms[c]

    for path_idx in range(len(channels)):
        for c in range(channels[path_idx].shape[0]):
            ## to retain stats
            channels[path_idx][c] = (channels[path_idx][c] - mns[c]) * factor + mns[c]
            channels[path_idx][c][channels[path_idx][c] < minms[c]] = minms[c]
            channels[path_idx][c][channels[path_idx][c] > maxms[c]] = maxms[c]

    return channels, Imgenlarge

def random_guassian_noise(channels, prms, local_state, rng, proof):
    # Add gaussian noise
    if prms is None or prms['std'] == 0:
        return channels

    if rng == None:
        ## training mode
        if prms['std'] > 0:
            noise_std = local_state.uniform(np.max((0, prms['std'] - 0.025)), prms['std'] + 0.025)
        else:
            noise_std = local_state.uniform(0, - prms['std'])
    else:
        noise_std = prms['std'] 

    # Intensity augmentation

    for path_idx in range(len(channels)):
        if proof == 0:
            shift_per_chan = local_state.normal(0, noise_std, [channels[path_idx].shape[0], channels[path_idx].shape[1],
                                                               channels[path_idx].shape[2], channels[path_idx].shape[3]])
            channels[path_idx] = channels[path_idx] + shift_per_chan
        else :
            # the ridiculous augmentation.
            if prms['std'] > 0 :
                # print('dangerous here')
                channels[path_idx] = channels[path_idx] * 0 + 1

    return channels

def random_guassian_blur(channels, prms, Imgenlarge, local_state, rng, proof):
    # Add gaussian noise
    if prms is None or prms['sigma'] == 0:
        return channels, Imgenlarge
    
    if rng == None:
        ## training mode
        if prms['sharpen'] == False:
            if prms['sigma'] > 0:
                blur_sigma = local_state.uniform(prms['sigma'] - 0.1, prms['sigma'] + 0.1)
            else:
                blur_sigma = local_state.uniform(0.4, - prms['sigma'])
        else:
            if prms['sigma'] > 0:
                blur_sigma = local_state.uniform(prms['sigma'] - 0.1, prms['sigma'] + 0.1)
            else:
                blur_sigma = local_state.uniform(- prms['sigma'], 1.)
    else:
        blur_sigma = prms['sigma']

    # save the statistic
    maxms = []
    minms = []
    for c in range(channels[0].shape[0]):
        maxms.append(Imgenlarge[c].max())
        minms.append(Imgenlarge[c].min())

    # blur
    for path_idx in range(len(channels)):
        for c in range(channels[path_idx].shape[0]):
            if prms['sharpen'] == False:
                channels[path_idx][c] = gaussian_filter(channels[path_idx][c], blur_sigma, order=0)
            else:
                channels[path_idx][c] = channels[path_idx][c] - gaussian_laplace(channels[path_idx][c], blur_sigma)
                channels[path_idx][c][channels[path_idx][c] < minms[c]] = minms[c]
                channels[path_idx][c][channels[path_idx][c] > maxms[c]] = maxms[c]
                

    return channels, Imgenlarge

def simulate_low_resolution(channels, prms, local_state, rng, proof):
    # simulate the low resolution
    order_downsample=1
    order_upsample=0
    if prms is None or prms['zoom'] == 1:
        return channels

    if rng == None:
        ## training mode
        if prms['zoom'] > 0:
            simlscale = local_state.uniform(prms['zoom'] - 0.1, prms['zoom'] + 0.1)
        else:
            simlscale = local_state.uniform(- prms['zoom'], 1.)
    else:
        simlscale = prms['zoom']

    # zoom in
    for path_idx in range(len(channels)):
        for c in range(channels[path_idx].shape[0]):
            shp = np.array(channels[path_idx].shape[1:])
            target_shape = np.round(shp * simlscale).astype(int)
            downsampled = resize(channels[path_idx][c].astype(float), target_shape, order=order_downsample, mode='edge',
                                anti_aliasing=False)
            channels[path_idx][c] = resize(downsampled, shp, order=order_upsample, mode='edge',
                                    anti_aliasing=False)

    return channels

def random_gamma_correction(channels, prms, Imgenlarge, local_state, rng, proof):
    # Gamma correction
    if prms is None or prms['gamma'] == 0:
        return channels, Imgenlarge

    if proof == 0:
        epsilon = 1e-6
        if rng == None:
            ## training mode
            rng = local_state.choice((1, -1))
            if prms['gamma'] > 0 :
                gamma = 1 + local_state.uniform(np.max((0, prms['gamma'] - 0.1)), prms['gamma'] + 0.1)
            else:
                gamma = 1 + local_state.uniform(0, - prms['gamma'])
        else:
            gamma = 1 + prms['gamma']
            # 
        if rng < 0:
            gamma = 1 / gamma
        
        minms = []
        rnges = []
        mns = []
        sds = []
        # in case we are sampling like Deepmedic, I should keep the pixel absolute intensity similar.
        for c in range(channels[0].shape[0]):
            minms.append(Imgenlarge[c].min())
            rnges.append(Imgenlarge[c].max() - Imgenlarge[c].min())
            mns.append(Imgenlarge[c].mean())
            sds.append(Imgenlarge[c].std())

        for path_idx in range(len(channels)):
            for c in range(channels[path_idx].shape[0]):
                ## to retain stats
                if rnges[c] != 0 and sds[c] != 0:
                    # Jan 27, 2021, fix a bug here.
                    minm = np.min((channels[path_idx][c].min(), Imgenlarge[c].min()))
                    maxm = np.max((channels[path_idx][c].max(), Imgenlarge[c].max()))
                    rnge = maxm - minm
                    # in case the minimum is sampled out, it should not happen very often I suppose
                    channels[path_idx][c] = np.power(((channels[path_idx][c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
                    
                    mn = Imgenlarge[c].mean()
                    sd = Imgenlarge[c].std()

                    Imgenlarge[c] = np.power(((Imgenlarge[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm

                    mnafter = Imgenlarge[c].mean()
                    
                    Imgenlarge[c] = Imgenlarge[c] - mnafter + mn
                    sdafter = Imgenlarge[c].std()

                    Imgenlarge[c] = Imgenlarge[c] / (sdafter + epsilon) * (sd + epsilon)

                    channels[path_idx][c] = channels[path_idx][c] - mnafter + mn
                    channels[path_idx][c] = channels[path_idx][c] / (sdafter + epsilon) * (sd + epsilon)

                else:
                    # if it is a blank, do not process.
                    # it is different from nnunet, it does not happen this case.
                    channels[path_idx][c] = channels[path_idx][c]
    else:
        # the ridiculous augmentation.
        for path_idx in range(len(channels)):
            if prms['gamma'] > 0 :
                channels[path_idx] = - channels[path_idx] * 0

    return channels, Imgenlarge

def random_invgamma_correction(channels, prms, Imgenlarge, local_state, rng, proof):
    # Inverted Gamma correction
    if prms is None or prms['invgamma'] == 0:
        return channels, Imgenlarge

    epsilon = 1e-6
    if rng == None:
        ## training mode
        rng = local_state.choice((1, -1))
        if prms['invgamma'] > 0 :
            gamma = 1 + local_state.uniform(np.max((0, prms['invgamma'] - 0.1)), prms['invgamma'] + 0.1)
        else:
            gamma = 1 + local_state.uniform(0, - prms['invgamma'])
    else:
        gamma = 1 + prms['invgamma']

    if rng < 0:
        gamma = 1 / gamma
    
    minms = []
    rnges = []
    mns = []
    sds = []
    Imgenlarge = - Imgenlarge
    for c in range(channels[0].shape[0]):
        minms.append(Imgenlarge[c].min())
        rnges.append(Imgenlarge[c].max() - Imgenlarge[c].min())
        mns.append(Imgenlarge[c].mean())
        sds.append(Imgenlarge[c].std())

    for path_idx in range(len(channels)):
        for c in range(channels[path_idx].shape[0]):
            channels[path_idx][c] = - channels[path_idx][c]
            if rnges[c] != 0 and sds[c] != 0:
                # Jan 27, 2021, fix a bug here.
                minm = np.min((channels[path_idx][c].min(), Imgenlarge[c].min()))
                maxm = np.max((channels[path_idx][c].max(), Imgenlarge[c].max()))
                rnge = maxm - minm
                # in case the minimum is sampled out, it should not happen very often I suppose
                channels[path_idx][c] = np.power(((channels[path_idx][c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
                
                mn = Imgenlarge[c].mean()
                sd = Imgenlarge[c].std()

                Imgenlarge[c] = np.power(((Imgenlarge[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm

                mnafter = Imgenlarge[c].mean()
                
                Imgenlarge[c] = Imgenlarge[c] - mnafter + mn
                sdafter = Imgenlarge[c].std()

                Imgenlarge[c] = Imgenlarge[c] / (sdafter + epsilon) * (sd + epsilon)

                channels[path_idx][c] = channels[path_idx][c] - mnafter + mn
                channels[path_idx][c] = channels[path_idx][c] / (sdafter + epsilon) * (sd + epsilon)

            else:
                # if it is a blank, do not process.
                # it is different from nnunet, it does not happen this case.
                channels[path_idx][c] = channels[path_idx][c]
            channels[path_idx][c] = - channels[path_idx][c]

    return channels, - Imgenlarge

def random_flip(channels, gt_lbls, probs_flip_axes, local_state):
    # Flip (reflect) along each axis.
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # gt_lbls: np array of shape [x,y,z]
    # probs_flip_axes: list of probabilities, one per axis.
    if probs_flip_axes is None:
        return channels, gt_lbls

    for axis_idx in range(len(channels[0].shape[1:])):  # 3 dims
        flip = local_state.choice(a=(True, False), size=1, p=(probs_flip_axes[axis_idx], 1. - probs_flip_axes[axis_idx]))
        if flip:
            for path_idx in range(len(channels)):
                channels[path_idx] = np.flip(channels[path_idx], axis=axis_idx + 1)  # + 1 because dim [0] is channels.
            if gt_lbls is not None:
                gt_lbls = np.flip(gt_lbls, axis=axis_idx)

    return channels, gt_lbls


def random_rotation_90(channels, gt_lbls, probs_rot_90, local_state, rng):
    # Rotate by 0/90/180/270 degrees.
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # gt_lbls: np array of shape [x,y,z]
    # probs_rot_90: {'xy': {'0': fl, '90': fl, '180': fl, '270': fl},
    #                'yz': {'0': fl, '90': fl, '180': fl, '270': fl},
    #                'xz': {'0': fl, '90': fl, '180': fl, '270': fl} }
    if probs_rot_90 is None:
        return channels, gt_lbls

    if rng == None:
        rng = local_state.choice((1, -1))

    for key, plane_axes in zip(['xy', 'yz', 'xz'], [(0, 1), (1, 2), (0, 2)]):
        probs_plane = probs_rot_90[key]

        if probs_plane is None:
            continue

        assert len(probs_plane) == 4  # rotation 0, rotation 90 degrees, 180, 270.
        # assert channels[0].shape[1 + plane_axes[0]] == channels[0].shape[1 + plane_axes[1]]  
        # # +1 cause [0] is channel. Image/patch must be isotropic.

        # Normalize probs
        sum_p = probs_plane['0'] + probs_plane['90'] + probs_plane['180'] + probs_plane['270']
        if sum_p == 0:
            continue
        for rot_k in probs_plane:
            probs_plane[rot_k] /= sum_p  # normalize p to 1.

        p_rot_90_x0123 = (probs_plane['0'], probs_plane['90'], probs_plane['180'], probs_plane['270'])

        if np.max(p_rot_90_x0123) < 1:
            # need rng to make the choice]
            if key == 'xz':
                ## this direction is inverse
                if rng == -1:
                    p_rot_90_x0123 = (probs_plane['0'], 1.0, probs_plane['180'], 0.0)
                else:
                    p_rot_90_x0123 = (probs_plane['0'], 0.0, probs_plane['180'], 1.0)
            else:    
                if rng == 1:
                    p_rot_90_x0123 = (probs_plane['0'], 1.0, probs_plane['180'], 0.0)
                else:
                    p_rot_90_x0123 = (probs_plane['0'], 0.0, probs_plane['180'], 1.0)

        rot_90_xtimes = local_state.choice(a=(0, 1, 2, 3), size=1, p=p_rot_90_x0123)

        for path_idx in range(len(channels)):
            channels[path_idx] = np.rot90(channels[path_idx], k=rot_90_xtimes,
                                          axes=[axis + 1 for axis in plane_axes])  # + 1 cause [0] is channels.
        if gt_lbls is not None:
            gt_lbls = np.rot90(gt_lbls, k=rot_90_xtimes, axes=plane_axes)

    return channels, gt_lbls