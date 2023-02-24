# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

from __future__ import absolute_import, print_function, division

import collections
import numpy as np
import scipy.ndimage
from utilities import create_zero_centered_coordinate_mesh

# Main function to call:
def augment_imgs_of_case(channels, gt_lbls, roi_mask, wmaps_per_cat, prms, patch_size, rng=None, transf_mtx=None):
    '''If I get a rng, I would assume I want to do test-time augmentation'''
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # By Zeju: now it uses map_coordinate, similar to nnunet.
    # I dont know why, but previous implementation is sub-optimal
    # gt_lbls: np array of shape [x,y,z]. Can be None.
    # roi_mask: np array of shape [x,y,z]. Can be None.
    # wmaps_per_cat: List of np.arrays (floats or ints), weightmaps for sampling. Can be None.
    # prms: None (for no augmentation) or Dictionary with parameters of each augmentation type. }

    if prms is not None:

        (channels,
         gt_lbls,
         roi_mask,
         wmaps_per_cat,
         transf_mtx) = random_affine_deformation(channels,
                                                    gt_lbls,
                                                    roi_mask,
                                                    wmaps_per_cat,
                                                    patch_size,
                                                    rng,
                                                    prms,
                                                    transf_mtx)
    return channels, gt_lbls, transf_mtx


def random_affine_deformation(channels, gt_lbls, roi_mask, wmaps_l, patch_size, rng, prms, transf_mtx):
    if prms is None:
        return channels, gt_lbls, roi_mask, wmaps_l

    augm = AugmenterAffine(prob=prms['prob'],
                           rot_xyz=prms['rot_xyz'],
                           scaling=prms['scaling'],
                           seed=prms['seed'])
    if transf_mtx is None:
        transf_mtx = augm.roll_dice_and_get_random_transformation(rng)
    assert transf_mtx is not None

    channels = augm(images_l=channels,
                    transf_mtx=transf_mtx,
                    interp_orders=prms['interp_order_imgs'],
                    boundary_modes=prms['boundary_mode'],
                    patch_size=patch_size,
                    imgchannels=True)
    if gt_lbls is not None:
        (gt_lbls,
        roi_mask) = augm(images_l=[gt_lbls, roi_mask],
                        transf_mtx=transf_mtx,
                        interp_orders=[prms['interp_order_lbls'], prms['interp_order_roi']],
                        boundary_modes=prms['boundary_mode'], patch_size=patch_size, imgchannels=False)
    wmaps_l = augm(images_l=wmaps_l,
                   transf_mtx=transf_mtx,
                   interp_orders=prms['interp_order_wmaps'],
                   boundary_modes=prms['boundary_mode'], patch_size=patch_size, imgchannels=True)

    return channels, gt_lbls, roi_mask, wmaps_l, transf_mtx


class AugmenterParams(object):
    # Parent class, for parameters of augmenters.
    def __init__(self, prms):
        # prms: dictionary
        self._prms = collections.OrderedDict()
        self._set_from_dict(prms)

    def __str__(self):
        return str(self._prms)

    def __getitem__(self, key):  # overriding the [] operator.
        # key: string.
        return self._prms[key] if key in self._prms else None

    def __setitem__(self, key, item):  # For instance[key] = item assignment
        self._prms[key] = item

    def _set_from_dict(self, prms):
        if prms is not None:
            for key in prms.keys():
                self._prms[key] = prms[key]


class AugmenterAffineParams(AugmenterParams):
    def __init__(self, prms):
        # Default values.
        self._prms = collections.OrderedDict([('prob', 0.0),
                                              ('rot_xyz', (45., 45., 45.)),
                                              ('scaling', .1),
                                              ('seed', None),
                                              # For calls.
                                              ('interp_order_imgs', 3),
                                              ('interp_order_lbls', 1),
                                              ('interp_order_roi', 0),
                                              ('interp_order_wmaps', 1),
                                            #   ('boundary_mode', 'nearest'),
                                              ('boundary_mode', 'constant'),
                                              ('cval', 0.)])
        # Overwrite defaults with given.
        self._set_from_dict(prms)

    def __str__(self):
        return str(self._prms)


class AugmenterAffine(object):
    def __init__(self, prob, rot_xyz, scaling, seed=None):
        self.prob = prob  # Probability of applying the transformation.
        self.rot_xyz = rot_xyz
        self.scaling = scaling
        self.rng = np.random.RandomState(seed)

    def roll_dice_and_get_random_transformation(self, rng):
        if self.rng.random_sample() > self.prob:
            return -1  # No augmentation
        else:
            return self._get_random_transformation(rng)  # transformation for augmentation

    def _get_random_transformation(self, rng):
        local_state = np.random.RandomState()

        if rng == None:
            ## training mode
            rng1 = local_state.choice((1, -1))
            rng2 = local_state.choice((1, -1))
            rng3 = local_state.choice((1, -1))
            rng4 = local_state.choice((1, -1))
            '''if it could be divided by 10, make it have range (N-10, N)'''
            '''However, if it is minus, I would assume it comes from DM, I just choose from a uniform distribution.'''
            if self.rot_xyz[0] == 0:
                theta_x = self.rot_xyz[0] * np.pi / 180.
            elif self.rot_xyz[0] > 0:
                ## this is what I want to get
                ## just make sure I do not do anything wrong...
                theta_x = rng1 * local_state.uniform(np.max((0, self.rot_xyz[0] - 5.)), self.rot_xyz[0] + 5.) * np.pi / 180.
            else:
                theta_x = rng1 * local_state.uniform(0, - self.rot_xyz[0]) * np.pi / 180.

            if self.rot_xyz[1] == 0:
                theta_y = self.rot_xyz[1] * np.pi / 180.
            elif self.rot_xyz[1] > 0:
                ## this is what I want to get
                ## just make sure I do not do anything wrong...
                theta_y = rng2 * local_state.uniform(np.max((0, self.rot_xyz[1] - 5.)), self.rot_xyz[1] + 5.) * np.pi / 180.
            else:
                theta_y = rng2 * local_state.uniform(0, - self.rot_xyz[1]) * np.pi / 180.
            
            if self.rot_xyz[2] == 0:
                theta_z = self.rot_xyz[2] * np.pi / 180.
            elif self.rot_xyz[2] > 0:
                ## this is what I want to get
                ## just make sure I do not do anything wrong...
                theta_z = rng3 * local_state.uniform(np.max((0, self.rot_xyz[2] - 5.)), self.rot_xyz[2] + 5.) * np.pi / 180.
            else:
                theta_z = rng3 * local_state.uniform(0, - self.rot_xyz[2]) * np.pi / 180.

            if self.scaling > 0:
                scalingfactor = 1 + local_state.uniform(np.max((0, self.scaling - 0.05)), self.scaling + 0.05)
            else:
                scalingfactor = 1 + local_state.uniform(0, - self.scaling)

            if rng4 < 1:
                scalingfactor = 1 / scalingfactor
            scale = np.eye(3, 3) * scalingfactor
        else:
            rng1 = rng
            rng2 = rng
            rng3 = rng
            rng4 = rng
            theta_x = rng1 * self.rot_xyz[0] * np.pi / 180.
            theta_y = rng2 * self.rot_xyz[1] * np.pi / 180.
            theta_z = rng3 * self.rot_xyz[2] * np.pi / 180.
            scalingfactor = 1 + self.scaling
            if rng4 < 1:
                scalingfactor = 1 / scalingfactor
            scale = np.eye(3, 3) * scalingfactor

        rot_x = np.array([[np.cos(theta_x), -np.sin(theta_x), 0.],
                          [np.sin(theta_x), np.cos(theta_x), 0.],
                          [0., 0., 1.]])

        rot_y = np.array([[np.cos(theta_y), 0., np.sin(theta_y)],
                          [0., 1., 0.],
                          [-np.sin(theta_y), 0., np.cos(theta_y)]])

        rot_z = np.array([[1., 0., 0.],
                          [0., np.cos(theta_z), -np.sin(theta_z)],
                          [0., np.sin(theta_z), np.cos(theta_z)]])

        # Sample the scale (zoom in/out)
        # TODO: Non isotropic?
        # Affine transformation matrix.
        transformation_mtx = np.dot(scale, np.dot(rot_z, np.dot(rot_x, rot_y)))

        return transformation_mtx

    def _apply_transformation(self, image, coords, interp_order=2., boundary_mode='nearest', cval=0., imgchannels=False):
        # image should be 3 dimensional (Height, Width, Depth). Not multi-channel.
        # interp_order: Integer. 1,2,3 for images, 0 for nearest neighbour on masks (GT & brainmasks)
        # boundary_mode = 'constant', 'min', 'nearest', 'mirror...
        # cval: float. value given to boundaries if mode is constant.
        assert interp_order in [0, 1, 2, 3]

        mode = boundary_mode
        if mode == 'min':
            cval = np.min(image)
            mode = 'constant'

        # For recentering
        '''be aware that it should be different for even and odd'''
        '''for the default setting, shape is even, so nnunet would not worry too much'''
        '''but for DM, it is odd, and I should pick the floor'''
        for d in range(len(image.shape)):
            ctr = int(np.floor(image.shape[d] / 2.))
            coords[d] += ctr

        if imgchannels==False and interp_order != 0:
            unique_labels = np.unique(image)
            result = np.zeros(coords.shape[1:], image.dtype)
            for i, c in enumerate(unique_labels):
                new_image = scipy.ndimage.map_coordinates((image == c).astype(float),
                                                    coords,
                                                    order=interp_order,
                                                    mode=mode,
                                                    cval=cval)
                result[new_image >= 0.5] = c
            return result
        else:
            new_image = scipy.ndimage.map_coordinates(image.astype(float),
                                                    coords,
                                                    order=interp_order,
                                                    mode=mode,
                                                    cval=cval).astype(image.dtype)
            return new_image

    def __call__(self, images_l, transf_mtx, interp_orders, boundary_modes, patch_size, cval=0., imgchannels=False):
        # images_l : List of images, or an array where first dimension is over images (eg channels).
        #            An image (element of the var) can be None, and it will be returned unchanged.
        #            If images_l is None, then returns None.
        # transf_mtx: Given (from get_random_transformation), -1, or None.
        #             If -1, no augmentation/transformation will be done.
        #             If None, new random will be made.
        # intrp_orders : Int or List of integers. Orders of bsplines for interpolation, one per image in images_l.
        #                Suggested: 3 for images. 1 is like linear. 0 for masks/labels, like NN.
        # boundary_mode = String or list of strings. 'constant', 'min', 'nearest', 'mirror...
        # cval: single float value. Value given to boundaries if mode is 'constant'.
        if images_l is None:
            return None
        if transf_mtx is None:  # Get random transformation.
            transf_mtx = self.roll_dice_and_get_random_transformation()
        if not isinstance(transf_mtx, np.ndarray) and transf_mtx == -1:  # Do not augment
            return images_l
        # If scalars/string was given, change it to list of scalars/strings, per image.
        if isinstance(interp_orders, int):
            interp_orders = [interp_orders] * len(images_l[0])
        if isinstance(boundary_modes, str):
            boundary_modes = [boundary_modes] * len(images_l[0])
        
        '''I should be careful here'''
        ## For Unet, the patch size is like 80 * 80 * 80
        ## Therefore I should have this to make the rotation center is 0.5, and in the middle, neat!
        coords = create_zero_centered_coordinate_mesh(patch_size)

        coords = np.dot(coords.reshape(len(coords), -1).transpose(), transf_mtx).transpose().reshape(coords.shape)

        # Deform images.
        if type(images_l) is list:
            new_images = images_l
        else:
            new_images = np.zeros((images_l.shape[0], *patch_size))
        for img_i, int_order, b_mode in zip(range(len(images_l)), interp_orders, boundary_modes):
            if images_l[img_i] is None:
                pass  # Dont do anything. Let it be None.
            else:
                new_images[img_i] = self._apply_transformation(images_l[img_i],
                                                                coords.copy(),
                                                                int_order,
                                                                b_mode,
                                                                cval,
                                                                imgchannels)
        return new_images


############# Currently not used ####################

# DON'T use on patches. Only on images. Cause I ll need to find min and max intensities, to move to range [0,1]
def random_gamma_correction(channels, gamma_std=0.05):
    # Gamma correction: I' = I^gamma
    # channels: list (x pathways) of np arrays [channels, x, y, z]. Whole volumes, channels of a case.
    # IMPORTANT: Does not work if intensities go to negatives.
    if gamma_std is None or gamma_std == 0.:
        return channels

    n_channs = channels[0].shape[0]
    gamma = np.random.normal(1, gamma_std, [n_channs, 1, 1, 1])
    for path_idx in range(len(channels)):
        assert np.min(channels[path_idx]) >= 0.
        channels[path_idx] = np.power(channels[path_idx], 1.5, dtype='float32')

    return channels
