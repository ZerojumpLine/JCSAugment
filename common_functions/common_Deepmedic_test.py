import os
import nibabel as nib
import torch
import numpy as np
from sampling.sampling_DM import Getimagepatchwithcoord, get_augment_par
from sampling.sampling_DM import getImagePartFromSubsampledImageForTraining
from utilities import ComputMetric, get_patch_size
from transformations.augmentSamplerng import augment_sample
from transformations.augmentImagerng import augment_imgs_of_case
from numpy.linalg import inv

def testatlas(model, saveresults, name, trainval = False, DatafileValFold=None, ttalist = [0], ttalistprob=[1], tta = False, NumsClass = 2):
    batch_size = 10
    NumsInputChannel = 1
    
    if trainval == False:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-eval.txt'
        DatafileLabel = DatafileFold + 'seg-eval.txt'
        DatafileMask = DatafileFold + 'mask-eval.txt'
    else:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-train.txt'
        DatafileLabel = DatafileFold + 'seg-train.txt'
        DatafileMask = DatafileFold + 'mask-train.txt'

    Imgfilec1 = open(DatafileImgc1)
    Imgreadc1 = Imgfilec1.read().splitlines()
    Maskfile = open(DatafileMask)
    Maskread = Maskfile.read().splitlines()
    Labelfile = open(DatafileLabel)
    Labelread = Labelfile.read().splitlines()

    DSClist = []
    SENSlist = []
    PREClist = []

    for numr in range(len(Imgreadc1)):
    # for numr in range(10, 11):

        Imgnamec1 = Imgreadc1[numr]
        Imgloadc1 = nib.load(Imgnamec1)
        Imgc1 = Imgloadc1.get_fdata()
        Maskname = Maskread[numr]
        Maskload = nib.load(Maskname)
        roi_mask = Maskload.get_fdata()
        Labelname = Labelread[numr]
        Labelload = nib.load(Labelname)
        gtlabel = Labelload.get_fdata()

        knamelist = Imgnamec1.split("/")
        kname = knamelist[-1][0:6]

        channels = Imgc1[None, ...]

        hp_results = tta_rolling(model, channels, batch_size, NumsInputChannel, NumsClass, ttalist, ttalistprob, tta)

        predSegmentation = np.argmax(hp_results, axis=0)
        PredSegmentationWithinRoi = predSegmentation * roi_mask
        imgToSave = PredSegmentationWithinRoi

        if saveresults:
            npDtype = np.dtype(np.float32)
            proxy_origin = nib.load(Imgnamec1)
            hdr_origin = proxy_origin.header
            affine_origin = proxy_origin.affine
            proxy_origin.uncache()

            newLabelImg = nib.Nifti1Image(imgToSave, affine_origin)
            newLabelImg.set_data_dtype(npDtype)

            dimsImgToSave = len(imgToSave.shape)
            newZooms = list(hdr_origin.get_zooms()[:dimsImgToSave])
            if len(newZooms) < dimsImgToSave:  # Eg if original image was 3D, but I need to save a multi-channel image.
                newZooms = newZooms + [1.0] * (dimsImgToSave - len(newZooms))
            newLabelImg.header.set_zooms(newZooms)

            directory = "./output/atlas/%s/" % (name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savename = directory + 'pred_' + kname + '_Segm.nii.gz'
            nib.save(newLabelImg, savename)

        #
        labelwt = gtlabel > 0
        predwt = imgToSave > 0

        DSCwt, SENSwt, PRECwt = ComputMetric(labelwt, predwt)

        # print(DSCwt)
        DSClist.append([DSCwt])
        SENSlist.append([SENSwt])
        PREClist.append([PRECwt])
        print('case ' + str(numr) + ' done')

    DSCmean = np.array(DSClist)
    DSCmean = DSCmean.mean(axis=0)
    SENSmean = np.array(SENSlist)
    SENSmean = SENSmean.mean(axis=0)
    PRECmean = np.array(PREClist)
    PRECmean = PRECmean.mean(axis=0)

    return DSCmean, SENSmean, PRECmean

def testprostate(model, saveresults, name, trainval = False, DatafileValFold=None, ttalist = [0], ttalistprob=[1], tta = False, NumsClass = 2):
    batch_size = 10
    NumsInputChannel = 1
    
    if trainval == False:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-eval.txt'
        DatafileLabel = DatafileFold + 'seg-eval.txt'
        DatafileMask = DatafileFold + 'mask-eval.txt'
    else:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-train.txt'
        DatafileLabel = DatafileFold + 'seg-train.txt'
        DatafileMask = DatafileFold + 'mask-train.txt'

    Imgfilec1 = open(DatafileImgc1)
    Imgreadc1 = Imgfilec1.read().splitlines()
    Maskfile = open(DatafileMask)
    Maskread = Maskfile.read().splitlines()
    Labelfile = open(DatafileLabel)
    Labelread = Labelfile.read().splitlines()

    DSClist = []
    SENSlist = []
    PREClist = []

    for numr in range(len(Imgreadc1)):
    # for numr in range(10, 11):

        Imgnamec1 = Imgreadc1[numr]
        Imgloadc1 = nib.load(Imgnamec1)
        Imgc1 = Imgloadc1.get_fdata()
        Maskname = Maskread[numr]
        Maskload = nib.load(Maskname)
        roi_mask = Maskload.get_fdata()
        Labelname = Labelread[numr]
        Labelload = nib.load(Labelname)
        gtlabel = Labelload.get_fdata()

        knamelist = Imgnamec1.split("/")
        kname = knamelist[-1][0:6]

        channels = Imgc1[None, ...]

        hp_results = tta_rolling(model, channels, batch_size, NumsInputChannel, NumsClass, ttalist, ttalistprob, tta)

        predSegmentation = np.argmax(hp_results, axis=0)
        PredSegmentationWithinRoi = predSegmentation * roi_mask
        imgToSave = PredSegmentationWithinRoi

        if saveresults:
            npDtype = np.dtype(np.float32)
            proxy_origin = nib.load(Imgnamec1)
            hdr_origin = proxy_origin.header
            affine_origin = proxy_origin.affine
            proxy_origin.uncache()

            newLabelImg = nib.Nifti1Image(imgToSave, affine_origin)
            newLabelImg.set_data_dtype(npDtype)

            dimsImgToSave = len(imgToSave.shape)
            newZooms = list(hdr_origin.get_zooms()[:dimsImgToSave])
            if len(newZooms) < dimsImgToSave:  # Eg if original image was 3D, but I need to save a multi-channel image.
                newZooms = newZooms + [1.0] * (dimsImgToSave - len(newZooms))
            newLabelImg.header.set_zooms(newZooms)

            directory = "./output/prostate/%s/" % (name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savename = directory + 'pred_' + kname + '_Segm.nii.gz'
            nib.save(newLabelImg, savename)

        #
        labelwt = gtlabel > 0
        predwt = imgToSave > 0

        DSCwt, SENSwt, PRECwt = ComputMetric(labelwt, predwt)

        # print(DSCwt)
        DSClist.append([DSCwt])
        SENSlist.append([SENSwt])
        PREClist.append([PRECwt])
        print('case ' + str(numr) + ' done')

    DSCmean = np.array(DSClist)
    DSCmean = DSCmean.mean(axis=0)
    SENSmean = np.array(SENSlist)
    SENSmean = SENSmean.mean(axis=0)
    PRECmean = np.array(PREClist)
    PRECmean = PRECmean.mean(axis=0)

    return DSCmean, SENSmean, PRECmean

def testcardiac(model, saveresults, name, trainval = False, DatafileValFold=None, ttalist = [0], ttalistprob=[1], tta = False, NumsClass = 4):
    batch_size = 10
    NumsInputChannel = 1
    
    if trainval == False:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-eval.txt'
        DatafileLabel = DatafileFold + 'seg-eval.txt'
        DatafileMask = DatafileFold + 'mask-eval.txt'
    else:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-train.txt'
        DatafileLabel = DatafileFold + 'seg-train.txt'
        DatafileMask = DatafileFold + 'mask-train.txt'

    Imgfilec1 = open(DatafileImgc1)
    Imgreadc1 = Imgfilec1.read().splitlines()
    Maskfile = open(DatafileMask)
    Maskread = Maskfile.read().splitlines()
    Labelfile = open(DatafileLabel)
    Labelread = Labelfile.read().splitlines()

    DSClist = []
    SENSlist = []
    PREClist = []

    for numr in range(len(Imgreadc1)):
    # for numr in range(10, 11):

        Imgnamec1 = Imgreadc1[numr]
        Imgloadc1 = nib.load(Imgnamec1)
        Imgc1 = Imgloadc1.get_fdata()
        Maskname = Maskread[numr]
        Maskload = nib.load(Maskname)
        roi_mask = Maskload.get_fdata()
        Labelname = Labelread[numr]
        Labelload = nib.load(Labelname)
        gtlabel = Labelload.get_fdata()

        knamelist = Imgnamec1.split("/")
        kname = knamelist[-1][0:8]

        channels = Imgc1[None, ...]

        hp_results = tta_rolling(model, channels, batch_size, NumsInputChannel, NumsClass, ttalist, ttalistprob, tta)

        predSegmentation = np.argmax(hp_results, axis=0)
        PredSegmentationWithinRoi = predSegmentation * roi_mask
        imgToSave = PredSegmentationWithinRoi

        if saveresults:
            npDtype = np.dtype(np.float32)
            proxy_origin = nib.load(Imgnamec1)
            hdr_origin = proxy_origin.header
            affine_origin = proxy_origin.affine
            proxy_origin.uncache()

            newLabelImg = nib.Nifti1Image(imgToSave, affine_origin)
            newLabelImg.set_data_dtype(npDtype)

            dimsImgToSave = len(imgToSave.shape)
            newZooms = list(hdr_origin.get_zooms()[:dimsImgToSave])
            if len(newZooms) < dimsImgToSave:  # Eg if original image was 3D, but I need to save a multi-channel image.
                newZooms = newZooms + [1.0] * (dimsImgToSave - len(newZooms))
            newLabelImg.header.set_zooms(newZooms)

            directory = "./output/cardiac/%s/" % (name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savename = directory + 'pred_' + kname + '_Segm.nii.gz'
            nib.save(newLabelImg, savename)

        #
        labelc1 = gtlabel == 1
        predc1 = imgToSave == 1
        labelc2 = gtlabel == 2
        predc2 = imgToSave == 2
        labelc3 = gtlabel == 3
        predc3 = imgToSave == 3

        DSCc1, SENSc1, PRECc1 = ComputMetric(labelc1, predc1)
        DSCc2, SENSc2, PRECc2 = ComputMetric(labelc2, predc2)
        DSCc3, SENSc3, PRECc3 = ComputMetric(labelc3, predc3)

        # print(DSCwt)
        DSClist.append([DSCc1, DSCc2, DSCc3])
        SENSlist.append([SENSc1, SENSc2, SENSc3])
        PREClist.append([PRECc1, PRECc2, PRECc3])
        print('case ' + str(numr) + ' done')

    DSCmean = np.array(DSClist)
    DSCmean = DSCmean.mean(axis=0)
    SENSmean = np.array(SENSlist)
    SENSmean = SENSmean.mean(axis=0)
    PRECmean = np.array(PREClist)
    PRECmean = PRECmean.mean(axis=0)

    return DSCmean, SENSmean, PRECmean

def testkits(model, saveresults, name, trainval = False, DatafileValFold=None, ttalist = [0], ttalistprob=[1], tta = False, NumsClass = 3):
    batch_size = 10
    NumsInputChannel = 1
    if trainval == False:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-eval.txt'
        DatafileLabel = DatafileFold + 'seg-eval.txt'
        DatafileMask = DatafileFold + 'mask-eval.txt'
    else:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-train.txt'
        DatafileLabel = DatafileFold + 'seg-train.txt'
        DatafileMask = DatafileFold + 'mask-train.txt'

    Imgfilec1 = open(DatafileImgc1)
    Imgreadc1 = Imgfilec1.read().splitlines()
    Maskfile = open(DatafileMask)
    Maskread = Maskfile.read().splitlines()
    Labelfile = open(DatafileLabel)
    Labelread = Labelfile.read().splitlines()

    DSClist = []
    SENSlist = []
    PREClist = []

    for numr in range(len(Imgreadc1)):
    # for numr in range(10, 11):

        Imgnamec1 = Imgreadc1[numr]
        Imgloadc1 = nib.load(Imgnamec1)
        Imgc1 = Imgloadc1.get_fdata()
        Maskname = Maskread[numr]
        Maskload = nib.load(Maskname)
        roi_mask = Maskload.get_fdata()
        Labelname = Labelread[numr]
        Labelload = nib.load(Labelname)
        gtlabel = Labelload.get_fdata()

        knamelist = Imgnamec1.split("/")
        kname = knamelist[-2]

        if kname[0:4] != 'case':
            kname = 'case_0' + knamelist[-1][0:4]

        channels = Imgc1[None, ...]

        hp_results = tta_rolling(model, channels, batch_size, NumsInputChannel, NumsClass, ttalist, ttalistprob, tta)

        predSegmentation = np.argmax(hp_results, axis=0)
        
        ## use the mask to constratin the results
        PredSegmentationWithinRoi = predSegmentation * roi_mask
        # PredSegmentationWithinRoi = predSegmentation
        # sio.savemat('./result.mat', {'results': PredSegmentationWithinRoi})
        imgToSave = PredSegmentationWithinRoi

        if saveresults:
            npDtype = np.dtype(np.float32)
            proxy_origin = nib.load(Imgnamec1)
            hdr_origin = proxy_origin.header
            affine_origin = proxy_origin.affine
            proxy_origin.uncache()

            newLabelImg = nib.Nifti1Image(imgToSave, affine_origin)
            newLabelImg.set_data_dtype(npDtype)

            dimsImgToSave = len(imgToSave.shape)
            newZooms = list(hdr_origin.get_zooms()[:dimsImgToSave])
            if len(newZooms) < dimsImgToSave:  # Eg if original image was 3D, but I need to save a multi-channel image.
                newZooms = newZooms + [1.0] * (dimsImgToSave - len(newZooms))
            newLabelImg.header.set_zooms(newZooms)

            directory = "./output/kits/%s/" % (name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savename = directory + 'pred_' + kname + '_Segm.nii.gz'
            nib.save(newLabelImg, savename)

        labelwt = gtlabel > 0
        predwt = imgToSave > 0
        labelc3 = gtlabel == 2
        predc3 = imgToSave == 2

        DSCwt, SENSwt, PRECwt = ComputMetric(labelwt, predwt)
        DSCen, SENSen, PRECen = ComputMetric(labelc3, predc3)

        DSClist.append([DSCwt, DSCen])
        SENSlist.append([SENSwt, SENSen])
        PREClist.append([PRECwt, PRECen])
        print('case ' + str(numr) + ' done')

    DSCmean = np.array(DSClist)
    DSCmean = DSCmean.mean(axis=0)
    SENSmean = np.array(SENSlist)
    SENSmean = SENSmean.mean(axis=0)
    PRECmean = np.array(PREClist)
    PRECmean = PRECmean.mean(axis=0)

    return DSCmean, SENSmean, PRECmean

def testorgan(model, saveresults, name, trainval = False, DatafileValFold=None, ttalist = [0], ttalistprob=[1], tta = False, NumsClass = 14):
    batch_size = 10
    NumsInputChannel = 1
    if trainval == False:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-eval.txt'
        DatafileLabel = DatafileFold + 'seg-eval.txt'
        DatafileMask = DatafileFold + 'mask-eval.txt'
    else:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-train.txt'
        DatafileLabel = DatafileFold + 'seg-train.txt'
        DatafileMask = DatafileFold + 'mask-train.txt'

    Imgfilec1 = open(DatafileImgc1)
    Imgreadc1 = Imgfilec1.read().splitlines()
    Maskfile = open(DatafileMask)
    Maskread = Maskfile.read().splitlines()
    Labelfile = open(DatafileLabel)
    Labelread = Labelfile.read().splitlines()

    DSClist = []
    SENSlist = []
    PREClist = []
    PredSumlist = []

    for numr in range(len(Imgreadc1)):
    # for numr in range(10, 11):

        Imgnamec1 = Imgreadc1[numr]
        Imgloadc1 = nib.load(Imgnamec1)
        Imgc1 = Imgloadc1.get_fdata()
        Maskname = Maskread[numr]
        Maskload = nib.load(Maskname)
        roi_mask = Maskload.get_fdata()
        Labelname = Labelread[numr]
        Labelload = nib.load(Labelname)
        gtlabel = Labelload.get_fdata()

        knamelist = Imgnamec1.split("/")
        kname = knamelist[-1][9:16]

        channels = Imgc1[None, ...]

        hp_results = tta_rolling(model, channels, batch_size, NumsInputChannel, NumsClass, ttalist, ttalistprob, tta)

        predSegmentation = np.argmax(hp_results, axis=0)
        ## use the mask to constratin the results
        PredSegmentationWithinRoi = predSegmentation * roi_mask
        # PredSegmentationWithinRoi = predSegmentation
        # sio.savemat('./result.mat', {'results': PredSegmentationWithinRoi})
        imgToSave = PredSegmentationWithinRoi

        if saveresults:
            npDtype = np.dtype(np.float32)
            proxy_origin = nib.load(Imgnamec1)
            hdr_origin = proxy_origin.header
            affine_origin = proxy_origin.affine
            proxy_origin.uncache()

            newLabelImg = nib.Nifti1Image(imgToSave, affine_origin)
            newLabelImg.set_data_dtype(npDtype)

            dimsImgToSave = len(imgToSave.shape)
            newZooms = list(hdr_origin.get_zooms()[:dimsImgToSave])
            if len(newZooms) < dimsImgToSave:  # Eg if original image was 3D, but I need to save a multi-channel image.
                newZooms = newZooms + [1.0] * (dimsImgToSave - len(newZooms))
            newLabelImg.header.set_zooms(newZooms)

            directory = "./output/organ/%s/" % (name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savename = directory + 'pred_' + kname + '_Segm.nii.gz'
            nib.save(newLabelImg, savename)

        labelc0 = gtlabel > 0
        predc0 = imgToSave > 0
        labelc1 = gtlabel == 1
        predc1 = imgToSave == 1
        labelc2 = gtlabel == 2
        predc2 = imgToSave == 2
        labelc3 = gtlabel == 3
        predc3 = imgToSave == 3
        labelc4 = gtlabel == 4
        predc4 = imgToSave == 4
        labelc5 = gtlabel == 5
        predc5 = imgToSave == 5
        labelc6 = gtlabel == 6
        predc6 = imgToSave == 6
        labelc7 = gtlabel == 7
        predc7 = imgToSave == 7
        labelc8 = gtlabel == 8
        predc8 = imgToSave == 8
        labelc9 = gtlabel == 9
        predc9 = imgToSave == 9
        labelc10 = gtlabel == 10
        predc10 = imgToSave == 10
        labelc11 = gtlabel == 11
        predc11 = imgToSave == 11
        labelc12 = gtlabel == 12
        predc12 = imgToSave == 12
        labelc13 = gtlabel == 13
        predc13 = imgToSave == 13

        DSCc0, SENSc0, PRECc0 = ComputMetric(labelc0, predc0)
        DSCc1, SENSc1, PRECc1 = ComputMetric(labelc1, predc1)
        DSCc2, SENSc2, PRECc2 = ComputMetric(labelc2, predc2)
        DSCc3, SENSc3, PRECc3 = ComputMetric(labelc3, predc3)
        DSCc4, SENSc4, PRECc4 = ComputMetric(labelc4, predc4)
        DSCc5, SENSc5, PRECc5 = ComputMetric(labelc5, predc5)
        DSCc6, SENSc6, PRECc6 = ComputMetric(labelc6, predc6)
        DSCc7, SENSc7, PRECc7 = ComputMetric(labelc7, predc7)
        DSCc8, SENSc8, PRECc8 = ComputMetric(labelc8, predc8)
        DSCc9, SENSc9, PRECc9 = ComputMetric(labelc9, predc9)
        DSCc10, SENSc10, PRECc10 = ComputMetric(labelc10, predc10)
        DSCc11, SENSc11, PRECc11 = ComputMetric(labelc11, predc11)
        DSCc12, SENSc12, PRECc12 = ComputMetric(labelc12, predc12)
        DSCc13, SENSc13, PRECc13 = ComputMetric(labelc13, predc13)

        DSClist.append([DSCc0, DSCc1, DSCc2, DSCc3, DSCc4, DSCc5, DSCc6, DSCc7, DSCc8, DSCc9, DSCc10, DSCc11, DSCc12, DSCc13])
        SENSlist.append([SENSc0, SENSc1, SENSc2, SENSc3, SENSc4, SENSc5, SENSc6, SENSc7, SENSc8, SENSc9, SENSc10, SENSc11, SENSc12, SENSc13])
        PREClist.append([PRECc0, PRECc1, PRECc2, PRECc3, PRECc4, PRECc5, PRECc6, PRECc7, PRECc8, PRECc9, PRECc10, PRECc11, PRECc12, PRECc13])
        PredSumlist.append(np.sum(labelc4))
        print('case ' + str(numr) + ' done')

    sel = [n for n, i in enumerate(PredSumlist) if i > 0]

    DSClist = np.array(DSClist)
    DSCmean = DSClist.mean(axis=0)
    SENSlist = np.array(SENSlist)
    SENSmean = SENSlist.mean(axis=0)
    PREClist = np.array(PREClist)
    PRECmean = PREClist.mean(axis=0)

    DSCmeanc4 = DSClist[sel, :].mean(axis=0)
    DSCmean[4] = DSCmeanc4[4]
    SENSmeanc4 = SENSlist[sel, :].mean(axis=0)
    SENSmean[4] = SENSmeanc4[4]
    PRECmeanc4 = PREClist[sel, :].mean(axis=0)
    PRECmean[4] = PRECmeanc4[4]

    return DSCmean, SENSmean, PRECmean

def tta_rolling(model, channels, batch_size, NumsInputChannel, NumsClass, ttalist, ttalistprob, tta):
    hp_results = 0

    augm_img_prms_tr, augm_sample_prms_tr = get_augment_par()
    augms_prms = {**augm_img_prms_tr, **augm_sample_prms_tr}

    for ttaindex, ttaindexprob in zip(ttalist, ttalistprob):
        
        ## I shall do the augmentation here, for the full image.
        ## maybe I should re-implement the inverse transform here.
        ## mainly for the spatial augmentation, methods in augmentImage.py.

        ## it should have 51 components.
        '''be careful, it is not the same as the paper notion, it is the order with sampling_multipreprocess.py'''
        augmentIDlist = [[0], [1, -2], [3, -4], [5, -6], [7, -8], [9, -10], \
                    [11, -12], [13, -14], [15, -16], [17, -18], [19, -20], [21, -22], [23, -24],[25, -26], [27, -28], \
                    [29], [30], [31], [32, -33], [34], [35, -36], [37], [38, -39], [40], \
                    [41, -42], [43, -44], [45, -46], [47, -48], [49, -50], [51, -52], \
                    [53, -54], [55, -56], [57, -58], [59, -60], [61, -62], [63, -64], [65, -66], [67, -68], [69, -70], \
                    [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83]]

        kcount = 0
        skipflag = False
        for key in augms_prms:
            for augmentIDs in augmentIDlist[kcount]:
                if ttaindex == np.abs(augmentIDs):
                    prmssel = augms_prms[key]
                    if augmentIDs > 0:
                        rng = 1
                    else:
                        rng = -1
                    skipflag = True
                    break
            if skipflag:
                break
            else:
                kcount += 1

        if ttaindex < 29 and ttaindex > 0:

            Imgsize_origin = channels.shape[1:]
            
            rotrange_z = prmssel['rot_xyz'][2]
            rotrange_y = prmssel['rot_xyz'][1]
            rotrange_x = prmssel['rot_xyz'][0]
            scalingrange = prmssel['scaling']
            
            new_patch_size_primary = get_patch_size(Imgsize_origin, 
                (-rotrange_z / 360 * 2. * np.pi, rotrange_z / 360 * 2. * np.pi), 
                (-rotrange_y / 360 * 2. * np.pi, rotrange_y / 360 * 2. * np.pi),
                (-rotrange_x / 360 * 2. * np.pi, rotrange_x / 360 * 2. * np.pi), 
                (1/(1+scalingrange), 1+scalingrange))

            ## return the center? to get the right transform.
            (channels_augment, _, transf_mtx) = augment_imgs_of_case(channels.copy(), None,
                                                            None, None, prmssel, new_patch_size_primary, rng)
            transf_mtx_inv = inv(transf_mtx)
            # Img = []
            # Img.append(channels[0, :, :, 87])
            # Img.append(channels_augment[0, :, :, 87])
            # Img.append(channels_augment_revert[0, :, :, 87])
            # show_threeimg(Img)
            # Img = []
            # Img.append(gtlabel[:, :, 87])
            # Img.append(gt_lbl_img_augment_revert[:, :, 87])
            # Img.append(gt_lbl_img_augment_revert[:, :, 87]-gtlabel[:, :, 87])
            # show_threeimg(Img)
            # print(np.sum(np.abs(gtlabel)))
            # print(np.sum(np.abs(gt_lbl_img_augment_revert)))
            # print(np.sum(np.abs(gtlabel-gt_lbl_img_augment_revert)))
        else:
            channels_per_path = []
            channels_per_path.append(channels.copy())
            
            Imgenlargeref = channels[:, ::4, ::4, ::4]

            (channs_of_sample_per_path, _) = augment_sample(channels_per_path,
                                                        None, prmssel, Imgenlargeref, rng, 0)
            channels_augment = channs_of_sample_per_path[0]
            '''debug: visualization'''
            # Img = []
            # Img.append(gtlabel[:, :, 87])
            # Img.append(lbls_predicted_part_of_sample[:, :, 87])
            # Img.append(lbls_predicted_part_of_sample_augment[:, :, 87])
            # show_threeimg(Img)

        _, range_x, range_y, range_z = channels_augment.shape

        ImgsegmentSize = 45
        offset = 8
        PredSizetest = ImgsegmentSize - 2 * offset

        ## the same as training
        Imgenlarge = np.zeros((NumsInputChannel, range_x + ImgsegmentSize, range_y + ImgsegmentSize, range_z + ImgsegmentSize))
        Imgenlarge[:, offset:range_x + offset, offset:range_y + offset, offset:range_z + offset] = channels_augment

        # Imgenlarge = np.flip(Imgenlarge, (3, ))

        hp = np.zeros((NumsClass, range_x + ImgsegmentSize, range_y + ImgsegmentSize, range_z + ImgsegmentSize))
        x_sample = np.floor(range_x / PredSizetest) + 1
        x_sample = x_sample.astype(np.int16)
        y_sample = np.floor(range_y / PredSizetest) + 1
        y_sample = y_sample.astype(np.int16)
        z_sample = np.floor(range_z / PredSizetest) + 1
        z_sample = z_sample.astype(np.int16)

        xpixels = []
        ypixels = []
        zpixels = []

        for jx in range(0, x_sample):
            for jy in range(0, y_sample):
                for jz in range(0, z_sample):
                    xstart = jx * PredSizetest
                    ystart = jy * PredSizetest
                    zstart = jz * PredSizetest
                    xpixels.append(xstart)
                    ypixels.append(ystart)
                    zpixels.append(zstart)

        inputxnor, inputxsub1, inputxsub2 = getallbatch(Imgenlarge, ImgsegmentSize, xpixels, ypixels, zpixels, offset)

        inputxnor = torch.tensor(np.array(inputxnor))
        inputxsub1 = torch.tensor(np.array(inputxsub1))
        inputxsub2 = torch.tensor(np.array(inputxsub2))
        inputxnor = inputxnor.float().cuda()
        inputxsub1 = inputxsub1.float().cuda()
        inputxsub2 = inputxsub2.float().cuda()
        for xlist in range(0, len(inputxnor), batch_size):
            batchxnor = inputxnor[xlist: xlist + batch_size, :, :, :, :]
            batchxsub1 = inputxsub1[xlist: xlist + batch_size, :, :, :, :]
            batchxsub2 = inputxsub2[xlist: xlist + batch_size, :, :, :, :]

            xstarts = xpixels[xlist: xlist + batch_size]
            ystarts = ypixels[xlist: xlist + batch_size]
            zstarts = zpixels[xlist: xlist + batch_size]

            with torch.no_grad():
                if tta:
                    output = 0
                    auglist = [0, 1, 2, 3, 4, 5, 6, 7]
                    # auglist = [1]
                    num_results = len(auglist)
                    mirror_axes = [0, 1, 2]
                    for m in auglist:
                        if m == 0:
                            pred = model(batchxnor, batchxsub1, batchxsub2)
                            output = 1 / num_results * pred
                        if m == 1:
                            pred = model(torch.flip(batchxnor, (4,)), 
                                        torch.flip(batchxsub1, (4,)), 
                                        torch.flip(batchxsub2, (4,)))
                            output += 1 / num_results * torch.flip(pred, (4,))
                        if m == 2:
                            pred = model(torch.flip(batchxnor, (3,)), 
                                        torch.flip(batchxsub1, (3,)), 
                                        torch.flip(batchxsub2, (3,)))
                            output += 1 / num_results * torch.flip(pred, (3,))
                        if m == 3:
                            pred = model(torch.flip(batchxnor, (4, 3)), 
                                        torch.flip(batchxsub1, (4, 3)), 
                                        torch.flip(batchxsub2, (4, 3)))
                            output += 1 / num_results * torch.flip(pred, (4, 3))
                        if m == 4:
                            pred = model(torch.flip(batchxnor, (2,)), 
                                        torch.flip(batchxsub1, (2,)), 
                                        torch.flip(batchxsub2, (2,)))
                            output += 1 / num_results * torch.flip(pred, (2,))
                        if m == 5:
                            pred = model(torch.flip(batchxnor, (4, 2)), 
                                        torch.flip(batchxsub1, (4, 2)), 
                                        torch.flip(batchxsub2, (4, 2)))
                            output += 1 / num_results * torch.flip(pred, (4, 2))
                        if m == 6:
                            pred = model(torch.flip(batchxnor, (2, 3)), 
                                        torch.flip(batchxsub1, (2, 3)), 
                                        torch.flip(batchxsub2, (2, 3)))
                            output += 1 / num_results * torch.flip(pred, (2, 3))
                        if m == 7:
                            pred = model(torch.flip(batchxnor, (4, 3, 2)), 
                                        torch.flip(batchxsub1, (4, 3, 2)), 
                                        torch.flip(batchxsub2, (4, 3, 2)))
                            output += 1 / num_results * torch.flip(pred, (4, 3, 2))
                else:
                    output = model(batchxnor, batchxsub1, batchxsub2)
            output = output.data.cpu().numpy()

            kbatch = 0
            for xstart, ystart, zstart in zip(xstarts, ystarts, zstarts):
                hp[:, xstart + offset:xstart + offset + PredSizetest,
                ystart + offset:ystart + offset + PredSizetest,
                zstart + offset:zstart + offset + PredSizetest] = output[kbatch, :, :, :, :]
                kbatch = kbatch + 1

        hp = hp[:, offset:range_x + offset,offset:range_y + offset, offset:range_z + offset]

        ## to see if the probability map needs invert spatial transformations 
        hp_revert = hp
        if ttaindex < 29 and ttaindex > 0:
            ## it is the revert transform.
            (hp_revert, _, _) = augment_imgs_of_case(hp.copy(), None,
            None, None, prmssel, Imgsize_origin, -rng, transf_mtx_inv)
        if (ttaindex >= 29 and ttaindex <= 40) or ttaindex == 83:
            hp_per_path = []
            hp_per_path.append(hp)
            (hp_revert_per_path, _) = augment_sample(hp_per_path,
                                                        None, prmssel, np.zeros(1), -rng, 0)
            hp_revert = hp_revert_per_path[0]
        hp_results += hp_revert / sum(ttalistprob) * ttaindexprob
    
    return hp_results
    
def getallbatch(Imgenlarge, ImgsegmentSize, xpixels, ypixels, zpixels, offset):

    inputxnor = []
    inputxsub1 = []
    inputxsub2 = []

    LabelsegmentSize = ImgsegmentSize - 2 * offset

    # normal pathway
    for (selindex_x, selindex_y, selindex_z) in zip(xpixels, ypixels, zpixels):
        coord_center = np.zeros(3, dtype=int)
        coord_center[0] = selindex_x + ImgsegmentSize // 2
        coord_center[1] = selindex_y + ImgsegmentSize // 2
        coord_center[2] = selindex_z + ImgsegmentSize // 2

        samplekernal_primary = 1
        channs_of_sample_per_path_normal = Getimagepatchwithcoord(Imgenlarge, ImgsegmentSize // 2, samplekernal_primary, coord_center[0], coord_center[1], coord_center[2])
        inputxnor.append(channs_of_sample_per_path_normal)

    ################################## subpathway 3x3 ##################################

    for (selindex_x, selindex_y, selindex_z) in zip(xpixels, ypixels, zpixels):
        coord_center = np.zeros(3, dtype=int)
        coord_center[0] = selindex_x + ImgsegmentSize // 2
        coord_center[1] = selindex_y + ImgsegmentSize // 2
        coord_center[2] = selindex_z + ImgsegmentSize // 2

        ## get the boundaries for following up sampling.
        leftBoundaryRcz = [coord_center[0] - samplekernal_primary * (ImgsegmentSize - 1) // 2,
                           coord_center[1] - samplekernal_primary * (ImgsegmentSize - 1) // 2,
                           coord_center[2] - samplekernal_primary * (ImgsegmentSize - 1) // 2]
        rightBoundaryRcz = [leftBoundaryRcz[0] + samplekernal_primary * ImgsegmentSize,
                            leftBoundaryRcz[1] + samplekernal_primary * ImgsegmentSize,
                            leftBoundaryRcz[2] + samplekernal_primary * ImgsegmentSize]
        dimsOfPrimarySegment = np.zeros(3, dtype=int)
        dimsOfPrimarySegment[0] = dimsOfPrimarySegment[1] = dimsOfPrimarySegment[2] = ImgsegmentSize
        subFactor = np.zeros(3, dtype=int)
        subFactor[0] = subFactor[1] = subFactor[2] = 3
        recFieldCnn = np.zeros(3, dtype=int)
        recFieldCnn[0] = recFieldCnn[1] = recFieldCnn[2] = offset * 2 + 1
        subsampledImagePartDimensions = np.zeros(3, dtype=int)
        subsampledImagePartDimensions[0] = subsampledImagePartDimensions[1] = subsampledImagePartDimensions[2] = int(np.ceil(LabelsegmentSize / subFactor[0]) + 2 * offset)
        slicesCoordsOfSegmForPrimaryPathway = [[leftBoundaryRcz[0], rightBoundaryRcz[0] - 1],
                                               [leftBoundaryRcz[1], rightBoundaryRcz[1] - 1],
                                               [leftBoundaryRcz[2], rightBoundaryRcz[2] - 1]]

        # this is brought from kostas
        channs_of_sample_per_path_sub1 = getImagePartFromSubsampledImageForTraining(
            dimsOfPrimarySegment=dimsOfPrimarySegment,
            recFieldCnn=recFieldCnn,
            subsampledImageChannels=Imgenlarge,
            image_part_slices_coords=slicesCoordsOfSegmForPrimaryPathway,
            subSamplingFactor=subFactor,
            subsampledImagePartDimensions=subsampledImagePartDimensions
        )
        inputxsub1.append(channs_of_sample_per_path_sub1)

    ################################## subpathway 5x5 ##################################

    for (selindex_x, selindex_y, selindex_z) in zip(xpixels, ypixels, zpixels):
        coord_center = np.zeros(3, dtype=int)
        coord_center[0] = selindex_x + ImgsegmentSize // 2
        coord_center[1] = selindex_y + ImgsegmentSize // 2
        coord_center[2] = selindex_z + ImgsegmentSize // 2

        ## get the boundaries for following up sampling.
        leftBoundaryRcz = [coord_center[0] - samplekernal_primary * (ImgsegmentSize - 1) // 2,
                           coord_center[1] - samplekernal_primary * (ImgsegmentSize - 1) // 2,
                           coord_center[2] - samplekernal_primary * (ImgsegmentSize - 1) // 2]
        rightBoundaryRcz = [leftBoundaryRcz[0] + samplekernal_primary * ImgsegmentSize,
                            leftBoundaryRcz[1] + samplekernal_primary * ImgsegmentSize,
                            leftBoundaryRcz[2] + samplekernal_primary * ImgsegmentSize]
        dimsOfPrimarySegment = np.zeros(3, dtype=int)
        dimsOfPrimarySegment[0] = dimsOfPrimarySegment[1] = dimsOfPrimarySegment[2] = ImgsegmentSize
        subFactor = np.zeros(3, dtype=int)
        subFactor[0] = subFactor[1] = subFactor[2] = 5
        recFieldCnn = np.zeros(3, dtype=int)
        recFieldCnn[0] = recFieldCnn[1] = recFieldCnn[2] = offset * 2 + 1
        subsampledImagePartDimensions = np.zeros(3, dtype=int)
        subsampledImagePartDimensions[0] = subsampledImagePartDimensions[1] = subsampledImagePartDimensions[2] = int(np.ceil(LabelsegmentSize / subFactor[0]) + 2 * offset)
        slicesCoordsOfSegmForPrimaryPathway = [[leftBoundaryRcz[0], rightBoundaryRcz[0] - 1],
                                               [leftBoundaryRcz[1], rightBoundaryRcz[1] - 1],
                                               [leftBoundaryRcz[2], rightBoundaryRcz[2] - 1]]

        # this is brought from kostas
        channs_of_sample_per_path_sub2 = getImagePartFromSubsampledImageForTraining(
            dimsOfPrimarySegment=dimsOfPrimarySegment,
            recFieldCnn=recFieldCnn,
            subsampledImageChannels=Imgenlarge,
            image_part_slices_coords=slicesCoordsOfSegmForPrimaryPathway,
            subSamplingFactor=subFactor,
            subsampledImagePartDimensions=subsampledImagePartDimensions
        )
        inputxsub2.append(channs_of_sample_per_path_sub2)

    return inputxnor, inputxsub1, inputxsub2