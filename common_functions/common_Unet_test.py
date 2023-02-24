import os
import nibabel as nib
import torch
import numpy as np
from sampling.sampling_DM import get_augment_par
from scipy.ndimage.filters import gaussian_filter
from utilities import ComputMetric, get_patch_size
from typing import Tuple, List
from transformations.augmentSamplerng import augment_sample
from transformations.augmentImagerng import augment_imgs_of_case
from numpy.linalg import inv

def nntestorgan(model, saveresults, name, trainval = False, ImgsegmentSize = [128, 128, 128], deepsupervision = False, DatafileValFold=None, tta=False, ttalist = [0], ttalistprob=[1], NumsClass = 14):
    batch_size = 1
    NumsInputChannel = 1
    if DatafileValFold == None:
        if trainval == False:
            DatafileFold = '/vol/medic02/users/zl9518/Multiorgan/datafileval/'
            DatafileImgc1 = DatafileFold + 'Imgpre-eval.txt'
            DatafileLabel = DatafileFold + 'seg-eval.txt'
            DatafileMask = DatafileFold + 'mask-eval.txt'
        else:
            DatafileFold = '/vol/medic02/users/zl9518/Multiorgan/datafiletraining/'
            DatafileImgc1 = DatafileFold + 'Imgpre-train.txt'
            DatafileLabel = DatafileFold + 'seg-train.txt'
            DatafileMask = DatafileFold + 'mask-train.txt'
    else:
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

        ## I should reverse for nnunet
        # Imgc1 = np.transpose(Imgc1, (2, 1, 0))
        # roi_mask = np.transpose(roi_mask, (2, 1, 0))
        # gtlabel = np.transpose(gtlabel, (2, 1, 0))

        Imgc1 = np.float32(Imgc1)

        knamelist = Imgnamec1.split("/")
        kname = knamelist[-1][9:16]

        channels = Imgc1[None, ...]

        hp_results = tta_rolling(model, channels, batch_size, ImgsegmentSize, NumsInputChannel, NumsClass, tta, ttalist, ttalistprob, deepsupervision)

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

def testkits(model, saveresults, name, trainval = False, ImgsegmentSize = [80, 160, 160], deepsupervision=False, DatafileValFold=None, tta=False, ttalist = [0], ttalistprob=[1], NumsClass = 3):
    batch_size = 1
    NumsInputChannel = 1
    if trainval:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-train.txt'
        DatafileLabel = DatafileFold + 'seg-train.txt'
        DatafileMask = DatafileFold + 'mask-train.txt'
    else:
        DatafileFold = DatafileValFold
        DatafileImgc1 = DatafileFold + 'Imgpre-eval.txt'
        DatafileLabel = DatafileFold + 'seg-eval.txt'
        DatafileMask = DatafileFold + 'mask-eval.txt'

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

        ## I should reverse for nnunet
        # Imgc1 = np.transpose(Imgc1, (2, 1, 0))
        # roi_mask = np.transpose(roi_mask, (2, 1, 0))
        # gtlabel = np.transpose(gtlabel, (2, 1, 0))

        Imgc1 = np.float32(Imgc1)

        knamelist = Imgnamec1.split("/")
        kname = knamelist[-2]

        if kname[0:4] != 'case':
            kname = 'case_0' + knamelist[-1][0:4]

        channels = Imgc1[None, ...]

        hp_results = tta_rolling(model, channels, batch_size, ImgsegmentSize, NumsInputChannel, NumsClass, tta, ttalist, ttalistprob, deepsupervision)
        
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

        # print(DSCen)
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

def testatlas(model, saveresults, name, trainval = False, ImgsegmentSize = [80, 160, 160], deepsupervision=False, DatafileValFold=None, tta=False, ttalist = [0], ttalistprob=[1], NumsClass = 2):
    batch_size = 1
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
    # for numr in range(6, 7):

        Imgnamec1 = Imgreadc1[numr]
        Imgloadc1 = nib.load(Imgnamec1)
        Imgc1 = Imgloadc1.get_fdata()
        Maskname = Maskread[numr]
        Maskload = nib.load(Maskname)
        roi_mask = Maskload.get_fdata()
        Labelname = Labelread[numr]
        Labelload = nib.load(Labelname)
        gtlabel = Labelload.get_fdata()

        ## I should reverse for nnunet
        # Imgc1 = np.transpose(Imgc1, (2, 1, 0))
        # roi_mask = np.transpose(roi_mask, (2, 1, 0))
        # gtlabel = np.transpose(gtlabel, (2, 1, 0))

        Imgc1 = np.float32(Imgc1)

        knamelist = Imgnamec1.split("/")
        kname = knamelist[-1][0:6]

        channels = Imgc1[None, ...]

        hp_results = tta_rolling(model, channels, batch_size, ImgsegmentSize, NumsInputChannel, NumsClass, tta, ttalist, ttalistprob, deepsupervision)
        
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

            directory = "./output/atlas/%s/" % (name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savename = directory + 'pred_' + kname + '_Segm.nii.gz'
            nib.save(newLabelImg, savename)

        #
        labelwt = gtlabel == 1
        predwt = imgToSave == 1

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

def testprostate(model, saveresults, name, trainval = False, ImgsegmentSize = [80, 160, 160], deepsupervision=False, DatafileValFold=None, tta=False, ttalist = [0], ttalistprob=[1], NumsClass = 2):
    batch_size = 1
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
    # for numr in range(6, 7):

        Imgnamec1 = Imgreadc1[numr]
        Imgloadc1 = nib.load(Imgnamec1)
        Imgc1 = Imgloadc1.get_fdata()
        Maskname = Maskread[numr]
        Maskload = nib.load(Maskname)
        roi_mask = Maskload.get_fdata()
        Labelname = Labelread[numr]
        Labelload = nib.load(Labelname)
        gtlabel = Labelload.get_fdata()

        ## I should reverse for nnunet
        # Imgc1 = np.transpose(Imgc1, (2, 1, 0))
        # roi_mask = np.transpose(roi_mask, (2, 1, 0))
        # gtlabel = np.transpose(gtlabel, (2, 1, 0))

        Imgc1 = np.float32(Imgc1)

        knamelist = Imgnamec1.split("/")
        kname = knamelist[-1][0:6]

        channels = Imgc1[None, ...]

        hp_results = tta_rolling(model, channels, batch_size, ImgsegmentSize, NumsInputChannel, NumsClass, tta, ttalist, ttalistprob, deepsupervision)
        
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

            directory = "./output/prostate/%s/" % (name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            savename = directory + 'pred_' + kname + '_Segm.nii.gz'
            nib.save(newLabelImg, savename)

        #
        labelwt = gtlabel == 1
        predwt = imgToSave == 1

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

def testcardiac(model, saveresults, name, trainval = False, ImgsegmentSize = [80, 160, 160], deepsupervision=False, DatafileValFold=None, tta=False, ttalist = [0], ttalistprob=[1], NumsClass = 2):
    batch_size = 1
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
    # for numr in range(6, 7):

        Imgnamec1 = Imgreadc1[numr]
        Imgloadc1 = nib.load(Imgnamec1)
        Imgc1 = Imgloadc1.get_fdata()
        Maskname = Maskread[numr]
        Maskload = nib.load(Maskname)
        roi_mask = Maskload.get_fdata()
        Labelname = Labelread[numr]
        Labelload = nib.load(Labelname)
        gtlabel = Labelload.get_fdata()

        ## I should reverse for nnunet
        # Imgc1 = np.transpose(Imgc1, (2, 1, 0))
        # roi_mask = np.transpose(roi_mask, (2, 1, 0))
        # gtlabel = np.transpose(gtlabel, (2, 1, 0))

        Imgc1 = np.float32(Imgc1)

        knamelist = Imgnamec1.split("/")
        # kname = knamelist[-1][0:8]
        kname = knamelist[-1].split(".")[0]

        channels = Imgc1[None, ...]

        hp_results = tta_rolling(model, channels, batch_size, ImgsegmentSize, NumsInputChannel, NumsClass, tta, ttalist, ttalistprob, deepsupervision)
        
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

def tta_rolling(model, channels, batch_size, ImgsegmentSize, NumsInputChannel, NumsClass, tta, ttalist, ttalistprob, deepsupervision):
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

            '''
            I do not need to do anything for anisotropy patches, because it operats with the whole images.
            '''

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

        offset = [0, 0, 0]

        pad_border_mode = 'constant'
        pad_kwargs = dict()
        pad_kwargs['constant_values'] = 0
        data, slicer = pad_nd_image(channels_augment, ImgsegmentSize, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape
        step_size = 0.5
        steps = _compute_steps_for_sliding_window(ImgsegmentSize, data_shape[1:], step_size)

        hp = np.zeros([NumsClass] + list(data.shape[1:]), dtype=np.float32)
        aggregated_nb_of_predictions = np.zeros([NumsClass] + list(data.shape[1:]), dtype=np.float32)
        xpixels = []
        ypixels = []
        zpixels = []

        for jx in steps[0]:
            for jy in steps[1]:
                for jz in steps[2]:
                    xpixels.append(jx)
                    ypixels.append(jy)
                    zpixels.append(jz)

        inputxnor = getallbatch(data, ImgsegmentSize, xpixels, ypixels, zpixels, offset)

        ## gaussian filter
        patch_size = [ImgsegmentSize[0], ImgsegmentSize[1], ImgsegmentSize[2]]
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * 1. / 8 for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        inputxnor = torch.tensor(np.array(inputxnor))
        inputxnor = inputxnor.float().cuda()
        for xlist in range(0, len(inputxnor), batch_size):
            batchxnor = inputxnor[xlist: xlist + batch_size, :, :, :, :]

            xstarts = xpixels[xlist: xlist + batch_size]
            ystarts = ypixels[xlist: xlist + batch_size]
            zstarts = zpixels[xlist: xlist + batch_size]

            with torch.no_grad():
                if tta:
                    auglist = [0, 1, 2, 3, 4, 5, 6, 7]
                    # auglist = [0, 4]
                    num_results = len(auglist)
                    mirror_axes = [0, 1, 2]
                    for m in range(num_results):
                        if m == 0:
                            pred = model(batchxnor)
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output = 1 / num_results * pred[0]
                            else:
                                output = 1 / num_results * pred
                        if m == 1 and (2 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (4,)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (4,))
                            else:
                                output += 1 / num_results * torch.flip(pred, (4,))
                        if m == 2 and (1 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (3,)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (3,))
                            else:
                                output += 1 / num_results * torch.flip(pred, (3,))
                        if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (4, 3)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (4, 3))
                            else:
                                output += 1 / num_results * torch.flip(pred, (4, 3))
                        if m == 4 and (0 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (2,)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (2,))
                            else:
                                output += 1 / num_results * torch.flip(pred, (2,))
                        if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (4, 2)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (4, 2))
                            else:
                                output += 1 / num_results * torch.flip(pred, (4, 2))
                        if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (2, 3)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (2, 3))
                            else:
                                output += 1 / num_results * torch.flip(pred, (2, 3))
                        if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                            pred = model(torch.flip(batchxnor, (4, 3, 2)))
                            if len(pred) == 2:
                                pred = pred[0]
                            if deepsupervision:
                                output += 1 / num_results * torch.flip(pred[0], (4, 3, 2))
                            else:
                                output += 1 / num_results * torch.flip(pred, (4, 3, 2))
                else:
                    pred = model(batchxnor)
                    ## in case it is multi-task model.
                    if len(pred) == 2:
                        pred = pred[0]
                    if deepsupervision:
                        output = pred[0]
                    else:
                        output = pred
            output = output.data.cpu().numpy()
            # if the one is features..
            if len(output.shape) == 4:
                output = output[np.newaxis, ...]

            kbatch = 0
            for xstart, ystart, zstart in zip(xstarts, ystarts, zstarts):
                # only crop the center parts.
                # maybe I should use gaussain? to do ..
                # hp[:, xstart + offset:xstart + offset + PredSizetest, ystart + offset:ystart + offset + PredSizetest,
                # zstart + offset:zstart + offset + PredSizetest] = output[kbatch, :, offset:offset + PredSizetest,
                #                                                   offset:offset + PredSizetest,
                #                                                   offset:offset + PredSizetest]
                hp[:, xstart:xstart + ImgsegmentSize[0], ystart:ystart + ImgsegmentSize[1],
                zstart:zstart + ImgsegmentSize[2]] += output[kbatch, :, :, :, :] * gaussian_importance_map
                aggregated_nb_of_predictions[:, xstart:xstart + ImgsegmentSize[0], ystart:ystart + ImgsegmentSize[1],
                zstart:zstart + ImgsegmentSize[2]] += gaussian_importance_map
                kbatch = kbatch + 1

        slicer = tuple(
            [slice(0, hp.shape[i]) for i in
             range(len(hp.shape) - (len(slicer) - 1))] + slicer[1:])
        hp = hp[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]
        hp = hp / aggregated_nb_of_predictions

        ## to see if the probability map needs revert spatial transformations 
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
        
        ## hp_revert with shape: 1, D, W, H
        ## Here I want to try to ensemble with prediction score.
        # hp_revert = np.exp(hp_revert) / np.sum(np.exp(hp_revert), axis=0)
        hp_results += hp_revert / sum(ttalistprob) * ttaindexprob
    
    return hp_results


def getallbatch(Imgenlarge, ImgsegmentSize, xpixels, ypixels, zpixels, offset):

    inputxnor = []

    # normal pathway
    for (selindex_x, selindex_y, selindex_z) in zip(xpixels, ypixels, zpixels):
        coord_center = np.zeros(3, dtype=int)
        coord_center[0] = selindex_x + ImgsegmentSize[0] // 2
        coord_center[1] = selindex_y + ImgsegmentSize[1] // 2
        coord_center[2] = selindex_z + ImgsegmentSize[2] // 2

        samplekernal_primary = 1
        channs_of_sample_per_path_normal = Imgenlarge[:,
                                    coord_center[0] - ImgsegmentSize[0] // 2: coord_center[0] + ImgsegmentSize[0] // 2,
                                    coord_center[1] - ImgsegmentSize[1] // 2: coord_center[1] + ImgsegmentSize[1] // 2,
                                    coord_center[2] - ImgsegmentSize[2] // 2: coord_center[2] + ImgsegmentSize[2] // 2]
        inputxnor.append(channs_of_sample_per_path_normal)

    return inputxnor

def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit

    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer

def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 32 and step_size of 0.5, then we want to make 4 steps starting at coordinate 0, 27, 55, 78
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps
