import os
import numpy as np
import nibabel as nib
from itertools import repeat
import multiprocessing as mp
from transformations.augmentSamplerng import augment_sample
from transformations.augmentImagerng import augment_imgs_of_case
from sampling.sampling_DM import get_augment_par
from utilities import get_patch_size

# np.random.seed(12345)
# random.seed(12345)
## take image as 37*37*37, and the target as 21*21*21

def getbatchkitsatlas(DatafileFold, batch_size, iteration, selind, maximumcase, offset, logging, proof = 0, ImgsegmentSize=[80,80,80], caselist=None, FGportion = 0.5):
    augm_img_prms_tr,augm_sample_prms_tr  = get_augment_par()
    augms_prms = {**augm_img_prms_tr, **augm_sample_prms_tr}

    # LabelsegmentSize = 21
    # ImgsegmentSize = LabelsegmentSize + 2 * offset

    LabelsegmentSize = ImgsegmentSize

    samplenum = batch_size * iteration

    DatafileImgc1 = DatafileFold + 'Imgpre-train.txt'
    DatafileLabel = DatafileFold + 'seg-train.txt'
    DatafileMask = DatafileFold + 'mask-train.txt'

    Imgfilec1 = open(DatafileImgc1)
    Imgreadc1 = Imgfilec1.read().splitlines()
    Labelfile = open(DatafileLabel)
    Labelread = Labelfile.read().splitlines()
    Maskfile = open(DatafileMask)
    Maskread = Maskfile.read().splitlines()

    samplingpool = iteration
    samplefromeachcase = batch_size

    if iteration > 100: # try not to print for meta validation...
        logging.info('Sampling training segments from ' + str(samplingpool) + ' cases')
        logging.info('Total patches are ' + str(samplenum))
    # else:
    #     logging.info('Augmentation selections are ' + str(selind))

    samplefromthiscase = np.ones((samplingpool)) * samplefromeachcase

    kimg = list(range(samplingpool))
    kstartpos = np.zeros(samplingpool)
    for kpos in kimg:
        for kiters in range(kpos):
            kstartpos[kpos] += samplefromthiscase[kiters]
    mp_pool = mp.Pool(processes=np.minimum(8, len(kimg)))
    mp_pool.daemon = False
    # print(os.getpid())
    # mp_pool.daemon = True

    if caselist is None:
        caselist = np.random.randint(0, len(Imgreadc1), samplingpool)
    if logging is not None:
        logging.info('Sampling caselist for validation: ' + str(caselist))
    seednum = np.random.randint(0, 1e6)

    try:
        with mp_pool as pool:
            results = pool.starmap(getsampleskitsatlas,
                                   zip(kimg, kstartpos[kimg], samplefromthiscase[kimg], repeat(Imgreadc1),
                                       repeat(Labelread), repeat(Maskread), repeat(augms_prms), repeat(offset)
                                       , repeat(ImgsegmentSize), repeat(LabelsegmentSize), repeat(selind), repeat(proof), caselist[kimg], repeat(seednum), repeat(FGportion)))

        # jobs = collections.OrderedDict()
        # for job_idx in kimg:
        #     jobs[job_idx] = mp_pool.apply_async(getsampleskitsatlas, (job_idx, samplefromeachcase, samplefromthiscase[job_idx], Imgreadc1
        #                                                           , Labelread, Maskread, augms_prms, offset
        #                                                           , ImgsegmentSize, LabelsegmentSize, selind, proof))
        # batchxnor = []
        # batchxp1 = []
        # batchxp2 = []
        # batchy = []
        # # augmentselind = []
        # for job_idx in kimg:
        #     results = jobs[job_idx].get(timeout=200)
        #     batchxnor.append(results[0])
        #     batchy.append(results[1])
        #     # augmentselind.append(results[5])

    except mp.TimeoutError:
        print("time out?")
    except:  # Catches everything, even a sys.exit(1) exception.
        mp_pool.terminate()
        mp_pool.join()
        raise Exception("Unexpected error.")
    else:  # Nothing went wrong
        # Needed in case any processes are hanging. mp_pool.close() does not solve this.

        batchxnor = []
        batchy = []
        clsselind = []
        for knum in range(samplingpool):
            batchxnor.append(results[knum][0])
            batchy.append(results[knum][1])
            clsselind.append(results[knum][3])

        # I should wait here and terminate.
        mp_pool.close()
        mp_pool.terminate()
        mp_pool.join()

    batchxnor = np.vstack(batchxnor)
    batchy = np.vstack(batchy)
    clsselind = np.vstack(clsselind)

    listr = list(range(samplenum))

    # batchy = one_hot_embedding(batchy, 4)

    return batchxnor, batchy, listr, clsselind

def getsampleskitsatlas(kimg, kstartpos, samplefromthiscase, Imgreadc1, Labelread, Maskread,
                    augms_prms, offset, ImgsegmentSize, LabelsegmentSize, selind, proof, numr, seednum, FGportion):

    np.random.seed(seednum + kimg)
    # print(os.getpid())

    # logging.info('Sampling training segments from case ' + str(numk + 1) + ' (' + str(numr + 1) + ')' + ' / ' + str(samplingpool))
    # print('Sampling training segments from ' + str(Imgreadc1[numr]))

    Imgnamec1 = Imgreadc1[numr]
    Imgloadc1 = nib.load(Imgnamec1)
    Imgc1 = Imgloadc1.get_fdata()
    Maskname = Maskread[numr]
    Maskload = nib.load(Maskname)
    roi_mask = Maskload.get_fdata()
    Labelname = Labelread[numr]
    Labelload = nib.load(Labelname)
    gt_lbl_img = Labelload.get_fdata()

    channels = Imgc1[None, ...] ## add one dimension

    batchxnor, batchy, clslist = getsamples(channels, gt_lbl_img, roi_mask, kimg, kstartpos, samplefromthiscase,
                    augms_prms, offset, ImgsegmentSize, LabelsegmentSize, selind, proof, False, FGportion)

    return batchxnor, batchy, numr, clslist

def getsamples(channels, gt_lbl_img, roi_mask, kimg, kstartpos, samplefromthiscase,
                    augms_prms, offset, ImgsegmentSize, LabelsegmentSize, selind, proof, bratsflag = False, FGportion = 0.5):
    # local_state = np.random.RandomState()

    range_x, range_y, range_z = roi_mask.shape

    if bratsflag == True:
        Imgenlarge = np.zeros(
            (4, max(ImgsegmentSize[0], range_x), max(ImgsegmentSize[1], range_y), max(ImgsegmentSize[2], range_z)))
    else:
        Imgenlarge = np.zeros(
            (1, max(ImgsegmentSize[0], range_x), max(ImgsegmentSize[1], range_y), max(ImgsegmentSize[2], range_z)))
    Maskenlarge = np.zeros(
        (max(ImgsegmentSize[0], range_x), max(ImgsegmentSize[1], range_y), max(ImgsegmentSize[2], range_z)))
    Labelenlarge = np.zeros(
        (max(ImgsegmentSize[0], range_x), max(ImgsegmentSize[1], range_y), max(ImgsegmentSize[2], range_z)))
    Imgenlarge[:, 0:range_x, 0:range_y, 0:range_z] = channels
    Labelenlarge[0:range_x, 0:range_y, 0:range_z] = gt_lbl_img
    Maskenlarge[0:range_x, 0:range_y, 0:range_z] = roi_mask

    if bratsflag == True:
        batchxnor = np.zeros((int(samplefromthiscase), 4, ImgsegmentSize[0], ImgsegmentSize[1], ImgsegmentSize[2]))
    else:
        batchxnor = np.zeros((int(samplefromthiscase), 1, ImgsegmentSize[0], ImgsegmentSize[1], ImgsegmentSize[2]))

    batchy = np.zeros((int(samplefromthiscase), LabelsegmentSize[0], LabelsegmentSize[1], LabelsegmentSize[2]))
    # numlist = np.zeros((int(samplefromthiscase)))
    clslist = np.zeros((int(samplefromthiscase), 1))

    # cls wise sampling would lead to werid results..
    # numofcls = np.int(np.max(gt_lbl_img)) + 1
    '''
    I notice nnunet just use FG/BG sampling, maybe I should follow his implementation to get the same results.
    '''
    # just use FG / BG sampling
    numofcls = 2

    # cls wise sampling...
    gt_lbl_img_inside_mask = Labelenlarge * Maskenlarge
    offsetx = [int(ImgsegmentSize[0] / 2), int(ImgsegmentSize[1] / 2), int(ImgsegmentSize[2] / 2)]
    lbx = offsetx[0]
    ubx = -offsetx[0]
    lby = offsetx[1]
    uby = -offsetx[1]
    lbz = offsetx[2]
    ubz = -offsetx[2]
    if gt_lbl_img_inside_mask.shape[0] == ImgsegmentSize[0]:
        lbx = int(ImgsegmentSize[0] / 2)
        ubx = int(ImgsegmentSize[0] / 2) + 1
    if gt_lbl_img_inside_mask.shape[1] == ImgsegmentSize[1]:
        lby = int(ImgsegmentSize[1] / 2)
        uby = int(ImgsegmentSize[1] / 2) + 1
    if gt_lbl_img_inside_mask.shape[2] == ImgsegmentSize[2]:
        lbz = int(ImgsegmentSize[2] / 2)
        ubz = int(ImgsegmentSize[2] / 2) + 1

    gt_lbl_img_inside_mask = gt_lbl_img_inside_mask[lbx:ubx, lby:uby, lbz:ubz]
    Maskenlargemask = Maskenlarge[lbx:ubx, lby:uby, lbz:ubz]

    # cls wise sampling...
    bgcls_mask = (gt_lbl_img_inside_mask == 0) * Maskenlargemask

    # if np.random.uniform() > FGportion :
    #     kcls = 0
    # else:
    #     kcls = np.random.randint(1, np.max((np.int(np.max(gt_lbl_img)+1), 2)))
    kcls = np.random.randint(1, np.max((np.int(np.max(gt_lbl_img)+1), 2)))
    Labelindex = np.where(gt_lbl_img_inside_mask == kcls)
    Labelindex_x = Labelindex[0]
    Labelindex_y = Labelindex[1]
    Labelindex_z = Labelindex[2]
    if len(Labelindex_x) == 0:
        ## it can be not Gt on the given slice
        ## find it near the center
        Labelindex_all = np.where(Labelenlarge == kcls)
        if len(Labelindex_all[0]) == 0:  # no cls in this map
            Labelindex_all = np.where(Maskenlarge > 0)
        Labelindex_all = list(Labelindex_all)
        Labelindex_all[0] = np.mean(Labelindex_all[0]) - int(ImgsegmentSize[0] / 2)
        Labelindex_all[1] = np.mean(Labelindex_all[1]) - int(ImgsegmentSize[1] / 2)
        Labelindex_all[2] = np.mean(Labelindex_all[2]) - int(ImgsegmentSize[2] / 2)
        Labelindex_x = [min(max(int(Labelindex_all[0]), 0), int(Labelenlarge.shape[0] - ImgsegmentSize[0]))]
        Labelindex_y = [min(max(int(Labelindex_all[1]), 0), int(Labelenlarge.shape[1] - ImgsegmentSize[1]))]
        Labelindex_z = [min(max(int(Labelindex_all[2]), 0), int(Labelenlarge.shape[2] - ImgsegmentSize[2]))]

    # print(Labelindex_x.size)
    numindex = np.random.randint(0, len(Labelindex_x))
    selindex_x = Labelindex_x[numindex]
    selindex_y = Labelindex_y[numindex]
    selindex_z = Labelindex_z[numindex]

    for k in range(np.int(samplefromthiscase)):  #

        channs_of_sample_per_path, lbls_predicted_part_of_sample = getaugmentpath(Imgenlarge, Labelenlarge, kimg, k,
                                                                                    selindex_x, selindex_y,
                                                                                    selindex_z, ImgsegmentSize,
                                                                                    LabelsegmentSize,
                                                                                    kstartpos, selind,
                                                                                    offset, augms_prms, proof)

        batchxnor[int(k), :, :, :, :] = channs_of_sample_per_path[0]
        batchy[int(k), :, :, :] = lbls_predicted_part_of_sample
        # numlist[int(k)] = selind[np.int(kstartpos + k)]
        clslist[int(k), 0] = kcls

    return batchxnor, batchy, clslist

def getaugmentpath(Imgenlarge, Labelenlarge, kimg, k, selindex_x, selindex_y, selindex_z, ImgsegmentSize,
                   LabelsegmentSize, kstartpos, selind, offset, augms_prms, proof):
    channs_of_sample_per_path = []

    # try to make it exactly like DM's implementation
    coord_center = np.zeros(3, dtype=int)
    coord_center[0] = int(selindex_x + ImgsegmentSize[0] // 2)
    coord_center[1] = int(selindex_y + ImgsegmentSize[1] // 2)
    coord_center[2] = int(selindex_z + ImgsegmentSize[2] // 2)
    # to accelarate the process if no augmentation is needed.
    '''be careful, it is not the same as the paper notion, it is the order with sampling_multipreprocess.py'''
    augmentIDlist = [[0], [1, -2], [3, -4], [5, -6], [7, -8], [9, -10], \
                    [11, -12], [13, -14], [15, -16], [17, -18], [19, -20], [21, -22], [23, -24],[25, -26], [27, -28], \
                    [29], [30], [31], [32, -33], [34], [35, -36], [37], [38, -39], [40], \
                    [41, -42], [43, -44], [45, -46], [47, -48], [49, -50], [51, -52], \
                    [53, -54], [55, -56], [57, -58], [59, -60], [61, -62], [63, -64], [65, -66], [67, -68], [69, -70], \
                    [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83]]

    kcount = 0
    Augindex = selind[np.int(kstartpos + k)]
    skipflag = False
    for key in augms_prms:
        for augmentIDs in augmentIDlist[kcount]:
            if Augindex == np.abs(augmentIDs):
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

    if Augindex < 29 and Augindex > 0:  # image level. totally 15.
        # cut a large img patch.

        samplekernal_primary = 1
        LabelSize_primary = LabelsegmentSize.copy()
        if Augindex == 32 or Augindex == 33:
            # swap the 2rd and 3th dimension
            tmp = LabelSize_primary[2]
            LabelSize_primary[2] = LabelSize_primary[1]
            LabelSize_primary[1] = tmp
        if Augindex == 35 or Augindex == 36:
            # swap the 1st and 3th dimension
            tmp = LabelSize_primary[2]
            LabelSize_primary[2] = LabelSize_primary[0]
            LabelSize_primary[0] = tmp
        half_LabelSize_primary = LabelsegmentSize.copy()

        rotrange_z = prmssel['rot_xyz'][2]
        rotrange_y = prmssel['rot_xyz'][1]
        rotrange_x = prmssel['rot_xyz'][0]
        scalingrange = prmssel['scaling']
        
        new_patch_size_primary = get_patch_size(LabelSize_primary, 
            (-rotrange_z / 360 * 2. * np.pi, rotrange_z / 360 * 2. * np.pi), 
            (-rotrange_y / 360 * 2. * np.pi, rotrange_y / 360 * 2. * np.pi),
            (-rotrange_x / 360 * 2. * np.pi, rotrange_x / 360 * 2. * np.pi), 
            (1/(1+scalingrange), 1+scalingrange))

        half_LabelSize_primary[0] = int(new_patch_size_primary[0]) // 2
        half_LabelSize_primary[1] = int(new_patch_size_primary[1]) // 2
        half_LabelSize_primary[2] = int(new_patch_size_primary[2]) // 2
        # half_LabelSize_primary[0] = int(LabelSize_primary[0] * 1.45) // 2
        # half_LabelSize_primary[1] = int(LabelSize_primary[1] * 1.45) // 2
        # half_LabelSize_primary[2] = int(LabelSize_primary[2] * 1.45) // 2
        lbls_of_sample_primary = Getimagepatchwithcoord(Labelenlarge[np.newaxis, ...], half_LabelSize_primary,
                                                        samplekernal_primary, coord_center[0],
                                                        coord_center[1], coord_center[2])
        lbls_of_sample_primary = lbls_of_sample_primary.squeeze()

        # context, fetch a larger context
        samplekernal_sub2 = 1
        half_ImgsegmentSize_sub2 = LabelsegmentSize.copy()
        half_ImgsegmentSize_sub2[0] = half_LabelSize_primary[0]
        half_ImgsegmentSize_sub2[1] = half_LabelSize_primary[1]
        half_ImgsegmentSize_sub2[2] = half_LabelSize_primary[2]
        channs_of_sample_sub2 = Getimagepatchwithcoord(Imgenlarge, half_ImgsegmentSize_sub2,
                                                       samplekernal_primary, coord_center[0],
                                                       coord_center[1], coord_center[2])

        (channels_augment,
         gt_lbl_img_augment, _) = augment_imgs_of_case(channs_of_sample_sub2,
                                                               lbls_of_sample_primary,
                                                               None, None, prmssel, LabelSize_primary, rng)

        coord_center[0] = channels_augment.shape[1] // 2
        coord_center[1] = channels_augment.shape[2] // 2
        coord_center[2] = channels_augment.shape[3] // 2
        ## I have finished image level augmentation.
        ## then I want extract samples from the center (primary subway)
        ImagetoSample = channels_augment.copy()
        LbltoSample = gt_lbl_img_augment.copy()
    else:
        ImagetoSample = Imgenlarge.copy()
        LbltoSample = Labelenlarge.copy()

    # note that for cardiac segmentation, with anisotropy patchsize 8 * 128 * 128, we do not want rot90 in xy or yz.
    if LabelsegmentSize[0] != LabelsegmentSize[1] and (Augindex == 35 or Augindex == 36 or Augindex == 38 or Augindex == 39):
        Augindex = 0

    '''
    Here, I want to make sure it is compatable with anisotropy patches, such as 160*160*80
    '''
    if Augindex == 32:
        # rot90Frontal_x1
        ImagetoSample = np.rot90(ImagetoSample, 1, (2, 3))
        LbltoSample = np.rot90(LbltoSample, 1, (1, 2))
        coord_tmp = coord_center[1]
        coord_center[1] = ImagetoSample.shape[2] - coord_center[2]
        coord_center[2] = coord_tmp
    
    if Augindex == 33:
        ImagetoSample = np.rot90(ImagetoSample, 3, (2, 3))
        LbltoSample = np.rot90(LbltoSample, 3, (1, 2))
        coord_tmp = coord_center[1]
        coord_center[1] = coord_center[2]
        coord_center[2] = ImagetoSample.shape[3] - coord_tmp

    if Augindex == 35:
        # rot90Sagittal_y1
        ImagetoSample = np.rot90(ImagetoSample, 3, (1, 3))
        LbltoSample = np.rot90(LbltoSample, 3, (0, 2))
        coord_tmp = coord_center[0]
        coord_center[0] = coord_center[2]
        coord_center[2] = ImagetoSample.shape[3] - coord_tmp
        
    if Augindex == 36:
        # rot 180
        ImagetoSample = np.rot90(ImagetoSample, 1, (1, 3))
        LbltoSample = np.rot90(LbltoSample, 1, (0, 2))
        coord_tmp = coord_center[0]
        coord_center[0] = ImagetoSample.shape[1] - coord_center[2]
        coord_center[2] = coord_tmp

    ## here I extract the samples
    ## primary pathway
    half_LabelSize_primary = LabelsegmentSize.copy()
    half_LabelSize_primary[0] = half_LabelSize_primary[0] // 2
    half_LabelSize_primary[1] = half_LabelSize_primary[1] // 2
    half_LabelSize_primary[2] = half_LabelSize_primary[2] // 2
    channs_of_sample_per_path_normal = Getimagepatchwithcoord(ImagetoSample, half_LabelSize_primary, 1, coord_center[0], coord_center[1], coord_center[2])
    channs_of_sample_per_path_normal = channs_of_sample_per_path_normal[:, :LabelsegmentSize[0], :LabelsegmentSize[1], :LabelsegmentSize[2]]
    channs_of_sample_per_path.append(channs_of_sample_per_path_normal)

    ## Label
    lbls_predicted_part_of_sample = Getimagepatchwithcoord(LbltoSample[np.newaxis, ...], half_LabelSize_primary, 1, coord_center[0], coord_center[1], coord_center[2])
    lbls_predicted_part_of_sample = lbls_predicted_part_of_sample[:, :LabelsegmentSize[0], :LabelsegmentSize[1], :LabelsegmentSize[2]]
    lbls_predicted_part_of_sample = lbls_predicted_part_of_sample.squeeze()

    if Augindex >= 29 and Augindex != 32 and Augindex != 33 and Augindex != 35 and Augindex != 36:  # segment level

        Imgenlargeref = Imgenlarge[:, ::4, ::4, ::4]

        (channs_of_sample_per_path,
         lbls_predicted_part_of_sample) = augment_sample(channs_of_sample_per_path,
                                                         lbls_predicted_part_of_sample, prmssel, Imgenlargeref, rng, proof)


    return channs_of_sample_per_path, lbls_predicted_part_of_sample

def Getimagepatchwithcoord(Imgenlarge, half_ImgsegmentSize_sub1, samplekernal, xcentercoordinate, ycentercoordinate, zcentercoordinate):
    xleftlist = np.arange(xcentercoordinate, -1, -samplekernal)
    if len(xleftlist) > half_ImgsegmentSize_sub1[0]:
        xleftlist = xleftlist[1:half_ImgsegmentSize_sub1[0] + 1]
    else:
        xleftlist = xleftlist[1:]
    xrightlist = np.arange(xcentercoordinate, Imgenlarge.shape[1], samplekernal)
    if len(xrightlist) > half_ImgsegmentSize_sub1[0]:
        xrightlist = xrightlist[1:half_ImgsegmentSize_sub1[0] + 1]
    else:
        xrightlist = xrightlist[1:]
    xcoordinatelist = np.concatenate([xleftlist[::-1], [xcentercoordinate], xrightlist])
    xleftpadding = half_ImgsegmentSize_sub1[0] - len(xleftlist)
    xrightpadding = half_ImgsegmentSize_sub1[0] - len(xrightlist)
    # for y direction
    yleftlist = np.arange(ycentercoordinate, -1, -samplekernal)
    if len(yleftlist) > half_ImgsegmentSize_sub1[1]:
        yleftlist = yleftlist[1:half_ImgsegmentSize_sub1[1] + 1]
    else:
        yleftlist = yleftlist[1:]
    yrightlist = np.arange(ycentercoordinate, Imgenlarge.shape[2], samplekernal)
    if len(yrightlist) > half_ImgsegmentSize_sub1[1]:
        yrightlist = yrightlist[1:half_ImgsegmentSize_sub1[1] + 1]
    else:
        yrightlist = yrightlist[1:]
    ycoordinatelist = np.concatenate([yleftlist[::-1], [ycentercoordinate], yrightlist])
    yleftpadding = half_ImgsegmentSize_sub1[1] - len(yleftlist)
    yrightpadding = half_ImgsegmentSize_sub1[1] - len(yrightlist)
    # for z direction
    zleftlist = np.arange(zcentercoordinate, -1, -samplekernal)
    if len(zleftlist) > half_ImgsegmentSize_sub1[2]:
        zleftlist = zleftlist[1:half_ImgsegmentSize_sub1[2] + 1]
    else:
        zleftlist = zleftlist[1:]
    zrightlist = np.arange(zcentercoordinate,Imgenlarge.shape[3], samplekernal)
    if len(zrightlist) > half_ImgsegmentSize_sub1[2]:
        zrightlist = zrightlist[1:half_ImgsegmentSize_sub1[2] + 1]
    else:
        zrightlist = zrightlist[1:]
    zcoordinatelist = np.concatenate([zleftlist[::-1], [zcentercoordinate], zrightlist])
    zleftpadding = half_ImgsegmentSize_sub1[2] - len(zleftlist)
    zrightpadding = half_ImgsegmentSize_sub1[2] - len(zrightlist)

    channs_of_sample_per_path = Imgenlarge[:, np.min(xcoordinatelist):np.max(xcoordinatelist) + 1:samplekernal,
                                     np.min(ycoordinatelist):np.max(ycoordinatelist) + 1:samplekernal,
                                     np.min(zcoordinatelist):np.max(zcoordinatelist) + 1:samplekernal]
    # pad x
    channs_of_sample_per_path = np.concatenate((np.zeros(
        (channs_of_sample_per_path.shape[0], np.int(xleftpadding), channs_of_sample_per_path.shape[2],
         channs_of_sample_per_path.shape[3])), channs_of_sample_per_path,
                                                np.zeros((channs_of_sample_per_path.shape[0],
                                                          np.int(xrightpadding),
                                                          channs_of_sample_per_path.shape[2],
                                                          channs_of_sample_per_path.shape[3]))),
        axis=1)
    # pad y
    channs_of_sample_per_path = np.concatenate((np.zeros(
        (channs_of_sample_per_path.shape[0], channs_of_sample_per_path.shape[1], np.int(yleftpadding),
         channs_of_sample_per_path.shape[3])), channs_of_sample_per_path,
                                                np.zeros((channs_of_sample_per_path.shape[0],
                                                          channs_of_sample_per_path.shape[1],
                                                          np.int(yrightpadding),
                                                          channs_of_sample_per_path.shape[3]))),
        axis=2)
    # pad z
    channs_of_sample_per_path = np.concatenate((np.zeros(
        (channs_of_sample_per_path.shape[0], channs_of_sample_per_path.shape[1],
         channs_of_sample_per_path.shape[2], np.int(zleftpadding))), channs_of_sample_per_path,
                                                np.zeros((channs_of_sample_per_path.shape[0],
                                                          channs_of_sample_per_path.shape[1],
                                                          channs_of_sample_per_path.shape[2],
                                                          np.int(zrightpadding)))), axis=3)
    return channs_of_sample_per_path

def calc_border_int_of_3d_img(img_3d):
    border_int = np.mean([img_3d[0, 0, 0],
                          img_3d[-1, 0, 0],
                          img_3d[0, -1, 0],
                          img_3d[-1, -1, 0],
                          img_3d[0, 0, -1],
                          img_3d[-1, 0, -1],
                          img_3d[0, -1, -1],
                          img_3d[-1, -1, -1]
                          ])
    return border_int

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = np.eye(num_classes)  # [D,D]

    return y[labels]  # [N,D]
