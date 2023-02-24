import numpy as np
import nibabel as nib
import math
from itertools import repeat
import multiprocessing as mp
from transformations.augmentSamplerng import augment_sample
from transformations.augmentImagerng import augment_imgs_of_case, AugmenterAffineParams
from utilities import get_patch_size

# np.random.seed(12345)
# random.seed(12345)
## take image as 37*37*37, and the target as 21*21*21

def getbatchkitsatlas(DatafileFold, batch_size, iteration, selind, maximumcase, offset, logging, proof = 0, ImgsegmentSize = 37, caselist = None):
    augm_img_prms_tr,augm_sample_prms_tr  = get_augment_par()
    augms_prms = {**augm_img_prms_tr, **augm_sample_prms_tr}

    # LabelsegmentSize = 21
    # ImgsegmentSize = LabelsegmentSize + 2 * offset

    LabelsegmentSize = ImgsegmentSize - 2 * offset

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

    samplingpool = np.minimum(int(maximumcase), len(Imgreadc1))
    samplingpool = np.minimum(samplingpool, samplenum)
    samplefromeachcase = np.floor(samplenum / samplingpool)
    residualcase = np.int(samplenum - samplefromeachcase * samplingpool)

    if iteration > 100: # try not to print for meta validation...
        logging.info('Sampling training segments from ' + str(samplingpool) + ' cases')
        logging.info('Total patches are ' + str(samplenum))
    # else:
    #     logging.info('Augmentation selections are ' + str(selind))

    samplefromthiscase = np.ones((samplingpool)) * samplefromeachcase
    for numiter in range(residualcase):
        samplefromthiscase[numiter] = samplefromthiscase[numiter] + 1

    kimg = list(range(samplingpool))
    kstartpos = np.zeros(samplingpool)
    for kpos in kimg:
        for kiters in range(kpos):
            kstartpos[kpos] += samplefromthiscase[kiters]
    
    if caselist is None:
        caselist = np.random.randint(0, len(Imgreadc1), samplingpool)

    logging.info('Sampling caselist for training: ' + str(caselist))
    seednum = np.random.randint(0, 1e6)

    try:
        mp_pool = mp.Pool(processes=np.minimum(8, len(kimg)))
        with mp_pool as pool:
            results = pool.starmap(getsampleskitsatlas,
                                   zip(kimg, kstartpos[kimg], samplefromthiscase[kimg], repeat(Imgreadc1),
                                       repeat(Labelread), repeat(Maskread), repeat(augms_prms), repeat(offset)
                                       , repeat(ImgsegmentSize), repeat(LabelsegmentSize), repeat(selind),repeat(proof), caselist[kimg], repeat(seednum)))

        # jobs = collections.OrderedDict()
        # for job_idx in kimg:
        #     jobs[job_idx] = mp_pool.apply_async(getsampleskitsatlas, (job_idx, samplefromeachcase, samplefromthiscase[job_idx], Imgreadc1
        #                                                           , Labelread, Maskread, augms_prms, offset
        #                                                           , ImgsegmentSize, LabelsegmentSize, selind, proof))
        # batchxnor = []
        # batchxp1 = []
        # batchxp2 = []
        # batchy = []
        # augmentselind = []
        # for job_idx in kimg:
        #     results = jobs[job_idx].get()
        #     batchxnor.append(results[0])
        #     batchxp1.append(results[1])
        #     batchxp2.append(results[2])
        #     batchy.append(results[3])
        #     augmentselind.append(results[5])


    except mp.TimeoutError:
        print("time out?")
    except:  # Catches everything, even a sys.exit(1) exception.
        mp_pool.terminate()
        mp_pool.join()
        raise Exception("Unexpected error.")
    else:  # Nothing went wrong
        # Needed in case any processes are hanging. mp_pool.close() does not solve this.
        mp_pool.terminate()
        mp_pool.join()
    # to be "safe", otherwise it can raise some OMP warnings


    batchxnor = []
    batchxp1 = []
    batchxp2 = []
    batchy = []
    clsselind = []
    for knum in range(samplingpool):
        batchxnor.append(results[knum][0])
        batchxp1.append(results[knum][1])
        batchxp2.append(results[knum][2])
        batchy.append(results[knum][3])
        clsselind.append(results[knum][5])

    batchxnor = np.vstack(batchxnor)
    batchxp1 = np.vstack(batchxp1)
    batchxp2 = np.vstack(batchxp2)
    batchy = np.vstack(batchy)
    clsselind = np.vstack(clsselind)
    listr = list(range(samplenum))
    np.random.shuffle(listr)
    batchxnor = batchxnor[listr, :, :, :, :]
    batchxp1 = batchxp1[listr, :, :, :, :]
    batchxp2 = batchxp2[listr, :, :, :, :]
    batchy = batchy[listr, :, :, :].astype('int')
    clsselind = clsselind[listr, :]

    # batchy = one_hot_embedding(batchy, 4)

    return batchxnor, batchxp1, batchxp2, batchy, listr, clsselind

def getsampleskitsatlas(kimg, kstartpos, samplefromthiscase, Imgreadc1, Labelread, Maskread,
                    augms_prms, offset, ImgsegmentSize, LabelsegmentSize, selind, proof, numr, seednum):

    np.random.seed(seednum + kimg)
    # if kimg == 0:
    #     print('sampling process ' + str(kimg) + ' from case ' + str(numr))
    #     print(selind[:, 0])

    # numr = kimg

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

    batchxnor, batchxsub1, batchxsub2, batchy, clslist = getsamples(channels, gt_lbl_img, roi_mask, kimg, kstartpos, samplefromthiscase,
                    augms_prms, offset, ImgsegmentSize, LabelsegmentSize, selind, proof)

    return batchxnor, batchxsub1, batchxsub2, batchy, numr, clslist

def getsamples(channels, gt_lbl_img, roi_mask, kimg, kstartpos, samplefromthiscase,
                    augms_prms, offset, ImgsegmentSize, LabelsegmentSize, selind, proof, bratsflag = False):

    # local_state = np.random.RandomState()

    range_x, range_y, range_z = roi_mask.shape

    Maskenlarge = np.zeros((range_x + ImgsegmentSize, range_y + ImgsegmentSize, range_z + ImgsegmentSize))
    Labelenlarge = np.zeros((range_x + ImgsegmentSize, range_y + ImgsegmentSize, range_z + ImgsegmentSize))
    if bratsflag == True:
        Imgenlarge = np.zeros((4, range_x + ImgsegmentSize, range_y + ImgsegmentSize, range_z + ImgsegmentSize))
    else :
        Imgenlarge = np.zeros((1, range_x + ImgsegmentSize, range_y + ImgsegmentSize, range_z + ImgsegmentSize))
    Maskenlarge[ImgsegmentSize // 2:range_x + ImgsegmentSize // 2, ImgsegmentSize // 2:range_y + ImgsegmentSize // 2,
    ImgsegmentSize // 2:range_z + ImgsegmentSize // 2] = roi_mask
    Labelenlarge[ImgsegmentSize // 2:range_x + ImgsegmentSize // 2, ImgsegmentSize // 2:range_y + ImgsegmentSize // 2,
    ImgsegmentSize // 2:range_z + ImgsegmentSize // 2] = gt_lbl_img
    Imgenlarge[:, ImgsegmentSize // 2:range_x + ImgsegmentSize // 2, ImgsegmentSize // 2:range_y + ImgsegmentSize // 2,
    ImgsegmentSize // 2:range_z + ImgsegmentSize // 2] = channels

    if bratsflag == True:
        batchxnor = np.zeros((int(samplefromthiscase), 4, ImgsegmentSize, ImgsegmentSize, ImgsegmentSize))
        batchxsub1 = np.zeros((int(samplefromthiscase), 4, int(np.ceil(LabelsegmentSize / 3) + 2 * offset),
                               int(np.ceil(LabelsegmentSize / 3) +  2 * offset), int(np.ceil(LabelsegmentSize / 3) +  2 * offset)))
        batchxsub2 = np.zeros((int(samplefromthiscase), 4, int(np.ceil(LabelsegmentSize / 5) +  2 * offset),
                               int(np.ceil(LabelsegmentSize / 5) +  2 * offset), int(np.ceil(LabelsegmentSize / 5) +  2 * offset)))
    else :
        batchxnor = np.zeros((int(samplefromthiscase), 1, ImgsegmentSize, ImgsegmentSize, ImgsegmentSize))
        batchxsub1 = np.zeros((int(samplefromthiscase), 1, int(np.ceil(LabelsegmentSize / 3) +  2 * offset),
                               int(np.ceil(LabelsegmentSize / 3) +  2 * offset), int(np.ceil(LabelsegmentSize / 3) +  2 * offset)))
        batchxsub2 = np.zeros((int(samplefromthiscase), 1, int(np.ceil(LabelsegmentSize / 5) +  2 * offset),
                               int(np.ceil(LabelsegmentSize / 5) +  2 * offset), int(np.ceil(LabelsegmentSize / 5) +  2 * offset)))
    batchy = np.zeros((int(samplefromthiscase), LabelsegmentSize, LabelsegmentSize, LabelsegmentSize))
    clslist = np.zeros((int(samplefromthiscase), 1))

    # cls wise sampling would lead to werid results..
    # just use FG / BG sampling
    numofcls = np.int(np.max(gt_lbl_img)) + 1
    # print(samplefromthiscase)
    # print(numofcls)

    if samplefromthiscase >= numofcls:  # I have enough samples for different cls inside one image, it is for normal training, sample 1000patches or more.

        for kcls in range(0, numofcls):

            # cls wise sampling...
            gt_lbl_img_inside_mask = gt_lbl_img * roi_mask
            if kcls == 0:
                bgcls_mask = (gt_lbl_img_inside_mask == 0) * roi_mask
                Labelindex = bgcls_mask.nonzero()
            else:
                Labelindex = np.where(gt_lbl_img_inside_mask == kcls)
            Labelindex_x = Labelindex[0]
            Labelindex_y = Labelindex[1]
            Labelindex_z = Labelindex[2]
            if len(Labelindex_x) == 0:
                ## unfortunately, it does not have a single pixel, just sample some foreground samples.
                Labelindex_x, Labelindex_y, Labelindex_z = gt_lbl_img_inside_mask.nonzero()
                if len(Labelindex_x) == 0:
                    ## it might have 0 labels because the augmentation...
                    Labelindex_x, Labelindex_y, Labelindex_z = roi_mask.nonzero()

            startpos = int(samplefromthiscase // (numofcls)) * kcls
            endpos = int(samplefromthiscase // (numofcls)) * (kcls + 1)

            for k in range(startpos, endpos):  # sampling from different classes
                # print(Labelindex_x.size)
                numindex = np.random.randint(0, Labelindex_x.size)
                # numindex = np.minimum(k, Labelindex_x.size - 1)
                # this line would lead to error when multi process of cas xxxx816/817 with augmentation
                # ...np.random.randint(0, Labelindex_x.size-1)
                # I dont need -1, anyway. remove -1, it would be fine?
                # maybe the operation -1 could lead to something werid.
                selindex_x = Labelindex_x[numindex]
                selindex_y = Labelindex_y[numindex]
                selindex_z = Labelindex_z[numindex]

                channs_of_sample_per_path, lbls_predicted_part_of_sample = getaugmentpath(Imgenlarge, Labelenlarge,
                                                                                          kimg, k,
                                                                                          selindex_x, selindex_y,
                                                                                          selindex_z, ImgsegmentSize,
                                                                                          LabelsegmentSize,
                                                                                          kstartpos, selind,
                                                                                          offset, augms_prms, kcls, proof)

                batchxnor[int(k), :, :, :, :] = channs_of_sample_per_path[0]
                batchxsub1[int(k), :, :, :, :] = channs_of_sample_per_path[1]
                batchxsub2[int(k), :, :, :, :] = channs_of_sample_per_path[2]
                batchy[int(k), :, :, :] = lbls_predicted_part_of_sample
                # numlist[int(k)] = selind[0, np.int(kstartpos + k)]
                clslist[int(k), 0] = kcls
         # if there are any residual cases.
        if endpos < int(samplefromthiscase):
            bgcls_mask = (gt_lbl_img_inside_mask == 0) * roi_mask

            for k in range(endpos, int(samplefromthiscase)):  #
                kcls = np.random.randint(0, numofcls)

                if kcls == 0:
                    Labelindex = bgcls_mask.nonzero()
                else:
                    Labelindex = np.where(gt_lbl_img_inside_mask == kcls)  ## it would take much time.
                    # the process is fundementally slow? for example, gt_lbl == 1 would also take long

                Labelindex_x = Labelindex[0]
                Labelindex_y = Labelindex[1]
                Labelindex_z = Labelindex[2]
                if len(Labelindex_x) == 0:
                    ## unfortunately, it does not have a single pixel, just sample some foreground samples.
                    Labelindex_x, Labelindex_y, Labelindex_z = gt_lbl_img_inside_mask.nonzero()
                    if len(Labelindex_x) == 0:
                        ## it might have 0 labels because the augmentation...
                        Labelindex_x, Labelindex_y, Labelindex_z = roi_mask.nonzero()


                # print(Labelindex_x.size)
                numindex = np.random.randint(0, len(Labelindex_x))
                selindex_x = Labelindex_x[numindex]
                selindex_y = Labelindex_y[numindex]
                selindex_z = Labelindex_z[numindex]

                channs_of_sample_per_path, lbls_predicted_part_of_sample = getaugmentpath(Imgenlarge, Labelenlarge, kimg, k,
                                                                                            selindex_x, selindex_y,
                                                                                            selindex_z, ImgsegmentSize,
                                                                                            LabelsegmentSize,
                                                                                            kstartpos, selind,
                                                                                            offset, augms_prms, kcls, proof)

                batchxnor[int(k), :, :, :, :] = channs_of_sample_per_path[0]
                batchxsub1[int(k), :, :, :, :] = channs_of_sample_per_path[1]
                batchxsub2[int(k), :, :, :, :] = channs_of_sample_per_path[2]
                batchy[int(k), :, :, :] = lbls_predicted_part_of_sample
                # numlist[int(k)] = selind[0, np.int(kstartpos + k)]
                clslist[int(k), 0] = kcls

    else:  # I do not have enough samples inside this image, sample randomly for each caes. it can be slow for sampling a lot
        # it is for the meta-training sampling.

        # cls wise sampling...
        gt_lbl_img_inside_mask = gt_lbl_img * roi_mask
        bgcls_mask = (gt_lbl_img_inside_mask == 0) * roi_mask

        for k in range(np.int(samplefromthiscase)):  #
            kcls = np.random.randint(0, numofcls)

            if kcls == 0:
                Labelindex = bgcls_mask.nonzero()
            else:
                Labelindex = np.where(gt_lbl_img_inside_mask == kcls)  ## it would take much time.
                # the process is fundementally slow? for example, gt_lbl == 1 would also take long

            Labelindex_x = Labelindex[0]
            Labelindex_y = Labelindex[1]
            Labelindex_z = Labelindex[2]
            if len(Labelindex_x) == 0:
                ## unfortunately, it does not have a single pixel, just sample some foreground samples.
                Labelindex_x, Labelindex_y, Labelindex_z = gt_lbl_img_inside_mask.nonzero()
                if len(Labelindex_x) == 0:
                    ## it might have 0 labels because the augmentation...
                    Labelindex_x, Labelindex_y, Labelindex_z = roi_mask.nonzero()


            # print(Labelindex_x.size)
            numindex = np.random.randint(0, len(Labelindex_x))
            selindex_x = Labelindex_x[numindex]
            selindex_y = Labelindex_y[numindex]
            selindex_z = Labelindex_z[numindex]

            channs_of_sample_per_path, lbls_predicted_part_of_sample = getaugmentpath(Imgenlarge, Labelenlarge, kimg, k,
                                                                                      selindex_x, selindex_y,
                                                                                      selindex_z, ImgsegmentSize,
                                                                                      LabelsegmentSize,
                                                                                      kstartpos, selind,
                                                                                      offset, augms_prms, kcls, proof)

            batchxnor[int(k), :, :, :, :] = channs_of_sample_per_path[0]
            batchxsub1[int(k), :, :, :, :] = channs_of_sample_per_path[1]
            batchxsub2[int(k), :, :, :, :] = channs_of_sample_per_path[2]
            batchy[int(k), :, :, :] = lbls_predicted_part_of_sample
            # numlist[int(k)] = selind[0, np.int(kstartpos + k)]
            clslist[int(k), 0] = kcls

    return batchxnor, batchxsub1, batchxsub2, batchy, clslist

def getaugmentpath(Imgenlarge, Labelenlarge, kimg, k, selindex_x, selindex_y, selindex_z, ImgsegmentSize,
                   LabelsegmentSize, kstartpos, selind, offset, augms_prms, kcls, proof):
    channs_of_sample_per_path = []

    # try to make it exactly like DM's implementation
    coord_center = np.zeros(3, dtype=int)
    coord_center[0] = int(selindex_x + ImgsegmentSize // 2)
    coord_center[1] = int(selindex_y + ImgsegmentSize // 2)
    coord_center[2] = int(selindex_z + ImgsegmentSize // 2)
    # to accelarate the process if no augmentation is needed.
    # I should get 10 policies here.
    # and I combine them.
    orgimgflag, orgsampleflag, augimg, augsample = get_all_prms(selind, augms_prms, np.int(kstartpos + k), kcls)

    if orgimgflag is False:  # image level. totally 15.
        # cut a large img patch.

        samplekernal_primary = 1
        LabelSize_primary = LabelsegmentSize

        rotrange_z = augimg['rot_xyz'][2] + 5
        rotrange_y = augimg['rot_xyz'][1] + 5
        rotrange_x = augimg['rot_xyz'][0] + 5
        scalingrange = augimg['scaling'] + 0.05

        new_patch_size_primary = get_patch_size([LabelSize_primary, LabelSize_primary, LabelSize_primary], 
            (-rotrange_z / 360 * 2. * np.pi, rotrange_z / 360 * 2. * np.pi), 
            (-rotrange_y / 360 * 2. * np.pi, rotrange_y / 360 * 2. * np.pi),
            (-rotrange_x / 360 * 2. * np.pi, rotrange_x / 360 * 2. * np.pi), 
            (1/(1+scalingrange), 1+scalingrange))
        half_LabelSize_primary = np.max(new_patch_size_primary) // 2
        lbls_of_sample_primary = Getimagepatchwithcoord(Labelenlarge[np.newaxis, ...], half_LabelSize_primary,
                                                        samplekernal_primary, coord_center[0],
                                                        coord_center[1], coord_center[2])
        lbls_of_sample_primary = lbls_of_sample_primary.squeeze()

        # context, fetch a larger context
        samplekernal_sub2 = 5
        ImgsegmentSize_sub2origin = int(LabelsegmentSize + (2 * offset + 1) * samplekernal_sub2)
        new_patch_size_primary = get_patch_size([ImgsegmentSize_sub2origin, ImgsegmentSize_sub2origin, ImgsegmentSize_sub2origin], 
            (-rotrange_z / 360 * 2. * np.pi, rotrange_z / 360 * 2. * np.pi), 
            (-rotrange_y / 360 * 2. * np.pi, rotrange_y / 360 * 2. * np.pi),
            (-rotrange_x / 360 * 2. * np.pi, rotrange_x / 360 * 2. * np.pi), 
            (1/(1+scalingrange), 1+scalingrange))
        half_ImgsegmentSize_sub2 = np.max(new_patch_size_primary) // 2
        channs_of_sample_sub2 = Getimagepatchwithcoord(Imgenlarge, half_ImgsegmentSize_sub2,
                                                       samplekernal_primary, coord_center[0],
                                                       coord_center[1], coord_center[2])

        # make it easy to do sampling (odd.)
        patch_size = [ImgsegmentSize_sub2origin // 2 * 2 + 1, ImgsegmentSize_sub2origin // 2 * 2 + 1, ImgsegmentSize_sub2origin // 2 * 2 + 1]

        (channels_augment,
         gt_lbl_img_augment,_) = augment_imgs_of_case(channs_of_sample_sub2,
                                                               lbls_of_sample_primary,
                                                               None, None, augimg, patch_size)

        coord_center[0] = channels_augment.shape[1] // 2
        coord_center[1] = channels_augment.shape[2] // 2
        coord_center[2] = channels_augment.shape[3] // 2
        gt_lbl_img_augment_full = np.zeros(
            (channels_augment.shape[1], channels_augment.shape[2], channels_augment.shape[3]))
        gt_lbl_img_augment_full[
        coord_center[0] - LabelSize_primary // 2:coord_center[0] + LabelSize_primary // 2 + 1,
        coord_center[1] - LabelSize_primary // 2:coord_center[1] + LabelSize_primary // 2 + 1,
        coord_center[2] - LabelSize_primary // 2:coord_center[2] + LabelSize_primary // 2 + 1] = gt_lbl_img_augment[gt_lbl_img_augment.shape[0] // 2 - LabelSize_primary // 2:
                                                                                                                    gt_lbl_img_augment.shape[0] // 2 + LabelSize_primary // 2 + 1,
                                                                                                 gt_lbl_img_augment.shape[
                                                                                                     1] // 2 - LabelSize_primary // 2:gt_lbl_img_augment.shape[
                                                                                                                                          1] // 2 + LabelSize_primary // 2 + 1,
                                                                                                 gt_lbl_img_augment.shape[
                                                                                                     2] // 2 - LabelSize_primary // 2:gt_lbl_img_augment.shape[
                                                                                                                                          2] // 2 + LabelSize_primary // 2 + 1]
        ## I have finished image level augmentation.
        ## then I want extract samples from the center (primary subway)
        ImagetoSample = channels_augment.copy()
        LbltoSample = gt_lbl_img_augment_full.copy()
    else:
        ImagetoSample = Imgenlarge.copy()
        LbltoSample = Labelenlarge.copy()

    ## here I extract the samples
    ## primary pathway
    samplekernal_primary = 1
    channs_of_sample_per_path_normal = Getimagepatchwithcoord(ImagetoSample, ImgsegmentSize // 2, samplekernal_primary, coord_center[0], coord_center[1], coord_center[2])
    channs_of_sample_per_path.append(channs_of_sample_per_path_normal)

    ## get the boundaries for following up sampling.
    leftBoundaryRcz = [coord_center[0] - samplekernal_primary * (ImgsegmentSize - 1) // 2,
                       coord_center[1] - samplekernal_primary * (ImgsegmentSize - 1) // 2,
                       coord_center[2] - samplekernal_primary * (ImgsegmentSize - 1) // 2]
    rightBoundaryRcz = [leftBoundaryRcz[0] + samplekernal_primary * ImgsegmentSize,
                        leftBoundaryRcz[1] + samplekernal_primary * ImgsegmentSize,
                        leftBoundaryRcz[2] + samplekernal_primary * ImgsegmentSize]

    ## first subpathway
    # create some parameters..
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
        subsampledImageChannels=ImagetoSample,
        image_part_slices_coords=slicesCoordsOfSegmForPrimaryPathway,
        subSamplingFactor=subFactor,
        subsampledImagePartDimensions=subsampledImagePartDimensions
    )
    channs_of_sample_per_path.append(channs_of_sample_per_path_sub1)

    ## second subpathway

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
        subsampledImageChannels=ImagetoSample,
        image_part_slices_coords=slicesCoordsOfSegmForPrimaryPathway,
        subSamplingFactor=subFactor,
        subsampledImagePartDimensions=subsampledImagePartDimensions
    )
    channs_of_sample_per_path.append(channs_of_sample_per_path_sub2)

    ## Label
    lbls_predicted_part_of_sample = LbltoSample[
                                    coord_center[0] - LabelsegmentSize // 2: coord_center[0] + LabelsegmentSize // 2 + 1,
                                    coord_center[1] - LabelsegmentSize // 2: coord_center[1] + LabelsegmentSize // 2 + 1,
                                    coord_center[2] - LabelsegmentSize // 2: coord_center[2] + LabelsegmentSize // 2 + 1]

    if orgsampleflag is False:  # segment level

        ## ok, I need to undersample Imgenlarge for gamma correction.
        ## /4 in three axis, it should be fine.
        ## note that I should use a hinge for the minms.

        Imgenlargeref = Imgenlarge[:, ::4, ::4, ::4]

        (channs_of_sample_per_path,
         lbls_predicted_part_of_sample) = augment_sample(channs_of_sample_per_path,
                                                         lbls_predicted_part_of_sample, augsample, Imgenlargeref, None, proof)


    return channs_of_sample_per_path, lbls_predicted_part_of_sample

def Getimagepatchwithcoord(Imgenlarge, half_ImgsegmentSize_sub1, samplekernal, xcentercoordinate, ycentercoordinate, zcentercoordinate):
    xleftlist = np.arange(xcentercoordinate, -1, -samplekernal)
    if len(xleftlist) > half_ImgsegmentSize_sub1:
        xleftlist = xleftlist[1:half_ImgsegmentSize_sub1 + 1]
    else:
        xleftlist = xleftlist[1:]
    xrightlist = np.arange(xcentercoordinate, Imgenlarge.shape[1], samplekernal)
    if len(xrightlist) > half_ImgsegmentSize_sub1:
        xrightlist = xrightlist[1:half_ImgsegmentSize_sub1 + 1]
    else:
        xrightlist = xrightlist[1:]
    xcoordinatelist = np.concatenate([xleftlist[::-1], [xcentercoordinate], xrightlist])
    xleftpadding = half_ImgsegmentSize_sub1 - len(xleftlist)
    xrightpadding = half_ImgsegmentSize_sub1 - len(xrightlist)
    # for y direction
    yleftlist = np.arange(ycentercoordinate, -1, -samplekernal)
    if len(yleftlist) > half_ImgsegmentSize_sub1:
        yleftlist = yleftlist[1:half_ImgsegmentSize_sub1 + 1]
    else:
        yleftlist = yleftlist[1:]
    yrightlist = np.arange(ycentercoordinate, Imgenlarge.shape[2], samplekernal)
    if len(yrightlist) > half_ImgsegmentSize_sub1:
        yrightlist = yrightlist[1:half_ImgsegmentSize_sub1 + 1]
    else:
        yrightlist = yrightlist[1:]
    ycoordinatelist = np.concatenate([yleftlist[::-1], [ycentercoordinate], yrightlist])
    yleftpadding = half_ImgsegmentSize_sub1 - len(yleftlist)
    yrightpadding = half_ImgsegmentSize_sub1 - len(yrightlist)
    # for z direction
    zleftlist = np.arange(zcentercoordinate, -1, -samplekernal)
    if len(zleftlist) > half_ImgsegmentSize_sub1:
        zleftlist = zleftlist[1:half_ImgsegmentSize_sub1 + 1]
    else:
        zleftlist = zleftlist[1:]
    zrightlist = np.arange(zcentercoordinate,Imgenlarge.shape[3], samplekernal)
    if len(zrightlist) > half_ImgsegmentSize_sub1:
        zrightlist = zrightlist[1:half_ImgsegmentSize_sub1 + 1]
    else:
        zrightlist = zrightlist[1:]
    zcoordinatelist = np.concatenate([zleftlist[::-1], [zcentercoordinate], zrightlist])
    zleftpadding = half_ImgsegmentSize_sub1 - len(zleftlist)
    zrightpadding = half_ImgsegmentSize_sub1 - len(zrightlist)

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

def getImagePartFromSubsampledImageForTraining(dimsOfPrimarySegment,
                                               recFieldCnn,
                                               subsampledImageChannels,
                                               image_part_slices_coords,
                                               subSamplingFactor,
                                               subsampledImagePartDimensions
                                               ):
    """
    This returns an image part from the sampled data, given the image_part_slices_coords,
    which has the coordinates where the normal-scale image part starts and ends (inclusive).
    (Actually, in this case, the right (end) part of image_part_slices_coords is not used.)

    The way it works is NOT optimal. From the beginning of the normal-resolution part,
    it goes further to the left 1 receptive-field and then forward xSubsamplingFactor receptive-fields.
    This stops it from being used with arbitrary size of subsampled segment (decoupled by the high-res segment).
    Now, the subsampled patch has to be of the same size as the normal-scale.
    To change this, I should find where THE FIRST TOP LEFT CENTRAL (predicted) VOXEL is,
    and do the back-one-(sub)patch + front-3-(sub)patches from there, not from the begining of the patch.

    Current way it works (correct):
    If I have eg subsample factor=3 and 9 central-pred-voxels, I get 3 "central" voxels/patches for the
    subsampled-part. Straightforward. If I have a number of central voxels that is not an exact multiple of
    the subfactor, eg 10 central-voxels, I get 3+1 central voxels in the subsampled-part.
    When the cnn is convolving them, they will get repeated to 4(last-layer-neurons)*3(factor) = 12,
    and will get sliced down to 10, in order to have same dimension with the 1st pathway.
    """
    subsampledImageDimensions = subsampledImageChannels[0].shape

    subsampledChannelsForThisImagePart = np.ones((len(subsampledImageChannels),
                                                  subsampledImagePartDimensions[0],
                                                  subsampledImagePartDimensions[1],
                                                  subsampledImagePartDimensions[2]),
                                                 dtype='float32')

    numberOfCentralVoxelsClassifiedForEachImagePart_rDim = dimsOfPrimarySegment[0] - recFieldCnn[0] + 1
    numberOfCentralVoxelsClassifiedForEachImagePart_cDim = dimsOfPrimarySegment[1] - recFieldCnn[1] + 1
    numberOfCentralVoxelsClassifiedForEachImagePart_zDim = dimsOfPrimarySegment[2] - recFieldCnn[2] + 1

    # Calculate the slice that I should get, and where I should put it in the imagePart
    # (eg if near the borders, and I cant grab a whole slice-imagePart).
    rSlotsPreviously = ((subSamplingFactor[0] - 1) // 2) * recFieldCnn[0] if subSamplingFactor[0] % 2 == 1 \
        else (subSamplingFactor[0] - 2) // 2 * recFieldCnn[0] + recFieldCnn[0] // 2
    cSlotsPreviously = ((subSamplingFactor[1] - 1) // 2) * recFieldCnn[1] if subSamplingFactor[1] % 2 == 1 \
        else (subSamplingFactor[1] - 2) // 2 * recFieldCnn[1] + recFieldCnn[1] // 2
    zSlotsPreviously = ((subSamplingFactor[2] - 1) // 2) * recFieldCnn[2] if subSamplingFactor[2] % 2 == 1 \
        else (subSamplingFactor[2] - 2) // 2 * recFieldCnn[2] + recFieldCnn[2] // 2
    # 1*17
    rToCentralVoxelOfAnAveragedArea = subSamplingFactor[0] // 2 if subSamplingFactor[0] % 2 == 1 else (
            subSamplingFactor[
                0] // 2 - 1)  # one closer to the beginning of dim. Same happens when I get parts of image.
    cToCentralVoxelOfAnAveragedArea = subSamplingFactor[1] // 2 if subSamplingFactor[1] % 2 == 1 else (
            subSamplingFactor[1] // 2 - 1)
    zToCentralVoxelOfAnAveragedArea = subSamplingFactor[2] // 2 if subSamplingFactor[2] % 2 == 1 else (
            subSamplingFactor[2] // 2 - 1)
    # This is where to start taking voxels from the subsampled image. From the beginning of the imagePart(1 st patch)...
    # ... go forward a few steps to the voxel that is like the "central" in this subsampled (eg 3x3) area.
    # ...Then go backwards -Patchsize to find the first voxel of the subsampled.

    # These indices can run out of image boundaries. I ll correct them afterwards.
    rlow = image_part_slices_coords[0][0] + rToCentralVoxelOfAnAveragedArea - rSlotsPreviously
    # If the patch is 17x17, I want a 17x17 subsampled Patch. BUT if the imgPART is 25x25 (9voxClass),
    # I want 3 subsampledPatches in my subsampPart to cover this area!
    # That is what the last term below is taking care of.
    # CAST TO INT because ceil returns a float, and later on when computing
    # rHighNonInclToPutTheNotPaddedInSubsampledImPart I need to do INTEGER DIVISION.
    rhighNonIncl = int(rlow + subSamplingFactor[0] * recFieldCnn[0] + (
            math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_rDim * 1.0) / subSamplingFactor[0]) - 1) *
                       subSamplingFactor[0])  # excluding index in segment
    clow = image_part_slices_coords[1][0] + cToCentralVoxelOfAnAveragedArea - cSlotsPreviously
    chighNonIncl = int(clow + subSamplingFactor[1] * recFieldCnn[1] + (
            math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_cDim * 1.0) / subSamplingFactor[1]) - 1) *
                       subSamplingFactor[1])
    zlow = image_part_slices_coords[2][0] + zToCentralVoxelOfAnAveragedArea - zSlotsPreviously
    zhighNonIncl = int(zlow + subSamplingFactor[2] * recFieldCnn[2] + (
            math.ceil((numberOfCentralVoxelsClassifiedForEachImagePart_zDim * 1.0) / subSamplingFactor[2]) - 1) *
                       subSamplingFactor[2])

    rlowCorrected = max(rlow, 0)
    clowCorrected = max(clow, 0)
    zlowCorrected = max(zlow, 0)
    rhighNonInclCorrected = min(rhighNonIncl, subsampledImageDimensions[0])
    chighNonInclCorrected = min(chighNonIncl, subsampledImageDimensions[1])
    zhighNonInclCorrected = min(zhighNonIncl, subsampledImageDimensions[2])  # This gave 7

    rLowToPutTheNotPaddedInSubsampledImPart = 0 if rlow >= 0 else abs(rlow) // subSamplingFactor[0]
    cLowToPutTheNotPaddedInSubsampledImPart = 0 if clow >= 0 else abs(clow) // subSamplingFactor[1]
    zLowToPutTheNotPaddedInSubsampledImPart = 0 if zlow >= 0 else abs(zlow) // subSamplingFactor[2]

    dimensionsOfTheSliceOfSubsampledImageNotPadded = [
        int(math.ceil((rhighNonInclCorrected - rlowCorrected) * 1.0 / subSamplingFactor[0])),
        int(math.ceil((chighNonInclCorrected - clowCorrected) * 1.0 / subSamplingFactor[1])),
        int(math.ceil((zhighNonInclCorrected - zlowCorrected) * 1.0 / subSamplingFactor[2]))
    ]

    # I now have exactly where to get the slice from and where to put it in the new array.
    for channel_i in range(len(subsampledImageChannels)):
        intensityZeroOfChannel = calc_border_int_of_3d_img(subsampledImageChannels[channel_i])
        subsampledChannelsForThisImagePart[channel_i] *= intensityZeroOfChannel

        sliceOfSubsampledImageNotPadded = subsampledImageChannels[channel_i][
                                          rlowCorrected: rhighNonInclCorrected: subSamplingFactor[0],
                                          clowCorrected: chighNonInclCorrected: subSamplingFactor[1],
                                          zlowCorrected: zhighNonInclCorrected: subSamplingFactor[2]
                                          ]
        subsampledChannelsForThisImagePart[
        channel_i,
        rLowToPutTheNotPaddedInSubsampledImPart: rLowToPutTheNotPaddedInSubsampledImPart +
                                                 dimensionsOfTheSliceOfSubsampledImageNotPadded[0],
        cLowToPutTheNotPaddedInSubsampledImPart: cLowToPutTheNotPaddedInSubsampledImPart +
                                                 dimensionsOfTheSliceOfSubsampledImageNotPadded[1],
        zLowToPutTheNotPaddedInSubsampledImPart: zLowToPutTheNotPaddedInSubsampledImPart +
                                                 dimensionsOfTheSliceOfSubsampledImageNotPadded[2]
        ] = sliceOfSubsampledImageNotPadded

    # placeholderReturn = np.ones([3,19,19,19], dtype="float32") #channel, dims
    return subsampledChannelsForThisImagePart

def get_all_prms(selind, augms_prms, knum, kcls):
    if type(selind) is list:
        # choose the right index
        if kcls == 0:
            selind = selind[kcls]
        else:
            selind = selind[1]

    augimg = []
    augsample = []
    Augindex_scale = selind[0, knum]
    # I need to be careful about these three.
    Augindex_rotF = (selind[1, knum] + 5) * np.float(selind[1, knum] != 0)
    if Augindex_rotF == 9:
        Augindex_rotF = 18
    Augindex_rotS = (selind[2, knum] + 9) * np.float(selind[2, knum] != 0)
    if Augindex_rotS == 10:
        Augindex_rotS = 9
    if Augindex_rotS == 11:
        Augindex_rotS = 10
    if Augindex_rotS == 12:
        Augindex_rotS = 11
    if Augindex_rotS == 13:
        Augindex_rotS = 20
    Augindex_rotL = (selind[3, knum] + 13) * np.float(selind[3, knum] != 0)
    if Augindex_rotL == 14:
        Augindex_rotL = 12
    if Augindex_rotL == 15:
        Augindex_rotL = 13
    if Augindex_rotL == 16:
        Augindex_rotL = 14
    if Augindex_rotL == 17:
        Augindex_rotL = 22
    Augindex_mirrorS = (selind[4, knum] + 20) * np.float(selind[4, knum] != 0)
    if Augindex_mirrorS == 21:
        Augindex_mirrorS = 15
    Augindex_mirrorF = (selind[5, knum] + 21) * np.float(selind[5, knum] != 0)
    if Augindex_mirrorF == 22:
        Augindex_mirrorF = 16
    Augindex_mirrorA = (selind[6, knum] + 22) * np.float(selind[6, knum] != 0)
    if Augindex_mirrorA == 23:
        Augindex_mirrorA = 17

    Augindex_gamma = (selind[7, knum] + 23) * np.float(selind[7, knum] != 0)
    Augindex_invgamma = (selind[8, knum] + 26) * np.float(selind[8, knum] != 0)
    Augindex_badd = (selind[9, knum] + 29) * np.float(selind[9, knum] != 0)
    Augindex_bmul = (selind[10, knum] + 32) * np.float(selind[10, knum] != 0)
    Augindex_contrast = (selind[11, knum] + 35) * np.float(selind[11, knum] != 0)
    
    Augindex_sharpen = (selind[12, knum] + 38) * np.float(selind[12, knum] != 0)
    Augindex_noise = (selind[13, knum] + 44) * np.float(selind[13, knum] != 0)
    Augindex_simulow = (selind[14, knum] + 47) * np.float(selind[14, knum] != 0)
    Augindex_all = [Augindex_scale, Augindex_rotF, Augindex_rotS, Augindex_rotL, Augindex_mirrorS, Augindex_mirrorF, Augindex_mirrorA,  
                    Augindex_gamma, Augindex_invgamma, Augindex_badd, Augindex_bmul, Augindex_contrast, 
                    Augindex_sharpen, Augindex_noise, Augindex_simulow]
    kcount = 0
    for key in augms_prms:
        if kcount in Augindex_all:
            prmssel = augms_prms[key]
            if kcount < 15:
                augimg.append(prmssel)
            else:
                augsample.append(prmssel)
        kcount += 1
    
    rotx = 0
    roty = 0
    rotz = 0
    scaling = 0
    for kaugimg in range(len(augimg)):
        rotx = rotx + augimg[kaugimg]['rot_xyz'][0]
        roty = roty + augimg[kaugimg]['rot_xyz'][1]
        rotz = rotz + augimg[kaugimg]['rot_xyz'][2]
        scaling = scaling + augimg[kaugimg]['scaling']

    augimgsum = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (rotx, roty, rotz), 'scaling': scaling, 'interp_order_imgs': 1})

    orgimgflag = rotx == 0 and roty == 0 and rotz == 0 and scaling == 0

    shiftmu = 0
    scalemu = 0
    contrast = 0
    reflectx = 0
    reflecty = 0
    reflectz = 0
    blursigma = 0
    blursharpen = 0
    rotxy90 = 0
    rotxy180 = 0
    rotxy270 = 0
    rotyz90 = 0
    rotyz180 = 0
    rotyz270 = 0
    rotxz90 = 0
    rotxz180 = 0
    rotxz270 = 0
    noitsestd = 0
    gamma = 0
    gammainvert = 0
    zoom = 1
    for kaugsample in range(len(augsample)):
        shiftmu = shiftmu + augsample[kaugsample]['hist_dist']['shift']['mu']
        scalemu = scalemu + augsample[kaugsample]['hist_dist']['scale']['mu']
        contrast = contrast + augsample[kaugsample]['contrast']['factor']
        reflectx = reflectx + augsample[kaugsample]['reflect'][0]
        reflecty = reflecty + augsample[kaugsample]['reflect'][1]
        reflectz = reflectz + augsample[kaugsample]['reflect'][2]
        blursigma = blursigma + augsample[kaugsample]['blur']['sigma']
        blursharpen = blursharpen + augsample[kaugsample]['blur']['sharpen']
        rotxy90 = rotxy90 + augsample[kaugsample]['rotate90']['xy']['90']
        rotxy180 = rotxy180 + augsample[kaugsample]['rotate90']['xy']['180']
        rotxy270 = rotxy270 + augsample[kaugsample]['rotate90']['xy']['270']
        rotyz90 = rotyz90 + augsample[kaugsample]['rotate90']['yz']['90']
        rotyz180 = rotyz180 + augsample[kaugsample]['rotate90']['yz']['180']
        rotyz270 = rotyz270 + augsample[kaugsample]['rotate90']['yz']['270']
        rotxz90 = rotxz90 + augsample[kaugsample]['rotate90']['xz']['90']
        rotxz180 = rotxz180 + augsample[kaugsample]['rotate90']['xz']['180']
        rotxz270 = rotxz270 + augsample[kaugsample]['rotate90']['xz']['270']
        noitsestd = noitsestd + augsample[kaugsample]['noise']['std']
        gamma = gamma + augsample[kaugsample]['gamma']['gamma']
        gammainvert = gammainvert + augsample[kaugsample]['gamma']['invgamma']
        zoom = zoom * augsample[kaugsample]['simulowres']['zoom']

    augsamplecompose = {'prob': 1.,
                        'hist_dist': {'shift': {'mu': shiftmu, 'std': 0.}, 'scale': {'mu': scalemu, 'std': 0.}},
                        'contrast': {'factor': contrast}, 'reflect': (reflectx, reflecty, reflectz), 'blur': {'sigma': blursigma, 'sharpen': np.bool(blursharpen)},
                        'rotate90': {'xy': {'0': 0., '90': rotxy90, '180': rotxy180, '270': rotxy270},
                                    'yz': {'0': 0., '90': rotyz90, '180': rotyz180, '270': rotyz270},
                                    'xz': {'0': 0., '90': rotxz90, '180': rotxz180, '270': rotxz270}},
                        'noise': {'std': noitsestd}, 'gamma': {'gamma': gamma, 'invgamma': gammainvert}, 'simulowres': {'zoom': zoom}}

    orgsampleflag = shiftmu == 0 and scalemu == 0 and contrast == 0 and reflectx == 0 and reflecty == 0 and reflectz == 0 and blursigma == 0 \
                    and blursharpen == 0 and rotxy90 == 0 and rotxy180 == 0 and rotxy270 == 0 and rotyz90 == 0 and rotyz180 == 0 \
                    and rotyz270 == 0 and rotxz90 == 0 and rotxz180 == 0 and rotxz270 == 0 and noitsestd == 0 and gamma == 0 \
                    and gammainvert == 0 and zoom == 1

    return orgimgflag, orgsampleflag, augimgsum, augsamplecompose

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

def get_augment_par():
    augm_img_prms_tr = {'origin': None}
    augm_img_prms_tr['origin'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.})
    augm_img_prms_tr['scaling1'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.05})
    augm_img_prms_tr['scaling2'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.15})
    augm_img_prms_tr['scaling3'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.25})
    augm_img_prms_tr['scaling4'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.35})
    augm_img_prms_tr['scaling5'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 0.), 'scaling': 0.45})
    augm_img_prms_tr['rotFrontal_x1'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 5.), 'scaling': 0.})
    augm_img_prms_tr['rotFrontal_x2'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 15.), 'scaling': 0.})
    augm_img_prms_tr['rotFrontal_x3'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 0., 25.), 'scaling': 0.})
    augm_img_prms_tr['rotSagittal_y1'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 5., 0.), 'scaling': 0.})
    augm_img_prms_tr['rotSagittal_y2'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 15., 0.), 'scaling': 0.})
    augm_img_prms_tr['rotSagittal_y3'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (0., 25., 0.), 'scaling': 0.})
    augm_img_prms_tr['rotLongitudinal_z1'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (5., 0., 0.), 'scaling': 0.})
    augm_img_prms_tr['rotLongitudinal_z2'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (15., 0., 0.), 'scaling': 0.})
    augm_img_prms_tr['rotLongitudinal_z3'] = AugmenterAffineParams(
        {'prob': 1., 'rot_xyz': (25., 0., 0.), 'scaling': 0.})

    augm_sample_prms_tr = {'mirror1': None}
    augm_sample_prms_tr['mirror1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (1., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['mirror2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 1., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['mirror3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 1.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Frontal_x1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 1., '180': 0., '270': 1.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Frontal_x2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 1., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Sagittal_y1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 1., '180': 0., '270': 1.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Sagittal_y2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 1., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Longitudinal_z1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 1., '180': 0., '270': 1.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['rot90Longitudinal_z2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 1., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0.1, 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0.3, 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0.5, 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma1invert'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.1}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma2invert'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.3}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['gamma3invert'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.5}, 'simulowres': {'zoom': 1.}}   
    augm_sample_prms_tr['brightnessadd1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0.05, 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['brightnessadd2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0.15, 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['brightnessadd3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0.25, 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}                          
    augm_sample_prms_tr['brightnessmul1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0.05, 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['brightnessmul2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0.15, 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['brightnessmul3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0.25, 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['contrast1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.05}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['contrast2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.15}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['contrast3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.25}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['blur1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.5, 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['blur2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.7, 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['blur3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.9, 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['sharpen1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.9, 'sharpen': True},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['sharpen2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.7, 'sharpen': True},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['sharpen3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0.5, 'sharpen': True},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['noise1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.025}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['noise2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.075}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['noise3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.125}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    augm_sample_prms_tr['simulowres1'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 0.9}}
    augm_sample_prms_tr['simulowres2'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 0.7}}
    augm_sample_prms_tr['simulowres3'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (0., 0., 0.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 0.5}}  
    ## this is only for the default test-time augmentation...mirror in three directions
    augm_sample_prms_tr['mirror123'] = {'prob': 1.,
                                      'hist_dist': {'shift': {'mu': 0., 'std': 0.}, 'scale': {'mu': 0., 'std': 0.}},
                                      'contrast': {'factor': 0.}, 'reflect': (1., 1., 1.), 'blur': {'sigma': 0., 'sharpen': False},
                                      'rotate90': {'xy': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'yz': {'0': 0., '90': 0., '180': 0., '270': 0.},
                                                   'xz': {'0': 0., '90': 0., '180': 0., '270': 0.}},
                                      'noise': {'std': 0.}, 'gamma': {'gamma': 0., 'invgamma': 0.}, 'simulowres': {'zoom': 1.}}
    return augm_img_prms_tr, augm_sample_prms_tr
