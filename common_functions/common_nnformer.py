import torch
import numpy as np
import torch.nn as nn
from utilities import SoftDiceLoss, accuracy, _concat, hessian_vector_product_Unet, resize_segmentation
from collections import OrderedDict, Counter
from models.nnformer import nnFormer
from tensorboard_logger import log_value
import time
from common_functions.common_Deepmedic import AverageMeter
from common_functions.common_Unet_test import testatlas, testkits, nntestorgan, testprostate
from sampling.sampling_DM import get_augment_par
from transformations.augmentSamplerng import augment_sample
from transformations.augmentImagerng import augment_imgs_of_case

def train(sampling_results, sampling_results_val, sampling_results_val_samepatch, Augindex, Augindex_ts, Augindex_ts_samepatch, model, samplemodel, samplemodel_ts, criterion, 
    optimizer, optimizer_arch, optimizer_arch_ts, alpha, epoch, logging, args, bratsflag = False):

    # maybe I do not use Augindex_ts here for now.

    '''Train for one epoch on the training set'''
    batch_time = AverageMeter()
    losses = AverageMeter()
    lossvales = AverageMeter()
    lossvalcees = AverageMeter()
    lossvaldsces = AverageMeter()
    top1 = AverageMeter()
    top1val = AverageMeter()

    # switch to train mode
    model.train()
    if args.cs:
        for samplemodelc in samplemodel:
            samplemodelc.train()
    else:
        samplemodel.train()
    samplemodel_ts.train()
    end = time.time()

    ## grep training samples
    inputnor = sampling_results[0] 
    target = sampling_results[1]
    listr = sampling_results[2]
    clslist = sampling_results[3]
    ## grep validation samples
    inputnor_val = sampling_results_val[0]
    target_val = sampling_results_val[1]
    listr_val = sampling_results_val[2]
    clslist_val = sampling_results_val[3]
    ## grep validation samples with the same patch
    inputnor_val_samepatch = sampling_results_val_samepatch[0]
    target_val_samepatch = sampling_results_val_samepatch[1]

    ## just use two classes now.
    clslist = clslist != 0
    clslist_val = clslist_val != 0

    if args.cs:
        Augindexs = Augindex
        Augindex = []
        for Augindexc in Augindexs:
            Augindex.append([Augindexc[:, i] for i in listr])
    else:
        Augindex = [Augindex[:, i] for i in listr]

    DSCvalcls = list(range(args.NumsClass))

    for iteration in range(int(args.numIteration)):

        # Img = []
        # for k in range(5) :
        #     Img.append(inputnor[k, 0, :, :, 18])
        #     Img.append(target[k, :, :, 10])
        #     Img.append(inputsubp1[k, 0, :, :, 11])
        #     Img.append(inputsubp2[k, 0, :, :, 10])
        # show_fivecase(Img)
            
        ## this is from learned distribution
        targetpick = torch.tensor(target[iteration * args.batch_size: (iteration + 1) * args.batch_size, :, :, :])
        target_var = targetpick.long().cuda()
        target_var = torch.autograd.Variable(target_var)
        inputnorpick = torch.tensor(inputnor[iteration * args.batch_size: (iteration + 1) * args.batch_size, :, :, :])
        inputnor_var = inputnorpick.float().cuda()
        inputnor_var = torch.autograd.Variable(inputnor_var)

        ####################################################################
        # this is sampled for learning training augmentation
        target_valpick = torch.tensor(target_val[iteration * args.batch_size_val: (iteration + 1) * args.batch_size_val, :, :, :])
        target_val_var = target_valpick.long().cuda()
        target_val_var = torch.autograd.Variable(target_val_var)
        inputnor_valpick = torch.tensor(
            inputnor_val[iteration * args.batch_size_val: (iteration + 1) * args.batch_size_val, :, :, :, :])
        inputnor_val_var = inputnor_valpick.float().cuda()
        inputnor_val_var = torch.autograd.Variable(inputnor_val_var)

        ####################################################################
        # this is sampled for learning test augmentation
        target_val_samepatchpick = torch.tensor(target_val_samepatch[iteration * (args.batch_size_val + 1): (iteration + 1) * (args.batch_size_val + 1), :, :, :])
        target_val_samepatch_var = target_val_samepatchpick.long().cuda()
        target_val_samepatch_var = torch.autograd.Variable(target_val_samepatch_var)
        inputnor_val_samepatchpick = torch.tensor(
            inputnor_val_samepatch[iteration * (args.batch_size_val + 1): (iteration + 1) * (args.batch_size_val + 1), :, :, :, :])
        inputnor_val_samepatch_var = inputnor_val_samepatchpick.float().cuda()
        inputnor_val_samepatch_var = torch.autograd.Variable(inputnor_val_samepatch_var)

        if iteration % args.augupdate == 0:

            if not args.wotrmeta:

                # compute output
                output = model(inputnor_var)

                # logging.info('model foward ' + str(time.time() - end) + ' seconds') # 300s ? for the first time
                # point1 = time.time()

                #######################################
                # fetch the weight from the current model using Augindex
                # it is used when the augmentation is chosen based on the current parameter
                # which means that I should sample per iteration, and can be super slow
                # Augweightpick = Augweight[iteration * args.batch_size: (iteration + 1) * args.batch_size]
                #######################################

                # I need to make the model pick this, in a *fake* way
                # because the parameters do not change *much*, I think it is reasonable
                if args.cs:
                    Augweightpick = []
                    for kcount in range(args.batch_size):
                        clspick = int(clslist[iteration * args.batch_size + kcount, 0])
                        Augindexpick = Augindex[clspick][iteration * args.batch_size + kcount: iteration * args.batch_size + kcount + 1]
                        Augindexpick = torch.LongTensor(np.int32(Augindexpick)).cuda()
                        Augweightpickc, _ = samplemodel[clspick](1, torch.transpose(Augindexpick, 0, 1))
                        Augweightpick.append(Augweightpickc[0])
                else:
                    Augindexpick = Augindex[iteration * args.batch_size: (iteration + 1) * args.batch_size]
                    Augindexpick = torch.LongTensor(np.int32(Augindexpick)).cuda()
                    Augweightpick, _ = samplemodel(args.batch_size, torch.transpose(Augindexpick, 0, 1))

                # print(len(Augweightpick))
                # just try use the normal training batch
                fast_weights = OrderedDict(model.named_parameters())
                # this is for ds.. (but I think I modify the code to facilite it.)
                # del fast_weights['seg_outputs.0.weight']
                # del fast_weights['seg_outputs.1.weight']

                losssample = 0
                for batch in list(range(inputnorpick.size()[0])):
                    weightsk = Augweightpick[batch]
                    targets = target_var[batch:batch+1, :, :, :]
                    # there is a problem, because it could not be the batch size
                    ''' Caculate training (inner) loss'''
                    if args.deepsupervision:
                        targetpicks = targets.data.cpu().numpy()
                        weights = np.array([1 / (2 ** i) for i in range(args.downsampling)])
                        mask = np.array([True] + [True if i < args.downsampling - 1 else False for i in range(1, args.downsampling)])
                        weights[~mask] = 0
                        weights = weights / weights.sum()
                        for kds in range(args.downsampling):
                            targetpickx = targetpicks[:, np.newaxis]
                            s = np.ones(3) * 0.5 ** kds
                            axes = list(range(2, len(targetpickx.shape)))
                            new_shape = np.array(targetpickx.shape).astype(float)
                            for i, a in enumerate(axes):
                                new_shape[a] *= s[i]
                            # in case it is something like 160 * 160 * 80
                            if args.patch_size[1] != args.patch_size[2]:
                                if kds > 0:
                                    new_shape[4] = new_shape[4] * 2
                            new_shape = np.round(new_shape).astype(int)
                            out_targetpickx = np.zeros(new_shape, dtype=targetpickx.dtype)
                            for b in range(targetpickx.shape[0]):
                                for c in range(targetpickx.shape[1]):
                                    out_targetpickx[b, c] = resize_segmentation(targetpickx[b, c], new_shape[2:], order=0, cval=0)
                            # if would be very slow if I used tensor from the begining.
                            target_varsx = torch.tensor(np.squeeze(out_targetpickx))

                            target_varsx = target_varsx.long().cuda()
                            target_varsx = torch.autograd.Variable(target_varsx)
                            losssample += weights[kds] * weightsk / inputnorpick.size()[0] * (
                                    criterion(output[kds][batch:batch+1, :, :, :, :], target_varsx.unsqueeze(0)) + SoftDiceLoss(
                                    output[kds][batch:batch+1, :, :, :, :], target_varsx.unsqueeze(0), list(range(args.NumsClass))))
                    else:
                        outputas = output[batch:batch+1, :, :, :, :]
                        losssample += weightsk / inputnorpick.size()[0] * (criterion(outputas, targets) 
                            + SoftDiceLoss(outputas, targets, list(range(args.NumsClass))))

                gradients = torch.autograd.grad(losssample, fast_weights.values(), create_graph=True)  # if order = high, I need the graph

                # it can be updated with moment/weightdecay, to find the real direction. 
                # But it may not be necessary, as long as all I need is angel < 180

                fast_weights = OrderedDict(
                    (name, param - alpha * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
                del gradients

                # logging.info('meta model forward ' + str(time.time() - point1) + ' seconds')  # 300s ? for the first time
                # point2 = time.time()
                
                theta = _concat(list(fast_weights.items())).data

                # create model
                embedding_dim=24
                depths=[2, 2, 2, 2]
                num_heads=[3, 6, 12, 24]
                embedding_patch_size=[4,4,4]
                window_size=[4,4,8,4]
                unrolled_model = nnFormer(crop_size=args.patch_size,
                            embedding_dim=embedding_dim,
                            input_channels=1, 
                            num_classes=args.NumsClass, 
                            conv_op=nn.Conv3d, 
                            depths=depths,
                            num_heads=num_heads,
                            patch_size=embedding_patch_size,
                            window_size=window_size,
                            deep_supervision=True)

                unrolled_model = unrolled_model.cuda()
                model_dict = model.state_dict()
                params, offset = {}, 0
                for k, v in model.named_parameters():
                    v_length = np.prod(v.size())
                    params[k] = theta[offset: offset + v_length].view(v.size())
                    offset += v_length
                assert offset == len(theta)
                model_dict.update(params)
                unrolled_model.load_state_dict(model_dict)
                outputval = unrolled_model(inputnor_val_var)

                ''' Caculate validation (outlier) loss'''
                lossval = calculate_loss_origin(args, target_val_var, outputval, epoch, valdata = True, DSCvalcls = DSCvalcls.copy())
                losscalflag = args.softdsc
                args.softdsc = 0
                lossvalce = calculate_loss_origin(args, target_val_var, outputval, epoch, valdata = True)
                args.softdsc = 1
                lossvaldsc = calculate_loss_origin(args, target_val_var, outputval, epoch, valdata = True)
                args.softdsc = losscalflag
                
                # outputval 10,4,21,21,21/ target_val_var 10,21,21,21

                optimizer_arch.zero_grad()
                optimizer_arch_ts.zero_grad()
                
                lossval.backward()
                vector = []
                for v in unrolled_model.parameters():
                    if v.grad is None:
                        vector.append(torch.zeros(v.shape).cuda())
                    else:
                        vector.append(v.grad.data)

                implicit_grads = hessian_vector_product_Unet(model, samplemodel, vector, inputnor_var, target_var, Augweightpick, Augindexpick, criterion, args, clslist, Augindex, iteration)

                if args.cs:
                    for kcls in range(len(samplemodel)):
                        if len(implicit_grads[kcls]) > 0:
                            for v, g in zip(samplemodel[kcls].parameters(), implicit_grads[kcls]):
                                if v.grad is None:
                                    v.grad = - alpha * g.data # it is the outer learning rate (for segmentor.)
                                else:
                                    v.grad.data.copy_(- alpha * g.data)
                else:
                    for v, g in zip(samplemodel.parameters(), implicit_grads):
                        if v.grad is None:
                            v.grad = - alpha * g.data # it is the outer learning rate (for segmentor.)
                        else:
                            v.grad.data.copy_(- alpha * g.data)

                ## for debugging....
                if iteration == 0:
                    logging.info('Chosen training-time augmentaiton is %s', Augindexpick)
                    if args.cs:
                        for v in samplemodel[1].parameters():
                            grad = v.grad
                            # clip_grad_value_(grad, args.clip)
                            # enhance the grad of which leads to bad performance
                            logging.info('Par grad of train augmentor is %s', grad)
                    else:
                        for v in samplemodel.parameters():
                            grad = v.grad
                            # clip_grad_value_(grad, args.clip)
                            # enhance the grad of which leads to bad performance
                            logging.info('Par grad of train augmentor is %s', grad)
                optimizer_arch.step()
            
            if args.wotrmeta or not args.wotsmeta:
                ## this is for the test-time augmentation learning
                ## and for retain loss for comparison
                '''I should restore a new model here for test-time augmentation'''
                '''Because I am a careful guy, I dont want to change the bn statistics based on validation data'''
                '''It would not affect current Unet, because it uses IN, but could happen when I change to BN.'''
                fast_weights = OrderedDict(model.named_parameters())
                theta = _concat(list(fast_weights.items())).data
                # create model
                embedding_dim=24
                depths=[2, 2, 2, 2]
                num_heads=[3, 6, 12, 24]
                embedding_patch_size=[4,4,4]
                window_size=[4,4,8,4]
                unrolled_model = nnFormer(crop_size=args.patch_size,
                            embedding_dim=embedding_dim,
                            input_channels=1, 
                            num_classes=args.NumsClass, 
                            conv_op=nn.Conv3d, 
                            depths=depths,
                            num_heads=num_heads,
                            patch_size=embedding_patch_size,
                            window_size=window_size,
                            deep_supervision=True)

                unrolled_model = unrolled_model.cuda()
                model_dict = model.state_dict()
                params, offset = {}, 0
                for k, v in model.named_parameters():
                    v_length = np.prod(v.size())
                    params[k] = theta[offset: offset + v_length].view(v.size())
                    offset += v_length
                assert offset == len(theta)
                model_dict.update(params)
                unrolled_model.load_state_dict(model_dict)

            if args.wotrmeta:
                # if I have training augmentation meta, I do not use is as indicator.
                outputval = unrolled_model(inputnor_val_var)

                lossval = calculate_loss_origin(args, target_val_var, outputval, epoch, valdata = True, DSCvalcls = DSCvalcls.copy())
                lossvalce = criterion(outputval[0], target_val_var)
                lossvaldsc = SoftDiceLoss(outputval[0], target_val_var, list(range(args.NumsClass)), batch_dice = True)

            if not args.wotsmeta:
                ## test-time augmentation

                outputval_samepatch = unrolled_model(inputnor_val_samepatch_var)
                Augindexpick_ts_samepatch = Augindex_ts_samepatch[iteration * (args.batch_size_val + 1) + 1: (iteration + 1) * (args.batch_size_val + 1)]
                Augweightpick_ts_samepatch, _, _ = samplemodel_ts(args.batch_size_val, Augindexpick_ts_samepatch)

                '''
                Implementation based on aggregated prediction...
                '''

                augm_img_prms_tr, augm_sample_prms_tr = get_augment_par()
                augms_prms = {**augm_img_prms_tr, **augm_sample_prms_tr}

                augmentIDlist = [[0], [1, -2], [3, -4], [5, -6], [7, -8], [9, -10], \
                                [11, -12], [13, -14], [15, -16], [17, -18], [19, -20], [21, -22], [23, -24],[25, -26], [27, -28], \
                                [29], [30], [31], [32, -33], [34], [35, -36], [37], [38, -39], [40], \
                                [41, -42], [43, -44], [45, -46], [47, -48], [49, -50], [51, -52], \
                                [53, -54], [55, -56], [57, -58], [59, -60], [61, -62], [63, -64], [65, -66], [67, -68], [69, -70], \
                                [71], [72], [73], [74], [75], [76], [77], [78], [79], [80], [81], [82], [83]]

                # output_tta = torch.zeros(outputval_samepatch[0].shape[1:])
                # output_tta = output_tta.cuda()

                outputall = []
                lossval_samepatch = 0
                for batch in list(range(inputnor_val_samepatch_var.size()[0] - 1)):
                    weightsk = Augweightpick_ts_samepatch[batch]
                    outputas = outputval_samepatch[0][batch + 1, :, :, :, :]

                    ttaindex = Augindexpick_ts_samepatch[batch]

                    if (ttaindex > 0 and ttaindex <= 40) or ttaindex == 83:
                        ## I want to revert the transformations..
                        ## It might cause some problems with boundaries.. for scaling and rotation...
                        ## I hope it can be good enough.

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

                        outputas_np = outputas.data.cpu().numpy()
                        
                        if ttaindex < 29 and ttaindex > 0:
                            ## it is the revert transform.
                            (outputas_revert, _, _) = augment_imgs_of_case(outputas_np, None,
                            None, None, prmssel, outputas_np.shape[1:], -rng, None)
                        if (ttaindex >= 29 and ttaindex <= 40) or ttaindex == 83:
                            outputas_per_path = []
                            outputas_per_path.append(outputas_np)
                            (outputas_revert_per_path, _) = augment_sample(outputas_per_path,
                                                                        None, prmssel, np.zeros(1), -rng, 0)
                            outputas_revert = outputas_revert_per_path[0]

                        # from utilities import show_fivecase
                        # Img = []
                        # for k in range(5) :
                        #     Img.append(inputnor_val_samepatchpick[k, 0, :, :, 32].data.cpu().numpy())
                        #     Img.append(target_val_samepatchpick[k, :, :, 32])
                        #     Img.append(outputas[1, :, :, 32].data.cpu().numpy())
                        #     Img.append(outputas_revert[1, :, :, 32])
                        # show_fivecase(Img)

                        # put the centralized image back to ensemble ones.
                        # it is done for rot90, where I use np.rot90.
                        outputas_revert_crop = np.zeros(outputval_samepatch[0].shape[1:]) # this is something like N x H x W x D
                        Hmin = np.min((outputas_revert.shape[1], outputas_revert_crop.shape[1]))
                        Wmin = np.min((outputas_revert.shape[2], outputas_revert_crop.shape[2]))
                        Dmin = np.min((outputas_revert.shape[3], outputas_revert_crop.shape[3]))
                        outputas_revert_crop[
                                            :, 
                                            outputas_revert_crop.shape[1] // 2 - Hmin // 2: outputas_revert_crop.shape[1] // 2 + Hmin // 2,
                                            outputas_revert_crop.shape[2] // 2 - Wmin // 2: outputas_revert_crop.shape[2] // 2 + Wmin // 2,
                                            outputas_revert_crop.shape[3] // 2 - Dmin // 2: outputas_revert_crop.shape[3] // 2 + Dmin // 2
                                            ] = outputas_revert[
                                            :, 
                                            outputas_revert.shape[1] // 2 - Hmin // 2: outputas_revert.shape[1] // 2 + Hmin // 2,
                                            outputas_revert.shape[2] // 2 - Wmin // 2: outputas_revert.shape[2] // 2 + Wmin // 2,
                                            outputas_revert.shape[3] // 2 - Dmin // 2: outputas_revert.shape[3] // 2 + Dmin // 2
                                            ]

                        outputas = torch.tensor(outputas_revert_crop.copy())
                        outputas = outputas.float()
                        outputas = outputas.cuda()

                    outputall.append(outputas)
                    # output_tta = output_tta + outputas * weightsk / sum(Augweightpick_ts_samepatch)
                    loss_mask = outputas[1, :, :, :] != 0
                    lossval_samepatch += weightsk / sum(Augweightpick_ts_samepatch) * (SoftDiceLoss(outputas.unsqueeze(0), target_val_samepatch_var[0, :, :, :].unsqueeze(0), list(range(2)), loss_mask = loss_mask.unsqueeze(0).unsqueeze(0).float()))
                    # lossval_samepatch += weightsk / sum(Augweightpick_ts_samepatch) * (criterion(outputas.unsqueeze(0), target_val_samepatch_var[0, :, :, :].unsqueeze(0)))
                    # from utilities import SoftPRECLoss
                    # lossval_samepatch += weightsk / sum(Augweightpick_ts_samepatch) * (SoftPRECLoss(outputas.unsqueeze(0), target_val_samepatch_var[0, :, :, :].unsqueeze(0), list(range(2)), loss_mask = loss_mask.unsqueeze(0).unsqueeze(0).float()))

                # there is a problem, because it could not be the batch size
                ''' Caculate training (inner) loss'''
                ## ts augment val - DSC

                # from utilities import show_tencase_Unet
                # Img = []
                # for k in range(10) :
                #     Img.append(inputnor_val_samepatchpick[k, 0, :, :, 32].data.cpu().numpy())
                #     Img.append(target_val_samepatchpick[k, :, :, 32])
                #     Img.append(outputall[k][1, :, :, 32].data.cpu().numpy())
                #     Img.append(output_tta[1, :, :, 32].data.cpu().numpy())
                # show_tencase_Unet(Img)

                # lossval_samepatch = SoftDiceLoss(output_tta.unsqueeze(0), target_val_samepatch_var[0, :, :, :].unsqueeze(0), DSCvalcls.copy())
                ## ts augment val - CE
                # lossval_samepatch = criterion(output_tta.unsqueeze(0), target_val_samepatch_var[0, :, :, :].unsqueeze(0))
                ## ts augment val - both
                # lossval_samepatch = criterion(output_tta.unsqueeze(0), target_val_samepatch_var[0, :, :, :].unsqueeze(0)) + SoftDiceLoss(output_tta.unsqueeze(0), target_val_samepatch_var[0, :, :, :].unsqueeze(0), DSCvalcls.copy())

                '''
                I should assign normalized weights here.
                Downweight the grad of which the augmnent policy is sampled more than once.
                '''
                grads_weight_val = torch.autograd.grad(lossval_samepatch, Augweightpick_ts_samepatch)

                grads_weight_val = torch.tensor(grads_weight_val)
                grads_weight_val = grads_weight_val - grads_weight_val.mean()

                Augindexpick_ts_samepatch_np = Augindexpick_ts_samepatch.data.cpu().numpy()
                Colounter = dict(Counter(Augindexpick_ts_samepatch_np))
                grads_on_samplerts = 0
                for batch in list(range(inputnor_val_samepatch_var.size()[0] - 1)):
                    grads_on_samplerts += grads_weight_val[batch] / Colounter[Augindexpick_ts_samepatch_np[batch]] * torch.autograd.grad(Augweightpick_ts_samepatch[batch], samplemodel_ts.parameters(), retain_graph=True)[0]
                grads_on_samplests_all = []
                grads_on_samplests_all.append(grads_on_samplerts)
                '''Here the grad ends'''

                for v, g in zip(samplemodel_ts.parameters(), grads_on_samplests_all):
                    if v.grad is None:
                        v.grad = g.data
                    else:
                        v.grad.data.copy_(g.data)

                ## for debugging....
                if iteration == 0:
                    logging.info('Chosen test-time augmentaiton is %s', Augindexpick_ts_samepatch)
                    logging.info('Par grad on the ensembling weight is %s', grads_weight_val)
                    logging.info('Par grad on the sampler is %s', grads_on_samplerts)
                optimizer_arch_ts.step()

            # logging.info('meta model backward ' + str(time.time() - point2) + ' seconds')  # 300s ? for the first epoch
            # point3 = time.time()

        ##### normal training
        # I calcualte the output again, recreate the graph, to save memory.

        if not args.woupdate:
            if args.wotrmeta or iteration % args.augupdate != 0:
                model.train()
                output = model(inputnor_var)
            # for appr, it can be omitted. not sure fore the 2rd order implementation, it also should be fine.
            # I can save 10% time here.
            '''
            It is a little cumbsome, if it uses deepsupervision, I calculate the loss like this.
            I repeat it for 2+1+2 times just in this script and Utilities, maybe it can be done in a better way.
            '''
            losssample = calculate_loss_origin(args, target_var, output, epoch)

            optimizer.zero_grad()
            losssample.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            optimizer.step()

        # logging.info('model backward ' + str(time.time() - point3) + ' seconds')  # 300s ? for the first time

        # measure accuracy and record loss
        if not args.woupdate:
            if args.deepsupervision:
                prec1 = accuracy(output[0].data, targetpick.long().cuda(), topk=(1,))[0]
            else:
                prec1 = accuracy(output.data, targetpick.long().cuda(), topk=(1,))[0]
            losses.update(losssample.data.item(), inputnorpick.size()[0])
            top1.update(prec1.item(), inputnorpick.size()[0])
        if args.deepsupervision:
            prec1val = accuracy(outputval[0].data, target_valpick.long().cuda(), topk=(1,))[0]
        else:
            prec1val = accuracy(outputval.data, target_valpick.long().cuda(), topk=(1,))[0]
        lossvales.update(lossval.data.item(), inputnor_valpick.size()[0])
        lossvalcees.update(lossvalce.data.item(), inputnor_valpick.size()[0])
        lossvaldsces.update(lossvaldsc.data.item(), inputnor_valpick.size()[0])
        top1val.update(prec1val.item(), inputnor_valpick.size()[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (iteration) % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Lossval {lossval.val:.4f} ({lossval.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, iteration, args.numIteration, batch_time=batch_time,
                loss=losses, lossval=lossvales, top1=top1))

    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, epoch)
        log_value('train_lossval', lossvales.avg, epoch)
        log_value('train_lossvalce', lossvalcees.avg, epoch)
        log_value('train_lossvaldsc', lossvaldsces.avg, epoch)
        log_value('train_acc', top1.avg, epoch)
        log_value('trainval_acc', top1val.avg, epoch)

def calculate_loss_origin(args, target_var, output, epoch, criterion = nn.CrossEntropyLoss().cuda(), valdata = False, DSCvalcls = [0]):
    DSCvalcls = list(range(args.NumsClass))
    if args.deepsupervision:
        if valdata:
            output = output[0]
            if args.softdsc == 0:
                losssample = criterion(output, target_var)
            if args.softdsc == 1:
                losssample = SoftDiceLoss(output, target_var, DSCvalcls, batch_dice = True)
            if args.softdsc == 2:
                losssample = criterion(output, target_var) + SoftDiceLoss(output, target_var, DSCvalcls, batch_dice = True)
            if args.softdsc == 3:
                trainingproces = epoch / (args.epochs - 1)
                losssample = (1 - trainingproces) * criterion(output, target_var) + trainingproces * SoftDiceLoss(output, target_var, DSCvalcls, batch_dice = True)
        else:
            losssample = 0
            targetpicks = target_var.data.cpu().numpy()
            weights = np.array([1 / (2 ** i) for i in range(args.downsampling)])
            mask = np.array([True] + [True if i < args.downsampling - 1 else False for i in range(1, args.downsampling)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            for kds in range(args.downsampling):
                targetpickx = targetpicks[:, np.newaxis]
                s = np.ones(3) * 0.5 ** kds
                axes = list(range(2, len(targetpickx.shape)))
                new_shape = np.array(targetpickx.shape).astype(float)
                for i, a in enumerate(axes):
                    new_shape[a] *= s[i]
                # in case it is something like 160 * 160 * 80
                if args.patch_size[1] != args.patch_size[2]:
                    if kds > 0:
                        new_shape[4] = new_shape[4] * 2
                new_shape = np.round(new_shape).astype(int)
                out_targetpickx = np.zeros(new_shape, dtype=targetpickx.dtype)
                for b in range(targetpickx.shape[0]):
                    for c in range(targetpickx.shape[1]):
                        out_targetpickx[b, c] = resize_segmentation(targetpickx[b, c], new_shape[2:], order=0, cval=0)
                # if would be very slow if I used tensor from the begining.
                target_vars = torch.tensor(np.squeeze(out_targetpickx))

                target_vars = target_vars.long().cuda()
                target_vars = torch.autograd.Variable(target_vars)
                if valdata:
                    if args.softdsc == 0:
                        losssample += weights[kds] * criterion(output[kds], target_vars)
                    if args.softdsc == 1:
                        losssample += weights[kds] * SoftDiceLoss(output[kds], target_vars, DSCvalcls, batch_dice = True)
                    if args.softdsc == 2:
                        losssample += weights[kds] * criterion(output[kds], target_vars)
                        losssample += weights[kds] * SoftDiceLoss(output[kds], target_vars, DSCvalcls, batch_dice = True)
                    if args.softdsc == 3:
                        trainingproces = epoch / (args.epochs - 1)
                        losssample += (1 - trainingproces) * weights[kds] * criterion(output[kds], target_vars)
                        losssample += trainingproces * weights[kds] * SoftDiceLoss(output[kds], target_vars, DSCvalcls, batch_dice = True)
                else:    
                    losssample += weights[kds] * (criterion(output[kds], target_vars) + 
                        SoftDiceLoss(output[kds], target_vars, list(range(args.NumsClass))))
    else:
        if valdata:
            if args.softdsc == 0:
                losssample = criterion(output, target_var)
            if args.softdsc == 1:
                losssample = SoftDiceLoss(output, target_var, DSCvalcls, batch_dice = True)
            if args.softdsc == 2:
                losssample = criterion(output, target_var) + SoftDiceLoss(output, target_var, DSCvalcls, batch_dice = True)
            if args.softdsc == 3:
                trainingproces = epoch / (args.epochs - 1)
                losssample = (1 - trainingproces) * criterion(output, target_var) + trainingproces * SoftDiceLoss(output, target_var, DSCvalcls, batch_dice = True)
        else:
            losssample = SoftDiceLoss(output, target_var, list(range(args.NumsClass))) + criterion(output, target_var)

    return losssample

def validateatlas(DatafileValFold, model, criterion, logging, epoch, Savename, args, NumsClass = 2):
    model.eval()

    DSC, SENS, PREC = testatlas(model, True, Savename + '/results/',
                            ImgsegmentSize=args.patch_size,
                            deepsupervision=args.deepsupervision, DatafileValFold=DatafileValFold, NumsClass = NumsClass)

    logging.info('DSC ' + str(DSC))
    logging.info('SENS ' + str(SENS))
    logging.info('PREC ' + str(PREC))
    # log to TensorBoard
    if args.tensorboard:
        log_value('DSClesion', DSC[0], epoch)
        log_value('SENSlesion', SENS[0], epoch)
        log_value('PREClesion', PREC[0], epoch)
    return DSC.mean()

def validateprostate(DatafileValFold, model, criterion, logging, epoch, Savename, args, NumsClass = 2):
    model.eval()

    DSC, SENS, PREC = testprostate(model, True, Savename + '/results/',
                            ImgsegmentSize=args.patch_size,
                            deepsupervision=args.deepsupervision, DatafileValFold=DatafileValFold, NumsClass = NumsClass)

    logging.info('DSC ' + str(DSC))
    logging.info('SENS ' + str(SENS))
    logging.info('PREC ' + str(PREC))
    # log to TensorBoard
    if args.tensorboard:
        log_value('DSCprostate', DSC[0], epoch)
        log_value('SENSprostate', SENS[0], epoch)
        log_value('PRECprostate', PREC[0], epoch)
    return DSC.mean()

def validatekits(DatafileValFold, model, criterion, logging, epoch, Savename, args):
    model.eval()

    DSC, SENS, PREC = testkits(model, True, Savename + '/results/',
                            ImgsegmentSize=args.patch_size,
                            deepsupervision=args.deepsupervision, DatafileValFold=DatafileValFold)

    logging.info('DSC ' + str(DSC))
    logging.info('SENS ' + str(SENS))
    logging.info('PREC ' + str(PREC))
    # log to TensorBoard
    if args.tensorboard:
        log_value('DSCkidney', DSC[0], epoch)
        log_value('DSCtumor', DSC[1], epoch)
        log_value('SENSkidney', SENS[0], epoch)
        log_value('SENStumor', SENS[1], epoch)
        log_value('PRECkidney', PREC[0], epoch)
        log_value('PRECtumor', PREC[1], epoch)
    return DSC.mean()

def validateorgan(DatafileValFold, model, criterion, logging, epoch, Savename, args):
    model.eval()

    DSC, SENS, PREC = nntestorgan(model, True, Savename + '/results/', False,
                        ImgsegmentSize=args.patch_size, 
                        deepsupervision=args.deepsupervision, DatafileValFold=DatafileValFold)

    logging.info('DSC ' + str(DSC))
    logging.info('SENS ' + str(SENS))
    logging.info('PREC ' + str(PREC))
    # log to TensorBoard
    if args.tensorboard:
        log_value('DSCc0', DSC[0], epoch)
        log_value('DSCc1', DSC[1], epoch)
        log_value('DSCc2', DSC[2], epoch)
        log_value('DSCc3', DSC[3], epoch)
        log_value('DSCc4', DSC[4], epoch)
        log_value('DSCc5', DSC[5], epoch)
        log_value('DSCc6', DSC[6], epoch)
        log_value('DSCc7', DSC[7], epoch)
        log_value('DSCc8', DSC[8], epoch)
        log_value('DSCc9', DSC[9], epoch)
        log_value('DSCc10', DSC[10], epoch)
        log_value('DSCc11', DSC[11], epoch)
        log_value('DSCc12', DSC[12], epoch)
        log_value('DSCc13', DSC[13], epoch)
        log_value('DSCavg', np.mean(DSC[1:]), epoch)
        log_value('SENSc0', SENS[0], epoch)
        log_value('SENSc1', SENS[1], epoch)
        log_value('SENSc2', SENS[2], epoch)
        log_value('SENSc3', SENS[3], epoch)
        log_value('SENSc4', SENS[4], epoch)
        log_value('SENSc5', SENS[5], epoch)
        log_value('SENSc6', SENS[6], epoch)
        log_value('SENSc7', SENS[7], epoch)
        log_value('SENSc8', SENS[8], epoch)
        log_value('SENSc9', SENS[9], epoch)
        log_value('SENSc10', SENS[10], epoch)
        log_value('SENSc11', SENS[11], epoch)
        log_value('SENSc12', SENS[12], epoch)
        log_value('SENSc13', SENS[13], epoch)
        log_value('SENSavg', np.mean(SENS[1:]), epoch)
        log_value('PRECc0', PREC[0], epoch)
        log_value('PRECc1', PREC[1], epoch)
        log_value('PRECc2', PREC[2], epoch)
        log_value('PRECc3', PREC[3], epoch)
        log_value('PRECc4', PREC[4], epoch)
        log_value('PRECc5', PREC[5], epoch)
        log_value('PRECc6', PREC[6], epoch)
        log_value('PRECc7', PREC[7], epoch)
        log_value('PRECc8', PREC[8], epoch)
        log_value('PRECc9', PREC[9], epoch)
        log_value('PRECc10', PREC[10], epoch)
        log_value('PRECc11', PREC[11], epoch)
        log_value('PRECc12', PREC[12], epoch)
        log_value('PRECc13', PREC[13], epoch)
        log_value('PRECavg', np.mean(PREC[1:]), epoch)
    return np.mean(DSC[1:])

    model.eval()

    DSC, SENS, PREC = nntestukbborgan(model, True, Savename + '/results/', False,
                        ImgsegmentSize=args.patch_size, 
                        deepsupervision=args.deepsupervision, DatafileValFold=DatafileValFold)

    logging.info('DSC ' + str(DSC))
    logging.info('SENS ' + str(SENS))
    logging.info('PREC ' + str(PREC))
    # log to TensorBoard
    if args.tensorboard:
        log_value('DSCc0', DSC[0], epoch)
        log_value('DSCc1', DSC[1], epoch)
        log_value('DSCc2', DSC[2], epoch)
        log_value('DSCc3', DSC[3], epoch)
        log_value('DSCc4', DSC[4], epoch)
        log_value('DSCc5', DSC[5], epoch)
        log_value('DSCc6', DSC[6], epoch)
        log_value('DSCc7', DSC[7], epoch)
        log_value('DSCc8', DSC[8], epoch)
        log_value('DSCc9', DSC[9], epoch)
        log_value('DSCc10', DSC[10], epoch)
        log_value('DSCc11', DSC[11], epoch)
        log_value('DSCc12', DSC[12], epoch)
        log_value('DSCc13', DSC[13], epoch)
        log_value('DSCc14', DSC[14], epoch)
        log_value('DSCc15', DSC[15], epoch)
        log_value('DSCc16', DSC[16], epoch)
        log_value('DSCavg', np.mean(DSC[1:]), epoch)
        log_value('SENSc0', SENS[0], epoch)
        log_value('SENSc1', SENS[1], epoch)
        log_value('SENSc2', SENS[2], epoch)
        log_value('SENSc3', SENS[3], epoch)
        log_value('SENSc4', SENS[4], epoch)
        log_value('SENSc5', SENS[5], epoch)
        log_value('SENSc6', SENS[6], epoch)
        log_value('SENSc7', SENS[7], epoch)
        log_value('SENSc8', SENS[8], epoch)
        log_value('SENSc9', SENS[9], epoch)
        log_value('SENSc10', SENS[10], epoch)
        log_value('SENSc11', SENS[11], epoch)
        log_value('SENSc12', SENS[12], epoch)
        log_value('SENSc13', SENS[13], epoch)
        log_value('SENSc14', SENS[14], epoch)
        log_value('SENSc15', SENS[15], epoch)
        log_value('SENSc16', SENS[16], epoch)
        log_value('SENSavg', np.mean(SENS[1:]), epoch)
        log_value('PRECc0', PREC[0], epoch)
        log_value('PRECc1', PREC[1], epoch)
        log_value('PRECc2', PREC[2], epoch)
        log_value('PRECc3', PREC[3], epoch)
        log_value('PRECc4', PREC[4], epoch)
        log_value('PRECc5', PREC[5], epoch)
        log_value('PRECc6', PREC[6], epoch)
        log_value('PRECc7', PREC[7], epoch)
        log_value('PRECc8', PREC[8], epoch)
        log_value('PRECc9', PREC[9], epoch)
        log_value('PRECc10', PREC[10], epoch)
        log_value('PRECc11', PREC[11], epoch)
        log_value('PRECc12', PREC[12], epoch)
        log_value('PRECc13', PREC[13], epoch)
        log_value('PRECc14', PREC[14], epoch)
        log_value('PRECc15', PREC[15], epoch)
        log_value('PRECc16', PREC[16], epoch)
        log_value('PRECavg', np.mean(PREC[1:]), epoch)
    return np.mean(DSC[1:])