import torch
import shutil
import numpy as np
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch.nn.functional as F
from collections import Counter
from torch import Tensor, einsum
from typing import Iterable, List, Tuple, Set
from scipy.ndimage import distance_transform_edt as distance

def get_tn_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tn = (1 - net_output) * (1 - y_onehot)
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tn = tn ** 2
        fp = fp ** 2
        fn = fn ** 2

    tn = sum_tensor(tn, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tn, fp, fn

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

def softmax_helper(x):
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def SoftDiceLoss(x, y, clschosen, loss_mask = None, smooth = 1e-5, do_bg = False, batch_dice = False):
    '''
    Batch_dice means that we want to calculate the dsc of all batch
    It would make more sense for small patchsize, aka DeepMedic based training.
    '''
    shp_x = x.shape
    apply_nonlin = softmax_helper
    square = False

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    if apply_nonlin is not None:
        x = apply_nonlin(x)

    tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, square)

    dc = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    if not do_bg:
        if batch_dice:
            clschosen.remove(0)
            dc_process = dc[clschosen]
        else:
            dc_process = []
            for ksel in clschosen:
                if ksel != 0 :
                    dc_process.append(dc[:, int(ksel)])
            dc_process = torch.cat(dc_process)
        dc_process = dc_process.mean()
    else:
        if batch_dice:
            dc_process = dc[clschosen]
        else:
            dc_process = []
            for ksel in clschosen:
                dc_process.append(dc[:, int(ksel)])
            dc_process = torch.cat(dc_process)
        dc_process = dc_process.mean()

    return -dc_process

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]

    return y[labels]  # [N,D]

def convert_seg_image_to_one_hot_encoding_batched(image, classes=None):
    '''
    same as convert_seg_image_to_one_hot_encoding, but expects image to be (b, x, y, z) or (b, x, y)
    '''
    if classes is None:
        classes = np.unique(image)
    output_shape = [image.shape[0]] + [len(classes)] + list(image.shape[1:])
    out_image = np.zeros(output_shape, dtype=image.dtype)
    for b in range(image.shape[0]):
        for i, c in enumerate(classes):
            out_image[b, i][image[b] == c] = 1
    return out_image

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def ComputMetric(ACTUAL, PREDICTED):
    ACTUAL = ACTUAL.flatten()
    PREDICTED = PREDICTED.flatten()
    idxp = ACTUAL == True
    idxn = ACTUAL == False

    tp = np.sum(ACTUAL[idxp] == PREDICTED[idxp])
    tn = np.sum(ACTUAL[idxn] == PREDICTED[idxn])
    fp = np.sum(idxn) - tn
    fn = np.sum(idxp) - tp
    FPR = fp / (fp + tn)
    if tp == 0 :
        dice = 0
        Precision = 0
        Sensitivity = 0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        Precision = tp / (tp + fp)
        Sensitivity = tp / (tp + fn)
    return dice, Sensitivity, Precision

def show_sevencase(images, titles=0):
    f, axarr = plt.subplots(7, 4, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for idx, ax in enumerate(f.axes):
        ax.imshow(images[idx], cmap='gray', vmin=-5, vmax=5)
        ax.axis("off")
        if idx % 4 == 1 :
            norm = mpl.colors.Normalize(vmin=-1, vmax=3)
            im2show = -1 * np.ones((images[idx].shape[0]+16, images[idx].shape[1]+16))
            im2show[8:8+images[idx].shape[0], 8:8+images[idx].shape[0]] = images[idx]
            ax.imshow(im2show, norm=norm)
        if titles: ax.set_title(titles[idx])
    plt.show()

def show_fivecase(images, titles=0):
    f, axarr = plt.subplots(5, 4, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for idx, ax in enumerate(f.axes):
        ax.imshow(images[idx], cmap='gray', vmin=-5, vmax=5)
        ax.axis("off")
        if idx % 4 == 1 :
            norm = mpl.colors.Normalize(vmin=-1, vmax=14)
            im2show = -1 * np.ones((images[idx].shape[0]+16, images[idx].shape[1]+16))
            im2show[8:8+images[idx].shape[0], 8:8+images[idx].shape[0]] = images[idx]
            ax.imshow(im2show, norm=norm)
        if titles: ax.set_title(titles[idx])
    plt.show()

def show_tencase_Unet(images, titles=0):
    f, axarr = plt.subplots(5, 8, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for idx, ax in enumerate(f.axes):
        ax.imshow(images[idx], cmap='gray', vmin=-5, vmax=5)
        ax.axis("off")
        if idx % 4 == 1 :
            norm = mpl.colors.Normalize(vmin=-1, vmax=14)
            ax.imshow(images[idx], norm=norm)
        if titles: ax.set_title(titles[idx])
    plt.show()

def show_fivecase_Unet(images, titles=0):
    f, axarr = plt.subplots(5, 2, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for idx, ax in enumerate(f.axes):
        ax.imshow(images[idx], cmap='gray', vmin=-5, vmax=5)
        ax.axis("off")
        if idx % 2 == 1 :
            norm = mpl.colors.Normalize(vmin=-1, vmax=14)
            ax.imshow(images[idx], norm=norm)
        if titles: ax.set_title(titles[idx])
    plt.show()

def show_threeimg(images, titles=0):
    f, axarr = plt.subplots(1, 3, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for idx, ax in enumerate(f.axes):
        ax.imshow(images[idx], cmap='gray', vmin=-5, vmax=5)
        ax.axis("off")
    if titles: ax.set_title(titles[idx])
    plt.show()

def show_twocasebrats(images, titles=0):
    f, axarr = plt.subplots(6, 6, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for idx, ax in enumerate(f.axes):
        ax.imshow(images[idx], cmap='gray', vmin=-5, vmax=5)
        ax.axis("off")
        if idx % 18 < 6 :
            norm = mpl.colors.Normalize(vmin=-1, vmax=3)
            im2show = -1 * np.ones((images[idx].shape[0]+16, images[idx].shape[1]+16))
            im2show[8:8+images[idx].shape[0], 8:8+images[idx].shape[0]] = images[idx]
            ax.imshow(im2show, norm=norm)
        if titles: ax.set_title(titles[idx])
    plt.show()

def show_sixty(images, titles=0):
    f, axarr = plt.subplots(6, 10, figsize=(15, 15), gridspec_kw={"wspace": 0, "hspace": 0})
    for idx, ax in enumerate(f.axes):
        ax.imshow(images[idx])
        ax.axis("off")
        if titles: ax.set_title(titles[idx])
    plt.show()

def _concat(xs):
  return torch.cat([x[1].view(-1) for x in xs])

def _concatmodel(xs):
  return torch.cat([x.view(-1) for x in xs])


def hessian_vector_product(args, model, samplemodel, vector, inputnor_var, inputsubp1_var, inputsubp2_var, target_var, Augweightpick, Augindexpick, criterion, clslist, Augindex, miteration, r=1e-2):
    '''
    I should make sure the gradients are not dominated by the most sampled cases.
    Most augmented cases would decrease the loss, therefore get high probabilities once it get sampled.
    Then, the learned policies would be dominated by the initialated ones.
    I should reweight the gradient of different chosen augment policies.
    e.g. We have 3 chosen policy1 and 1 chosen policy2, we want 1/3 gradient from cases from cases of policy1
    '''
    R = r / _concatmodel(vector).norm()
    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)
    outputprime = model(inputnor_var, inputsubp1_var, inputsubp2_var)
    losssampleprime = 0
    for batch in list(range(inputnor_var.shape[0])):
        weightsk = Augweightpick[batch]
        outputas = outputprime[batch, :, :, :, :]
        targets = target_var[batch, :, :, :]
        # there is a problem, because it could not be the batch size
        losssampleprime += weightsk / inputnor_var.shape[0] * criterion(outputas.unsqueeze(0), targets.unsqueeze(0))
    grads_p = torch.autograd.grad(losssampleprime, Augweightpick, retain_graph=True)

    for p, v in zip(model.parameters(), vector):
        p.data.sub_(2 * R, v)
    outputprime = model(inputnor_var, inputsubp1_var, inputsubp2_var)
    losssampleprime = 0
    for batch in list(range(inputnor_var.shape[0])):
        weightsk = Augweightpick[batch]
        outputas = outputprime[batch, :, :, :, :]
        targets = target_var[batch, :, :, :]
        # there is a problem, because it could not be the batch size
        losssampleprime += weightsk / inputnor_var.shape[0] * criterion(outputas.unsqueeze(0), targets.unsqueeze(0))
    grads_n = torch.autograd.grad(losssampleprime, Augweightpick)
    # it shares a lot with the large batch (700), always retain gradph

    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)

    grad_on_weight = [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
    '''normalize the the gradient'''
    clspick = np.array(clslist[miteration * args.batch_size: (miteration + 1) * args.batch_size, 0])
    grad_on_weight = torch.tensor(grad_on_weight)
    for kcls in range(2): # only FG / BG
        grad_on_weight[clspick == kcls] = grad_on_weight[clspick == kcls] - grad_on_weight[clspick == kcls].mean()
    # grad_on_weight = torch.tensor(grad_on_weight)
    # grad_on_weight = grad_on_weight - grad_on_weight.mean()

    '''Here I start the gradient weight normalization process'''
    if type(samplemodel) is list:
        Colcountersall = []
        for kcls in range(len(samplemodel)):
            Colcounters = []
            Augindexpick = Augindex[kcls][miteration * args.batch_size: (miteration + 1) * args.batch_size]
            Augindexpick = torch.LongTensor(np.int32(Augindexpick))
            Augindexpick_np = torch.transpose(Augindexpick, 0, 1).data.cpu().numpy()
            for kpar in range(len(Augindexpick_np)):
                Augindexpick_npcls = Augindexpick_np[:, clspick == kcls]
                Colcounters.append(dict(Counter(Augindexpick_npcls[kpar])))
            Colcountersall.append(Colcounters)

        grads_on_sampler = []
        for kcls in range(len(samplemodel)):
            Augindexpick = Augindex[kcls][miteration * args.batch_size: (miteration + 1) * args.batch_size]
            Augindexpick = torch.LongTensor(np.int32(Augindexpick))
            Augindexpick_np = torch.transpose(Augindexpick, 0, 1).data.cpu().numpy()
            newflag = True
            grads_on_samplerc = []
            for batch in list(range(inputnor_var.shape[0])):
                if clspick[batch] == kcls:
                    grads_weight_on_sampler = torch.autograd.grad(Augweightpick[batch], samplemodel[int(clspick[batch])].parameters(), retain_graph=True)
                    if newflag:
                        for kpar in range(len(grads_weight_on_sampler)):
                            grads_on_samplerc.append(grads_weight_on_sampler[kpar] / Colcountersall[int(clspick[batch])][kpar][Augindexpick_np[kpar][batch]] * grad_on_weight[batch])
                        newflag = False
                    else:
                        for kpar in range(len(grads_weight_on_sampler)):
                            grads_on_samplerc[kpar] += grads_weight_on_sampler[kpar] / Colcountersall[int(clspick[batch])][kpar][Augindexpick_np[kpar][batch]] * grad_on_weight[batch]
            grads_on_sampler.append(grads_on_samplerc)
    else:
        Augindexpick_np = torch.transpose(Augindexpick, 0, 1).data.cpu().numpy()
        Colcounters = []
        for kpar in range(len(Augindexpick_np)):
            Colcounters.append(dict(Counter(Augindexpick_np[kpar])))
        grads_on_sampler = []
        for batch in list(range(inputnor_var.shape[0])):
            grads_weight_on_sampler = torch.autograd.grad(Augweightpick[batch], samplemodel.parameters(), retain_graph=True)
            if batch == 0:
                for kpar in range(len(grads_weight_on_sampler)):
                    grads_on_sampler.append(grads_weight_on_sampler[kpar] / Colcounters[kpar][Augindexpick_np[kpar][batch]] * grad_on_weight[batch])
            else:
                for kpar in range(len(grads_weight_on_sampler)):
                    grads_on_sampler[kpar] += grads_weight_on_sampler[kpar] / Colcounters[kpar][Augindexpick_np[kpar][batch]] * grad_on_weight[batch]

    # If I want to revert it to the previous naive version, I can just delete the Colcounters term.

    return grads_on_sampler

def hessian_vector_product_Unet(model, samplemodel, vector, inputnor_var, target_var, Augweightpick, Augindexpick, criterion, args, clslist, Augindex, miteration, r=1e-2):
    '''
    I should make sure the gradients are not dominated by the most sampled cases.
    Most augmented cases would decrease the loss, therefore get high probabilities once it get sampled.
    Then, the learned policies would be dominated by the initialated ones.
    I should reweight the gradient of different chosen augment policies.
    e.g. We have 3 chosen policy1 and 1 chosen policy2, we want 1/3 gradient from cases from cases of policy1
    '''
    R = r / _concatmodel(vector).norm()
    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)
    outputprime = model(inputnor_var)
    losssampleprime = 0
    for batch in list(range(inputnor_var.shape[0])):
        weightsk = Augweightpick[batch]
        targets = target_var[batch:batch+1, :, :, :]
        '''Also calculate the loss in different scales'''
        if args.deepsupervision:
            targetpicks = targets.data.cpu().numpy()
            weights = np.array([1 / (2 ** i) for i in range(args.downsampling)])
            mask = np.array([True] + [True if i < args.downsampling - 1 else False for i in range(1, args.downsampling)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            for kds in range(args.downsampling):
                targetpickx = targetpicks[:, np.newaxis]
                s = np.ones(3) * 0.5 ** kds
                if args.kits0pros1 == 2: # training with 128*128*8
                    s[2] = 1
                axes = list(range(2, len(targetpickx.shape)))
                new_shape = np.array(targetpickx.shape).astype(float)
                for i, a in enumerate(axes):
                    new_shape[a] *= s[i]
                # in case it is something like 160 * 160 * 80
                if args.kits0pros1 == 1: # training with 64*64*32
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
                losssampleprime += weights[kds] * weightsk / inputnor_var.shape[0] * (
                        criterion(outputprime[kds][batch:batch+1, :, :, :, :], target_vars.unsqueeze(0)) + SoftDiceLoss(
                        outputprime[kds][batch:batch+1, :, :, :, :], target_vars.unsqueeze(0), list(range(args.NumsClass))))
        else:
            outputas = outputprime[batch:batch+1, :, :, :, :]
            losssampleprime += weightsk / inputnor_var.shape[0] * (criterion(outputas, targets) + 
                SoftDiceLoss(outputas, targets, list(range(args.NumsClass))))
    
    grads_p = torch.autograd.grad(losssampleprime, Augweightpick, retain_graph=True)

    for p, v in zip(model.parameters(), vector):
        p.data.sub_(2 * R, v)
    outputprime = model(inputnor_var)
    losssampleprime = 0
    for batch in list(range(inputnor_var.shape[0])):
        weightsk = Augweightpick[batch]
        targets = target_var[batch:batch+1, :, :, :]
        # there is a problem, because it could not be the batch size
        if args.deepsupervision:
            targetpicks = targets.data.cpu().numpy()
            weights = np.array([1 / (2 ** i) for i in range(args.downsampling)])
            mask = np.array([True] + [True if i < args.downsampling - 1 else False for i in range(1, args.downsampling)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            for kds in range(args.downsampling):
                targetpickx = targetpicks[:, np.newaxis]
                s = np.ones(3) * 0.5 ** kds
                if args.kits0pros1 == 2: # training with 128*128*8
                    s[2] = 1
                axes = list(range(2, len(targetpickx.shape)))
                new_shape = np.array(targetpickx.shape).astype(float)
                for i, a in enumerate(axes):
                    new_shape[a] *= s[i]
                # in case it is something like 160 * 160 * 80
                if args.kits0pros1 == 1: # training with 64*64*32
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
                losssampleprime += weights[kds] * weightsk / inputnor_var.shape[0] * (
                        criterion(outputprime[kds][batch:batch+1, :, :, :, :], target_vars.unsqueeze(0)) + SoftDiceLoss(
                        outputprime[kds][batch:batch+1, :, :, :, :], target_vars.unsqueeze(0), list(range(args.NumsClass))))
        else:
            outputas = outputprime[batch:batch+1, :, :, :, :]
            losssampleprime += weightsk / inputnor_var.shape[0] * (criterion(outputas, targets) + 
                SoftDiceLoss(outputas, targets, list(range(args.NumsClass))))
                    
    grads_n = torch.autograd.grad(losssampleprime, Augweightpick)
    # it shares a lot with the large batch (700), always retain gradph

    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)

    grad_on_weight = [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
    clspick = np.array(clslist[miteration * args.batch_size: (miteration + 1) * args.batch_size, 0])
    grad_on_weight = torch.tensor(grad_on_weight)
    for kcls in range(2): # only FG / BG
        grad_on_weight[clspick == kcls] = grad_on_weight[clspick == kcls] - grad_on_weight[clspick == kcls].mean()
    # grad_on_weight = torch.tensor(grad_on_weight)
    # grad_on_weight = grad_on_weight - grad_on_weight.mean()

    '''Here I start the gradient weight normalization process'''
    if type(samplemodel) is list:
        Colcountersall = []
        for kcls in range(len(samplemodel)):
            Colcounters = []
            Augindexpick = Augindex[kcls][miteration * args.batch_size: (miteration + 1) * args.batch_size]
            Augindexpick = torch.LongTensor(np.int32(Augindexpick))
            Augindexpick_np = torch.transpose(Augindexpick, 0, 1).data.cpu().numpy()
            for kpar in range(len(Augindexpick_np)):
                Augindexpick_npcls = Augindexpick_np[:, clspick == kcls]
                Colcounters.append(dict(Counter(Augindexpick_npcls[kpar])))
            Colcountersall.append(Colcounters)

        grads_on_sampler = []
        for kcls in range(len(samplemodel)):
            Augindexpick = Augindex[kcls][miteration * args.batch_size: (miteration + 1) * args.batch_size]
            Augindexpick = torch.LongTensor(np.int32(Augindexpick))
            Augindexpick_np = torch.transpose(Augindexpick, 0, 1).data.cpu().numpy()
            newflag = True
            grads_on_samplerc = []
            for batch in list(range(inputnor_var.shape[0])):
                if clspick[batch] == kcls:
                    grads_weight_on_sampler = torch.autograd.grad(Augweightpick[batch], samplemodel[int(clspick[batch])].parameters(), retain_graph=True)
                    if newflag:
                        for kpar in range(len(grads_weight_on_sampler)):
                            grads_on_samplerc.append(grads_weight_on_sampler[kpar] / Colcountersall[int(clspick[batch])][kpar][Augindexpick_np[kpar][batch]] * grad_on_weight[batch])
                        newflag = False
                    else:
                        for kpar in range(len(grads_weight_on_sampler)):
                            grads_on_samplerc[kpar] += grads_weight_on_sampler[kpar] / Colcountersall[int(clspick[batch])][kpar][Augindexpick_np[kpar][batch]] * grad_on_weight[batch]
            grads_on_sampler.append(grads_on_samplerc)
    else:
        Augindexpick_np = torch.transpose(Augindexpick, 0, 1).data.cpu().numpy()
        Colcounters = []
        for kpar in range(len(Augindexpick_np)):
            Colcounters.append(dict(Counter(Augindexpick_np[kpar])))
        grads_on_sampler = []
        for batch in list(range(inputnor_var.shape[0])):
            grads_weight_on_sampler = torch.autograd.grad(Augweightpick[batch], samplemodel.parameters(), retain_graph=True)
            if batch == 0:
                for kpar in range(len(grads_weight_on_sampler)):
                    grads_on_sampler.append(grads_weight_on_sampler[kpar] / Colcounters[kpar][Augindexpick_np[kpar][batch]] * grad_on_weight[batch])
            else:
                for kpar in range(len(grads_weight_on_sampler)):
                    grads_on_sampler[kpar] += grads_weight_on_sampler[kpar] / Colcounters[kpar][Augindexpick_np[kpar][batch]] * grad_on_weight[batch]

    # If I want to revert it to the previous naive version, I can just delete the Colcounters term.

    return grads_on_sampler

def hessian_vector_product_Unet_mutitask(model, samplemodel, vector, inputnor_var, target_var, taskGenerated, criterion, args, r=1e-2):
    R = r / _concatmodel(vector).norm()
    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)
    _, outputprimeaux = model(inputnor_var)
    losssampleprime = SoftDiceLoss(outputprimeaux, torch.softmax(taskGenerated, 1), list(range(args.taskcls)))
    outputprimeaux = outputprimeaux.transpose(1, 2)
    outputprimeaux = outputprimeaux.transpose(2, 3)
    outputprimeaux = outputprimeaux.transpose(3, 4).contiguous()
    outputprimeaux = outputprimeaux.view(-1, outputprimeaux.shape[4])
    taskGeneratedstretch = taskGenerated.transpose(1, 2)
    taskGeneratedstretch = taskGeneratedstretch.transpose(2, 3)
    taskGeneratedstretch = taskGeneratedstretch.transpose(3, 4).contiguous()
    taskGeneratedstretch = taskGeneratedstretch.view(-1, taskGeneratedstretch.shape[4])
    e1 = 1e-6
    p_y_given_x_train = torch.softmax(outputprimeaux, 1)
    log_p_y_given_x_train = (p_y_given_x_train + e1).log()
    y_aux_given_x_train = torch.softmax(taskGeneratedstretch, 1)
    lossaux = - (1. / p_y_given_x_train.shape[0]) * log_p_y_given_x_train * y_aux_given_x_train
    losssampleprime += lossaux.sum()
    
    grads_p = torch.autograd.grad(losssampleprime, samplemodel.parameters(), retain_graph=True)

    for p, v in zip(model.parameters(), vector):
        p.data.sub_(2 * R, v)
    _, outputprimeaux = model(inputnor_var)
    losssampleprime = SoftDiceLoss(outputprimeaux, torch.softmax(taskGenerated, 1), list(range(args.taskcls)))
    outputprimeaux = outputprimeaux.transpose(1, 2)
    outputprimeaux = outputprimeaux.transpose(2, 3)
    outputprimeaux = outputprimeaux.transpose(3, 4).contiguous()
    outputprimeaux = outputprimeaux.view(-1, outputprimeaux.shape[4])
    taskGeneratedstretch = taskGenerated.transpose(1, 2)
    taskGeneratedstretch = taskGeneratedstretch.transpose(2, 3)
    taskGeneratedstretch = taskGeneratedstretch.transpose(3, 4).contiguous()
    taskGeneratedstretch = taskGeneratedstretch.view(-1, taskGeneratedstretch.shape[4])
    e1 = 1e-6
    p_y_given_x_train = torch.softmax(outputprimeaux, 1)
    log_p_y_given_x_train = (p_y_given_x_train + e1).log()
    y_aux_given_x_train = torch.softmax(taskGeneratedstretch, 1)
    lossaux = - (1. / p_y_given_x_train.shape[0]) * log_p_y_given_x_train * y_aux_given_x_train
    losssampleprime += lossaux.sum()
                    
    grads_n = torch.autograd.grad(losssampleprime, samplemodel.parameters(), retain_graph=(args.entrweight > 0))
    # it shares a lot with the large batch (700), always retain gradph

    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

def hessian_vector_product_Unet_task(model, samplemodel, vector, inputnor_var, target_var, taskGenerated, criterion, args, loss_mask, BGclsindex, r=1e-2):

    from common_Unet_task import calculate_loss

    R = r / _concatmodel(vector).norm()

    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)
    outputprimeaux = model(inputnor_var)

    losssampleprime = calculate_loss(args, target_var, outputprimeaux, taskGenerated, loss_masks = loss_mask, BGcls = BGclsindex)
    grads_p = torch.autograd.grad(losssampleprime, samplemodel.parameters(), retain_graph=True)

    for p, v in zip(model.parameters(), vector):
        p.data.sub_(2 * R, v)
    
    outputprimeaux = model(inputnor_var)
    losssampleprime = calculate_loss(args, target_var, outputprimeaux, taskGenerated, loss_masks = loss_mask, BGcls = BGclsindex)
    
    grads_n = torch.autograd.grad(losssampleprime, samplemodel.parameters(), retain_graph=(args.entrweight > 0) or (args.shconsweight > 0))
    # it shares a lot with the large batch (700), always retain gradph

    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

def hessian_vector_product_Unet_mutitask_map(model, vector, inputnor_var, target_var, taskGeneratedall, taskGenerated, criterion, args, loss_mask, BGclsindex, r=1e-2):
    
    from common_Unet_task import calculate_loss

    R = r / _concatmodel(vector).norm()
    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)
    outputprime = model(inputnor_var)

    losssampleprime = calculate_loss(args, target_var, outputprime, taskGeneratedall, loss_masks = loss_mask, BGcls = BGclsindex)

    grads_p = torch.autograd.grad(losssampleprime, taskGenerated, retain_graph=True)

    for p, v in zip(model.parameters(), vector):
        p.data.sub_(2 * R, v)
    outputprime = model(inputnor_var)
    losssampleprime = calculate_loss(args, target_var, outputprime, taskGeneratedall, loss_masks = loss_mask, BGcls = BGclsindex)
                    
    grads_n = torch.autograd.grad(losssampleprime, taskGenerated)
    # it shares a lot with the large batch (700), always retain gradph

    for p, v in zip(model.parameters(), vector):
        p.data.add_(R, v)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

def clip_grad_value_(parameters, clip_value):
    """Clips gradient of an iterable of parameters at specified value.
    Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in parameters:
        # logging.info(p.data.max())
        p.data.clamp_(min=-clip_value, max=clip_value)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    ## output, N, C, H, W, D to N, C
    ## target N, H, W, D to N,

    target = target.view(-1)
    output = output.permute(0, 2, 3, 4, 1).contiguous()
    output = output.view(-1, output.shape[4])

    maxk = max(topk)
    batch_size = target.size(0)


    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in')

def resize_segmentation(segmentation, new_shape, order=3, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped

def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators_local.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        # it should consider both directions, I think.
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, -rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, -rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, -rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, -rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)

def printaugment(samplemodel, samplemodel_ts, args, logging):
    '''
    Print the current polices
    '''
    if type(samplemodel) is list:
        for kcls in range(len(samplemodel)):
            policyprob_scale = F.softmax(samplemodel[kcls].scale.outwn, dim=0).data.cpu().numpy() * 6
            rawweights_scale = samplemodel[kcls].scale.outwn.data.cpu().numpy()
            policyprob_rotF = F.softmax(samplemodel[kcls].rotF.outwn, dim=0).data.cpu().numpy() * 5
            rawweights_rotF = samplemodel[kcls].rotF.outwn.data.cpu().numpy()
            policyprob_rotS = F.softmax(samplemodel[kcls].rotS.outwn, dim=0).data.cpu().numpy() * 5
            rawweights_rotS = samplemodel[kcls].rotS.outwn.data.cpu().numpy()
            policyprob_rotL = F.softmax(samplemodel[kcls].rotL.outwn, dim=0).data.cpu().numpy() * 5
            rawweights_rotL = samplemodel[kcls].rotL.outwn.data.cpu().numpy()
            policyprob_mirrorS = F.softmax(samplemodel[kcls].mirrorS.outwn, dim=0).data.cpu().numpy() * 2
            rawweights_mirrorS = samplemodel[kcls].mirrorS.outwn.data.cpu().numpy()
            policyprob_mirrorF = F.softmax(samplemodel[kcls].mirrorF.outwn, dim=0).data.cpu().numpy() * 2
            rawweights_mirrorF = samplemodel[kcls].mirrorF.outwn.data.cpu().numpy()
            policyprob_mirrorA = F.softmax(samplemodel[kcls].mirrorA.outwn, dim=0).data.cpu().numpy() * 2
            rawweights_mirrorA = samplemodel[kcls].mirrorA.outwn.data.cpu().numpy()

            policyprob_gamma = F.softmax(samplemodel[kcls].gamma.outwn, dim=0).data.cpu().numpy() * 4
            rawweights_gamma = samplemodel[kcls].gamma.outwn.data.cpu().numpy()
            policyprob_invgamma = F.softmax(samplemodel[kcls].invgamma.outwn, dim=0).data.cpu().numpy() * 4
            rawweights_invgamma = samplemodel[kcls].invgamma.outwn.data.cpu().numpy()
            policyprob_badd = F.softmax(samplemodel[kcls].badd.outwn, dim=0).data.cpu().numpy() * 4
            rawweights_badd = samplemodel[kcls].badd.outwn.data.cpu().numpy()
            policyprob_bmul = F.softmax(samplemodel[kcls].bmul.outwn, dim=0).data.cpu().numpy() * 4
            rawweights_bmul = samplemodel[kcls].bmul.outwn.data.cpu().numpy()
            policyprob_contrast = F.softmax(samplemodel[kcls].contrast.outwn, dim=0).data.cpu().numpy() * 4
            rawweights_contrast = samplemodel[kcls].contrast.outwn.data.cpu().numpy()      

            policyprob_sharpen = F.softmax(samplemodel[kcls].sharpen.outwn, dim=0).data.cpu().numpy() * 7
            rawweights_sharpen = samplemodel[kcls].sharpen.outwn.data.cpu().numpy()
            policyprob_noise = F.softmax(samplemodel[kcls].noise.outwn, dim=0).data.cpu().numpy() * 4
            rawweights_noise = samplemodel[kcls].noise.outwn.data.cpu().numpy()
            policyprob_simulow = F.softmax(samplemodel[kcls].simulow.outwn, dim=0).data.cpu().numpy() * 4
            rawweights_simulow = samplemodel[kcls].simulow.outwn.data.cpu().numpy()

            policyprob = np.concatenate((policyprob_scale, policyprob_rotF, policyprob_rotS, policyprob_rotL, 
                    policyprob_mirrorS, policyprob_mirrorF, policyprob_mirrorA, 
                    policyprob_gamma, policyprob_invgamma, policyprob_badd, policyprob_bmul, policyprob_contrast, 
                    policyprob_sharpen, policyprob_noise, policyprob_simulow))
            rawweights = np.concatenate((rawweights_scale, rawweights_rotF, rawweights_rotS, rawweights_rotL, 
                    rawweights_mirrorS, rawweights_mirrorF, rawweights_mirrorA, 
                    rawweights_gamma, rawweights_invgamma, rawweights_badd, rawweights_bmul, rawweights_contrast, 
                    rawweights_sharpen, rawweights_noise, rawweights_simulow))
            logging.info('Policy probability for training augmentation of class %s = %s', kcls, policyprob)
            logging.info('Raw weights = %s', rawweights)
    else:
        policyprob_scale = F.softmax(samplemodel.scale.outwn, dim=0).data.cpu().numpy() * 6
        rawweights_scale = samplemodel.scale.outwn.data.cpu().numpy()
        policyprob_rotF = F.softmax(samplemodel.rotF.outwn, dim=0).data.cpu().numpy() * 5
        rawweights_rotF = samplemodel.rotF.outwn.data.cpu().numpy()
        policyprob_rotS = F.softmax(samplemodel.rotS.outwn, dim=0).data.cpu().numpy() * 5
        rawweights_rotS = samplemodel.rotS.outwn.data.cpu().numpy()
        policyprob_rotL = F.softmax(samplemodel.rotL.outwn, dim=0).data.cpu().numpy() * 5
        rawweights_rotL = samplemodel.rotL.outwn.data.cpu().numpy()
        policyprob_mirrorS = F.softmax(samplemodel.mirrorS.outwn, dim=0).data.cpu().numpy() * 2
        rawweights_mirrorS = samplemodel.mirrorS.outwn.data.cpu().numpy()
        policyprob_mirrorF = F.softmax(samplemodel.mirrorF.outwn, dim=0).data.cpu().numpy() * 2
        rawweights_mirrorF = samplemodel.mirrorF.outwn.data.cpu().numpy()
        policyprob_mirrorA = F.softmax(samplemodel.mirrorA.outwn, dim=0).data.cpu().numpy() * 2
        rawweights_mirrorA = samplemodel.mirrorA.outwn.data.cpu().numpy()

        policyprob_gamma = F.softmax(samplemodel.gamma.outwn, dim=0).data.cpu().numpy() * 4
        rawweights_gamma = samplemodel.gamma.outwn.data.cpu().numpy()
        policyprob_invgamma = F.softmax(samplemodel.invgamma.outwn, dim=0).data.cpu().numpy() * 4
        rawweights_invgamma = samplemodel.invgamma.outwn.data.cpu().numpy()
        policyprob_badd = F.softmax(samplemodel.badd.outwn, dim=0).data.cpu().numpy() * 4
        rawweights_badd = samplemodel.badd.outwn.data.cpu().numpy()
        policyprob_bmul = F.softmax(samplemodel.bmul.outwn, dim=0).data.cpu().numpy() * 4
        rawweights_bmul = samplemodel.bmul.outwn.data.cpu().numpy()
        policyprob_contrast = F.softmax(samplemodel.contrast.outwn, dim=0).data.cpu().numpy() * 4
        rawweights_contrast = samplemodel.contrast.outwn.data.cpu().numpy()   

        policyprob_sharpen = F.softmax(samplemodel.sharpen.outwn, dim=0).data.cpu().numpy() * 7
        rawweights_sharpen = samplemodel.sharpen.outwn.data.cpu().numpy()
        policyprob_noise = F.softmax(samplemodel.noise.outwn, dim=0).data.cpu().numpy() * 4
        rawweights_noise = samplemodel.noise.outwn.data.cpu().numpy()
        policyprob_simulow = F.softmax(samplemodel.simulow.outwn, dim=0).data.cpu().numpy() * 4
        rawweights_simulow = samplemodel.simulow.outwn.data.cpu().numpy()     

        policyprob = np.concatenate((policyprob_scale, policyprob_rotF, policyprob_rotS, policyprob_rotL, 
                    policyprob_mirrorS, policyprob_mirrorF, policyprob_mirrorA, 
                    policyprob_gamma, policyprob_invgamma, policyprob_badd, policyprob_bmul, policyprob_contrast, 
                    policyprob_sharpen, policyprob_noise, policyprob_simulow))
        rawweights = np.concatenate((rawweights_scale, rawweights_rotF, rawweights_rotS, rawweights_rotL, 
                rawweights_mirrorS, rawweights_mirrorF, rawweights_mirrorA, 
                rawweights_gamma, rawweights_invgamma, rawweights_badd, rawweights_bmul, rawweights_contrast,  
                rawweights_sharpen, rawweights_noise, rawweights_simulow))
        logging.info('Policy probability for training augmentation = %s', policyprob)
        logging.info('Raw weights = %s', rawweights)

    # no need to print the policies, if we dont want it to transfer to DM.
    # printpolicies(logging, policyprob)
    ## this is for test augmentation, also output the best 8 indexes.
    policyprob_ts = F.softmax(samplemodel_ts.outwn, dim=0).data.cpu().numpy()
    rawweights_ts = samplemodel_ts.outwn.data.cpu().numpy()
    sort_index = np.argsort(rawweights_ts)[::-1]
    logging.info('The indexes for test-time augmentation = %s', sort_index)
    logging.info('Policy probability for sorted test-time augmentation = %s', policyprob_ts[sort_index])
    logging.info('Policy probability for test-time  augmentation = %s', policyprob_ts)
    logging.info('Raw weights = %s', rawweights_ts)

def get_composed_augmentation(policyprob, samplenum):
    local_state = np.random.RandomState()
    if type(policyprob) is list:
        selind = []
        for kcls in range(len(policyprob)):
            policyprobc = policyprob[kcls]
            policyprob_scale = policyprobc[:6] / np.sum(policyprobc[:6])
            selind_scale = local_state.choice(np.arange(len(policyprob_scale)), samplenum, p=policyprob_scale)
            policyprob_rotF = policyprobc[6:11] / np.sum(policyprobc[6:11])
            selind_rotF = local_state.choice(np.arange(len(policyprob_rotF)), samplenum, p=policyprob_rotF)
            policyprob_rotS = policyprobc[11:16] / np.sum(policyprobc[11:16])
            selind_rotS = local_state.choice(np.arange(len(policyprob_rotS)), samplenum, p=policyprob_rotS)
            policyprob_rotL = policyprobc[16:21] / np.sum(policyprobc[16:21])
            selind_rotL = local_state.choice(np.arange(len(policyprob_rotL)), samplenum, p=policyprob_rotL)
            policyprob_mirrorS = policyprobc[21:23] / np.sum(policyprobc[21:23])
            selind_mirrorS = local_state.choice(np.arange(len(policyprob_mirrorS)), samplenum, p=policyprob_mirrorS)
            policyprob_mirrorF = policyprobc[23:25] / np.sum(policyprobc[23:25])
            selind_mirrorF = local_state.choice(np.arange(len(policyprob_mirrorF)), samplenum, p=policyprob_mirrorF)
            policyprob_mirrorA = policyprobc[25:27] / np.sum(policyprobc[25:27])
            selind_mirrorA = local_state.choice(np.arange(len(policyprob_mirrorA)), samplenum, p=policyprob_mirrorA)
            
            policyprob_gamma = policyprobc[27:31] / np.sum(policyprobc[27:31])
            selind_gamma = local_state.choice(np.arange(len(policyprob_gamma)), samplenum, p=policyprob_gamma)
            policyprob_invgamma = policyprobc[31:35] / np.sum(policyprobc[31:35])
            selind_invgamma = local_state.choice(np.arange(len(policyprob_invgamma)), samplenum, p=policyprob_invgamma)
            policyprob_badd = policyprobc[35:39] / np.sum(policyprobc[35:39])
            selind_badd = local_state.choice(np.arange(len(policyprob_badd)), samplenum, p=policyprob_badd)
            policyprob_bmul = policyprobc[39:43] / np.sum(policyprobc[39:43])
            selind_bmul = local_state.choice(np.arange(len(policyprob_bmul)), samplenum, p=policyprob_bmul)
            policyprob_contrast = policyprobc[43:47] / np.sum(policyprobc[43:47])
            selind_contrast = local_state.choice(np.arange(len(policyprob_contrast)), samplenum, p=policyprob_contrast)

            policyprob_sharpen = policyprobc[47:54] / np.sum(policyprobc[47:54])
            selind_sharpen = local_state.choice(np.arange(len(policyprob_sharpen)), samplenum, p=policyprob_sharpen)
            policyprob_noise = policyprobc[54:58] / np.sum(policyprobc[54:58])
            selind_noise = local_state.choice(np.arange(len(policyprob_noise)), samplenum, p=policyprob_noise)
            policyprob_simulow = policyprobc[58:] / np.sum(policyprobc[58:])
            selind_simulow = local_state.choice(np.arange(len(policyprob_simulow)), samplenum, p=policyprob_simulow)

            selindc = np.array([selind_scale, selind_rotF, selind_rotS, selind_rotL, selind_mirrorS, selind_mirrorF, selind_mirrorA, 
                            selind_gamma, selind_invgamma, selind_badd, selind_bmul, selind_contrast, 
                            selind_sharpen, selind_noise, selind_simulow])
            selind.append(selindc)
    else:
        # policyprob: the probability of composed augmentatoin, not need to be normalized
        policyprob_scale = policyprob[:6] / np.sum(policyprob[:6])
        selind_scale = local_state.choice(np.arange(len(policyprob_scale)), samplenum, p=policyprob_scale)
        policyprob_rotF = policyprob[6:11] / np.sum(policyprob[6:11])
        selind_rotF = local_state.choice(np.arange(len(policyprob_rotF)), samplenum, p=policyprob_rotF)
        policyprob_rotS = policyprob[11:16] / np.sum(policyprob[11:16])
        selind_rotS = local_state.choice(np.arange(len(policyprob_rotS)), samplenum, p=policyprob_rotS)
        policyprob_rotL = policyprob[16:21] / np.sum(policyprob[16:21])
        selind_rotL = local_state.choice(np.arange(len(policyprob_rotL)), samplenum, p=policyprob_rotL)
        policyprob_mirrorS = policyprob[21:23] / np.sum(policyprob[21:23])
        selind_mirrorS = local_state.choice(np.arange(len(policyprob_mirrorS)), samplenum, p=policyprob_mirrorS)
        policyprob_mirrorF = policyprob[23:25] / np.sum(policyprob[23:25])
        selind_mirrorF = local_state.choice(np.arange(len(policyprob_mirrorF)), samplenum, p=policyprob_mirrorF)
        policyprob_mirrorA = policyprob[25:27] / np.sum(policyprob[25:27])
        selind_mirrorA = local_state.choice(np.arange(len(policyprob_mirrorA)), samplenum, p=policyprob_mirrorA)

        policyprob_gamma = policyprob[27:31] / np.sum(policyprob[27:31])
        selind_gamma = local_state.choice(np.arange(len(policyprob_gamma)), samplenum, p=policyprob_gamma)
        policyprob_invgamma = policyprob[31:35] / np.sum(policyprob[31:35])
        selind_invgamma = local_state.choice(np.arange(len(policyprob_invgamma)), samplenum, p=policyprob_invgamma)
        policyprob_badd = policyprob[35:39] / np.sum(policyprob[35:39])
        selind_badd = local_state.choice(np.arange(len(policyprob_badd)), samplenum, p=policyprob_badd)
        policyprob_bmul = policyprob[39:43] / np.sum(policyprob[39:43])
        selind_bmul = local_state.choice(np.arange(len(policyprob_bmul)), samplenum, p=policyprob_bmul)
        policyprob_contrast = policyprob[43:47] / np.sum(policyprob[43:47])
        selind_contrast = local_state.choice(np.arange(len(policyprob_contrast)), samplenum, p=policyprob_contrast)

        policyprob_sharpen = policyprob[47:54] / np.sum(policyprob[47:54])
        selind_sharpen = local_state.choice(np.arange(len(policyprob_sharpen)), samplenum, p=policyprob_sharpen)
        policyprob_noise = policyprob[54:58] / np.sum(policyprob[54:58])
        selind_noise = local_state.choice(np.arange(len(policyprob_noise)), samplenum, p=policyprob_noise)
        policyprob_simulow = policyprob[58:] / np.sum(policyprob[58:])
        selind_simulow = local_state.choice(np.arange(len(policyprob_simulow)), samplenum, p=policyprob_simulow)

        selind = np.array([selind_scale, selind_rotF, selind_rotS, selind_rotL, selind_mirrorS, selind_mirrorF, selind_mirrorA, 
                        selind_gamma, selind_invgamma, selind_badd, selind_bmul, selind_contrast, 
                        selind_sharpen, selind_noise, selind_simulow])
    return selind 

def get_composed_augmentation_uniform(samplemodel, samplenum, args):
    '''
    Not use for the current implemnetation.
    '''

    samplingthreshold = 1e-3 
    ## We sample from those policies if the policies have probability larger than this
    ## it would save time for meta-training.

    # policyprob: the probability of composed augmentatoin, not need to be normalized
    policyprob_scale = F.softmax(samplemodel.scale.outwn, dim=0).data.cpu().numpy()
    selind_scale = removesmallprob(policyprob_scale, samplingthreshold, samplenum)
    policyprob_rotF = F.softmax(samplemodel.rotF.outwn, dim=0).data.cpu().numpy()
    selind_rotF = removesmallprob(policyprob_rotF, samplingthreshold, samplenum)
    policyprob_rotS = F.softmax(samplemodel.rotS.outwn, dim=0).data.cpu().numpy()
    selind_rotS = removesmallprob(policyprob_rotS, samplingthreshold, samplenum)
    policyprob_rotL = F.softmax(samplemodel.rotL.outwn, dim=0).data.cpu().numpy()
    selind_rotL = removesmallprob(policyprob_rotL, samplingthreshold, samplenum)
    policyprob_mirrorS = F.softmax(samplemodel.mirrorS.outwn, dim=0).data.cpu().numpy()
    selind_mirrorS = removesmallprob(policyprob_mirrorS, samplingthreshold, samplenum)
    policyprob_mirrorF = F.softmax(samplemodel.mirrorF.outwn, dim=0).data.cpu().numpy()
    selind_mirrorF = removesmallprob(policyprob_mirrorF, samplingthreshold, samplenum)
    policyprob_mirrorA = F.softmax(samplemodel.mirrorA.outwn, dim=0).data.cpu().numpy()
    selind_mirrorA = removesmallprob(policyprob_mirrorA, samplingthreshold, samplenum)

    policyprob_gamma = F.softmax(samplemodel.gamma.outwn, dim=0).data.cpu().numpy()
    selind_gamma = removesmallprob(policyprob_gamma, samplingthreshold, samplenum)
    policyprob_invgamma = F.softmax(samplemodel.invgamma.outwn, dim=0).data.cpu().numpy()
    selind_invgamma = removesmallprob(policyprob_invgamma, samplingthreshold, samplenum)
    policyprob_badd = F.softmax(samplemodel.badd.outwn, dim=0).data.cpu().numpy()
    selind_badd = removesmallprob(policyprob_badd, samplingthreshold, samplenum)
    policyprob_bmul = F.softmax(samplemodel.bmul.outwn, dim=0).data.cpu().numpy()
    selind_bmul = removesmallprob(policyprob_bmul, samplingthreshold, samplenum)
    policyprob_contrast = F.softmax(samplemodel.contrast.outwn, dim=0).data.cpu().numpy()
    selind_contrast = removesmallprob(policyprob_contrast, samplingthreshold, samplenum)

    policyprob_sharpen = F.softmax(samplemodel.sharpen.outwn, dim=0).data.cpu().numpy()
    selind_sharpen = removesmallprob(policyprob_sharpen, samplingthreshold, samplenum)
    policyprob_noise = F.softmax(samplemodel.noise.outwn, dim=0).data.cpu().numpy()
    selind_noise = removesmallprob(policyprob_noise, samplingthreshold, samplenum)
    policyprob_simulow = F.softmax(samplemodel.simulow.outwn, dim=0).data.cpu().numpy()
    selind_simulow = removesmallprob(policyprob_simulow, samplingthreshold, samplenum)

    selind = np.array([selind_scale, selind_rotF, selind_rotS, selind_rotL, selind_mirrorS, selind_mirrorF, selind_mirrorA, 
                    selind_gamma, selind_invgamma, selind_badd, selind_bmul, selind_contrast, 
                    selind_sharpen, selind_noise, selind_simulow])
    return selind 

def removesmallprob(policyprob_aug, samplingthreshold, samplenum):

    aug_pool = np.arange(len(policyprob_aug))
    policyprob_chosenindex = policyprob_aug > samplingthreshold

    selind_aug = np.random.choice(aug_pool[policyprob_chosenindex], samplenum)

    return selind_aug

def save_checkpoint(state, is_best, dataset, Savename, filename='checkpoint.pth.tar', record=0):
    """Saves checkpoint to disk"""
    directory = "./output/%s/%s/"%(dataset, Savename)
    if record > 0 and state['epoch'] % record == 0:
        filename = directory + str(state['epoch']) + filename
    else:
        filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './output/%s/%s/'%(dataset, Savename) + 'model_best.pth.tar')

def save_checkpoint_meta(state, augmentor_state, augmentor_state_ts, is_best, dataset, Savename, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "./output/%s/%s/"%(dataset, Savename)
    filename = directory + filename
    torch.save(state, filename)

    if type(augmentor_state['state_dict']) is list:
        samplemodels = augmentor_state['state_dict']
        for kcls in range(len(samplemodels)):
            augmentor_state['state_dict'] = samplemodels[kcls].state_dict()
            augmentor_filename = directory + 'augmentorcls' + str(kcls) + '.pth.tar'
            torch.save(augmentor_state, augmentor_filename)
    else:
        samplemodel = augmentor_state['state_dict']
        augmentor_state['state_dict'] = samplemodel.state_dict()
        augmentor_filename = directory + 'augmentor.pth.tar'
        torch.save(augmentor_state, augmentor_filename)

    augmentor_ts_filename = directory + 'augmentor_ts.pth.tar'
    torch.save(augmentor_state_ts, augmentor_ts_filename)
    if is_best:
        shutil.copyfile(filename, './output/%s/%s/'%(dataset, Savename) + 'model_best.pth.tar')

def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None) -> np.ndarray:
    # assert one_hot(torch.tensor(seg), axis=0)
    K: int = len(seg)

    res = np.zeros_like(seg)
    for k in range(K):
        posmask = seg[k].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = distance(negmask, sampling=resolution) * negmask \
                - (distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res

class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]  # 'wh' for 2d case, 'whd' for 3d case
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor) -> Tensor:
        probs = softmax_helper(probs)
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum(f"bk{self.nd},bk{self.nd}->bk{self.nd}", pc, dc)

        loss = multipled.mean()

        return loss

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h, d = seg.shape  # type: Tuple[int, int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h, d)
    assert one_hot(res)

    return res

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords

def SoftPRECLoss(x, y, cls, loss_mask = None, smooth = 1e-5):
    shp_x = x.shape
    batch_dice = False
    apply_nonlin = softmax_helper
    square = False
    do_bg = False

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    if apply_nonlin is not None:
        x = apply_nonlin(x)

    tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, square)

    dc = (tp + smooth) / (tp + fp + smooth)

    if not do_bg:
        if batch_dice:
            dc = dc[1:]
        else:
            dc = dc[:, 1:]

    dc_process = []
    for ksel in cls:
        if ksel != 0 :
            dc_process.append(dc[:, int(ksel)-1])

    dc_process = torch.cat(dc_process)
    dc_process = dc_process.mean()

    return -dc_process

def SoftSPECLoss(x, y, cls, loss_mask = None, smooth = 1e-5):
    shp_x = x.shape
    batch_dice = False
    apply_nonlin = softmax_helper
    square = False
    do_bg = False

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    if apply_nonlin is not None:
        x = apply_nonlin(x)

    tn, fp, fn = get_tn_fp_fn(x, y, axes, loss_mask, square)

    dc = (tn + smooth) / (tn + fp + smooth)

    if not do_bg:
        if batch_dice:
            dc = dc[1:]
        else:
            dc = dc[:, 1:]

    dc_process = []
    for ksel in cls:
        if ksel != 0 :
            dc_process.append(dc[:, int(ksel)-1])

    dc_process = torch.cat(dc_process)
    dc_process = dc_process.mean()

    return -dc_process

def SoftSENSLoss(x, y, cls, loss_mask = None, smooth = 1e-5):
    shp_x = x.shape
    batch_dice = False
    apply_nonlin = softmax_helper
    square = False
    do_bg = False

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    if apply_nonlin is not None:
        x = apply_nonlin(x)

    tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, square)

    dc = (tp + smooth) / (tp + fn + smooth)

    if not do_bg:
        if batch_dice:
            dc = dc[1:]
        else:
            dc = dc[:, 1:]

    dc_process = []
    for ksel in cls:
        if ksel != 0 :
            dc_process.append(dc[:, int(ksel)-1])

    dc_process = torch.cat(dc_process)
    dc_process = dc_process.mean()

    return -dc_process

def SoftFPLoss(x, y, cls, loss_mask = None, smooth = 1e-5):
    shp_x = x.shape
    batch_dice = False
    apply_nonlin = softmax_helper
    square = False
    do_bg = False

    if batch_dice:
        axes = [0] + list(range(2, len(shp_x)))
    else:
        axes = list(range(2, len(shp_x)))

    if apply_nonlin is not None:
        x = apply_nonlin(x)

    tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, square)

    dc = - fp

    if not do_bg:
        if batch_dice:
            dc = dc[1:]
        else:
            dc = dc[:, 1:]

    dc_process = []
    for ksel in cls:
        if ksel != 0 :
            dc_process.append(dc[:, int(ksel)-1])

    dc_process = torch.cat(dc_process)
    dc_process = dc_process.mean()

    return -dc_process