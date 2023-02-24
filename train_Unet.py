import argparse
import os
import shutil
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
from multiprocessing.pool import ThreadPool
# loading models, functions.
from models.Unet import Generic_UNet, InitWeights_He
from sampling.sampler_gambels_test import SampleNet
from sampling.sampler_gambels import SampleNet_compose
from common_functions.common_Unet import train
from common_functions.common_Deepmedic import adjust_learning_rate, adjust_learning_rate_arch, adjust_learning_rate_arch_ts
from utilities import save_checkpoint_meta, get_composed_augmentation, printaugment
# loading sampling code
from sampling.sampling_Unet import getbatchkitsatlas as getbatch
from sampling.sampling_Unet_test import getbatchkitsatlas as getbatch_ts
from sampling.sampling_Unet_test_samepatch import getbatchkitsatlas as getbatch_ts_samepatch
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
os.environ['KMP_WARNINGS'] = 'off'

parser = argparse.ArgumentParser(description='PyTorch JCSAugment for 3D U-Net Training')
# General configures.
parser.add_argument('--name', default='3DUnet', type=str, help='name of experiment')
parser.add_argument('--print-freq', '-p', default=40, type=int, help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--resumeaugmentor', default='', type=str, help='path to augmentor checkpoint (default: none)')
parser.add_argument('--startover', action='store_true', help='do not care about the training info in the ckpt')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# Training configures.
parser.add_argument('--epochs', default=2000, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=10, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--batch-size-val', default=10, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--numIteration', default=100, type=int, help='num of iteration per epoch')
parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--sgd0orAdam1orRms2', default=2, type=float, help='choose the optimizer')
parser.add_argument('--arch_learning_rate', type=float, default=1e-3, help='learning rate for arch encoding')
parser.add_argument('--archts_learning_rate', type=float, default=5e-3, help='learning rate for arch encoding, test augment')
# Network configures.
parser.add_argument('--maxsample', type=float, default=50, help='sample from cases, large number leads to longer time')
parser.add_argument('--evalevery', type=float, default=200, help='evaluation every epoches')
parser.add_argument('--downsampling', default=3, type=int, help='too see if I need deeper arch')
parser.add_argument('--features', default=10, type=int, help='feature map')
parser.add_argument('--deepsupervision', action='store_true', help='use deep supervision, just like nnunet')
parser.add_argument('--patch-size', default=[64,64,64], nargs='+', type=int, help='patch size')
parser.add_argument('--patch-size-val', default=[64,64,64], nargs='+', type=int, help='patch size of val')
# Dataset configures.
parser.add_argument('--kits0pros1', default=0, type=float, help='choose the dataset')
parser.add_argument('--split', default=0, type=int, help='using different portion of training data')
# Data Augmentaion parameters.
parser.add_argument('--wotrmeta', action='store_true', help='do not have meta update for TRA')
parser.add_argument('--wotsmeta', action='store_true', help='do not have meta update for TEA')
parser.add_argument('--woupdate', action='store_true', help='do not update the segmentation model')
parser.add_argument('--printevery', type=float, default=10, help='print policies every epoches')
parser.add_argument('--softdsc', default=0, type=float, help='0 use ce, 1 use dsc, 2 use both for validation criterion')
parser.add_argument('--org', default=0, type=int, help='set 2 as predefined as nnunet, 3 as predefined as DeepMedic, 5 as predefined for nnformer')
parser.add_argument('--orgts', default=0, type=int, help='set 2 as predefined as nnunet, set 4 as predefined as DeepMedic')
parser.add_argument('--cs', action='store_true', help='use class specific augmentor (different augmentator for different cls)')
parser.add_argument('--det', action='store_true', help='control seed to for control experiments')
parser.add_argument('--vanilla', action='store_true', help='do not use augmentation')
parser.add_argument('--predefineda', default=0, type=int, help='set 0 as vanilla, 1 as small, 2 as mid, 3 as large')
parser.add_argument('--augupdate', default=1, type=int, help='I update the augmentor every this epoch, to save time.')
parser.add_argument('--seed', default=79, type=int, help='set seed for training.')
args = parser.parse_args()
best_prec1 = 0

if args.det:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    ## note that I can only control the sampling case, but not the sampling patches (it is controled by a global seed).
else:
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False

def main():
    global best_prec1

    # some dataset specific configs.
    if args.kits0pros1 == 0:
        dataset = 'kits'
        # this is for kidney tumor segmentation.
        from common_functions.common_Unet import validatekits as validate
        if args.split == 0 : ## training with 50% training data
            DatafileTrainqueueFold = './datafiles/KiTS/datafiletrain50percent/'
            DatafileValqueueFold = './datafiles/KiTS/datafileval/'
            DatafileValFold = './datafiles/KiTS/datafiletest/'
        if args.split == 1 : ## training with 100% training data
            DatafileTrainqueueFold = './datafiles/KiTS/datafiletrain/'
            DatafileValqueueFold = './datafiles/KiTS/datafileval/'
            DatafileValFold = './datafiles/KiTS/datafiletest/'
        args.NumsInputChannel = 1
        args.NumsClass = 3
    if args.kits0pros1 == 1:
        dataset = 'prostate'
        # this is for cross-dataset prostate MRI
        from common_functions.common_Unet import validateprostate as validate
        if args.split == 0: ## training with data from the source domain
            DatafileTrainqueueFold = './datafiles/Prostate/datafiletrainingSource/'
            DatafileValqueueFold = './datafiles/Prostate/datafilevalTarget/'
            DatafileValFold = './datafiles/Prostate/datafiletestTarget/'
        if args.split == 1: ## training with limited data from the target domain 
            DatafileTrainqueueFold = './datafiles/Prostate/datafiletestTarget/'
            DatafileValqueueFold = './datafiles/Prostate/datafilevalTarget/'
            DatafileValFold = './datafiles/Prostate/datafiletestTarget/'
        if args.split == 2: ## training with data from both the source and target domain
            DatafileTrainqueueFold = './datafiles/Prostate/datafiletrainingSourceandTarget/'
            DatafileValqueueFold = './datafiles/Prostate/datafilevalTarget/'
            DatafileValFold = './datafiles/Prostate/datafiletestTarget/'
        args.NumsInputChannel = 1
        args.NumsClass = 2
    if args.kits0pros1 == 2:
        dataset = 'cardiac'
        # this is for cardiac segmentation.
        from common_functions.common_Unet import validatecardiac as validate
        if args.split == 0 : ## training with source training data
            DatafileTrainqueueFold = './datafiles/MnMCardiac/datafiletrainingSource/'
            DatafileValqueueFold = './datafiles/MnMCardiac/datafilevalSourceTarget/'
            DatafileValFold = './datafiles/MnMCardiac/datafiletestTarget/'
        args.NumsInputChannel = 1
        args.NumsClass = 4

    Savename = args.name + 'archlr' + str(args.arch_learning_rate) + 'archtslr' + str(args.archts_learning_rate)
    directory = "./output/%s/%s/"%(dataset, Savename)

    if not os.path.exists(directory):
        os.makedirs(directory)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(directory, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(args.gpu)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.tensorboard: configure("./output/%s/%s"%(dataset, Savename))

    ## prepare the vanilla augmentation
    if args.vanilla:
        if args.predefineda == 0:
            policyprob = np.zeros(62, )
            policyprob[0] = 1
            policyprob[6] = 1
            policyprob[11] = 1
            policyprob[16] = 1
            policyprob[21] = 1
            policyprob[23] = 1
            policyprob[25] = 1
            policyprob[27] = 1
            policyprob[31] = 1
            policyprob[35] = 1
            policyprob[39] = 1
            policyprob[43] = 1
            policyprob[47] = 1
            policyprob[54] = 1
            policyprob[58] = 1
        if args.predefineda == 1:
            policyprob = '4.8        1.20000005 0. 0. 0. 0. ' \
            '4.5        0.49999996 0. 0. 0. 4.5 ' \
            '0.49999996 0. 0. 0. 4.5        0.49999996 ' \
            '0. 0. 0.      1.         1.         1. ' \
            '1.         1.         1.         2.8        1.20000012 0. ' \
            '0. 3.6000001  0.39999999 0. 0. 3.8 ' \
            '0.19999998 0. 0. 3.3999999  0.59999997 0. ' \
            '0. 3.3999999  0.59999997 0. 0. 5.6 ' \
            '0.69999996 0. 0. 0.69999996 0. 0. ' \
            '3.6000001  0.39999999 0. 0. 3.6000001         0.39999999 ' \
            '0.  0.'
            policyprob = policyprob.split(' ')
            policyprob = list(filter(None, policyprob))
            policyprob = [float(policyprob[i]) for i in range(len(policyprob))]
            policyprob = np.array(policyprob)
        if args.predefineda == 2:
            policyprob = '4.8        0. 0. 1.20000005 0. 0. ' \
            '4.5        0. 0.49999996 0. 0. 4.5 ' \
            '0. 0.49999996 0. 0. 4.5        0. ' \
            '0.49999996 0. 0.       1.         1.         1. ' \
            '1.         1.         1.         2.8        0. 1.20000012 ' \
            '0. 3.6000001  0. 0.39999999 0. 3.8 ' \
            '0. 0.19999998 0. 3.3999999  0. 0.59999997 ' \
            '0. 3.3999999  0. 0.59999997 0. 5.6 ' \
            '0. 0.69999996 0. 0. 0.69999996 0. ' \
            '3.6000001  0. 0.39999999 0. 3.6000001         0. ' \
            '0.39999999  0.'
            policyprob = policyprob.split(' ')
            policyprob = list(filter(None, policyprob))
            policyprob = [float(policyprob[i]) for i in range(len(policyprob))]
            policyprob = np.array(policyprob)
        if args.predefineda == 3:
            policyprob = '4.8        0. 0. 0. 0. 1.20000005 ' \
            '4.5        0. 0. 0.49999996 0. 4.5 ' \
            '0. 0. 0.49999996 0. 4.5        0. ' \
            '0. 0.49999996 0.       1.         1.         1. ' \
            '1.         1.         1.         2.8        0. 0. ' \
            '1.20000012 3.6000001  0. 0. 0.39999999 3.8 ' \
            '0. 0. 0.19999998 3.3999999  0. 0. ' \
            '0.59999997 3.3999999  0. 0. 0.59999997 5.6 ' \
            '0. 0. 0.69999996 0. 0. 0.69999996 ' \
            '3.6000001  0. 0. 0.39999999 3.6000001         0. ' \
            '0.  0.39999999'
            policyprob = policyprob.split(' ')
            policyprob = list(filter(None, policyprob))
            policyprob = [float(policyprob[i]) for i in range(len(policyprob))]
            policyprob = np.array(policyprob)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    # create model
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    conv_per_stage = 2
    base_num_features = args.features

    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': False}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': False}
    net_num_pool_op_kernel_sizes = []
    if args.kits0pros1 == 1: # training with 64*64*32
        net_num_pool_op_kernel_sizes.append([2, 2, 1])
        for kiter in range(0, args.downsampling - 1):  # (0,5)
            net_num_pool_op_kernel_sizes.append([2, 2, 2])
    elif args.kits0pros1 == 2: # training with 128*128*8
        for kiter in range(0, args.downsampling):  # (0,5)
            net_num_pool_op_kernel_sizes.append([2, 2, 1])
    else:
        for kiter in range(0, args.downsampling):  # (0,5)
            net_num_pool_op_kernel_sizes.append([2, 2, 2])
    net_conv_kernel_sizes = []
    for kiter in range(0, args.downsampling + 1):  # (0,6)
        net_conv_kernel_sizes.append([3, 3, 3])

    model = Generic_UNet(args.NumsInputChannel, base_num_features, args.NumsClass,
                         len(net_num_pool_op_kernel_sizes),
                         conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                         dropout_op_kwargs,
                         net_nonlin, net_nonlin_kwargs, args.deepsupervision, False, lambda x: x, InitWeights_He(1e-2),
                         net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)
    
    if args.cs:
        samplemodel = []
        for _ in range(2):
            samplemodel.append(SampleNet_compose(args.org, 1).cuda())
        parameters_arch = []
        for samplemodelc in samplemodel:
            parameters_arch += list(samplemodelc.parameters())
        optimizer_arch = torch.optim.Adam(parameters_arch, lr=args.arch_learning_rate, betas=(0.9, 0.999), weight_decay=0)
    else:
        samplemodel = SampleNet_compose(args.org, 1)
        samplemodel = samplemodel.cuda()
        optimizer_arch = torch.optim.Adam(samplemodel.parameters(), lr=args.arch_learning_rate, betas=(0.9, 0.999), weight_decay=0)

    nninitindex = [0, 29, 30, 31, 34, 37, 40, 83]
    augpool = 84
    samplemodel_ts = SampleNet(augpool, args.orgts, 1, nninitindex)

    if args.sgd0orAdam1orRms2 == 0 :
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    if args.sgd0orAdam1orRms2 == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-5, amsgrad=True)
    if args.sgd0orAdam1orRms2 == 2:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1e-04, weight_decay=0.0001, momentum=0.6)

    optimizer_arch_ts = torch.optim.Adam(samplemodel_ts.parameters(), lr=args.archts_learning_rate, betas=(0.9, 0.999), weight_decay=0)

    # get the number of model parameters
    logging.info('Number of model parameters: {} MB'.format(sum([p.data.nelement() for p in model.parameters()])/1e6))

    model = model.cuda()
    samplemodel_ts = samplemodel_ts.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:' + str(args.gpu))
            if not args.startover:
                args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_prec1 = checkpoint['best_prec1']
            # when started, it needs initalization of prec1.
            prec1 = 0
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.resumeaugmentor:
        if os.path.isfile(args.resumeaugmentor):
            logging.info("=> loading augmentor checkpoint '{}'".format(args.resumeaugmentor))
            if args.cs:
                for kcls in range(2):
                    knamelist = args.resumeaugmentor.split("/")
                    ckt_tr = args.resumeaugmentor[:-len(knamelist[-1])] + 'augmentorcls' + str(kcls) + '.pth.tar'
                    checkpoint = torch.load(ckt_tr, map_location='cuda:' + str(args.gpu))
                    samplemodel[kcls].load_state_dict(checkpoint['state_dict'])
            else:
                checkpoint = torch.load(args.resumeaugmentor, map_location='cuda:' + str(args.gpu))
                samplemodel.load_state_dict(checkpoint['state_dict'])
            optimizer_arch.load_state_dict(checkpoint['optimizer_arch'])
            knamelist = args.resumeaugmentor.split("/")
            ckt_ts = args.resumeaugmentor[:-len(knamelist[-1])] + 'augmentor_ts.pth.tar'
            checkpoint_ts = torch.load(ckt_ts, map_location='cuda:' + str(args.gpu))
            samplemodel_ts.load_state_dict(checkpoint_ts['state_dict'])
            optimizer_arch_ts.load_state_dict(checkpoint_ts['optimizer_arch_ts'])
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resumeaugmentor, checkpoint['epoch']))
        else:
            logging.info("=> no augmentor checkpoint found at '{}'".format(args.resumeaugmentor))

    mp_pool = None
    mp_pool = ThreadPool(processes=1)
    # take from DeepMedic.

    for epoch in range(args.start_epoch, args.epochs):
        alpha = adjust_learning_rate(optimizer, epoch, args)
        adjust_learning_rate_arch(optimizer_arch, epoch, args)
        adjust_learning_rate_arch_ts(optimizer_arch_ts, epoch, args)

        if epoch % args.printevery == 0:
            logging.info('For epoch = %s', epoch)
            printaugment(samplemodel, samplemodel_ts, args, logging)

        if mp_pool is None:
            # sequence processing
            # sample more validation cases in one iteration
            '''Training augmentation policies'''
            if args.vanilla:
                Augindex = get_composed_augmentation(policyprob, args.batch_size * args.numIteration)
            else:
                if args.cs:
                    Augindex = []
                    for samplemodelc in samplemodel:
                        _, Augindexc = samplemodelc(args.batch_size * args.numIteration, None)
                        Augindex.append(Augindexc.data.cpu().numpy())
                else:
                    _, Augindex = samplemodel(args.batch_size * args.numIteration, None)
                    Augindex = Augindex.data.cpu().numpy()

            '''Test augmentation policies'''
            if args.wotsmeta:
                Augindex_ts = torch.zeros(args.batch_size_val * args.numIteration)
                Augindex_ts_samepatch = torch.zeros((args.batch_size_val + 1) * args.numIteration)
            else:
                _, Augindex_ts, _ = samplemodel_ts(args.batch_size_val * args.numIteration, None)
                ## I need to make some random sampling policies here.
                # Augindex_ts_samepatch = torch.randint(0, args.augpool, (args.batch_size_val * args.numIteration,))
                _, Augindex_ts_samepatch, _ = samplemodel_ts((args.batch_size_val + 1) * args.numIteration, None)
                Augindex_ts_samepatch_pos0 = np.zeros((args.batch_size_val + 1) * args.numIteration)
                Augindex_ts_samepatch_pos0[::args.batch_size_val + 1] = 1
                Augindex_ts_samepatch[Augindex_ts_samepatch_pos0 == 1] = 0

            sampling_results = getbatch(DatafileTrainqueueFold, args.batch_size, args.numIteration, Augindex,
                                                    args.maxsample, 0, logging, 0, args.patch_size)
            sampling_results_val = getbatch_ts(DatafileValqueueFold, args.batch_size_val, args.numIteration, Augindex_ts.data.cpu().numpy(),
                                                    args.maxsample, 0, logging, 0, args.patch_size_val)
            sampling_results_val_samepatch = getbatch_ts_samepatch(DatafileValqueueFold, args.batch_size_val + 1, args.numIteration, Augindex_ts_samepatch.data.cpu().numpy(),
                                                    args.maxsample, 0, logging, 0, args.patch_size_val)
            
        elif epoch == args.start_epoch:  # Not previously submitted in case of first epoch
            # to get the sampling from the multiprocess. the sampling parameters might have mismatch
            # get one results.
            '''Training augmentation policies'''
            if args.vanilla:
                Augindex = get_composed_augmentation(policyprob, args.batch_size * args.numIteration)
            else:
                if args.cs:
                    Augindex = []
                    for samplemodelc in samplemodel:
                        _, Augindexc = samplemodelc(args.batch_size * args.numIteration, None)
                        Augindex.append(Augindexc.data.cpu().numpy())
                else:
                    _, Augindex = samplemodel(args.batch_size * args.numIteration, None)
                    Augindex = Augindex.data.cpu().numpy()

            '''Test augmentation policies'''
            if args.wotsmeta:
                Augindex_ts = torch.zeros(args.batch_size_val * args.numIteration)
                Augindex_ts_samepatch = torch.zeros((args.batch_size_val + 1) * args.numIteration)
            else:
                _, Augindex_ts, _ = samplemodel_ts((args.batch_size_val + 1) * args.numIteration, None)
                ## I need to make some random sampling policies here.
                # Augindex_ts_samepatch = torch.randint(0, args.augpool, (args.batch_size_val * args.numIteration,))
                _, Augindex_ts_samepatch, _ = samplemodel_ts((args.batch_size_val + 1) * args.numIteration, None)
                Augindex_ts_samepatch_pos0 = np.zeros((args.batch_size_val + 1) * args.numIteration)
                Augindex_ts_samepatch_pos0[::args.batch_size_val + 1] = 1
                Augindex_ts_samepatch[Augindex_ts_samepatch_pos0 == 1] = 0

            sampling_results = getbatch(DatafileTrainqueueFold, args.batch_size, args.numIteration, Augindex,
                                                    args.maxsample, 0, logging, 0, args.patch_size)
            sampling_results_val = getbatch_ts(DatafileValqueueFold, args.batch_size_val, args.numIteration, Augindex_ts.data.cpu().numpy(),
                                                    args.maxsample, 0, logging, 0, args.patch_size_val)
            sampling_results_val_samepatch = getbatch_ts_samepatch(DatafileValqueueFold, args.batch_size_val + 1, args.numIteration, Augindex_ts_samepatch.data.cpu().numpy(),
                                                    args.maxsample, 0, logging, 0, args.patch_size_val)

            '''Training augmentation policies'''
            # sub new job.
            if args.vanilla:
                Augindexsampling = get_composed_augmentation(policyprob, args.batch_size * args.numIteration)
            else:
                if args.cs:
                    Augindexsampling = []
                    for samplemodelc in samplemodel:
                        _, Augindexsamplingc = samplemodelc(args.batch_size * args.numIteration, None)
                        Augindexsampling.append(Augindexsamplingc.data.cpu().numpy())
                else:
                    _, Augindexsampling = samplemodel(args.batch_size * args.numIteration, None)
                    Augindexsampling = Augindexsampling.data.cpu().numpy()

            '''Test augmentation policies'''
            if args.wotsmeta:
                Augindexsampling_ts = torch.zeros(args.batch_size_val * args.numIteration)
                Augindexsampling_ts_samepatch = torch.zeros((args.batch_size_val + 1)* args.numIteration)
            else:
                _, Augindexsampling_ts, _ = samplemodel_ts(args.batch_size_val * args.numIteration, None)
                ## I need to make some random sampling policies here.
                # Augindexsampling_ts_samepatch = torch.randint(0, args.augpool, (args.batch_size_val * args.numIteration,))
                _, Augindexsampling_ts_samepatch, _ = samplemodel_ts((args.batch_size_val + 1) * args.numIteration, None)
                Augindexsampling_ts_samepatch_pos0 = np.zeros((args.batch_size_val + 1) * args.numIteration)
                Augindexsampling_ts_samepatch_pos0[::args.batch_size_val + 1] = 1
                Augindexsampling_ts_samepatch[Augindexsampling_ts_samepatch_pos0 == 1] = 0

            sampling_job = mp_pool.apply_async(getbatch, (DatafileTrainqueueFold, args.batch_size, args.numIteration, Augindexsampling,
                                                                args.maxsample, 0, logging, 0, args.patch_size))
            sampling_job_ts = mp_pool.apply_async(getbatch_ts, (DatafileValqueueFold, args.batch_size_val, args.numIteration, Augindexsampling_ts.data.cpu().numpy(),
                                                                args.maxsample, 0, logging, 0, args.patch_size_val))
            sampling_job_ts_samepatch = mp_pool.apply_async(getbatch_ts_samepatch, (DatafileValqueueFold, args.batch_size_val + 1, args.numIteration, Augindexsampling_ts_samepatch.data.cpu().numpy(),
                                                                args.maxsample, 0, logging, 0, args.patch_size_val))
        elif epoch == args.epochs - 1: # last iteration
            # do not need to submit job
            Augindex = Augindexsampling
            sampling_results = sampling_job.get()
            Augindex_ts = Augindexsampling_ts
            Augindex_ts_samepatch = Augindexsampling_ts_samepatch
            sampling_results_val = sampling_job_ts.get()
            sampling_results_val_samepatch = sampling_job_ts_samepatch.get()
            mp_pool.close()
            mp_pool.join()
        else:
            # get old job and submit new job
            Augindex = Augindexsampling
            sampling_results = sampling_job.get()
            Augindex_ts = Augindexsampling_ts
            Augindex_ts_samepatch = Augindexsampling_ts_samepatch
            sampling_results_val = sampling_job_ts.get()
            sampling_results_val_samepatch = sampling_job_ts_samepatch.get()

            '''submit new'''
            '''Training augmentation policies'''
            if args.vanilla:
                Augindexsampling = get_composed_augmentation(policyprob, args.batch_size * args.numIteration)
            else:
                if args.cs:
                    Augindexsampling = []
                    for samplemodelc in samplemodel:
                        _, Augindexsamplingc = samplemodelc(args.batch_size * args.numIteration, None)
                        Augindexsampling.append(Augindexsamplingc.data.cpu().numpy())
                else:
                    _, Augindexsampling = samplemodel(args.batch_size * args.numIteration, None)
                    Augindexsampling = Augindexsampling.data.cpu().numpy()

            '''Test augmentation policies'''
            if args.wotsmeta:
                Augindexsampling_ts = torch.zeros(args.batch_size_val * args.numIteration)
                Augindexsampling_ts_samepatch = torch.zeros((args.batch_size_val + 1) * args.numIteration)
            else:
                _, Augindexsampling_ts, _ = samplemodel_ts((args.batch_size_val) * args.numIteration, None)
                ## I need to make some random sampling policies here.
                # Augindexsampling_ts_samepatch = torch.randint(0, args.augpool, (args.batch_size_val * args.numIteration,))
                _, Augindexsampling_ts_samepatch, _ = samplemodel_ts((args.batch_size_val + 1) * args.numIteration, None)
                Augindexsampling_ts_samepatch_pos0 = np.zeros((args.batch_size_val + 1) * args.numIteration)
                Augindexsampling_ts_samepatch_pos0[::args.batch_size_val + 1] = 1
                Augindexsampling_ts_samepatch[Augindexsampling_ts_samepatch_pos0 == 1] = 0
            
            sampling_job = mp_pool.apply_async(getbatch, (DatafileTrainqueueFold, args.batch_size, args.numIteration, Augindexsampling,
                                                                args.maxsample, 0, logging, 0, args.patch_size))
            sampling_job_ts = mp_pool.apply_async(getbatch_ts, (DatafileValqueueFold, args.batch_size_val, args.numIteration, Augindexsampling_ts.data.cpu().numpy(),
                                                                args.maxsample, 0, logging, 0, args.patch_size_val))
            sampling_job_ts_samepatch = mp_pool.apply_async(getbatch_ts_samepatch, (DatafileValqueueFold, args.batch_size_val + 1, args.numIteration, Augindexsampling_ts_samepatch.data.cpu().numpy(),
                                                                args.maxsample, 0, logging, 0, args.patch_size_val))

        # train for one epoch
        train(sampling_results, sampling_results_val, sampling_results_val_samepatch, Augindex, Augindex_ts, Augindex_ts_samepatch, model, samplemodel, samplemodel_ts, criterion, 
            optimizer, optimizer_arch, optimizer_arch_ts, alpha, epoch, logging, args)

        if epoch == args.epochs - 1: # last
            logging.info('Finally policies:')
            printaugment(samplemodel, samplemodel_ts, args, logging)

        # evaluate on validation set every 5 epoches
        if args.evalevery > 0:
            if epoch % args.evalevery == 0 or epoch == args.epochs-1 :
                prec1 = validate(DatafileValFold, model, criterion, logging, epoch, Savename, args)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint_meta({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1,
        }, {
            'epoch': epoch + 1,
            'state_dict': samplemodel,
            'optimizer_arch': optimizer_arch.state_dict(),
            'best_prec1': best_prec1,
        }, {
            'epoch': epoch + 1,
            'state_dict': samplemodel_ts.state_dict(),
            'optimizer_arch_ts': optimizer_arch_ts.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, dataset, Savename)
    logging.info('Best overall DSCuracy: %s ', best_prec1)

if __name__ == '__main__':
    main()