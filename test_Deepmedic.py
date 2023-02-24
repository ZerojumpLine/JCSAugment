import os
import argparse
import numpy as np
from models.Deepmedic import Deepmedic
import torch

parser = argparse.ArgumentParser(description='PyTorch DeepMedic Test')
# General configures.
parser.add_argument('--name', default='Deepmedic', type=str, help='name of experiment')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# Test configures.
parser.add_argument('--saveresults', help='To save results in name', action='store_true')
# Network configures.
parser.add_argument('--nores', help='Exclude res block', action='store_true')
# Dataset configures.
parser.add_argument('--kits0pros1', default=0, type=float, help='choose the dataset')
parser.add_argument('--trainval', help='to test on the training data, just for debugging', action='store_true')
# Data Augmentaion parameters.
parser.add_argument('--ttalist', default=[0], nargs='+', type=int, help='way to do test time augmentation')
parser.add_argument('--ttalistprob', default=[1], nargs='+', type=float, help='weights to integrade test time augmentation')
parser.add_argument('--tta', help='test time augmentation (x8), not for use here', action='store_true')
parser.set_defaults(augment=True)
args = parser.parse_args()

np.random.seed(12345)
torch.manual_seed(12345)
torch.cuda.manual_seed_all(12345)

if __name__ == '__main__':
    if args.kits0pros1 == 0:
        from common_functions.common_Deepmedic_test import testkits as test
        NumsInputChannel = 1
        NumsClass = 3
        if args.trainval:
            DatafileValFold = './datafiles/KiTS/datafiletrain/'
        else:
            DatafileValFold = './datafiles/KiTS/datafiletest/'
    if args.kits0pros1 == 1:
        from common_functions.common_Deepmedic_test import validateprostate as test
        NumsInputChannel = 1
        NumsClass = 2
        if args.trainval:
            DatafileValFold='./datafiles/Prostate/datafiletrainingSourceandTarget/'
        else:
            DatafileValFold='./datafiles/Prostate/datafiletestTarget/'
        
    torch.cuda.set_device(args.gpu)
    model = Deepmedic(NumsInputChannel, NumsClass, args.nores)
    model = model.cuda()
    model.eval()

    # ttalist = [0, 1, 20]
    ttalist = args.ttalist
    ttalistprob = args.ttalistprob

    ttalistprob = ttalistprob / np.sum(ttalistprob) * len(ttalistprob)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:' + str(args.gpu))
            # args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    DSC, SENS, PREC = test(model, args.saveresults, args.name + '/results/', DatafileValFold=DatafileValFold, 
                    trainval=args.trainval, ttalist=ttalist, ttalistprob=ttalistprob, tta=args.tta, NumsClass = NumsClass)
    print('DSC ' + str(DSC))
    print('SENS ' + str(SENS))
    print('PREC ' + str(PREC))
    
    if len(DSC) > 1:
        print('DSCavg ' + str(np.mean(DSC[1:])))
        print('SENSavg ' + str(np.mean(SENS[1:])))
        print('PRECavg ' + str(np.mean(PREC[1:])))