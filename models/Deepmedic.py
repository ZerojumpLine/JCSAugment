import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def convblock(in_planes, out_planes, ksize=3, padding = 0, dropoutrate = 0.):
    #  Input -> [ BatchNorm OR biases applied] -> NonLinearity -> DropOut -> Pooling --> Conv ] #

    if padding > 0 : # pad with reflect
        out = nn.Sequential(
            nn.BatchNorm3d(in_planes),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Dropout(p=dropoutrate),

            # reflect padding
            # ReflectionPad3d(padding = padding),
            # nn.Conv3d(in_planes, out_planes, ksize)

            # zero padding
            # the paddding mode is not supported by conv3d, for this pytorch version.
            # nn.Conv3d(in_planes, out_planes, ksize, padding = padding)

            # mirror padding, which is actually adapted by DM
            nn.ReplicationPad3d(padding = padding),
            nn.Conv3d(in_planes, out_planes, ksize)
        )
    else:
        out = nn.Sequential(
            nn.BatchNorm3d(in_planes),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Dropout(p=dropoutrate),
            nn.Conv3d(in_planes, out_planes, ksize, padding = padding)
        )

    return out

class ReflectionPad3d(nn.Module):
    # not use here..
    def __init__(self, padding):
        super(ReflectionPad3d, self).__init__()
        self.padding = padding
        # now only support pad = 1
        assert self.padding == 1
    def forward(self, out):
        out = torch.cat((out[:, :, 1:2, :, :], out, out[:, :, -2:-1, :, :]), dim=2)
        out = torch.cat((out[:, :, :, 1:2, :], out, out[:, :, :, -2:-1, :]), dim=3)
        out = torch.cat((out[:, :, :, :, 1:2], out, out[:, :, :, :, -2:-1]), dim=4)
        return out

class Deepmedic(nn.Module):
    def __init__(self, NumsInputChannel, NumsClass, nores, simplearch=False, dropoutrate = 0.0):
        super(Deepmedic, self).__init__()
        if simplearch:
            # self.FMsPerLayerNormal = [10, 10, 10, 10, 10, 10, 10, 10]
            # self.FMsPerLayerFC = [30, 30]
            # self.FMsPerLayerNormal = [10, 10, 20, 20, 25, 25, 30, 30]
            # self.FMsPerLayerFC = [120, 120]
            self.FMsPerLayerNormal = [15, 15, 20, 20, 25, 25, 30, 30]
            self.FMsPerLayerFC = [150, 150]
        else:
            self.FMsPerLayerNormal = [30, 30, 40, 40, 40, 40, 50, 50]
            self.FMsPerLayerFC = [250, 250]
        self.nores = nores
        # Normal pathway, 8 conv block

        ## Normal pathway
        self.NormalPathwayp0 = nn.Sequential(
            nn.Conv3d(NumsInputChannel, self.FMsPerLayerNormal[0], 3),
            convblock(self.FMsPerLayerNormal[0], self.FMsPerLayerNormal[1]))

        self.NormalPathwayp1 = nn.Sequential(
            convblock(self.FMsPerLayerNormal[1], self.FMsPerLayerNormal[2]),
            convblock(self.FMsPerLayerNormal[2], self.FMsPerLayerNormal[3]))

        self.NormalPathwayp2 = nn.Sequential(
            convblock(self.FMsPerLayerNormal[3], self.FMsPerLayerNormal[4]),
            convblock(self.FMsPerLayerNormal[4], self.FMsPerLayerNormal[5]))

        self.NormalPathwayp3 = nn.Sequential(
            convblock(self.FMsPerLayerNormal[5], self.FMsPerLayerNormal[6]),
            convblock(self.FMsPerLayerNormal[6], self.FMsPerLayerNormal[7]))

        # subpathway 1
        self.SubPathway1p0 = nn.Sequential(
            nn.Conv3d(NumsInputChannel, self.FMsPerLayerNormal[0], 3),
            convblock(self.FMsPerLayerNormal[0], self.FMsPerLayerNormal[1]))

        self.SubPathway1p1 = nn.Sequential(
            convblock(self.FMsPerLayerNormal[1], self.FMsPerLayerNormal[2]),
            convblock(self.FMsPerLayerNormal[2], self.FMsPerLayerNormal[3]))

        self.SubPathway1p2 = nn.Sequential(
            convblock(self.FMsPerLayerNormal[3], self.FMsPerLayerNormal[4]),
            convblock(self.FMsPerLayerNormal[4], self.FMsPerLayerNormal[5]))

        self.SubPathway1p3 = nn.Sequential(
            convblock(self.FMsPerLayerNormal[5], self.FMsPerLayerNormal[6]),
            convblock(self.FMsPerLayerNormal[6], self.FMsPerLayerNormal[7]))

        ## subpathway 2
        self.SubPathway2p0 = nn.Sequential(
            nn.Conv3d(NumsInputChannel, self.FMsPerLayerNormal[0], 3),
            convblock(self.FMsPerLayerNormal[0], self.FMsPerLayerNormal[1]))

        self.SubPathway2p1 = nn.Sequential(
            convblock(self.FMsPerLayerNormal[1], self.FMsPerLayerNormal[2]),
            convblock(self.FMsPerLayerNormal[2], self.FMsPerLayerNormal[3]))

        self.SubPathway2p2 = nn.Sequential(
            convblock(self.FMsPerLayerNormal[3], self.FMsPerLayerNormal[4]),
            convblock(self.FMsPerLayerNormal[4], self.FMsPerLayerNormal[5]))

        self.SubPathway2p3 = nn.Sequential(
            convblock(self.FMsPerLayerNormal[5], self.FMsPerLayerNormal[6]),
            convblock(self.FMsPerLayerNormal[6], self.FMsPerLayerNormal[7]))

        ## FC
        self.fc = nn.Sequential(
            # the first one do not have dropout.
            convblock(self.FMsPerLayerNormal[7] * 3, self.FMsPerLayerFC[0], 3, padding= 1 ),
            # convblock(self.FMsPerLayerNormal[7], self.FMsPerLayerFC[0], 3, padding=1),
            convblock(self.FMsPerLayerFC[0], self.FMsPerLayerFC[1], 1, dropoutrate = dropoutrate))

        self.outconv = convblock(self.FMsPerLayerFC[1], NumsClass, 1, dropoutrate = dropoutrate)

    def forward(self, xnor, xsubp1, xsubp2):

        # Normal pathway
        xl2 = self.NormalPathwayp0(xnor)
        xl4 = self.NormalPathwayp1(xl2)
        if self.nores:
            xres0 = xl4
        else:
            xl2padding = torch.zeros(xl4.shape)
            xl2padding = xl2padding.cuda()
            xl2padding[:, :xl2.shape[1], :, :, :] = xl2[:, :, 2:-2, 2:-2, 2:-2]
            xres0 = xl4 + xl2padding
        xl6 = self.NormalPathwayp2(xres0)
        if self.nores:
            xres1 = xl6
        else:
            xres0padding = torch.zeros(xl6.shape)
            xres0padding = xres0padding.cuda()
            xres0padding[:, :xres0.shape[1], :, :, :] = xres0[:, :, 2:-2, 2:-2, 2:-2]
            xres1 = xl6 + xres0padding
        xl8 = self.NormalPathwayp3(xres1)
        if self.nores:
            xres2 = xl8
        else:
            xres1padding = torch.zeros(xl8.shape)
            xres1padding = xres1padding.cuda()
            xres1padding[:, :xres1.shape[1], :, :, :] = xres1[:, :, 2:-2, 2:-2, 2:-2]
            xres2 = xl8 + xres1padding

        # Sub pathway1
        xsubp1l2 = self.SubPathway1p0(xsubp1)
        xsubp1l4 = self.SubPathway1p1(xsubp1l2)
        if self.nores:
            xsubp1res0 = xsubp1l4
        else:
            xsubp1l2padding = torch.zeros(xsubp1l4.shape)
            xsubp1l2padding = xsubp1l2padding.cuda()
            xsubp1l2padding[:, :xsubp1l2.shape[1], :, :, :] = xsubp1l2[:, :, 2:-2, 2:-2, 2:-2]
            xsubp1res0 = xsubp1l4 + xsubp1l2padding
        xsubp1l6 = self.SubPathway1p2(xsubp1res0)
        if self.nores:
            xsubp1res1 = xsubp1l6
        else:
            xsubp1res0padding = torch.zeros(xsubp1l6.shape)
            xsubp1res0padding = xsubp1res0padding.cuda()
            xsubp1res0padding[:, :xsubp1res0.shape[1], :, :, :] = xsubp1res0[:, :, 2:-2, 2:-2, 2:-2]
            xsubp1res1 = xsubp1l6 + xsubp1res0padding
        xsubp1l8 = self.SubPathway1p3(xsubp1res1)
        if self.nores:
            xsubp1res2 = xsubp1l8
        else:
            xsubp1res1padding = torch.zeros(xsubp1l8.shape)
            xsubp1res1padding = xsubp1res1padding.cuda()
            xsubp1res1padding[:, :xsubp1res1.shape[1], :, :, :] = xsubp1res1[:, :, 2:-2, 2:-2, 2:-2]
            xsubp1res2 = xsubp1l8 + xsubp1res1padding

        # Sub pathway2
        xsubp2l2 = self.SubPathway2p0(xsubp2)
        xsubp2l4 = self.SubPathway2p1(xsubp2l2)
        if self.nores:
            xsubp2res0 = xsubp2l4
        else:
            xsubp2l2padding = torch.zeros(xsubp2l4.shape)
            xsubp2l2padding = xsubp2l2padding.cuda()
            xsubp2l2padding[:, :xsubp2l2.shape[1], :, :, :] = xsubp2l2[:, :, 2:-2, 2:-2, 2:-2]
            xsubp2res0 = xsubp2l4 + xsubp2l2padding
        xsubp2l6 = self.SubPathway2p2(xsubp2res0)
        if self.nores:
            xsubp2res1 = xsubp2l6
        else:
            xsubp2res0padding = torch.zeros(xsubp2l6.shape)
            xsubp2res0padding = xsubp2res0padding.cuda()
            xsubp2res0padding[:, :xsubp2res0.shape[1], :, :, :] = xsubp2res0[:, :, 2:-2, 2:-2, 2:-2]
            xsubp2res1 = xsubp2l6 + xsubp2res0padding
        xsubp2l8 = self.SubPathway2p3(xsubp2res1)
        if self.nores:
            xsubp2res2 = xsubp2l8
        else:
            xsubp2res1padding = torch.zeros(xsubp2l8.shape)
            xsubp2res1padding = xsubp2res1padding.cuda()
            xsubp2res1padding[:, :xsubp2res1.shape[1], :, :, :] = xsubp2res1[:, :, 2:-2, 2:-2, 2:-2]
            xsubp2res2 = xsubp2l8 + xsubp2res1padding

        # repeat is not what kostas proposed..although it works when train/test with the size patchsize
        # can be wrong when the input dimension changes
        # xsubp1res2 = xsubp1res2.repeat(1, 1, 3, 3, 3)
        xsubp1res2 = xsubp1res2.repeat_interleave(3, dim = 2)
        xsubp1res2 = xsubp1res2.repeat_interleave(3, dim = 3)
        xsubp1res2 = xsubp1res2.repeat_interleave(3, dim = 4)
        # take some offset to have the center crop. It is omited in the original deepmedic.
        xsubp1res2 = xsubp1res2[:, :, :xres2.shape[2], :xres2.shape[3], :xres2.shape[4]]

        # xsubp2res2 = xsubp2res2.repeat(1, 1, 5, 5, 5)
        xsubp2res2 = xsubp2res2.repeat_interleave(5, dim = 2)
        xsubp2res2 = xsubp2res2.repeat_interleave(5, dim = 3)
        xsubp2res2 = xsubp2res2.repeat_interleave(5, dim = 4)
        xsubp2res2 = xsubp2res2[:, :, :xres2.shape[2], :xres2.shape[3], :xres2.shape[4]]

        xcat = torch.cat([xres2, xsubp1res2, xsubp2res2], 1)
        # xcat = xres2

        xfc = self.fc(xcat)
        if self.nores:
            xres3 = xfc
        else:
            xres2padding = torch.zeros(xfc.shape)
            xres2padding = xres2padding.cuda()
            xres2padding[:, :xcat.shape[1], :, :, :] = xcat[:, :, :, :, :]
            xres3 = xfc + xres2padding
        out = xres3

        out = self.outconv(out)

        return out

    def functional_forward(self, xnor, xsubp1, xsubp2, weights, BN_runing_mean, BN_runing_var):
        BNcount = 0
        listparameters = list(weights.items())

        # 126 pars
        # (5*7 + 1) * 3 + 5 * 3

        # Normal pathway
        out = xnor
        out = F.conv3d(out, weight=listparameters[0][1], bias=listparameters[1][1])
        for klayer in range(0, 1):
            # if I set training=True, it would get exactly the same output. But I need validation mode
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 2][1],
                               bias=listparameters[klayer * 5 + 3][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 4][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 5][1], bias=listparameters[klayer * 5 + 6][1])
        xl2 = out

        for klayer in range(1, 3):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 2][1],
                               bias=listparameters[klayer * 5 + 3][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 4][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 5][1], bias=listparameters[klayer * 5 + 6][1])
        if self.nores:
            out = out
        else:
            xl4 = out
            xl2padding = torch.zeros(xl4.shape)
            xl2padding = xl2padding.cuda()
            xl2padding[:, :xl2.shape[1], :, :, :] = xl2[:, :, 2:-2, 2:-2, 2:-2]
            xres0 = xl4 + xl2padding
            out = xres0

        for klayer in range(3, 5):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 2][1],
                               bias=listparameters[klayer * 5 + 3][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 4][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 5][1], bias=listparameters[klayer * 5 + 6][1])
        if self.nores:
            out = out
        else:
            xl6 = out
            xres0padding = torch.zeros(xl6.shape)
            xres0padding = xres0padding.cuda()
            xres0padding[:, :xres0.shape[1], :, :, :] = xres0[:, :, 2:-2, 2:-2, 2:-2]
            xres1 = xl6 + xres0padding
            out= xres1

        for klayer in range(5, 7):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 2][1],
                               bias=listparameters[klayer * 5 + 3][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 4][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 5][1], bias=listparameters[klayer * 5 + 6][1])
        if self.nores:
            xres2 = out
        else:
            xl8 = out
            xres1padding = torch.zeros(xl8.shape)
            xres1padding = xres1padding.cuda()
            xres1padding[:, :xres1.shape[1], :, :, :] = xres1[:, :, 2:-2, 2:-2, 2:-2]
            xres2 = xl8 + xres1padding

        ## subpathway 1
        out = xsubp1
        out = F.conv3d(out, weight=listparameters[klayer * 5 + 7][1], bias=listparameters[klayer * 5 + 8][1])
        for klayer in range(7, 8):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 4][1],
                               bias=listparameters[klayer * 5 + 5][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 6][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 7][1], bias=listparameters[klayer * 5 + 8][1])
        xsubp1l2 = out

        for klayer in range(8, 10):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 4][1],
                               bias=listparameters[klayer * 5 + 5][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 6][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 7][1], bias=listparameters[klayer * 5 + 8][1])
        if self.nores:
            out = out
        else:
            xsubp1l4 = out
            xsubp1l2padding = torch.zeros(xsubp1l4.shape)
            xsubp1l2padding = xsubp1l2padding.cuda()
            xsubp1l2padding[:, :xsubp1l2.shape[1], :, :, :] = xsubp1l2[:, :, 2:-2, 2:-2, 2:-2]
            xsubp1res0 = xsubp1l4 + xsubp1l2padding
            out = xsubp1res0

        for klayer in range(10, 12):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 4][1],
                               bias=listparameters[klayer * 5 + 5][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 6][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 7][1], bias=listparameters[klayer * 5 + 8][1])
        if self.nores:
            out = out
        else:
            xsubp1l6 = out
            xsubp1res0padding = torch.zeros(xsubp1l6.shape)
            xsubp1res0padding = xsubp1res0padding.cuda()
            xsubp1res0padding[:, :xsubp1res0.shape[1], :, :, :] = xsubp1res0[:, :, 2:-2, 2:-2, 2:-2]
            xsubp1res1 = xsubp1l6 + xsubp1res0padding
            out = xsubp1res1

        for klayer in range(12, 14):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 4][1],
                               bias=listparameters[klayer * 5 + 5][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 6][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 7][1], bias=listparameters[klayer * 5 + 8][1])
        if self.nores:
            xsubp1res2 = out
        else:
            xsubp1l8 = out
            xsubp1res1padding = torch.zeros(xsubp1l8.shape)
            xsubp1res1padding = xsubp1res1padding.cuda()
            xsubp1res1padding[:, :xsubp1res1.shape[1], :, :, :] = xsubp1res1[:, :, 2:-2, 2:-2, 2:-2]
            xsubp1res2 = xsubp1l8 + xsubp1res1padding

        ## subpathway 2
        out = xsubp2
        out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1])
        for klayer in range(14, 15):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1])
        xsubp2l2 = out

        for klayer in range(15, 17):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1])
        if self.nores:
            out = out
        else:
            xsubp2l4 = out
            xsubp2l2padding = torch.zeros(xsubp2l4.shape)
            xsubp2l2padding = xsubp2l2padding.cuda()
            xsubp2l2padding[:, :xsubp2l2.shape[1], :, :, :] = xsubp2l2[:, :, 2:-2, 2:-2, 2:-2]
            xsubp2res0 = xsubp2l4 + xsubp2l2padding
            out = xsubp2res0

        for klayer in range(17, 19):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1])
        if self.nores:
            out = out
        else:
            xsubp2l6 = out
            xsubp2res0padding = torch.zeros(xsubp2l6.shape)
            xsubp2res0padding = xsubp2res0padding.cuda()
            xsubp2res0padding[:, :xsubp2res0.shape[1], :, :, :] = xsubp2res0[:, :, 2:-2, 2:-2, 2:-2]
            xsubp2res1 = xsubp2l6 + xsubp2res0padding
            out = xsubp2res1

        for klayer in range(19, 21):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1])
        if self.nores:
            xsubp2res2 = out
        else:
            xsubp2l8 = out
            xsubp2res1padding = torch.zeros(xsubp2l8.shape)
            xsubp2res1padding = xsubp2res1padding.cuda()
            xsubp2res1padding[:, :xsubp2res1.shape[1], :, :, :] = xsubp2res1[:, :, 2:-2, 2:-2, 2:-2]
            xsubp2res2 = xsubp2l8 + xsubp2res1padding

        # xsubp1res2 = xsubp1res2.repeat(1, 1, 3, 3, 3)
        xsubp1res2 = xsubp1res2.repeat_interleave(3, dim=2)
        xsubp1res2 = xsubp1res2.repeat_interleave(3, dim=3)
        xsubp1res2 = xsubp1res2.repeat_interleave(3, dim=4)
        # take some offset to have the center crop. It is omited in the original deepmedic.
        xsubp1res2 = xsubp1res2[:, :, :xres2.shape[2], :xres2.shape[3], :xres2.shape[4]]

        # xsubp2res2 = xsubp2res2.repeat(1, 1, 5, 5, 5)
        xsubp2res2 = xsubp2res2.repeat_interleave(5, dim=2)
        xsubp2res2 = xsubp2res2.repeat_interleave(5, dim=3)
        xsubp2res2 = xsubp2res2.repeat_interleave(5, dim=4)
        xsubp2res2 = xsubp2res2[:, :, :xres2.shape[2], :xres2.shape[3], :xres2.shape[4]]

        xcat = torch.cat([xres2, xsubp1res2, xsubp2res2], 1)
        out = xcat

        ## fc
        for klayer in range(21, 23):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            if klayer == 21 :  ## fisrt fc has conv kernel of 3.
                # I should pad here, with mirror images
                # reflect padding
                # out = torch.cat((out[:, :, 1:2, :, :], out, out[:, :, -2:-1, :, :]), dim=2)
                # out = torch.cat((out[:, :, :, 1:2, :], out, out[:, :, :, -2:-1, :]), dim=3)
                # out = torch.cat((out[:, :, :, :, 1:2], out, out[:, :, :, :, -2:-1]), dim=4)
                # out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1], padding = 0)

                # zero padding
                # out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1], padding = 1)

                # mirror padding
                out = torch.cat((out[:, :, 0:1, :, :], out, out[:, :, -1:, :, :]), dim=2)
                out = torch.cat((out[:, :, :, 0:1, :], out, out[:, :, :, -1:, :]), dim=3)
                out = torch.cat((out[:, :, :, :, 0:1], out, out[:, :, :, :, -1:]), dim=4)
                out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1], padding=0)
            else :
                # out = F.dropout(out, p=0.5)
                out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1], padding = 0)

        xfc = out
        if self.nores:
            out = xfc
        else:
            xres2padding = torch.zeros(xfc.shape)
            xres2padding = xres2padding.cuda()
            xres2padding[:, :xcat.shape[1], :, :, :] = xcat[:, :, :, :, :]
            xres3 = xfc + xres2padding
            out = xres3

        # the operation of the output conv3d
        for klayer in range(23, 24):
            out = F.batch_norm(out, running_mean=BN_runing_mean[BNcount], running_var=BN_runing_var[BNcount], weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=False)
            BNcount = BNcount + 1
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            # out = F.dropout(out, p=0.5)
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1], padding = 0)

        return out

    def functional_forward_trmode(self, xnor, xsubp1, xsubp2, weights):
        listparameters = list(weights.items())

        # 126 pars
        # (5*7 + 1) * 3 + 5 * 3

        # Normal pathway
        out = xnor
        out = F.conv3d(out, weight=listparameters[0][1], bias=listparameters[1][1])
        for klayer in range(0, 1):
            # if I set training=True, it would get exactly the same output. But I need validation mode
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 2][1],
                               bias=listparameters[klayer * 5 + 3][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 4][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 5][1], bias=listparameters[klayer * 5 + 6][1])
        xl2 = out

        for klayer in range(1, 3):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 2][1],
                               bias=listparameters[klayer * 5 + 3][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 4][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 5][1], bias=listparameters[klayer * 5 + 6][1])
        if self.nores:
            out = out
        else:
            xl4 = out
            xl2padding = torch.zeros(xl4.shape)
            xl2padding = xl2padding.cuda()
            xl2padding[:, :xl2.shape[1], :, :, :] = xl2[:, :, 2:-2, 2:-2, 2:-2]
            xres0 = xl4 + xl2padding
            out = xres0

        for klayer in range(3, 5):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 2][1],
                               bias=listparameters[klayer * 5 + 3][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 4][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 5][1], bias=listparameters[klayer * 5 + 6][1])
        if self.nores:
            out = out
        else:
            xl6 = out
            xres0padding = torch.zeros(xl6.shape)
            xres0padding = xres0padding.cuda()
            xres0padding[:, :xres0.shape[1], :, :, :] = xres0[:, :, 2:-2, 2:-2, 2:-2]
            xres1 = xl6 + xres0padding
            out= xres1

        for klayer in range(5, 7):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 2][1],
                               bias=listparameters[klayer * 5 + 3][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 4][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 5][1], bias=listparameters[klayer * 5 + 6][1])
        if self.nores:
            xres2 = out
        else:
            xl8 = out
            xres1padding = torch.zeros(xl8.shape)
            xres1padding = xres1padding.cuda()
            xres1padding[:, :xres1.shape[1], :, :, :] = xres1[:, :, 2:-2, 2:-2, 2:-2]
            xres2 = xl8 + xres1padding

        ## subpathway 1
        out = xsubp1
        out = F.conv3d(out, weight=listparameters[klayer * 5 + 7][1], bias=listparameters[klayer * 5 + 8][1])
        for klayer in range(7, 8):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 4][1],
                               bias=listparameters[klayer * 5 + 5][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 6][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 7][1], bias=listparameters[klayer * 5 + 8][1])
        xsubp1l2 = out

        for klayer in range(8, 10):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 4][1],
                               bias=listparameters[klayer * 5 + 5][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 6][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 7][1], bias=listparameters[klayer * 5 + 8][1])
        if self.nores:
            out = out
        else:
            xsubp1l4 = out
            xsubp1l2padding = torch.zeros(xsubp1l4.shape)
            xsubp1l2padding = xsubp1l2padding.cuda()
            xsubp1l2padding[:, :xsubp1l2.shape[1], :, :, :] = xsubp1l2[:, :, 2:-2, 2:-2, 2:-2]
            xsubp1res0 = xsubp1l4 + xsubp1l2padding
            out = xsubp1res0

        for klayer in range(10, 12):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 4][1],
                               bias=listparameters[klayer * 5 + 5][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 6][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 7][1], bias=listparameters[klayer * 5 + 8][1])
        if self.nores:
            out = out
        else:
            xsubp1l6 = out
            xsubp1res0padding = torch.zeros(xsubp1l6.shape)
            xsubp1res0padding = xsubp1res0padding.cuda()
            xsubp1res0padding[:, :xsubp1res0.shape[1], :, :, :] = xsubp1res0[:, :, 2:-2, 2:-2, 2:-2]
            xsubp1res1 = xsubp1l6 + xsubp1res0padding
            out = xsubp1res1

        for klayer in range(12, 14):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 4][1],
                               bias=listparameters[klayer * 5 + 5][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 6][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 7][1], bias=listparameters[klayer * 5 + 8][1])
        if self.nores:
            xsubp1res2 = out
        else:
            xsubp1l8 = out
            xsubp1res1padding = torch.zeros(xsubp1l8.shape)
            xsubp1res1padding = xsubp1res1padding.cuda()
            xsubp1res1padding[:, :xsubp1res1.shape[1], :, :, :] = xsubp1res1[:, :, 2:-2, 2:-2, 2:-2]
            xsubp1res2 = xsubp1l8 + xsubp1res1padding

        ## subpathway 2
        out = xsubp2
        out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1])
        for klayer in range(14, 15):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1])
        xsubp2l2 = out

        for klayer in range(15, 17):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1])
        if self.nores:
            out = out
        else:
            xsubp2l4 = out
            xsubp2l2padding = torch.zeros(xsubp2l4.shape)
            xsubp2l2padding = xsubp2l2padding.cuda()
            xsubp2l2padding[:, :xsubp2l2.shape[1], :, :, :] = xsubp2l2[:, :, 2:-2, 2:-2, 2:-2]
            xsubp2res0 = xsubp2l4 + xsubp2l2padding
            out = xsubp2res0

        for klayer in range(17, 19):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1])
        if self.nores:
            out = out
        else:
            xsubp2l6 = out
            xsubp2res0padding = torch.zeros(xsubp2l6.shape)
            xsubp2res0padding = xsubp2res0padding.cuda()
            xsubp2res0padding[:, :xsubp2res0.shape[1], :, :, :] = xsubp2res0[:, :, 2:-2, 2:-2, 2:-2]
            xsubp2res1 = xsubp2l6 + xsubp2res0padding
            out = xsubp2res1

        for klayer in range(19, 21):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1])
        if self.nores:
            xsubp2res2 = out
        else:
            xsubp2l8 = out
            xsubp2res1padding = torch.zeros(xsubp2l8.shape)
            xsubp2res1padding = xsubp2res1padding.cuda()
            xsubp2res1padding[:, :xsubp2res1.shape[1], :, :, :] = xsubp2res1[:, :, 2:-2, 2:-2, 2:-2]
            xsubp2res2 = xsubp2l8 + xsubp2res1padding

        # xsubp1res2 = xsubp1res2.repeat(1, 1, 3, 3, 3)
        xsubp1res2 = xsubp1res2.repeat_interleave(3, dim=2)
        xsubp1res2 = xsubp1res2.repeat_interleave(3, dim=3)
        xsubp1res2 = xsubp1res2.repeat_interleave(3, dim=4)
        # take some offset to have the center crop. It is omited in the original deepmedic.
        xsubp1res2 = xsubp1res2[:, :, :xres2.shape[2], :xres2.shape[3], :xres2.shape[4]]

        # xsubp2res2 = xsubp2res2.repeat(1, 1, 5, 5, 5)
        xsubp2res2 = xsubp2res2.repeat_interleave(5, dim=2)
        xsubp2res2 = xsubp2res2.repeat_interleave(5, dim=3)
        xsubp2res2 = xsubp2res2.repeat_interleave(5, dim=4)
        xsubp2res2 = xsubp2res2[:, :, :xres2.shape[2], :xres2.shape[3], :xres2.shape[4]]

        xcat = torch.cat([xres2, xsubp1res2, xsubp2res2], 1)
        out = xcat

        ## fc
        for klayer in range(21, 23):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            if klayer == 21 :  ## fisrt fc has conv kernel of 3.
                # I should pad here, with mirror images
                # reflect padding
                # out = torch.cat((out[:, :, 1:2, :, :], out, out[:, :, -2:-1, :, :]), dim=2)
                # out = torch.cat((out[:, :, :, 1:2, :], out, out[:, :, :, -2:-1, :]), dim=3)
                # out = torch.cat((out[:, :, :, :, 1:2], out, out[:, :, :, :, -2:-1]), dim=4)
                # out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1], padding = 0)

                # zero padding
                # out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1], padding = 1)

                # mirror padding
                out = torch.cat((out[:, :, 0:1, :, :], out, out[:, :, -1:, :, :]), dim=2)
                out = torch.cat((out[:, :, :, 0:1, :], out, out[:, :, :, -1:, :]), dim=3)
                out = torch.cat((out[:, :, :, :, 0:1], out, out[:, :, :, :, -1:]), dim=4)
                out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1], padding = 0)
            else :
                # out = F.dropout(out, p=0.5)
                out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1], padding = 0)

        xfc = out
        if self.nores:
            out = xfc
        else:
            xres2padding = torch.zeros(xfc.shape)
            xres2padding = xres2padding.cuda()
            xres2padding[:, :xcat.shape[1], :, :, :] = xcat[:, :, :, :, :]
            xres3 = xfc + xres2padding
            out = xres3

        # the operation of the output conv3d
        for klayer in range(23, 24):
            out = F.batch_norm(out, running_mean=None, running_var=None, weight=listparameters[klayer * 5 + 6][1],
                               bias=listparameters[klayer * 5 + 7][1], training=True)
            out = F.prelu(out, listparameters[klayer * 5 + 8][1])
            # out = F.dropout(out, p=0.5)
            out = F.conv3d(out, weight=listparameters[klayer * 5 + 9][1], bias=listparameters[klayer * 5 + 10][1], padding = 0)

        return out
