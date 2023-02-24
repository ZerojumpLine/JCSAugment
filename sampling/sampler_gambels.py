import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sampling.sampler_gambels_test import SampleNet

class SampleNet_compose(nn.Module):
    def __init__(self, origininit, temperature):
        super(SampleNet_compose, self).__init__()

        self.temperature = temperature
        if origininit <= 2:
            ''' the last argument is only for predefined v2, refering to nnunet '''
            self.scale = SampleNet(6, origininit, self.temperature, 0.2)
            self.rotF = SampleNet(5, origininit, self.temperature, 0.1)
            self.rotS = SampleNet(5, origininit, self.temperature, 0.1)
            self.rotL = SampleNet(5, origininit, self.temperature, 0.1)
            self.mirrorS = SampleNet(2, origininit, self.temperature, 0.5)
            self.mirrorF = SampleNet(2, origininit, self.temperature, 0.5)
            self.mirrorA = SampleNet(2, origininit, self.temperature, 0.5)

            self.gamma = SampleNet(4, origininit, self.temperature, 0.3)
            self.invgamma = SampleNet(4, origininit, self.temperature, 0.1)
            self.badd = SampleNet(4, origininit, self.temperature, 0.05)
            self.bmul = SampleNet(4, origininit, self.temperature, 0.15)
            self.contrast = SampleNet(4, origininit, self.temperature, 0.15)
            
            self.sharpen = SampleNet(7, origininit, self.temperature, 0.2)
            self.noise = SampleNet(4, origininit, self.temperature, 0.1)
            self.simulow = SampleNet(4, origininit, self.temperature, 0.1)
        if origininit == 3:
            ''' the last argument is only for predefined v2DM, refering to DM '''
            self.scale = SampleNet(6, origininit, self.temperature, 0.2)
            self.rotF = SampleNet(5, origininit, self.temperature, 0.2)
            self.rotS = SampleNet(5, origininit, self.temperature, 0.2)
            self.rotL = SampleNet(5, origininit, self.temperature, 0.2)
            self.mirrorS = SampleNet(2, origininit, self.temperature, 0.5)
            self.mirrorF = SampleNet(2, origininit, self.temperature, 0.05)
            self.mirrorA = SampleNet(2, origininit, self.temperature, 0.05)

            self.gamma = SampleNet(4, origininit, self.temperature, 0.05)
            self.invgamma = SampleNet(4, origininit, self.temperature, 0.05)
            self.badd = SampleNet(4, origininit, self.temperature, 0.15)
            self.bmul = SampleNet(4, origininit, self.temperature, 0.05)
            self.contrast = SampleNet(4, origininit, self.temperature, 0.05)
            
            self.sharpen = SampleNet(7, origininit, self.temperature, 0.05)
            self.noise = SampleNet(4, origininit, self.temperature, 0.05)
            self.simulow = SampleNet(4, origininit, self.temperature, 0.05)
        if origininit == 4:
            ''' this is the case I do not want any large trasformations at the begining '''
            self.scale = SampleNet(6, origininit, self.temperature, 0.1)
            self.rotF = SampleNet(5, origininit, self.temperature, 0.1)
            self.rotS = SampleNet(5, origininit, self.temperature, 0.1)
            self.rotL = SampleNet(5, origininit, self.temperature, 0.1)
            self.mirrorS = SampleNet(2, origininit, self.temperature, 0.1)
            self.mirrorF = SampleNet(2, origininit, self.temperature, 0.1)
            self.mirrorA = SampleNet(2, origininit, self.temperature, 0.1)

            self.gamma = SampleNet(4, origininit, self.temperature, 0.1)
            self.invgamma = SampleNet(4, origininit, self.temperature, 0.1)
            self.badd = SampleNet(4, origininit, self.temperature, 0.1)
            self.bmul = SampleNet(4, origininit, self.temperature, 0.1)
            self.contrast = SampleNet(4, origininit, self.temperature, 0.1)
            
            self.sharpen = SampleNet(7, origininit, self.temperature, 0.1)
            self.noise = SampleNet(4, origininit, self.temperature, 0.1)
            self.simulow = SampleNet(4, origininit, self.temperature, 0.1)
        if origininit == 5:
            ''' furhter decreased. '''
            self.scale = SampleNet(6, origininit, self.temperature, 0.05)
            self.rotF = SampleNet(5, origininit, self.temperature, 0.05)
            self.rotS = SampleNet(5, origininit, self.temperature, 0.05)
            self.rotL = SampleNet(5, origininit, self.temperature, 0.05)
            self.mirrorS = SampleNet(2, origininit, self.temperature, 0.05)
            self.mirrorF = SampleNet(2, origininit, self.temperature, 0.05)
            self.mirrorA = SampleNet(2, origininit, self.temperature, 0.05)

            self.gamma = SampleNet(4, origininit, self.temperature, 0.05)
            self.invgamma = SampleNet(4, origininit, self.temperature, 0.05)
            self.badd = SampleNet(4, origininit, self.temperature, 0.05)
            self.bmul = SampleNet(4, origininit, self.temperature, 0.05)
            self.contrast = SampleNet(4, origininit, self.temperature, 0.05)
            
            self.sharpen = SampleNet(7, origininit, self.temperature, 0.05)
            self.noise = SampleNet(4, origininit, self.temperature, 0.05)
            self.simulow = SampleNet(4, origininit, self.temperature, 0.05)

    def forward(self, samplenum, indsel):
        ind = torch.zeros(15, samplenum)
        if indsel is None:
            outputsel_scale, ind_scale,_ = self.scale(samplenum, indsel)
            outputsel_rotF, ind_rotF,_ = self.rotF(samplenum, indsel)
            outputsel_rotS, ind_rotS,_ = self.rotS(samplenum, indsel)
            outputsel_rotL, ind_rotL,_ = self.rotL(samplenum, indsel)
            outputsel_mirrorS, ind_mirrorS,_ = self.mirrorS(samplenum, indsel)
            outputsel_mirrorF, ind_mirrorF,_ = self.mirrorF(samplenum, indsel)
            outputsel_mirrorA, ind_mirrorA,_ = self.mirrorA(samplenum, indsel)

            outputsel_gamma, ind_gamma,_ = self.gamma(samplenum, indsel)
            outputsel_invgamma, ind_invgamma,_ = self.invgamma(samplenum, indsel)
            outputsel_badd, ind_badd,_ = self.badd(samplenum, indsel)
            outputsel_bmul, ind_bmul,_ = self.bmul(samplenum, indsel)
            outputsel_contrast, ind_contrast,_ = self.contrast(samplenum, indsel)

            outputsel_sharpen, ind_sharpen,_ = self.sharpen(samplenum, indsel)
            outputsel_noise, ind_noise,_ = self.noise(samplenum, indsel)
            outputsel_simulow, ind_simulow,_ = self.simulow(samplenum, indsel)
        else:
            outputsel_scale, ind_scale,_ = self.scale(samplenum, indsel[0, :])
            outputsel_rotF, ind_rotF,_ = self.rotF(samplenum, indsel[1, :])
            outputsel_rotS, ind_rotS,_ = self.rotS(samplenum, indsel[2, :])
            outputsel_rotL, ind_rotL,_ = self.rotL(samplenum, indsel[3, :])
            outputsel_mirrorS, ind_mirrorS,_ = self.mirrorS(samplenum, indsel[4, :])
            outputsel_mirrorF, ind_mirrorF,_ = self.mirrorF(samplenum, indsel[5, :])
            outputsel_mirrorA, ind_mirrorA,_ = self.mirrorA(samplenum, indsel[6, :])

            outputsel_gamma, ind_gamma,_ = self.gamma(samplenum, indsel[7, :])
            outputsel_invgamma, ind_invgamma,_ = self.invgamma(samplenum, indsel[8, :])
            outputsel_badd, ind_badd,_ = self.badd(samplenum, indsel[9, :])
            outputsel_bmul, ind_bmul,_ = self.bmul(samplenum, indsel[10, :])
            outputsel_contrast, ind_contrast,_ = self.contrast(samplenum, indsel[11, :])

            outputsel_sharpen, ind_sharpen,_ = self.sharpen(samplenum, indsel[12, :])
            outputsel_noise, ind_noise,_ = self.noise(samplenum, indsel[13, :])
            outputsel_simulow, ind_simulow,_ = self.simulow(samplenum, indsel[14, :])
        
        outputsel = []
        for klist in range(samplenum):
            outputsel.append(outputsel_scale[klist] * outputsel_rotF[klist] * outputsel_rotS[klist] * outputsel_rotL[klist] \
                * outputsel_mirrorS[klist] * outputsel_mirrorF[klist] * outputsel_mirrorA[klist] \
                * outputsel_gamma[klist] * outputsel_invgamma[klist] * outputsel_badd[klist] * outputsel_bmul[klist] * outputsel_contrast[klist] \
                * outputsel_sharpen[klist] * outputsel_noise[klist] * outputsel_simulow[klist])

        ind[0, :] = ind_scale
        ind[1, :] = ind_rotF
        ind[2, :] = ind_rotS
        ind[3, :] = ind_rotL
        ind[4, :] = ind_mirrorS
        ind[5, :] = ind_mirrorF
        ind[6, :] = ind_mirrorA
        
        ind[7, :] = ind_gamma
        ind[8, :] = ind_invgamma
        ind[9, :] = ind_badd
        ind[10, :] = ind_bmul
        ind[11, :] = ind_contrast
        
        ind[12, :] = ind_sharpen
        ind[13, :] = ind_noise
        ind[14, :] = ind_simulow
          
        return outputsel, ind
