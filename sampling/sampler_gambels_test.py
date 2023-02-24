import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SampleNet(nn.Module):
    def __init__(self, augpool, origininit, temperature, initprob = 0.5, sharpen = False):
        super(SampleNet, self).__init__()
        self.temperature = temperature
        if origininit == 0:
            # uniform initial
            alphas_augment = torch.zeros(augpool).cuda()
            self.outwn = nn.Parameter(alphas_augment)
        if origininit == 1:
            # predefined policy v1, Identity always have 0.5 prob
            alphas_origin = torch.ones(1).cuda()
            alphas_augment = torch.log(torch.exp(torch.ones(augpool - 1))/(augpool-1)).cuda()
            self.outwn = nn.Parameter(torch.cat((alphas_origin, alphas_augment)))
        if origininit >= 2:
            # predefined policy v2, similar to nnunet policy
            if type(initprob) is list:
                ''' this is for test augmentation policy '''
                if origininit == 2:
                    ## this init prob contains the augment index having probability 10%.
                    setprob = 0.1
                    tnum = len(initprob)
                    alphas = torch.ones(augpool).cuda()
                    for kcount in range(augpool):
                        if kcount not in initprob:
                            alphas[kcount] = torch.log( (1 - setprob * tnum) * torch.exp(torch.ones(1)) / (augpool - tnum) / setprob ).cuda()
                    self.outwn = nn.Parameter(alphas)
                if origininit == 3:
                    ## 0.5 + 0.05 * 7 + others.
                    setprob = 0.05
                    tnum = len(initprob) - 1
                    alphas = torch.ones(augpool).cuda()
                    for kcount in range(augpool):
                        if kcount not in initprob:
                            alphas[kcount] = torch.log( (0.5 - setprob * tnum) * torch.exp(torch.ones(1)) / (0.5 * (augpool - tnum - 1) )).cuda()
                    initprob.remove(0)
                    for kcount in range(augpool):
                        if kcount in initprob:
                            alphas[kcount] = torch.log( setprob * torch.exp(torch.ones(1)) / 0.5).cuda()
                    self.outwn = nn.Parameter(alphas)
                if origininit == 4:
                    ## origin 20%, fliping 20% * 3 + others
                    initprob = initprob[:4]
                    setprob = 0.2
                    tnum = len(initprob)
                    alphas = torch.ones(augpool).cuda()
                    for kcount in range(augpool):
                        if kcount not in initprob:
                            alphas[kcount] = torch.log( (1 - setprob * tnum) * torch.exp(torch.ones(1)) / (augpool - tnum) / setprob ).cuda()
                    self.outwn = nn.Parameter(alphas)
                if origininit == 5:
                    ## 0.5 + 0.1 * 3 + others.
                    setprob = 0.1
                    initprob = initprob[:4]
                    tnum = len(initprob) - 1
                    alphas = torch.ones(augpool).cuda()
                    for kcount in range(augpool):
                        if kcount not in initprob:
                            alphas[kcount] = torch.log( (0.5 - setprob * tnum) * torch.exp(torch.ones(1)) / (0.5 * (augpool - tnum - 1) )).cuda()
                    initprob.remove(0)
                    for kcount in range(augpool):
                        if kcount in initprob:
                            alphas[kcount] = torch.log( setprob * torch.exp(torch.ones(1)) / 0.5).cuda()
                    self.outwn = nn.Parameter(alphas)
                if origininit == 6:
                    ## 0.4 + 0.4 + others.
                    setprob = 0.4
                    initprob = initprob[:2]
                    tnum = len(initprob)
                    alphas = torch.ones(augpool).cuda()
                    for kcount in range(augpool):
                        if kcount not in initprob:
                            alphas[kcount] = torch.log( (1 - setprob * tnum) * torch.exp(torch.ones(1)) / (augpool - tnum) / setprob ).cuda()
                    self.outwn = nn.Parameter(alphas)
            else:
                ''' this is for training augmentation policy '''
                # the init identity prob is (1 - initprob)
                alphas_origin = torch.ones(1).cuda()
                if sharpen:
                    ## nn default does not have sharpen...so make it have smaller probablity
                    initprob_blur = initprob * 0.75
                    alphas_augment_blur = torch.log( initprob_blur * torch.exp(torch.ones((augpool - 1)//2)) / ((augpool-1)//2) / (1-initprob_blur) ).cuda()
                    initprob_sharpen = initprob * 0.25
                    alphas_augment_sharpen = torch.log( initprob_sharpen * torch.exp(torch.ones((augpool - 1)//2)) / ((augpool-1)//2) / (1-initprob_sharpen) ).cuda()
                    self.outwn = nn.Parameter(torch.cat((alphas_origin, alphas_augment_blur, alphas_augment_sharpen)))
                else:
                    alphas_augment = torch.log( initprob * torch.exp(torch.ones(augpool - 1)) / (augpool-1) / (1-initprob) ).cuda()
                    self.outwn = nn.Parameter(torch.cat((alphas_origin, alphas_augment)))

    def forward(self, samplenum, indsel):
        ProbabilityConcat = self.outwn.repeat(samplenum, 1)
        # I am not sure if it is necessary here. log(softmax(o)) or just o.
        # ProbabilityConcat = F.softmax(ProbabilityConcat, dim=-1).log()
        outputsel, ind, realprob = self.gumbel_softmax(ProbabilityConcat, self.temperature, indsel)
        return outputsel, ind, realprob

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, indsel):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        if indsel is None:
            y = self.gumbel_softmax_sample(logits, temperature)
            shape = y.size()
            _, ind = y.max(dim=-1)
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            output = (y_hard - y).detach() + y
            outputsel = []
            for k in range(ind.shape[0]):
                outputsel.append(output[k, ind[k]])
        else:
            '''I should be careful here, it draws fake samples as indsel'''
            '''I have to make sure that the chosen indsel has the largest probability'''
            shape = logits.size()
            logtis_uq = logits[0:1, :]
            ind = indsel
            y = torch.zeros(shape)
            y = y.cuda()

            for kbatch in range(logits.shape[0]):
                indk = indsel[kbatch]
                
                while 1 > 0:
                    y_trial = self.gumbel_softmax_sample(logtis_uq, temperature)
                    _, indtrial = y_trial.max(dim=-1)
                    if indtrial == indk:
                        y[kbatch, :] = y_trial
                        break
                
            y_hard = torch.zeros_like(y).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            output = (y_hard - y).detach() + y
            outputsel = []
            for k in range(ind.shape[0]):
                outputsel.append(output[k, ind[k]])

        return outputsel, ind, y