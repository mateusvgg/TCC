from typing import Union
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

from projections_dataclasses import PairProjections


class L2pooling(nn.Module):

    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 ) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        # shape = [channels, 1, filter_size-2, filter_size-2]
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels, 1, 1, 1))) # type: ignore


    def forward(self, input):
        input = input**2
        out = F.conv2d(
            input,
            self.filter, # type: ignore
            stride=self.stride,
            padding=self.padding, # type: ignore
            groups=input.shape[1]
        )
        return (out+1e-12).sqrt()



class DISTS(torch.nn.Module):

    def __init__(self, load_weights: bool = True, use_pooling: bool = False):
        super(DISTS, self).__init__()

        self.use_pooling = use_pooling

        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x]) # type: ignore
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x]) # type: ignore
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x]) # type: ignore
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x]) # type: ignore
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x]) # type: ignore
    
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)) # shape = [1, 3, 1, 1]
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)) # shape = [1, 3, 1, 1]

        # number of feature maps at each stage
        self.chns = [3, 64, 128, 256, 512, 512]
        
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))) # shape = [1, 1475, 1, 1]
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))) # shape = [1, 1475, 1, 1]
        
        self.alpha.data.normal_(0.1, 0.01) # type: ignore
        self.beta.data.normal_(0.1, 0.01) # type: ignore

        if load_weights:
            weights = torch.load('weights.pt')
            self.alpha.data = weights['alpha']
            self.beta.data = weights['beta']

        
    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        # original image concatenated with the features -> injective funtion
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]


    def forward(
        self,
        pair_projections: PairProjections,
        require_grad: bool = False,
        batch_average: bool = False
    ) -> Union[float, list[float]]:
        view_scores = []
        for view_ref, view_deg in zip(pair_projections.ref, pair_projections.deg):
            if require_grad:
                feats0 = self.forward_once(view_ref)
                feats1 = self.forward_once(view_deg)
            else:
                with torch.no_grad():
                    feats0 = self.forward_once(view_ref)
                    feats1 = self.forward_once(view_deg)
            dist1 = 0 
            dist2 = 0 
            c1 = 1e-6
            c2 = 1e-6
            w_sum = self.alpha.sum() + self.beta.sum() # type: ignore
            # len(6), shape of elements = [1, num_filters, 1, 1]
            alpha = torch.split(self.alpha/w_sum, self.chns, dim=1) 
            beta = torch.split(self.beta/w_sum, self.chns, dim=1)
            for k in range(len(self.chns)):
                # feats0[k] has shape (1, num_filters, feat_map.shape[0], feat_map.shape[1])

                x_mean = feats0[k].mean([2, 3], keepdim=True) # shape = [1, num_filters, 1, 1]
                y_mean = feats1[k].mean([2, 3], keepdim=True) # shape = [1, num_filters, 1, 1]
                S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1) # shape = [1, num_filters, 1, 1]
                dist1 = dist1+(alpha[k]*S1).sum(1, keepdim=True) # shape = [1, 1, 1, 1]

                x_var = ((feats0[k]-x_mean)**2).mean([2, 3], keepdim=True) # shape = [1, num_filters, 1, 1]
                y_var = ((feats1[k]-y_mean)**2).mean([2, 3], keepdim=True) # shape = [1, num_filters, 1, 1]
                xy_cov = (feats0[k]*feats1[k]).mean([2, 3], keepdim=True) - x_mean*y_mean # shape = [1, num_filters, 1, 1]
                S2 = (2*xy_cov+c2)/(x_var+y_var+c2) # shape = [1, num_filters, 1, 1]
                dist2 = dist2+(beta[k]*S2).sum(1, keepdim=True) # shape = [1, 1, 1, 1]

            score = 1 - (dist1+dist2).squeeze() # type: ignore
            if batch_average:
                view_scores.append(score.mean())
            else:
                view_scores.append(score)

        if self.use_pooling:
            final_score = sum(view_scores)/len(view_scores)
            return final_score     
        return view_scores